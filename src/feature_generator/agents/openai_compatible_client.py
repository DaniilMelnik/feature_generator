"""LLM backend for any server exposing an OpenAI-compatible /chat/completions
endpoint (Groq, a local Ollama/vLLM/LM Studio server, etc.) -- a free-or-local
alternative to ``llm_client.LLMClient`` (Anthropic), used to compare feature
quality across backends on the same pipeline.

``OpenAICompatibleLLMClient`` satisfies the same duck-typed interface as
``LLMClient`` (``call(tier, *, system_prompt, messages, output_schema,
max_tokens) -> LLMResponse`` plus a ``total_usage: LLMUsage`` attribute), so
every caller in ``orchestration/graph.py`` and the three role agents
(hypothesize/codegen/leakage_review) works against either backend unmodified.

The one real translation problem: ``hypothesis_agent.py``/``codegen_agent.py``
build up a long-lived ``messages`` list across turns, appending each
response's ``raw_content`` verbatim as the next assistant turn -- and that
shape is Anthropic's content-block list (``[{"type": "text", "text": ...}]``),
regardless of which backend produced it, since the calling code has no
backend-awareness. So this client accepts that same block-list shape on the
way in (flattening it to a plain string for the OpenAI-style request) and
always re-wraps its own response back into that shape on the way out --
conversation history round-trips identically whichever backend produced each
turn.

Fields on ``ModelTierEntry`` that only make sense for Anthropic (``effort``,
``thinking``, ``use_compaction``, ``use_context_editing``) have no equivalent
here and are ignored; only ``tier.model`` and ``tier.max_output_tokens`` are
read.
"""

from __future__ import annotations

import json
from typing import Any

import openai

from feature_generator.agents.llm_client import LLMResponse, LLMResponseError, LLMUsage
from feature_generator.config import ModelTierEntry


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def _to_chat_messages(system_prompt: str, messages: list[dict]) -> list[dict]:
    chat_messages = [{"role": "system", "content": system_prompt}]
    for message in messages:
        chat_messages.append({"role": message["role"], "content": _flatten_content(message["content"])})
    return chat_messages


class OpenAICompatibleLLMClient:
    def __init__(self, *, base_url: str, api_key: str, max_retries: int = 8) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.total_usage = LLMUsage()

    def call(
        self,
        tier: ModelTierEntry,
        *,
        system_prompt: str,
        messages: list[dict],
        output_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        response_format: dict | None = None
        if output_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": output_schema},
            }

        request_kwargs: dict[str, Any] = {
            "model": tier.model,
            "messages": _to_chat_messages(system_prompt, messages),
            # The current OpenAI field is `max_completion_tokens`, but it's
            # silently ignored by Ollama's compat layer (confirmed live: a
            # response ran to its natural end regardless of the cap) while
            # the deprecated `max_tokens` is honored correctly there and is
            # also accepted by Groq -- `max_tokens` is the field with the
            # broadest actual support across OpenAI-compatible backends.
            "max_tokens": max_tokens if max_tokens is not None else tier.max_output_tokens,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**request_kwargs)

        usage = LLMUsage()
        if response.usage is not None:
            usage.input_tokens = response.usage.prompt_tokens or 0
            usage.output_tokens = response.usage.completion_tokens or 0
        self.total_usage.input_tokens += usage.input_tokens
        self.total_usage.output_tokens += usage.output_tokens

        choice = response.choices[0]
        text = choice.message.content or ""
        stop_reason = choice.finish_reason or "stop"

        parsed = None
        if output_schema is not None:
            if not text:
                raise LLMResponseError(
                    f"expected structured JSON output but got no text content "
                    f"(finish_reason={stop_reason!r} -- a truncated response due to max_tokens is "
                    "the most common cause)",
                    stop_reason,
                )
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMResponseError(
                    f"structured output was not valid JSON (finish_reason={stop_reason!r}): {exc}",
                    stop_reason,
                ) from exc

        return LLMResponse(
            text=text,
            parsed=parsed,
            raw_content=[{"type": "text", "text": text}],
            usage=usage,
            stop_reason=stop_reason,
        )

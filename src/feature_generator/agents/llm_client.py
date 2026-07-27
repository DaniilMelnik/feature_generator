"""Thin wrapper around the Anthropic SDK used by every LLM-driven node.

Calls the SDK directly (not a LangChain chat-model abstraction) so this
module has precise control over the things that matter for a long,
many-call pipeline run: prompt-cache placement/TTL, the `effort` parameter,
adaptive thinking, structured JSON outputs, and the compaction /
context-editing betas that keep a long-lived per-role conversation from
blowing its context window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from feature_generator.config import ModelTierEntry

COMPACTION_BETA = "compact-2026-01-12"
CONTEXT_EDITING_BETA = "context-management-2025-06-27"

# Soft threshold (in tokens) above which a role's conversation is routed
# through the compaction/context-editing beta path. Comfortably under the
# API's own ~150K default compaction trigger.
DEFAULT_CONTEXT_TOKEN_THRESHOLD = 100_000

DEFAULT_MAX_TOKENS = 8000
CACHE_TTL = "1h"  # sandbox execution + CatBoost training between turns can run
# minutes-to-tens-of-minutes; the default 5-minute ephemeral TTL would
# cold-miss on almost every turn.


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0


@dataclass
class LLMResponse:
    text: str
    parsed: Any
    raw_content: list[dict]  # response.content, appendable verbatim as the next assistant turn
    usage: LLMUsage = field(default_factory=LLMUsage)
    stop_reason: str = "end_turn"


def build_request(
    tier: ModelTierEntry,
    *,
    system_prompt: str,
    messages: list[dict],
    output_schema: dict | None,
    max_tokens: int,
) -> dict:
    """Pure request-construction, independent of the SDK call itself -- kept
    separate so its shape (cache placement, effort, thinking, structured
    output) can be unit-tested without a live API or a mocked client.
    """
    system_blocks = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}
    ]
    thinking: dict = {"type": "adaptive"} if tier.thinking == "adaptive" else {"type": "disabled"}
    output_config: dict = {"effort": tier.effort}
    if output_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": output_schema}

    return {
        "model": tier.model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": messages,
        "thinking": thinking,
        "output_config": output_config,
    }


def estimate_tokens_heuristic(messages: list[dict]) -> int:
    """Crude fallback (chars/4) used only if a real count_tokens() call
    itself fails (e.g. transient network issue) -- never the primary path.
    """
    return sum(len(json.dumps(m)) for m in messages) // 4


def _block_to_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return dict(block)


class LLMClient:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        context_token_threshold: int = DEFAULT_CONTEXT_TOKEN_THRESHOLD,
    ) -> None:
        self.client = client if client is not None else anthropic.Anthropic()
        self.context_token_threshold = context_token_threshold
        self.total_usage = LLMUsage()

    def _count_tokens(self, tier: ModelTierEntry, system_prompt: str, messages: list[dict]) -> int:
        try:
            result = self.client.messages.count_tokens(
                model=tier.model,
                system=system_prompt,
                messages=messages,
            )
            return result.input_tokens
        except Exception:
            return estimate_tokens_heuristic(messages)

    def _beta_requirements(
        self, tier: ModelTierEntry, system_prompt: str, messages: list[dict]
    ) -> tuple[list[str], dict] | None:
        if not (tier.use_compaction or tier.use_context_editing):
            return None
        if self._count_tokens(tier, system_prompt, messages) < self.context_token_threshold:
            return None
        if tier.use_compaction:
            return [COMPACTION_BETA], {"edits": [{"type": "compact_20260112"}]}
        return (
            [CONTEXT_EDITING_BETA],
            {"edits": [{"type": "clear_tool_uses_20250919", "clear_tool_inputs": True}]},
        )

    def call(
        self,
        tier: ModelTierEntry,
        *,
        system_prompt: str,
        messages: list[dict],
        output_schema: dict | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        request = build_request(
            tier, system_prompt=system_prompt, messages=messages, output_schema=output_schema,
            max_tokens=max_tokens,
        )

        beta_reqs = self._beta_requirements(tier, system_prompt, messages)
        if beta_reqs is not None:
            betas, context_management = beta_reqs
            response = self.client.beta.messages.create(
                **request, betas=betas, context_management=context_management
            )
        else:
            response = self.client.messages.create(**request)

        return self._to_llm_response(response, output_schema)

    def _to_llm_response(self, response: Any, output_schema: dict | None) -> LLMResponse:
        usage = LLMUsage()
        usage.add(response.usage)
        self.total_usage.add(response.usage)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        parsed = json.loads(text) if (output_schema is not None and text) else None
        raw_content = [_block_to_dict(block) for block in response.content]

        return LLMResponse(
            text=text,
            parsed=parsed,
            raw_content=raw_content,
            usage=usage,
            stop_reason=getattr(response, "stop_reason", "end_turn"),
        )

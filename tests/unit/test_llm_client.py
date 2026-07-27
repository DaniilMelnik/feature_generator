"""Unit tests for the Anthropic SDK wrapper. Fully mocked -- no real API
calls, no ANTHROPIC_API_KEY needed. Focuses on request SHAPE (cache
placement, effort, thinking, structured output, compaction/context-editing
dispatch) since that's what a long-running pipeline's cost profile depends on.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from feature_generator.agents.llm_client import (
    CACHE_TTL,
    COMPACTION_BETA,
    CONTEXT_EDITING_BETA,
    LLMClient,
    build_request,
    estimate_tokens_heuristic,
)
from feature_generator.config import ModelTierEntry


# --- lightweight fakes for the Anthropic SDK surface we touch -------------


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"

    def model_dump(self) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass
class FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 5
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeResponse:
    content: list = field(default_factory=list)
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"


class FakeMessagesResource:
    def __init__(self, response: FakeResponse | None = None, count_tokens_value: int = 100) -> None:
        self.response = response or FakeResponse(content=[FakeTextBlock('{"ok": true}')])
        self.count_tokens_value = count_tokens_value
        self.create_calls: list[dict] = []
        self.count_tokens_calls: list[dict] = []

    def create(self, **kwargs) -> FakeResponse:
        self.create_calls.append(kwargs)
        return self.response

    def count_tokens(self, **kwargs) -> SimpleNamespace:
        self.count_tokens_calls.append(kwargs)
        return SimpleNamespace(input_tokens=self.count_tokens_value)


class FakeAnthropicClient:
    def __init__(self, response: FakeResponse | None = None, count_tokens_value: int = 100) -> None:
        self.messages = FakeMessagesResource(response, count_tokens_value)
        self.beta = SimpleNamespace(messages=FakeMessagesResource(response, count_tokens_value))


def _tier(**overrides) -> ModelTierEntry:
    defaults = dict(model="claude-sonnet-5", effort="high", thinking="adaptive")
    defaults.update(overrides)
    return ModelTierEntry(**defaults)


# --- build_request (pure) --------------------------------------------------


def test_build_request_places_cache_control_on_system_block() -> None:
    request = build_request(
        _tier(), system_prompt="SYS", messages=[], output_schema=None, max_tokens=100
    )
    assert request["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}
    ]


def test_build_request_uses_adaptive_thinking_by_default() -> None:
    request = build_request(_tier(thinking="adaptive"), system_prompt="S", messages=[], output_schema=None, max_tokens=10)
    assert request["thinking"] == {"type": "adaptive"}


def test_build_request_disables_thinking_when_configured() -> None:
    request = build_request(_tier(thinking="disabled"), system_prompt="S", messages=[], output_schema=None, max_tokens=10)
    assert request["thinking"] == {"type": "disabled"}


def test_build_request_sets_effort_from_tier() -> None:
    request = build_request(_tier(effort="xhigh"), system_prompt="S", messages=[], output_schema=None, max_tokens=10)
    assert request["output_config"]["effort"] == "xhigh"


def test_build_request_includes_structured_output_schema_when_provided() -> None:
    schema = {"type": "object", "properties": {}}
    request = build_request(_tier(), system_prompt="S", messages=[], output_schema=schema, max_tokens=10)
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": schema}


def test_build_request_omits_format_when_no_schema() -> None:
    request = build_request(_tier(), system_prompt="S", messages=[], output_schema=None, max_tokens=10)
    assert "format" not in request["output_config"]


def test_estimate_tokens_heuristic_scales_with_message_size() -> None:
    small = estimate_tokens_heuristic([{"role": "user", "content": "hi"}])
    large = estimate_tokens_heuristic([{"role": "user", "content": "hi " * 10_000}])
    assert large > small


# --- LLMClient.call dispatch -------------------------------------------------


def test_call_uses_non_beta_path_when_no_compaction_or_context_editing() -> None:
    fake = FakeAnthropicClient()
    client = LLMClient(fake)

    response = client.call(
        _tier(),
        system_prompt="S",
        messages=[{"role": "user", "content": "hi"}],
        output_schema={"type": "object"},
    )

    assert len(fake.messages.create_calls) == 1
    assert len(fake.beta.messages.create_calls) == 0
    assert response.parsed == {"ok": True}
    assert response.text == '{"ok": true}'
    assert response.stop_reason == "end_turn"


def test_call_uses_non_beta_path_when_below_context_threshold() -> None:
    fake = FakeAnthropicClient(count_tokens_value=500)  # well under threshold
    client = LLMClient(fake, context_token_threshold=100_000)
    tier = _tier(use_compaction=True)

    client.call(tier, system_prompt="S", messages=[{"role": "user", "content": "hi"}])

    assert len(fake.messages.create_calls) == 1
    assert len(fake.beta.messages.create_calls) == 0


def test_call_routes_through_compaction_beta_above_threshold() -> None:
    fake = FakeAnthropicClient(count_tokens_value=200_000)
    client = LLMClient(fake, context_token_threshold=100_000)
    tier = _tier(use_compaction=True)

    client.call(tier, system_prompt="S", messages=[{"role": "user", "content": "hi"}])

    assert len(fake.beta.messages.create_calls) == 1
    call = fake.beta.messages.create_calls[0]
    assert call["betas"] == [COMPACTION_BETA]
    assert call["context_management"] == {"edits": [{"type": "compact_20260112"}]}


def test_call_routes_through_context_editing_beta_above_threshold() -> None:
    fake = FakeAnthropicClient(count_tokens_value=200_000)
    client = LLMClient(fake, context_token_threshold=100_000)
    tier = _tier(use_context_editing=True)

    client.call(tier, system_prompt="S", messages=[{"role": "user", "content": "hi"}])

    assert len(fake.beta.messages.create_calls) == 1
    call = fake.beta.messages.create_calls[0]
    assert call["betas"] == [CONTEXT_EDITING_BETA]
    assert call["context_management"]["edits"][0]["type"] == "clear_tool_uses_20250919"


def test_call_never_applies_both_compaction_and_context_editing() -> None:
    # tier config should only ever set one of these, but confirm compaction
    # wins deterministically if both were somehow set
    fake = FakeAnthropicClient(count_tokens_value=200_000)
    client = LLMClient(fake, context_token_threshold=100_000)
    tier = _tier(use_compaction=True, use_context_editing=True)

    client.call(tier, system_prompt="S", messages=[{"role": "user", "content": "hi"}])

    call = fake.beta.messages.create_calls[0]
    assert call["betas"] == [COMPACTION_BETA]


def test_count_tokens_falls_back_to_heuristic_on_error() -> None:
    fake = FakeAnthropicClient()

    def _raise(**kwargs):
        raise RuntimeError("network error")

    fake.messages.count_tokens = _raise
    client = LLMClient(fake, context_token_threshold=1)  # tiny threshold forces the beta check
    tier = _tier(use_compaction=True)

    # must not raise even though count_tokens() failed
    client.call(tier, system_prompt="S", messages=[{"role": "user", "content": "hi"}])
    assert len(fake.beta.messages.create_calls) == 1  # heuristic count exceeded the tiny threshold


def test_response_usage_is_accumulated_across_calls() -> None:
    fake = FakeAnthropicClient(
        response=FakeResponse(content=[FakeTextBlock("hi")], usage=FakeUsage(input_tokens=10, output_tokens=20))
    )
    client = LLMClient(fake)

    client.call(_tier(), system_prompt="S", messages=[])
    client.call(_tier(), system_prompt="S", messages=[])

    assert client.total_usage.input_tokens == 20
    assert client.total_usage.output_tokens == 40


def test_raw_content_preserves_full_blocks_for_replay() -> None:
    fake = FakeAnthropicClient(response=FakeResponse(content=[FakeTextBlock("hello world")]))
    client = LLMClient(fake)

    response = client.call(_tier(), system_prompt="S", messages=[])

    assert response.raw_content == [{"type": "text", "text": "hello world"}]


def test_parsed_is_none_when_no_output_schema_requested() -> None:
    fake = FakeAnthropicClient(response=FakeResponse(content=[FakeTextBlock("plain text reply")]))
    client = LLMClient(fake)

    response = client.call(_tier(), system_prompt="S", messages=[], output_schema=None)

    assert response.parsed is None
    assert response.text == "plain text reply"

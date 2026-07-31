"""Unit tests for the Groq/OpenAI-compatible backend. Fully mocked -- no real
API calls, no GROQ_API_KEY needed. Focuses on the message-translation
boundary (Anthropic block-list <-> plain OpenAI chat messages), since that's
what lets hypothesis_agent.py/codegen_agent.py/leakage_reviewer_agent.py stay
backend-agnostic.
"""

from dataclasses import dataclass, field

import pytest

from feature_generator.agents.llm_client import LLMResponseError
from feature_generator.agents.openai_compatible_client import OpenAICompatibleLLMClient
from feature_generator.config import ModelTierEntry


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5


@dataclass
class FakeCompletion:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage | None = field(default_factory=FakeUsage)


class FakeCompletionsResource:
    def __init__(self, response: FakeCompletion) -> None:
        self.response = response
        self.create_calls: list[dict] = []

    def create(self, **kwargs) -> FakeCompletion:
        self.create_calls.append(kwargs)
        return self.response


class FakeChatResource:
    def __init__(self, response: FakeCompletion) -> None:
        self.completions = FakeCompletionsResource(response)


class FakeOpenAIClient:
    def __init__(self, response: FakeCompletion | None = None) -> None:
        default = FakeCompletion(choices=[FakeChoice(message=FakeMessage('{"ok": true}'))])
        self.chat = FakeChatResource(response or default)


def _tier(**overrides) -> ModelTierEntry:
    defaults = dict(model="openai/gpt-oss-120b", max_output_tokens=4000)
    defaults.update(overrides)
    return ModelTierEntry(**defaults)


def _client(response: FakeCompletion | None = None) -> tuple[OpenAICompatibleLLMClient, FakeOpenAIClient]:
    client = OpenAICompatibleLLMClient(base_url="https://api.groq.com/openai/v1", api_key="test-key")
    fake = FakeOpenAIClient(response)
    client.client = fake
    return client, fake


def test_call_prepends_system_prompt_and_flattens_string_content() -> None:
    client, fake = _client()

    client.call(_tier(), system_prompt="SYS", messages=[{"role": "user", "content": "hi"}])

    sent_messages = fake.chat.completions.create_calls[0]["messages"]
    assert sent_messages[0] == {"role": "system", "content": "SYS"}
    assert sent_messages[1] == {"role": "user", "content": "hi"}


def test_call_flattens_anthropic_block_list_assistant_turn_to_a_string() -> None:
    # exactly the shape hypothesis_agent.py/codegen_agent.py build for a prior
    # assistant turn: response.raw_content round-tripped as the next message.
    client, fake = _client()
    prior_turn = [{"role": "assistant", "content": [{"type": "text", "text": "previous reply"}]}]

    client.call(_tier(), system_prompt="SYS", messages=prior_turn)

    sent_messages = fake.chat.completions.create_calls[0]["messages"]
    assert sent_messages[-1] == {"role": "assistant", "content": "previous reply"}


def test_call_uses_tier_max_output_tokens_when_max_tokens_not_given() -> None:
    client, fake = _client()

    client.call(_tier(max_output_tokens=777), system_prompt="S", messages=[])

    assert fake.chat.completions.create_calls[0]["max_tokens"] == 777


def test_call_prefers_explicit_max_tokens_over_tier_default() -> None:
    client, fake = _client()

    client.call(_tier(max_output_tokens=777), system_prompt="S", messages=[], max_tokens=123)

    assert fake.chat.completions.create_calls[0]["max_tokens"] == 123


def test_call_builds_strict_json_schema_response_format_when_schema_given() -> None:
    client, fake = _client()
    schema = {"type": "object", "properties": {}}

    client.call(_tier(), system_prompt="S", messages=[], output_schema=schema)

    assert fake.chat.completions.create_calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "output", "strict": True, "schema": schema},
    }


def test_call_omits_response_format_when_no_schema() -> None:
    client, fake = _client()

    client.call(_tier(), system_prompt="S", messages=[])

    assert "response_format" not in fake.chat.completions.create_calls[0]


def test_call_parses_structured_output_json() -> None:
    response = FakeCompletion(choices=[FakeChoice(message=FakeMessage('{"hypotheses": []}'))])
    client, _ = _client(response)

    result = client.call(_tier(), system_prompt="S", messages=[], output_schema={"type": "object"})

    assert result.parsed == {"hypotheses": []}


def test_call_raw_content_wraps_response_text_as_a_text_block() -> None:
    response = FakeCompletion(choices=[FakeChoice(message=FakeMessage("hello world"))])
    client, _ = _client(response)

    result = client.call(_tier(), system_prompt="S", messages=[])

    assert result.raw_content == [{"type": "text", "text": "hello world"}]


def test_call_raises_llm_response_error_when_structured_output_has_no_text() -> None:
    response = FakeCompletion(choices=[FakeChoice(message=FakeMessage(None), finish_reason="length")])
    client, _ = _client(response)

    with pytest.raises(LLMResponseError) as exc_info:
        client.call(_tier(), system_prompt="S", messages=[], output_schema={"type": "object"})

    assert exc_info.value.stop_reason == "length"


def test_call_raises_llm_response_error_on_malformed_json() -> None:
    response = FakeCompletion(choices=[FakeChoice(message=FakeMessage("not valid json{"))])
    client, _ = _client(response)

    with pytest.raises(LLMResponseError):
        client.call(_tier(), system_prompt="S", messages=[], output_schema={"type": "object"})


def test_usage_is_accumulated_across_calls() -> None:
    response = FakeCompletion(
        choices=[FakeChoice(message=FakeMessage("hi"))], usage=FakeUsage(prompt_tokens=10, completion_tokens=20)
    )
    client, _ = _client(response)

    client.call(_tier(), system_prompt="S", messages=[])
    client.call(_tier(), system_prompt="S", messages=[])

    assert client.total_usage.input_tokens == 20
    assert client.total_usage.output_tokens == 40

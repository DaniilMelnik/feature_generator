from feature_generator.agents.llm_client import LLMResponse
from feature_generator.agents.utility_agent import summarize_kb_excerpt
from feature_generator.config import ModelTierEntry


class _FakeLLMClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def call(self, tier, *, system_prompt, messages, output_schema=None, max_tokens=8000):
        self.calls.append(
            {"tier": tier, "system_prompt": system_prompt, "messages": messages, "output_schema": output_schema}
        )
        return LLMResponse(text=self.text, parsed=None, raw_content=[{"type": "text", "text": self.text}])


def test_summarize_kb_excerpt_returns_llm_text() -> None:
    fake = _FakeLLMClient("compressed catalog text")
    tier = ModelTierEntry(model="claude-haiku-4-5", thinking="disabled")

    result = summarize_kb_excerpt(fake, tier, excerpt="a very long catalog " * 100, target_chars=500)

    assert result == "compressed catalog text"
    assert fake.calls[0]["output_schema"] is None
    assert "500 characters" in fake.calls[0]["messages"][0]["content"]

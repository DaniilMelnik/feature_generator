from feature_generator.agents.leakage_reviewer_agent import adversarial_review
from feature_generator.agents.llm_client import LLMResponse
from feature_generator.config import ModelTierEntry
from feature_generator.schemas import FeatureHypothesis, FeatureSpec


class _FakeLLMClient:
    def __init__(self, parsed: dict) -> None:
        self.parsed = parsed
        self.calls: list[dict] = []

    def call(self, tier, *, system_prompt, messages, output_schema=None, max_tokens=8000):
        self.calls.append({"messages": messages, "system_prompt": system_prompt, "max_tokens": max_tokens})
        return LLMResponse(
            text="...", parsed=self.parsed, raw_content=[{"type": "text", "text": "..."}], stop_reason="end_turn"
        )


def _pair(feature_name: str, requires_target_in_fit: bool = False) -> tuple[FeatureHypothesis, FeatureSpec]:
    hypothesis = FeatureHypothesis(
        iteration=0, rationale="r", description="d", input_columns=["Age"], input_tables=["t"],
        feature_type="ratio", proposed_by_model="claude-sonnet-5", requires_target_in_fit=requires_target_in_fit,
    )
    spec = FeatureSpec(
        hypothesis_id=hypothesis.id, feature_name=feature_name, module_source="FEATURE_NAME = 'x'\n",
        declared_input_columns=["Age"], declared_input_tables=["t"], output_dtype="float64",
        codegen_model="claude-sonnet-5",
    )
    return hypothesis, spec


def test_adversarial_review_returns_verdict_per_feature() -> None:
    fake = _FakeLLMClient(
        parsed={
            "verdicts": [
                {"feature_name": "f1", "verdict": "approve", "concerns": [], "confidence": 0.9},
                {"feature_name": "f2", "verdict": "reject", "concerns": ["leaks target"], "confidence": 0.95},
            ]
        }
    )
    tier = ModelTierEntry(model="claude-opus-5")
    batch = [_pair("f1"), _pair("f2")]

    verdicts, updated_messages, response = adversarial_review(fake, tier, messages=[], batch=batch)

    assert verdicts["f1"].verdict == "approve"
    assert verdicts["f2"].verdict == "reject"
    assert verdicts["f2"].concerns == ["leaks target"]
    assert verdicts["f1"].reviewer_model == "claude-opus-5"


def test_adversarial_review_short_circuits_on_empty_batch() -> None:
    fake = _FakeLLMClient(parsed={"verdicts": []})
    tier = ModelTierEntry(model="claude-opus-5")

    verdicts, updated_messages, response = adversarial_review(
        fake, tier, messages=[{"role": "user", "content": "prior"}], batch=[]
    )

    assert verdicts == {}
    assert updated_messages == [{"role": "user", "content": "prior"}]  # unchanged
    assert response is None
    assert fake.calls == []  # no LLM call made for an empty batch


def test_adversarial_review_includes_rejected_patterns_and_source_in_prompt() -> None:
    fake = _FakeLLMClient(parsed={"verdicts": [{"feature_name": "f1", "verdict": "approve", "concerns": [], "confidence": 0.5}]})
    tier = ModelTierEntry(model="claude-opus-5")
    batch = [_pair("f1", requires_target_in_fit=True)]

    adversarial_review(
        fake, tier, messages=[], batch=batch, rejected_patterns_excerpt="memoized globals are a common bug"
    )

    user_text = fake.calls[0]["messages"][-1]["content"]
    assert "memoized globals are a common bug" in user_text
    assert "requires_target_in_fit: True" in user_text
    assert "FEATURE_NAME = 'x'" in user_text


def test_adversarial_review_passes_max_output_tokens_from_tier() -> None:
    fake = _FakeLLMClient(parsed={"verdicts": []})
    tier = ModelTierEntry(model="openai/gpt-oss-120b", max_output_tokens=4000)

    adversarial_review(fake, tier, messages=[], batch=[_pair("f1")])

    assert fake.calls[0]["max_tokens"] == 4000

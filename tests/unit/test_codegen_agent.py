from datetime import datetime

from feature_generator.agents.codegen_agent import build_user_turn, generate_feature_module
from feature_generator.agents.llm_client import LLMResponse
from feature_generator.config import ModelTierEntry
from feature_generator.profiling.schemas import ColumnProfile, DataProfile, NumericStats, TableProfile
from feature_generator.schemas import FeatureHypothesis


class _FakeLLMClient:
    def __init__(self, parsed: dict) -> None:
        self.parsed = parsed
        self.calls: list[dict] = []

    def call(self, tier, *, system_prompt, messages, output_schema=None, max_tokens=8000):
        self.calls.append(
            {"tier": tier, "system_prompt": system_prompt, "messages": messages, "output_schema": output_schema}
        )
        return LLMResponse(
            text="...", parsed=self.parsed, raw_content=[{"type": "text", "text": "..."}], stop_reason="end_turn"
        )


def _profile() -> DataProfile:
    col = ColumnProfile(
        name="Age", source_table="passengers", dtype="float64", role="feature",
        n_missing=0, pct_missing=0.0, n_unique=50, is_constant=False,
        numeric_stats=NumericStats(min=0, max=79, mean=29, std=14, skew=0.1, kurtosis=-0.2, quantiles={}),
    )
    table = TableProfile(
        table_name="passengers", file_path="x.csv", n_rows=100, n_cols=1, role="primary", columns=[col]
    )
    return DataProfile(
        dataset_name="ds", tables=[table], target_column="Transported", target_table="passengers",
        id_columns=["PassengerId"], positive_rate=0.5, generated_at=datetime.utcnow(),
    )


def _hypothesis(**overrides) -> FeatureHypothesis:
    defaults = dict(
        iteration=0, rationale="r", description="d", input_columns=["Age"], input_tables=["passengers"],
        feature_type="ratio", proposed_by_model="claude-sonnet-5",
    )
    defaults.update(overrides)
    return FeatureHypothesis(**defaults)


def test_build_user_turn_includes_feedback_when_retrying() -> None:
    text = build_user_turn(_hypothesis(), _profile(), feedback="fix the import")
    assert "fix the import" in text
    assert "previous attempt was rejected" in text


def test_build_user_turn_omits_feedback_on_first_attempt() -> None:
    text = build_user_turn(_hypothesis(), _profile(), feedback=None)
    assert "previous attempt" not in text


def test_generate_feature_module_builds_feature_spec() -> None:
    fake = _FakeLLMClient(
        parsed={"feature_name": "age_bucket", "source_code": "FEATURE_NAME = 'age_bucket'\n", "output_dtype": "float64"}
    )
    tier = ModelTierEntry(model="claude-sonnet-5")
    hypothesis = _hypothesis()

    spec, updated_messages, response = generate_feature_module(
        fake, tier, messages=[], hypothesis=hypothesis, data_profile=_profile(), attempt=0
    )

    assert spec.feature_name == "age_bucket"
    assert spec.hypothesis_id == hypothesis.id
    assert spec.codegen_attempt == 0
    assert spec.codegen_model == "claude-sonnet-5"
    assert spec.declared_input_columns == ["Age"]
    assert len(updated_messages) == 2  # user + assistant


def test_generate_feature_module_increments_attempt_and_carries_feedback() -> None:
    fake = _FakeLLMClient(
        parsed={"feature_name": "age_bucket", "source_code": "x", "output_dtype": "float64"}
    )
    tier = ModelTierEntry(model="claude-sonnet-5")
    hypothesis = _hypothesis()

    spec, _, _ = generate_feature_module(
        fake, tier, messages=[{"role": "user", "content": "first"}], hypothesis=hypothesis,
        data_profile=_profile(), attempt=1, feedback="static check failed: denied import 'os'",
    )

    assert spec.codegen_attempt == 1
    last_user_message = fake.calls[0]["messages"][-1]
    assert "denied import 'os'" in last_user_message["content"]

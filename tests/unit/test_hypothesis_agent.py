from datetime import datetime

from feature_generator.agents.hypothesis_agent import propose_hypotheses, render_data_profile_summary
from feature_generator.agents.llm_client import LLMResponse
from feature_generator.config import ModelTierEntry
from feature_generator.profiling.schemas import ColumnProfile, DataProfile, NumericStats, TableProfile


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
        correlation_with_target=0.1,
    )
    table = TableProfile(
        table_name="passengers", file_path="x.csv", n_rows=100, n_cols=1, role="primary", columns=[col]
    )
    return DataProfile(
        dataset_name="ds", tables=[table], target_column="Transported", target_table="passengers",
        id_columns=["PassengerId"], positive_rate=0.5, generated_at=datetime.utcnow(),
    )


def test_render_data_profile_summary_includes_correlation_and_range() -> None:
    summary = render_data_profile_summary(_profile())
    assert "Age" in summary
    assert "corr_with_target=0.100" in summary
    assert "range=[0, 79]" in summary


def test_render_data_profile_summary_skips_ignored_columns() -> None:
    profile = _profile()
    profile.tables[0].columns[0].role = "ignore"
    summary = render_data_profile_summary(profile)
    assert "Age" not in summary


def test_propose_hypotheses_parses_response_into_feature_hypotheses() -> None:
    fake = _FakeLLMClient(
        parsed={
            "hypotheses": [
                {
                    "rationale": "r1", "description": "d1", "input_columns": ["Age"],
                    "input_tables": ["passengers"], "feature_type": "ratio",
                    "requires_target_in_fit": False,
                }
            ]
        }
    )
    tier = ModelTierEntry(model="claude-sonnet-5")

    hypotheses, updated_messages, response = propose_hypotheses(
        fake, tier, messages=[], data_profile=_profile(), kb_excerpt="none yet",
        feature_selection_summary="none yet", iteration=2, iterations_since_improvement=1,
        hypotheses_per_iteration=1,
    )

    assert len(hypotheses) == 1
    h = hypotheses[0]
    assert h.description == "d1"
    assert h.iteration == 2
    assert h.proposed_by_model == "claude-sonnet-5"
    assert h.requires_target_in_fit is False


def test_propose_hypotheses_appends_user_and_assistant_turns() -> None:
    fake = _FakeLLMClient(parsed={"hypotheses": []})
    tier = ModelTierEntry(model="claude-sonnet-5")

    _, updated_messages, _ = propose_hypotheses(
        fake, tier, messages=[{"role": "user", "content": "prior turn"}], data_profile=_profile(),
        kb_excerpt="x", feature_selection_summary="y", iteration=0, iterations_since_improvement=0,
        hypotheses_per_iteration=3,
    )

    assert len(updated_messages) == 3  # prior turn + new user turn + assistant reply
    assert updated_messages[0] == {"role": "user", "content": "prior turn"}
    assert updated_messages[1]["role"] == "user"
    assert "Iteration 0" in updated_messages[1]["content"]
    assert updated_messages[2]["role"] == "assistant"


def test_propose_hypotheses_passes_output_schema_and_system_prompt() -> None:
    fake = _FakeLLMClient(parsed={"hypotheses": []})
    tier = ModelTierEntry(model="claude-sonnet-5")

    propose_hypotheses(
        fake, tier, messages=[], data_profile=_profile(), kb_excerpt="x", feature_selection_summary="y",
        iteration=0, iterations_since_improvement=0, hypotheses_per_iteration=3,
    )

    call = fake.calls[0]
    assert call["output_schema"]["required"] == ["hypotheses"]
    assert "Data Analyst" in call["system_prompt"]

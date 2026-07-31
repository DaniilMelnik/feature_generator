"""Control-flow tests for the LangGraph wiring -- stubbed hypothesize/codegen/
leakage_review functions (zero API calls), real sandbox execution via
InProcessFeatureComputer wrapping dynamically-exec'd (but test-authored,
trusted) module sources, real DatasetBuilder/CatBoost/etc. This tests
retry-bounding, budget exhaustion, and escalation-ladder logic exactly as
the build plan specifies.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from feature_generator.agents.llm_client import LLMResponse, LLMResponseError, LLMUsage
from feature_generator.config import (
    BudgetConfig,
    DatasetConfig,
    ModelTierEntry,
    RetryConfig,
    RunConfig,
    TableConfig,
)
from feature_generator.dataset.builder import DatasetBuilder
from feature_generator.dataset.feature_store import FeatureStore
from feature_generator.knowledge_base.db import KnowledgeBase
from feature_generator.knowledge_base.results_store import ResultsStore
from feature_generator.orchestration.budget import BudgetTracker
from feature_generator.orchestration.graph import GraphDependencies, build_iteration_graph, run_pipeline
from feature_generator.profiling.schemas import DataProfile, TableProfile
from feature_generator.sandbox.contract import InProcessFeatureComputer
from feature_generator.schemas import FeatureHypothesis, FeatureSpec

COMPLIANT_SOURCE = """
FEATURE_NAME = "avg_room_service_by_home_planet"
REQUIRED_COLUMNS = ["HomePlanet", "RoomService"]

def fit(train_df, context):
    means = train_df.groupby("HomePlanet")["RoomService"].mean()
    return {"means": {str(k): float(v) for k, v in means.items()}, "global_mean": float(train_df["RoomService"].mean())}

def transform(df, params):
    return df["HomePlanet"].map(params["means"]).fillna(params["global_mean"])
"""

MALICIOUS_SOURCE = """
import os

FEATURE_NAME = "malicious"
REQUIRED_COLUMNS = ["Age"]

def fit(train_df, context):
    return {"v": os.environ.get("HOME", "")}

def transform(df, params):
    return df["Age"] * 0
"""


def _feature_computer_factory(spec: FeatureSpec, allow_target_in_fit: bool):
    namespace: dict = {}
    exec(compile(spec.module_source, spec.feature_name, "exec"), namespace)  # noqa: S102 -- test-authored trusted source

    class _Module:
        FEATURE_NAME = namespace["FEATURE_NAME"]
        REQUIRED_COLUMNS = namespace["REQUIRED_COLUMNS"]
        fit = staticmethod(namespace["fit"])
        transform = staticmethod(namespace["transform"])

    return InProcessFeatureComputer(_Module(), allow_target_in_fit=allow_target_in_fit)


def _run_config(**retry_overrides) -> RunConfig:
    return RunConfig(
        dataset=DatasetConfig(
            name="test",
            tables=[TableConfig(name="passengers", path="unused.csv", role="primary")],
            target_column="Transported",
            target_table="passengers",
            id_columns=["PassengerId"],
        ),
        retries=RetryConfig(**retry_overrides),
        hypotheses_per_iteration=1,
    )


def _data_profile() -> DataProfile:
    return DataProfile(
        dataset_name="test",
        tables=[TableProfile(table_name="passengers", file_path="x.csv", n_rows=10, n_cols=1, columns=[])],
        target_column="Transported",
        target_table="passengers",
        id_columns=["PassengerId"],
        positive_rate=0.5,
        generated_at=datetime.utcnow(),
    )


def _base_df(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    home_planet = rng.choice(["Earth", "Mars"], size=n)
    # RoomService is genuinely higher for Mars passengers -- the groupby-mean
    # feature has real signal, and the target is driven by that same effect,
    # so `avg_room_service_by_home_planet` should reliably clear the
    # feature-selection lift threshold.
    room_service = np.where(home_planet == "Mars", rng.normal(80, 10, size=n), rng.normal(20, 10, size=n))
    logit = np.where(home_planet == "Mars", 1.5, -1.5)
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="Transported")
    df = pd.DataFrame({"HomePlanet": home_planet, "RoomService": room_service, "Transported": y})
    return df, y


def _hypothesis(**overrides) -> FeatureHypothesis:
    defaults = dict(
        iteration=0, rationale="r", description="avg room service by home planet",
        input_columns=["HomePlanet", "RoomService"], input_tables=["passengers"],
        feature_type="aggregation", proposed_by_model="stub",
    )
    defaults.update(overrides)
    return FeatureHypothesis(**defaults)


def _make_llm_response(parsed: dict) -> LLMResponse:
    return LLMResponse(text="...", parsed=parsed, raw_content=[{"type": "text", "text": "..."}])


class ScriptedHypothesize:
    """Stub matching hypothesis_agent.propose_hypotheses's signature."""

    def __init__(self, hypotheses_per_call: list[list[FeatureHypothesis]]) -> None:
        self.hypotheses_per_call = hypotheses_per_call
        self.calls = 0

    def __call__(self, llm_client, tier, **kwargs):
        hypotheses = self.hypotheses_per_call[min(self.calls, len(self.hypotheses_per_call) - 1)]
        self.calls += 1
        return hypotheses, kwargs["messages"] + [{"role": "assistant", "content": "ok"}], _make_llm_response({})


class ScriptedCodegen:
    """Stub matching codegen_agent.generate_feature_module's signature."""

    def __init__(self, source: str = COMPLIANT_SOURCE, feature_name: str = "avg_room_service_by_home_planet") -> None:
        self.source = source
        self.feature_name = feature_name
        self.calls = 0

    def __call__(self, llm_client, tier, *, messages, hypothesis, data_profile, attempt, feedback=None):
        self.calls += 1
        spec = FeatureSpec(
            hypothesis_id=hypothesis.id, feature_name=self.feature_name, module_source=self.source,
            declared_input_columns=hypothesis.input_columns, declared_input_tables=hypothesis.input_tables,
            output_dtype="float64", codegen_model="stub", codegen_attempt=attempt,
        )
        return spec, [*messages, {"role": "assistant", "content": "ok"}], _make_llm_response({})


class ScriptedFailingCodegen:
    """Stub that raises -- e.g. a truncated/malformed structured output
    (LLMResponseError) -- simulating a real failure mode observed in
    production (see llm_client.LLMResponseError). Must be routed through the
    same retry/give-up machinery as a static-check violation, never crash
    the iteration. `fail_after` successful calls succeed first (e.g. to let
    an initial codegen succeed and only the leakage-review revision retry
    fail), defaulting to 0 (always fails).
    """

    def __init__(self, fail_after: int = 0, feature_name: str = "flaky") -> None:
        self.fail_after = fail_after
        self.feature_name = feature_name
        self.calls = 0

    def __call__(self, llm_client, tier, *, messages, hypothesis, data_profile, attempt, feedback=None):
        self.calls += 1
        if self.calls > self.fail_after:
            raise LLMResponseError("expected structured JSON output but got no text content", "max_tokens")
        spec = FeatureSpec(
            hypothesis_id=hypothesis.id, feature_name=self.feature_name, module_source=COMPLIANT_SOURCE,
            declared_input_columns=hypothesis.input_columns, declared_input_tables=hypothesis.input_tables,
            output_dtype="float64", codegen_model="stub", codegen_attempt=attempt,
        )
        return spec, [*messages, {"role": "assistant", "content": "ok"}], _make_llm_response({})


class ScriptedLeakageReview:
    def __init__(self, verdict: str = "approve") -> None:
        self.verdict = verdict
        self.calls = 0

    def __call__(self, llm_client, tier, *, messages, batch, rejected_patterns_excerpt="(none yet)"):
        self.calls += 1
        if not batch:
            return {}, messages, None
        from feature_generator.schemas import LeakageReviewVerdict

        verdicts = {
            spec.feature_name: LeakageReviewVerdict(reviewer_model="stub", verdict=self.verdict, confidence=0.9)
            for _hyp, spec in batch
        }
        return verdicts, messages, _make_llm_response({})


class _FakeLLMClientHolder:
    """Stands in for a real LLMClient -- role functions are stubbed in these
    tests, but `run_pipeline` still reads `llm_client.total_usage` for its
    own budget bookkeeping, so a real Anthropic-backed client isn't needed.
    """

    def __init__(self) -> None:
        self.total_usage = LLMUsage()


@pytest.fixture
def deps(tmp_path):
    df, y = _base_df()
    config = _run_config()
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    results = ResultsStore(tmp_path / "results.duckdb")
    store = FeatureStore(tmp_path / "feature_store")
    builder = DatasetBuilder(cv_folds=4, random_seed=42)

    return GraphDependencies(
        run_config=config,
        data_profile=_data_profile(),
        llm_client=_FakeLLMClientHolder(),  # role fns are stubbed; run_pipeline still reads .total_usage
        knowledge_base=kb,
        results_store=results,
        feature_store=store,
        dataset_builder=builder,
        base_df=df,
        y=y,
        feature_computer_factory=_feature_computer_factory,
        hypothesis_tier=ModelTierEntry(
            model="claude-sonnet-5", escalate_to_model="claude-opus-5", escalate_to_effort="xhigh"
        ),
        codegen_tier=ModelTierEntry(model="claude-sonnet-5"),
        leakage_reviewer_tier=ModelTierEntry(model="claude-opus-5"),
        hypothesize_fn=ScriptedHypothesize([[_hypothesis()]]),
        codegen_fn=ScriptedCodegen(),
        leakage_review_fn=ScriptedLeakageReview("approve"),
    )


def test_single_iteration_validates_and_promotes_a_compliant_feature(deps) -> None:
    graph = build_iteration_graph(deps)
    initial_state = {
        "run_id": "run-1", "iteration": 0, "kb_excerpt": "none", "feature_selection_summary": "",
        "hypotheses_per_iteration": 1, "iterations_since_improvement": 0, "escalation_tier": "base",
        "hypothesis_agent_messages": [], "accepted_feature_names": [], "best_metric_so_far": float("-inf"),
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": "t1"}})

    assert result["newly_validated_feature_names"] == ["avg_room_service_by_home_planet"]
    assert result["current_training_metrics"] is not None
    assert "avg_room_service_by_home_planet" in result["accepted_feature_names"]

    kb_results = deps.knowledge_base.list_validation_results("run-1")
    assert any(r.final_status == "validated" for r in kb_results)
    assert deps.results_store.is_promoted("avg_room_service_by_home_planet")

    selection_rounds = deps.knowledge_base.list_feature_selection_rounds("run-1")
    assert len(selection_rounds) == 1
    assert selection_rounds[0]["iteration"] == 0
    assert "avg_room_service_by_home_planet" in selection_rounds[0]["final_selected"]

    detail = deps.results_store.get_promoted_feature_detail("avg_room_service_by_home_planet")
    assert detail["iteration"] == 0
    assert detail["lift_over_baseline"] is not None


def test_holdout_auc_is_populated_and_never_leaked_to_llm_context(tmp_path) -> None:
    df, y = _base_df()
    config = _run_config()
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    results = ResultsStore(tmp_path / "results.duckdb")
    store = FeatureStore(tmp_path / "feature_store")
    builder = DatasetBuilder(cv_folds=4, random_seed=42)
    dev_index, holdout_index = builder.split_dev_holdout(y, 0.2)

    deps = GraphDependencies(
        run_config=config,
        data_profile=_data_profile(),
        llm_client=_FakeLLMClientHolder(),
        knowledge_base=kb,
        results_store=results,
        feature_store=store,
        dataset_builder=builder,
        base_df=df,
        y=y,
        dev_index=dev_index,
        holdout_index=holdout_index,
        feature_computer_factory=_feature_computer_factory,
        hypothesis_tier=ModelTierEntry(
            model="claude-sonnet-5", escalate_to_model="claude-opus-5", escalate_to_effort="xhigh"
        ),
        codegen_tier=ModelTierEntry(model="claude-sonnet-5"),
        leakage_reviewer_tier=ModelTierEntry(model="claude-opus-5"),
        hypothesize_fn=ScriptedHypothesize([[_hypothesis()]]),
        codegen_fn=ScriptedCodegen(),
        leakage_review_fn=ScriptedLeakageReview("approve"),
    )

    graph = build_iteration_graph(deps)
    initial_state = {
        "run_id": "run-holdout", "iteration": 0, "kb_excerpt": "none", "feature_selection_summary": "",
        "hypotheses_per_iteration": 1, "iterations_since_improvement": 0, "escalation_tier": "base",
        "hypothesis_agent_messages": [], "accepted_feature_names": [], "best_metric_so_far": float("-inf"),
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": "t-holdout"}})

    metrics_dict = result["current_training_metrics"]
    assert metrics_dict is not None
    assert metrics_dict["holdout_auc"] is not None
    assert 0.0 <= metrics_dict["holdout_auc"] <= 1.0
    assert metrics_dict["train_auc_mean"] > 0.0

    # the one safety-relevant assertion: holdout must never reach anything
    # the hypothesis agent's prompt is built from
    kb_excerpt = deps.knowledge_base.render_kb_excerpt("run-holdout")
    assert "holdout" not in kb_excerpt.lower()
    assert "holdout" not in result["feature_selection_summary"].lower()


def test_codegen_retry_bounded_and_ends_static_failed(deps) -> None:
    deps.run_config = _run_config(max_codegen_retries=2)
    deps.codegen_fn = ScriptedCodegen(source=MALICIOUS_SOURCE, feature_name="malicious")

    graph = build_iteration_graph(deps)
    initial_state = {
        "run_id": "run-2", "iteration": 0, "kb_excerpt": "none", "feature_selection_summary": "",
        "hypotheses_per_iteration": 1, "iterations_since_improvement": 0, "escalation_tier": "base",
        "hypothesis_agent_messages": [], "accepted_feature_names": [], "best_metric_so_far": float("-inf"),
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": "t2"}})

    # 1 initial attempt + 2 retries = 3 total codegen calls
    assert deps.codegen_fn.calls == 3
    statuses = {r["feature_name"]: r["final_status"] for r in result["current_validation_results"]}
    assert statuses["malicious"] == "static_failed"
    assert result["newly_validated_feature_names"] == []
    assert not deps.feature_store.exists("run-2", "malicious")


def test_codegen_llm_failure_is_retried_not_crashed(deps) -> None:
    """Regression test: a codegen call that raises LLMResponseError (e.g. a
    truncated max_tokens response) must be retried like a static-check
    violation and give up gracefully once retries are exhausted -- not
    propagate and crash the whole run (this happened for real in production
    with hypotheses_per_iteration=15: a revision-retry codegen call was
    truncated and the resulting TypeError killed an in-progress run).
    """
    deps.run_config = _run_config(max_codegen_retries=2)
    deps.codegen_fn = ScriptedFailingCodegen()

    graph = build_iteration_graph(deps)
    initial_state = {
        "run_id": "run-flaky", "iteration": 0, "kb_excerpt": "none", "feature_selection_summary": "",
        "hypotheses_per_iteration": 1, "iterations_since_improvement": 0, "escalation_tier": "base",
        "hypothesis_agent_messages": [], "accepted_feature_names": [], "best_metric_so_far": float("-inf"),
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": "t-flaky"}})

    assert deps.codegen_fn.calls == 3  # 1 initial attempt + 2 retries, never crashes
    statuses = [r["final_status"] for r in result["current_validation_results"]]
    assert statuses == ["static_failed"]
    assert result["newly_validated_feature_names"] == []


def test_leakage_review_reject_prevents_sandbox_execution(deps) -> None:
    deps.leakage_review_fn = ScriptedLeakageReview("reject")

    graph = build_iteration_graph(deps)
    initial_state = {
        "run_id": "run-3", "iteration": 0, "kb_excerpt": "none", "feature_selection_summary": "",
        "hypotheses_per_iteration": 1, "iterations_since_improvement": 0, "escalation_tier": "base",
        "hypothesis_agent_messages": [], "accepted_feature_names": [], "best_metric_so_far": float("-inf"),
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": "t3"}})

    statuses = {r["feature_name"]: r["final_status"] for r in result["current_validation_results"]}
    assert statuses["avg_room_service_by_home_planet"] == "leakage_rejected"
    assert not deps.feature_store.exists("run-3", "avg_room_service_by_home_planet")
    assert result["newly_validated_feature_names"] == []


def test_leakage_review_needs_revision_retries_bounded(deps) -> None:
    deps.run_config = _run_config(max_leakage_revision_retries=1)
    deps.leakage_review_fn = ScriptedLeakageReview("needs_revision")

    graph = build_iteration_graph(deps)
    initial_state = {
        "run_id": "run-4", "iteration": 0, "kb_excerpt": "none", "feature_selection_summary": "",
        "hypotheses_per_iteration": 1, "iterations_since_improvement": 0, "escalation_tier": "base",
        "hypothesis_agent_messages": [], "accepted_feature_names": [], "best_metric_so_far": float("-inf"),
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": "t4"}})

    # leakage review is called once per round: initial batch + 1 retry round = 2 calls
    assert deps.leakage_review_fn.calls == 2
    statuses = [r["final_status"] for r in result["current_validation_results"]]
    assert "leakage_rejected" in statuses
    assert result["newly_validated_feature_names"] == []


def test_leakage_revision_retry_codegen_failure_is_handled_not_crashed(deps) -> None:
    """Regression test for the exact production crash: leakage_review's
    revision-retry path calls `_regenerate_with_feedback`, which calls
    `codegen_fn` with a fresh, empty `messages=[]` list every time (unlike
    the codegen_branch subgraph's own retry loop) -- a separate code path
    that needs its own coverage for the same LLMResponseError resilience.
    """
    deps.run_config = _run_config(max_leakage_revision_retries=1)
    deps.leakage_review_fn = ScriptedLeakageReview("needs_revision")
    deps.codegen_fn = ScriptedFailingCodegen(fail_after=1)  # initial codegen succeeds, revision retries fail

    graph = build_iteration_graph(deps)
    initial_state = {
        "run_id": "run-flaky-revision", "iteration": 0, "kb_excerpt": "none", "feature_selection_summary": "",
        "hypotheses_per_iteration": 1, "iterations_since_improvement": 0, "escalation_tier": "base",
        "hypothesis_agent_messages": [], "accepted_feature_names": [], "best_metric_so_far": float("-inf"),
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": "t-flaky-revision"}})

    statuses = [r["final_status"] for r in result["current_validation_results"]]
    assert "static_failed" in statuses  # the failed regeneration attempt, not a crash
    assert result["newly_validated_feature_names"] == []


def test_no_hypotheses_iteration_still_completes(deps) -> None:
    deps.hypothesize_fn = ScriptedHypothesize([[]])

    graph = build_iteration_graph(deps)
    initial_state = {
        "run_id": "run-5", "iteration": 0, "kb_excerpt": "none", "feature_selection_summary": "",
        "hypotheses_per_iteration": 1, "iterations_since_improvement": 0, "escalation_tier": "base",
        "hypothesis_agent_messages": [], "accepted_feature_names": [], "best_metric_so_far": float("-inf"),
    }

    result = graph.invoke(initial_state, config={"configurable": {"thread_id": "t5"}})

    assert result["newly_validated_feature_names"] == []
    assert result.get("iterations_since_improvement", 0) >= 1  # promote node ran and incremented it


def test_run_pipeline_stops_at_max_iterations(deps) -> None:
    deps.hypothesize_fn = ScriptedHypothesize([[]])  # trivial iterations for speed
    tracker = BudgetTracker(BudgetConfig(max_wall_clock_minutes=None, max_iterations=3))

    run_result = run_pipeline(deps, "run-6", budget_tracker=tracker)

    assert run_result.iterations_completed == 3
    assert run_result.stopped_reason == "budget_exhausted"


def test_run_pipeline_escalates_after_stalled_iterations(deps) -> None:
    deps.hypothesize_fn = ScriptedHypothesize([[]])
    tracker = BudgetTracker(
        BudgetConfig(max_wall_clock_minutes=None, max_iterations=5, stalled_iterations_before_escalation=2)
    )

    seen_tiers: list[str] = []
    original_call = deps.hypothesize_fn

    def _spy(llm_client, tier, **kwargs):
        seen_tiers.append(tier.model)
        return original_call(llm_client, tier, **kwargs)

    deps.hypothesize_fn = _spy

    run_pipeline(deps, "run-7", budget_tracker=tracker)

    # first couple of iterations use the base tier; later ones escalate
    assert seen_tiers[0] == deps.hypothesis_tier.model
    assert deps.hypothesis_tier.escalate_to_model in seen_tiers


def test_run_pipeline_carries_forward_accepted_features_across_iterations(deps) -> None:
    tracker = BudgetTracker(BudgetConfig(max_wall_clock_minutes=None, max_iterations=1))

    run_result = run_pipeline(deps, "run-8", budget_tracker=tracker)

    assert "avg_room_service_by_home_planet" in run_result.final_state["accepted_feature_names"]

from pathlib import Path

from feature_generator.knowledge_base.db import KnowledgeBase
from feature_generator.modeling.feature_selection import FeatureSelectionOutput
from feature_generator.schemas import (
    DynamicCheckResult,
    FeatureHypothesis,
    FeatureSpec,
    LeakageReviewVerdict,
    StabilityReport,
    StaticCheckResult,
    TrainingMetrics,
    ValidationResult,
)


def _hypothesis(iteration: int = 0, description: str = "log ratio of spend columns") -> FeatureHypothesis:
    return FeatureHypothesis(
        iteration=iteration,
        rationale="spend columns are skewed",
        description=description,
        input_columns=["FoodCourt", "RoomService"],
        input_tables=["passengers"],
        feature_type="ratio",
        proposed_by_model="claude-sonnet-5",
    )


def test_add_and_get_hypothesis_round_trips(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    h = _hypothesis()
    kb.add_hypothesis("run-1", h)

    fetched = kb.get_hypothesis(h.id)
    assert fetched == h
    kb.close()


def test_list_hypotheses_scoped_by_run_and_ordered_by_iteration(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    h1 = _hypothesis(iteration=1, description="first")
    h2 = _hypothesis(iteration=0, description="second")
    other_run = _hypothesis(iteration=0, description="other run")

    kb.add_hypothesis("run-1", h1)
    kb.add_hypothesis("run-1", h2)
    kb.add_hypothesis("run-2", other_run)

    results = kb.list_hypotheses("run-1")
    assert [h.description for h in results] == ["second", "first"]  # ordered by iteration
    assert len(kb.list_hypotheses("run-2")) == 1
    kb.close()


def test_feature_specs_round_trip_and_multiple_attempts(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    h = _hypothesis()
    kb.add_hypothesis("run-1", h)

    spec1 = FeatureSpec(
        hypothesis_id=h.id, feature_name="f", module_source="v1", declared_input_columns=["a"],
        declared_input_tables=["t"], output_dtype="float64", codegen_model="claude-sonnet-5",
        codegen_attempt=0,
    )
    spec2 = spec1.model_copy(update={"module_source": "v2", "codegen_attempt": 1})
    kb.add_feature_spec(spec1)
    kb.add_feature_spec(spec2)

    specs = kb.get_feature_specs(h.id)
    assert [s.module_source for s in specs] == ["v1", "v2"]
    kb.close()


def test_validation_results_round_trip_and_filter_by_status(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    h = _hypothesis()
    kb.add_hypothesis("run-1", h)

    validated = ValidationResult(
        feature_name="f1", hypothesis_id=h.id, static=StaticCheckResult(passed=True),
        final_status="validated",
    )
    rejected = ValidationResult(
        feature_name="f2", hypothesis_id=h.id, static=StaticCheckResult(passed=False, violations=["x"]),
        final_status="static_failed",
    )
    kb.add_validation_result("run-1", validated)
    kb.add_validation_result("run-1", rejected)

    all_results = kb.list_validation_results("run-1")
    assert len(all_results) == 2

    only_validated = kb.list_validation_results("run-1", final_status="validated")
    assert [r.feature_name for r in only_validated] == ["f1"]
    kb.close()


def test_training_metrics_latest_returns_most_recent(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    m1 = TrainingMetrics(
        run_id="run-1", feature_set_id="fs-1", iteration=0, cv_folds=5, auc_mean=0.7, auc_std=0.01,
        logloss_mean=0.5, pr_auc_mean=0.6, ks_stat_mean=0.3, brier_score_mean=0.2,
        train_rows=100, val_rows=20,
    )
    m2 = m1.model_copy(update={"feature_set_id": "fs-2", "iteration": 1, "auc_mean": 0.8})
    kb.add_training_metrics("run-1", m1)
    kb.add_training_metrics("run-1", m2)

    latest = kb.get_latest_training_metrics("run-1")
    assert latest.feature_set_id == "fs-2"
    assert len(kb.list_training_metrics("run-1")) == 2
    kb.close()


def test_training_metrics_same_feature_set_id_across_iterations_does_not_raise(tmp_path: Path) -> None:
    """Regression test: `feature_set_id` is content-derived (sorted accepted
    feature names) and legitimately repeats across iterations -- e.g. an
    iteration where no new candidate survives leaves the accepted set, and
    therefore this string, unchanged. The primary key must be `(run_id,
    iteration)`, not `(run_id, feature_set_id)`, or this raises a duplicate-
    key constraint violation (caught for real running the pipeline live).
    """
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    same_set = TrainingMetrics(
        run_id="run-1", feature_set_id="a|b", iteration=0, cv_folds=5, auc_mean=0.8, auc_std=0.01,
        logloss_mean=0.5, pr_auc_mean=0.6, ks_stat_mean=0.3, brier_score_mean=0.2,
        train_rows=100, val_rows=20,
    )
    kb.add_training_metrics("run-1", same_set)
    kb.add_training_metrics("run-1", same_set.model_copy(update={"iteration": 1}))  # must not raise

    assert len(kb.list_training_metrics("run-1")) == 2
    kb.close()


def test_stability_report_round_trip(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    report = StabilityReport(
        feature_set_id="fs-1", features=[], overall_stability_flag="stable",
        method_note="bootstrap stand-in",
    )
    kb.add_stability_report("run-1", report)
    kb.close()  # just confirming the write doesn't raise; read path covered elsewhere


def test_render_kb_excerpt_empty_run() -> None:
    kb = KnowledgeBase(":memory:")
    assert kb.render_kb_excerpt("nonexistent-run") == "No feature hypotheses have been tried yet."
    kb.close()


def test_render_kb_excerpt_includes_status_and_metrics(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    h = _hypothesis(description="cryo sleep spend ratio")
    kb.add_hypothesis("run-1", h)
    result = ValidationResult(
        feature_name="f1", hypothesis_id=h.id, static=StaticCheckResult(passed=True),
        dynamic=DynamicCheckResult(fold_fit_transform_ok=True, single_feature_auc=0.55),
        leakage_review=LeakageReviewVerdict(reviewer_model="claude-opus-5", verdict="approve"),
        final_status="validated",
    )
    kb.add_validation_result("run-1", result)

    excerpt = kb.render_kb_excerpt("run-1")
    assert "cryo sleep spend ratio" in excerpt
    assert "status=validated" in excerpt
    assert "0.55" in excerpt
    assert "approve" in excerpt
    kb.close()


def test_render_kb_excerpt_truncates_long_output(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    for i in range(50):
        kb.add_hypothesis("run-1", _hypothesis(iteration=i, description=f"hypothesis number {i} " * 5))

    excerpt = kb.render_kb_excerpt("run-1", max_chars=500)
    assert len(excerpt) <= 500
    assert excerpt.endswith("(truncated)")
    kb.close()


def test_feature_selection_round_round_trips(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    output = FeatureSelectionOutput(
        accepted_candidates=["a", "b"],
        rejected_by_lift=["c"],
        redundancy_clusters=[["a", "b"]],
        kept_after_redundancy=["a"],
        final_selected=["a"],
        auc_trace=[("a", 0.81)],
        baseline_auc=0.80,
        final_auc=0.81,
    )
    kb.add_feature_selection_round("run-1", 0, output)

    rounds = kb.list_feature_selection_rounds("run-1")
    assert len(rounds) == 1
    assert rounds[0]["iteration"] == 0
    assert rounds[0]["baseline_auc"] == 0.80
    assert rounds[0]["final_auc"] == 0.81
    assert rounds[0]["accepted_candidates"] == ["a", "b"]
    assert rounds[0]["auc_trace"] == [["a", 0.81]]  # tuples become lists through JSON
    kb.close()


def test_list_feature_selection_rounds_scoped_by_run_and_ordered(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    base = dict(
        accepted_candidates=[], rejected_by_lift=[], redundancy_clusters=[], kept_after_redundancy=[],
        final_selected=[], auc_trace=[], baseline_auc=0.5, final_auc=0.5,
    )
    kb.add_feature_selection_round("run-1", 1, FeatureSelectionOutput(**base))
    kb.add_feature_selection_round("run-1", 0, FeatureSelectionOutput(**base))
    kb.add_feature_selection_round("run-2", 0, FeatureSelectionOutput(**base))

    rounds = kb.list_feature_selection_rounds("run-1")
    assert [r["iteration"] for r in rounds] == [0, 1]
    assert len(kb.list_feature_selection_rounds("run-2")) == 1
    kb.close()


def test_run_metadata_round_trips(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    kb.set_run_metadata("run-1", dataset_name="spaceship_titanic", baseline_auc=0.8796, raw_feature_columns=["Age", "HomePlanet"])

    metadata = kb.get_run_metadata("run-1")
    assert metadata["dataset_name"] == "spaceship_titanic"
    assert metadata["baseline_auc"] == 0.8796
    assert metadata["raw_feature_columns"] == ["Age", "HomePlanet"]
    kb.close()


def test_run_metadata_upsert_replaces_previous_value(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    kb.set_run_metadata("run-1", dataset_name="ds", baseline_auc=0.5, raw_feature_columns=["a"])
    kb.set_run_metadata("run-1", dataset_name="ds", baseline_auc=0.6, raw_feature_columns=["a", "b"])

    metadata = kb.get_run_metadata("run-1")
    assert metadata["baseline_auc"] == 0.6
    assert metadata["raw_feature_columns"] == ["a", "b"]
    kb.close()


def test_get_run_metadata_returns_none_when_missing() -> None:
    kb = KnowledgeBase(":memory:")
    assert kb.get_run_metadata("nonexistent-run") is None
    kb.close()


def test_get_best_training_metrics_returns_highest_auc(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    low = TrainingMetrics(
        run_id="run-1", feature_set_id="fs-low", iteration=0, cv_folds=5, auc_mean=0.7, auc_std=0.01,
        logloss_mean=0.5, pr_auc_mean=0.6, ks_stat_mean=0.3, brier_score_mean=0.2,
        train_rows=100, val_rows=20,
    )
    high = low.model_copy(update={"feature_set_id": "fs-high", "iteration": 1, "auc_mean": 0.9})
    kb.add_training_metrics("run-1", low)
    kb.add_training_metrics("run-1", high)

    best = kb.get_best_training_metrics("run-1")
    assert best.feature_set_id == "fs-high"
    kb.close()

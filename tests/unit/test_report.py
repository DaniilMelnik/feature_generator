from pathlib import Path

from feature_generator.config import DatasetConfig, OutputConfig, RunConfig, TableConfig
from feature_generator.knowledge_base.db import KnowledgeBase
from feature_generator.knowledge_base.results_store import ResultsStore
from feature_generator.modeling.feature_selection import FeatureSelectionOutput
from feature_generator.reporting.report import build_run_report
from feature_generator.schemas import (
    DynamicCheckResult,
    FeatureHypothesis,
    FeatureSpec,
    FeatureStability,
    StabilityReport,
    StaticCheckResult,
    TrainingMetrics,
    ValidationResult,
)


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        dataset=DatasetConfig(
            name="mini_dataset",
            tables=[TableConfig(name="t", path="unused.csv", role="primary")],
            target_column="y",
            target_table="t",
        ),
        output=OutputConfig(
            run_dir=str(tmp_path),
            knowledge_base_path=str(tmp_path / "kb.duckdb"),
            results_store_path=str(tmp_path / "results.duckdb"),
        ),
    )


def _seed_promoted_feature(tmp_path: Path, run_id: str, *, rationale: str = "spend is skewed") -> None:
    kb = KnowledgeBase(tmp_path / "kb.duckdb")
    store = ResultsStore(tmp_path / "results.duckdb")

    hypothesis = FeatureHypothesis(
        iteration=0,
        rationale=rationale,
        description="log1p of total spend",
        input_columns=["RoomService"],
        input_tables=["passengers"],
        feature_type="ratio",
        proposed_by_model="claude-sonnet-5",
    )
    kb.add_hypothesis(run_id, hypothesis)

    spec = FeatureSpec(
        hypothesis_id=hypothesis.id,
        feature_name="log_total_spend",
        module_source="def transform(df, params):\n    return df['RoomService']\n",
        declared_input_columns=["RoomService"],
        declared_input_tables=["passengers"],
        output_dtype="float64",
        codegen_model="claude-sonnet-5",
    )
    result = ValidationResult(
        feature_name=spec.feature_name,
        hypothesis_id=hypothesis.id,
        static=StaticCheckResult(passed=True),
        dynamic=DynamicCheckResult(fold_fit_transform_ok=True, single_feature_auc=0.61),
        final_status="validated",
    )
    metrics = TrainingMetrics(
        run_id=run_id, feature_set_id="log_total_spend", iteration=1, cv_folds=5,
        auc_mean=0.8931, auc_std=0.01, train_auc_mean=0.9520, train_auc_std=0.005, holdout_auc=0.8850,
        logloss_mean=0.4, pr_auc_mean=0.7, ks_stat_mean=0.5, brier_score_mean=0.15,
        train_rows=100, val_rows=20,
    )
    kb.add_training_metrics(run_id, metrics)
    stability = StabilityReport(
        feature_set_id="log_total_spend",
        features=[FeatureStability(feature_name=spec.feature_name, csi=0.03, method="bootstrap_resampling")],
        overall_stability_flag="stable",
        method_note="bootstrap stand-in",
    )
    store.promote(run_id, spec, result, metrics, stability, iteration=1, lift_over_baseline=0.0135)

    output = FeatureSelectionOutput(
        accepted_candidates=[spec.feature_name],
        rejected_by_lift=[],
        redundancy_clusters=[],
        kept_after_redundancy=[spec.feature_name],
        final_selected=[spec.feature_name],
        auc_trace=[(spec.feature_name, 0.8931)],
        baseline_auc=0.8796,
        final_auc=0.8931,
    )
    kb.add_feature_selection_round(run_id, 1, output)
    kb.set_run_metadata(run_id, dataset_name="mini_dataset", baseline_auc=0.8796, raw_feature_columns=["Age", "RoomService"])

    kb.close()
    store.close()


def test_report_includes_promoted_feature_detail(tmp_path: Path) -> None:
    _seed_promoted_feature(tmp_path, "run-1")

    html = build_run_report(_config(tmp_path), "run-1")

    assert "log_total_spend" in html
    assert "spend is skewed" in html
    assert "def transform(df, params):" in html
    assert "0.8931" in html  # model AUC at promotion
    assert "0.0135" in html  # lift over baseline
    assert "0.8796" in html  # features-off baseline
    assert "0.9520" in html  # train AUC
    assert "0.8850" in html  # holdout AUC


def test_report_includes_selection_round_trace(tmp_path: Path) -> None:
    _seed_promoted_feature(tmp_path, "run-1")

    html = build_run_report(_config(tmp_path), "run-1")

    assert "Iteration 1" in html
    assert "log_total_spend" in html


def test_report_includes_model_performance_over_run_table(tmp_path: Path) -> None:
    _seed_promoted_feature(tmp_path, "run-1")

    html = build_run_report(_config(tmp_path), "run-1")

    assert "Model performance over the run" in html
    start = html.find("Model performance over the run")
    end = html.find("<h2>Promoted features</h2>", start)
    section = html[start:end]
    assert "0.9520" in section  # train AUC
    assert "0.8931" in section  # CV AUC
    assert "0.8850" in section  # holdout AUC


def test_report_escapes_llm_authored_rationale(tmp_path: Path) -> None:
    _seed_promoted_feature(tmp_path, "run-1", rationale="<script>alert(1)</script>")

    html = build_run_report(_config(tmp_path), "run-1")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_handles_run_with_no_data(tmp_path: Path) -> None:
    KnowledgeBase(tmp_path / "kb.duckdb").close()
    ResultsStore(tmp_path / "results.duckdb").close()

    html = build_run_report(_config(tmp_path), "nonexistent-run")

    assert "No features have been promoted" in html
    assert "No hypotheses recorded yet" in html

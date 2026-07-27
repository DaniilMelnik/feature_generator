from datetime import datetime

from feature_generator.profiling.schemas import (
    ColumnProfile,
    DataProfile,
    NumericStats,
    TableProfile,
)
from feature_generator.schemas import (
    DynamicCheckResult,
    FeatureHypothesis,
    FeatureSpec,
    LeakageReviewVerdict,
    ServingParityResult,
    StabilityReport,
    StaticCheckResult,
    TrainingMetrics,
    ValidationResult,
)


def _sample_data_profile() -> DataProfile:
    numeric_col = ColumnProfile(
        name="Age",
        source_table="passengers",
        dtype="float64",
        role="feature",
        n_missing=5,
        pct_missing=0.02,
        n_unique=80,
        is_constant=False,
        numeric_stats=NumericStats(
            min=0, max=79, mean=28.5, std=14.2, skew=0.3, kurtosis=-0.2,
            quantiles={"p1": 1, "p25": 19, "p50": 27, "p75": 38, "p99": 70},
        ),
        correlation_with_target=0.05,
    )
    table = TableProfile(
        table_name="passengers",
        file_path="data/spaceship_titanic/train.csv",
        n_rows=8693,
        n_cols=14,
        role="primary",
        columns=[numeric_col],
    )
    return DataProfile(
        dataset_name="spaceship_titanic",
        tables=[table],
        target_column="Transported",
        target_table="passengers",
        id_columns=["PassengerId"],
        time_column=None,
        positive_rate=0.503,
        generated_at=datetime.utcnow(),
    )


def test_data_profile_round_trips_and_helpers_work() -> None:
    profile = _sample_data_profile()
    dumped = profile.model_dump(mode="json")
    reloaded = DataProfile.model_validate(dumped)
    assert reloaded == profile
    assert reloaded.primary_table().table_name == "passengers"
    assert len(reloaded.all_feature_columns()) == 1


def test_feature_hypothesis_default_id_and_flags() -> None:
    h1 = FeatureHypothesis(
        iteration=0,
        rationale="Spend columns are heavily right-skewed; log-ratio may help.",
        description="log1p(FoodCourt) / log1p(RoomService + 1)",
        input_columns=["FoodCourt", "RoomService"],
        input_tables=["passengers"],
        feature_type="ratio",
        proposed_by_model="claude-sonnet-5",
    )
    h2 = FeatureHypothesis(
        iteration=0,
        rationale="x",
        description="x",
        input_columns=["x"],
        input_tables=["passengers"],
        feature_type="other",
        proposed_by_model="claude-sonnet-5",
    )
    assert h1.id != h2.id  # each hypothesis gets a fresh uuid
    assert h1.requires_target_in_fit is False


def test_validation_result_composes_and_round_trips() -> None:
    result = ValidationResult(
        feature_name="spend_log_ratio",
        hypothesis_id="abc-123",
        static=StaticCheckResult(passed=True),
        dynamic=DynamicCheckResult(fold_fit_transform_ok=True, single_feature_auc=0.55),
        leakage_review=LeakageReviewVerdict(
            reviewer_model="claude-opus-5", verdict="approve", confidence=0.9
        ),
        serving_parity=ServingParityResult(ran=True, matched=True, mode="iid_shuffle"),
        final_status="validated",
    )
    dumped = result.model_dump(mode="json")
    reloaded = ValidationResult.model_validate(dumped)
    assert reloaded == result
    assert reloaded.final_status == "validated"


def test_training_metrics_and_stability_report_round_trip() -> None:
    metrics = TrainingMetrics(
        run_id="run-1",
        feature_set_id="fs-1",
        cv_folds=5,
        auc_mean=0.81,
        auc_std=0.01,
        logloss_mean=0.42,
        pr_auc_mean=0.79,
        ks_stat_mean=0.45,
        brier_score_mean=0.15,
        shap_importance={"Age": 0.12},
        train_rows=6954,
        val_rows=1739,
    )
    assert TrainingMetrics.model_validate(metrics.model_dump(mode="json")) == metrics

    report = StabilityReport(
        feature_set_id="fs-1",
        features=[],
        overall_stability_flag="stable",
        method_note="bootstrap stand-in: no time axis in Spaceship Titanic",
    )
    assert StabilityReport.model_validate(report.model_dump(mode="json")) == report


def test_feature_spec_round_trip() -> None:
    spec = FeatureSpec(
        hypothesis_id="abc-123",
        feature_name="spend_log_ratio",
        module_source="FEATURE_NAME = 'spend_log_ratio'\n",
        declared_input_columns=["FoodCourt", "RoomService"],
        declared_input_tables=["passengers"],
        output_dtype="float64",
        codegen_model="claude-sonnet-5",
    )
    assert FeatureSpec.model_validate(spec.model_dump(mode="json")) == spec

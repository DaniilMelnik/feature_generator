"""Cross-cutting pydantic schemas shared by agents, sandbox, dataset, modeling,
stability and knowledge_base. Centralized here (rather than split per-module)
to avoid import cycles: e.g. ``dataset.builder`` needs ``FeatureSpec``,
``knowledge_base.db`` needs nearly everything, and ``agents.*`` produce most
of these types.

See ``profiling.schemas`` for the (separate, non-cyclical) ``DataProfile`` tree.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

FeatureType = Literal[
    "aggregation",
    "ratio",
    "interaction",
    "temporal",
    "target_encoding",
    "categorical_encoding",
    "text",
    "other",
]


class FeatureHypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    iteration: int
    rationale: str
    description: str
    input_columns: list[str]
    input_tables: list[str]
    feature_type: FeatureType
    # Explicit opt-in: computing this feature's `fit()` legitimately needs to see
    # the target column (e.g. target/likelihood encoding). Triggers extra
    # leakage-reviewer scrutiny rather than being silently allowed or silently
    # blocked.
    requires_target_in_fit: bool = False
    parent_hypothesis_ids: list[str] = Field(default_factory=list)
    proposed_by_model: str


class FeatureSpec(BaseModel):
    hypothesis_id: str
    feature_name: str
    module_source: str
    declared_input_columns: list[str]
    declared_input_tables: list[str]
    output_dtype: Literal["float64", "int64", "bool", "category"]
    codegen_model: str
    codegen_attempt: int = 0


# --- Validation -------------------------------------------------------------


class StaticCheckResult(BaseModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
    ast_denylist_hits: list[str] = Field(default_factory=list)


class DynamicCheckResult(BaseModel):
    fold_fit_transform_ok: bool
    exceptions: list[str] = Field(default_factory=list)
    single_feature_auc: float | None = None
    single_feature_auc_flag: bool = False
    runtime_seconds: float = 0.0
    memory_mb_peak: float | None = None


class LeakageReviewVerdict(BaseModel):
    reviewer_model: str
    verdict: Literal["approve", "reject", "needs_revision"]
    concerns: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ServingParityMismatch(BaseModel):
    row_id: str
    offline_value: float | str | bool | None
    online_value: float | str | bool | None
    abs_diff: float | None = None


class ServingParityResult(BaseModel):
    ran: bool
    matched: bool
    mode: Literal["iid_shuffle", "temporal_replay"]
    n_rows_checked: int = 0
    n_mismatches: int = 0
    max_abs_diff: float | None = None
    mismatch_examples: list[ServingParityMismatch] = Field(default_factory=list)


FinalStatus = Literal[
    "pending",
    "static_failed",
    "dynamic_failed",
    "leakage_rejected",
    "serving_mismatch",
    "validated",
    "promoted",
]


class ValidationResult(BaseModel):
    feature_name: str
    hypothesis_id: str
    static: StaticCheckResult
    dynamic: DynamicCheckResult | None = None
    leakage_review: LeakageReviewVerdict | None = None
    serving_parity: ServingParityResult | None = None
    final_status: FinalStatus = "pending"


# --- Modeling & stability -----------------------------------------------------


class TrainingMetrics(BaseModel):
    run_id: str
    feature_set_id: str
    iteration: int = 0
    cv_folds: int
    auc_mean: float  # mean of the per-fold VALIDATION AUCs (the "CV AUC")
    auc_std: float
    train_auc_mean: float = 0.0  # mean of the per-fold TRAIN (in-sample) AUCs -- overfit-gap signal
    train_auc_std: float = 0.0
    # a model fit on "dev" rows only, scored once against rows the CV loop and
    # feature selection never see -- None only when holdout is disabled.
    # Never surfaced to the hypothesis/feature-selection agents -- see
    # orchestration.graph's holdout wiring.
    holdout_auc: float | None = None
    logloss_mean: float
    pr_auc_mean: float
    ks_stat_mean: float
    brier_score_mean: float
    shap_importance: dict[str, float] = Field(default_factory=dict)
    permutation_importance: dict[str, float] = Field(default_factory=dict)
    catboost_params: dict = Field(default_factory=dict)
    train_rows: int
    val_rows: int
    trained_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureStability(BaseModel):
    feature_name: str
    csi: float | None = None
    psi_over_time: list[tuple[str, float]] = Field(default_factory=list)
    method: Literal["temporal_oot", "bootstrap_resampling"]


class StabilityReport(BaseModel):
    feature_set_id: str
    iteration: int = 0
    features: list[FeatureStability]
    score_decile_stability: dict = Field(default_factory=dict)
    target_rate_by_decile_over_time: dict = Field(default_factory=dict)
    overall_stability_flag: Literal["stable", "watch", "unstable"]
    method_note: str

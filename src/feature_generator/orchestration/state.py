"""The LangGraph state schema for one pipeline run.

Fields annotated with ``operator.add`` are the ones LangGraph merges across
parallel ``Send`` branches (one branch per hypothesis in the codegen fan-out)
rather than overwrites -- see ``orchestration.graph``.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from feature_generator.config import RunConfig
from feature_generator.profiling.schemas import DataProfile
from feature_generator.schemas import (
    FeatureHypothesis,
    FeatureSpec,
    StabilityReport,
    TrainingMetrics,
    ValidationResult,
)

EscalationTier = Literal["base", "escalated"]
RunStatus = Literal["running", "completed_budget", "completed_error", "paused"]


class BudgetState(TypedDict):
    started_at: datetime
    deadline: datetime | None
    input_tokens_spent: int
    output_tokens_spent: int
    cache_read_tokens: int
    cache_creation_tokens: int


class PipelineState(TypedDict, total=False):
    run_id: str
    config: RunConfig
    data_profile: DataProfile

    iteration: int
    kb_excerpt: str

    current_hypotheses: list[FeatureHypothesis]
    current_feature_specs: Annotated[list[FeatureSpec], operator.add]
    current_validation_results: Annotated[list[ValidationResult], operator.add]
    current_training_metrics: TrainingMetrics | None
    current_stability_report: StabilityReport | None

    best_metric_so_far: float
    iterations_since_improvement: int
    escalation_tier: EscalationTier

    budget: BudgetState
    retry_counts: dict[str, int]

    hypothesis_agent_messages: list[dict]
    codegen_agent_messages: dict[str, list[dict]]  # keyed by hypothesis_id
    leakage_reviewer_messages: list[dict]

    accepted_feature_names: list[str]  # the running, promoted-eligible feature set

    status: RunStatus
    error_log: Annotated[list[str], operator.add]


def new_pipeline_state(run_id: str, config: RunConfig) -> PipelineState:
    """Initial state for a fresh run (used by ``orchestration.graph.init_run``)."""
    return PipelineState(
        run_id=run_id,
        config=config,
        iteration=0,
        kb_excerpt="No feature hypotheses have been tried yet.",
        current_hypotheses=[],
        current_feature_specs=[],
        current_validation_results=[],
        current_training_metrics=None,
        current_stability_report=None,
        best_metric_so_far=float("-inf"),
        iterations_since_improvement=0,
        escalation_tier="base",
        retry_counts={},
        hypothesis_agent_messages=[],
        codegen_agent_messages={},
        leakage_reviewer_messages=[],
        accepted_feature_names=[],
        status="running",
        error_log=[],
    )

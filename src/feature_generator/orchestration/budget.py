"""Wall-clock + token budget tracking, and the escalation-ladder decision
(switch a role to its configured escalated model/effort after N consecutive
iterations with no metric improvement). The operator's only lever over a run
is this budget -- everything else about when the loop ends is derived here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from feature_generator.agents.llm_client import LLMUsage
from feature_generator.config import BudgetConfig
from feature_generator.orchestration.state import BudgetState


def new_budget_state(config: BudgetConfig, *, now: datetime | None = None) -> BudgetState:
    started_at = now or datetime.utcnow()
    deadline = (
        started_at + timedelta(minutes=config.max_wall_clock_minutes)
        if config.max_wall_clock_minutes is not None
        else None
    )
    return BudgetState(
        started_at=started_at,
        deadline=deadline,
        input_tokens_spent=0,
        output_tokens_spent=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def record_usage(budget: BudgetState, usage: LLMUsage) -> BudgetState:
    return BudgetState(
        started_at=budget["started_at"],
        deadline=budget["deadline"],
        input_tokens_spent=budget["input_tokens_spent"] + usage.input_tokens,
        output_tokens_spent=budget["output_tokens_spent"] + usage.output_tokens,
        cache_read_tokens=budget["cache_read_tokens"] + usage.cache_read_tokens,
        cache_creation_tokens=budget["cache_creation_tokens"] + usage.cache_creation_tokens,
    )


def is_budget_exhausted(
    budget: BudgetState, config: BudgetConfig, *, now: datetime | None = None, iteration: int = 0
) -> bool:
    now = now or datetime.utcnow()
    if budget["deadline"] is not None and now >= budget["deadline"]:
        return True
    if config.max_iterations is not None and iteration >= config.max_iterations:
        return True
    if config.max_total_tokens is not None:
        total = budget["input_tokens_spent"] + budget["output_tokens_spent"]
        if total >= config.max_total_tokens:
            return True
    return False


def should_escalate(iterations_since_improvement: int, config: BudgetConfig) -> bool:
    return iterations_since_improvement >= config.stalled_iterations_before_escalation


class BudgetTracker:
    """Thin, stateless-except-for-config wrapper -- callers own the
    `BudgetState` (it lives in `PipelineState`, checkpointed by LangGraph),
    this class just knows how to create/update/interpret it.
    """

    def __init__(self, config: BudgetConfig) -> None:
        self.config = config

    def new_state(self, *, now: datetime | None = None) -> BudgetState:
        return new_budget_state(self.config, now=now)

    def record_usage(self, budget: BudgetState, usage: LLMUsage) -> BudgetState:
        return record_usage(budget, usage)

    def is_exhausted(self, budget: BudgetState, *, now: datetime | None = None, iteration: int = 0) -> bool:
        return is_budget_exhausted(budget, self.config, now=now, iteration=iteration)

    def should_escalate(self, iterations_since_improvement: int) -> bool:
        return should_escalate(iterations_since_improvement, self.config)

from datetime import datetime, timedelta

from feature_generator.agents.llm_client import LLMUsage
from feature_generator.config import BudgetConfig
from feature_generator.orchestration.budget import BudgetTracker


def test_new_state_sets_deadline_from_wall_clock_minutes() -> None:
    config = BudgetConfig(max_wall_clock_minutes=30)
    tracker = BudgetTracker(config)
    now = datetime(2026, 1, 1, 12, 0, 0)

    state = tracker.new_state(now=now)

    assert state["deadline"] == now + timedelta(minutes=30)
    assert state["input_tokens_spent"] == 0


def test_new_state_has_no_deadline_when_unbounded() -> None:
    tracker = BudgetTracker(BudgetConfig(max_wall_clock_minutes=None))
    state = tracker.new_state()
    assert state["deadline"] is None


def test_record_usage_accumulates_across_calls() -> None:
    tracker = BudgetTracker(BudgetConfig())
    state = tracker.new_state()

    state = tracker.record_usage(state, LLMUsage(input_tokens=100, output_tokens=50))
    state = tracker.record_usage(state, LLMUsage(input_tokens=10, output_tokens=5))

    assert state["input_tokens_spent"] == 110
    assert state["output_tokens_spent"] == 55


def test_is_exhausted_false_before_deadline() -> None:
    tracker = BudgetTracker(BudgetConfig(max_wall_clock_minutes=60))
    now = datetime(2026, 1, 1, 12, 0, 0)
    state = tracker.new_state(now=now)

    assert tracker.is_exhausted(state, now=now + timedelta(minutes=30)) is False


def test_is_exhausted_true_after_deadline() -> None:
    tracker = BudgetTracker(BudgetConfig(max_wall_clock_minutes=60))
    now = datetime(2026, 1, 1, 12, 0, 0)
    state = tracker.new_state(now=now)

    assert tracker.is_exhausted(state, now=now + timedelta(minutes=61)) is True


def test_is_exhausted_true_at_max_iterations() -> None:
    tracker = BudgetTracker(BudgetConfig(max_wall_clock_minutes=None, max_iterations=5))
    state = tracker.new_state()

    assert tracker.is_exhausted(state, iteration=4) is False
    assert tracker.is_exhausted(state, iteration=5) is True


def test_is_exhausted_true_at_max_total_tokens() -> None:
    tracker = BudgetTracker(BudgetConfig(max_wall_clock_minutes=None, max_total_tokens=100))
    state = tracker.new_state()
    state = tracker.record_usage(state, LLMUsage(input_tokens=60, output_tokens=39))
    assert tracker.is_exhausted(state) is False

    state = tracker.record_usage(state, LLMUsage(input_tokens=0, output_tokens=1))
    assert tracker.is_exhausted(state) is True


def test_is_exhausted_false_when_fully_unbounded() -> None:
    tracker = BudgetTracker(BudgetConfig(max_wall_clock_minutes=None))
    state = tracker.new_state()
    assert tracker.is_exhausted(state, iteration=10_000) is False


def test_should_escalate_at_configured_threshold() -> None:
    tracker = BudgetTracker(BudgetConfig(stalled_iterations_before_escalation=3))
    assert tracker.should_escalate(2) is False
    assert tracker.should_escalate(3) is True
    assert tracker.should_escalate(4) is True

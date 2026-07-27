"""The data-analyst half of the "feature engineer" agent: reads the (already
computed, deterministic) data profile plus the running knowledge-base
excerpt, and proposes the next batch of feature hypotheses. Runs as one
long-lived conversation per pipeline run (`messages` accumulates turn over
turn) -- never rebuilt from scratch each iteration.
"""

from __future__ import annotations

import json
from pathlib import Path

from feature_generator.agents.llm_client import LLMClient, LLMResponse
from feature_generator.config import ModelTierEntry
from feature_generator.profiling.schemas import DataProfile
from feature_generator.schemas import FeatureHypothesis

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "hypothesis_system.md").read_text()
OUTPUT_SCHEMA = json.loads((PROMPTS_DIR / "hypothesis_output_schema.json").read_text())


def render_data_profile_summary(profile: DataProfile) -> str:
    lines = [
        f"Dataset: {profile.dataset_name} (target={profile.target_column}, "
        f"positive_rate={profile.positive_rate:.3f})"
    ]
    if profile.time_column:
        lines.append(f"Time column: {profile.time_column}")
    if profile.profiling_notes:
        lines.append("Notes: " + "; ".join(profile.profiling_notes))

    for table in profile.tables:
        lines.append(f"\nTable '{table.table_name}' ({table.n_rows} rows, role={table.role}):")
        for col in table.columns:
            if col.role == "ignore":
                continue
            desc = f"  - {col.name} [{col.role}, {col.dtype}]"
            if col.correlation_with_target is not None:
                desc += f" corr_with_target={col.correlation_with_target:.3f}"
            if col.pct_missing:
                desc += f" missing={col.pct_missing:.1%}"
            if col.numeric_stats:
                desc += f" range=[{col.numeric_stats.min:.3g}, {col.numeric_stats.max:.3g}]"
            if col.categorical_stats:
                desc += f" cardinality={col.categorical_stats.cardinality}"
            lines.append(desc)
    return "\n".join(lines)


def build_user_turn(
    *,
    data_profile_summary: str,
    kb_excerpt: str,
    feature_selection_summary: str,
    iteration: int,
    iterations_since_improvement: int,
    hypotheses_per_iteration: int,
) -> str:
    return (
        f"Iteration {iteration}. You have not improved the model in the last "
        f"{iterations_since_improvement} iteration(s).\n\n"
        f"## Data profile\n{data_profile_summary}\n\n"
        f"## Feature catalog so far\n{kb_excerpt}\n\n"
        f"## Latest feature-selection results\n{feature_selection_summary}\n\n"
        f"Propose exactly {hypotheses_per_iteration} new feature hypotheses. "
        "Prefer ideas that are stable over time and cannot leak the target."
    )


def propose_hypotheses(
    llm_client: LLMClient,
    tier: ModelTierEntry,
    *,
    messages: list[dict],
    data_profile: DataProfile,
    kb_excerpt: str,
    feature_selection_summary: str,
    iteration: int,
    iterations_since_improvement: int,
    hypotheses_per_iteration: int,
) -> tuple[list[FeatureHypothesis], list[dict], LLMResponse]:
    user_text = build_user_turn(
        data_profile_summary=render_data_profile_summary(data_profile),
        kb_excerpt=kb_excerpt,
        feature_selection_summary=feature_selection_summary,
        iteration=iteration,
        iterations_since_improvement=iterations_since_improvement,
        hypotheses_per_iteration=hypotheses_per_iteration,
    )
    request_messages = [*messages, {"role": "user", "content": user_text}]

    response = llm_client.call(
        tier, system_prompt=SYSTEM_PROMPT, messages=request_messages, output_schema=OUTPUT_SCHEMA
    )

    updated_messages = [*request_messages, {"role": "assistant", "content": response.raw_content}]
    hypotheses = [
        FeatureHypothesis(iteration=iteration, proposed_by_model=tier.model, **h)
        for h in response.parsed["hypotheses"]
    ]
    return hypotheses, updated_messages, response

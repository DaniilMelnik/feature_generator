"""The code-generation half of the "feature engineer" agent: turns one
FeatureHypothesis into a FeatureSpec (Python source implementing the
fit/transform contract). Each hypothesis gets its own per-hypothesis
conversation (keyed by hypothesis_id in PipelineState.codegen_agent_messages)
so retry feedback (static-check violations, sandbox tracebacks, leakage-
reviewer concerns) stays scoped to that one hypothesis's thread.
"""

from __future__ import annotations

import json
from pathlib import Path

from feature_generator.agents.llm_client import LLMClient, LLMResponse
from feature_generator.config import ModelTierEntry
from feature_generator.profiling.schemas import DataProfile
from feature_generator.schemas import FeatureHypothesis, FeatureSpec

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "codegen_system.md").read_text()
OUTPUT_SCHEMA = json.loads((PROMPTS_DIR / "codegen_output_schema.json").read_text())


def render_hypothesis_context(hypothesis: FeatureHypothesis, data_profile: DataProfile) -> str:
    columns_by_name = {c.name: c for t in data_profile.tables for c in t.columns}
    lines = [
        f"Hypothesis: {hypothesis.description}",
        f"Rationale: {hypothesis.rationale}",
        f"Feature type: {hypothesis.feature_type}",
        f"requires_target_in_fit: {hypothesis.requires_target_in_fit}",
        "Input columns:",
    ]
    for name in hypothesis.input_columns:
        col = columns_by_name.get(name)
        if col is None:
            lines.append(f"  - {name} (not found in data profile)")
            continue
        detail = f"  - {name} [{col.dtype}]"
        if col.numeric_stats:
            detail += f" range=[{col.numeric_stats.min:.3g}, {col.numeric_stats.max:.3g}]"
        if col.categorical_stats:
            detail += f" cardinality={col.categorical_stats.cardinality}"
        lines.append(detail)
    return "\n".join(lines)


def build_user_turn(
    hypothesis: FeatureHypothesis, data_profile: DataProfile, feedback: str | None
) -> str:
    text = render_hypothesis_context(hypothesis, data_profile)
    if feedback:
        text += f"\n\nYour previous attempt was rejected. Fix this and resubmit:\n{feedback}"
    return text


def generate_feature_module(
    llm_client: LLMClient,
    tier: ModelTierEntry,
    *,
    messages: list[dict],
    hypothesis: FeatureHypothesis,
    data_profile: DataProfile,
    attempt: int,
    feedback: str | None = None,
) -> tuple[FeatureSpec, list[dict], LLMResponse]:
    user_text = build_user_turn(hypothesis, data_profile, feedback)
    request_messages = [*messages, {"role": "user", "content": user_text}]

    response = llm_client.call(
        tier, system_prompt=SYSTEM_PROMPT, messages=request_messages, output_schema=OUTPUT_SCHEMA
    )

    updated_messages = [*request_messages, {"role": "assistant", "content": response.raw_content}]
    parsed = response.parsed
    spec = FeatureSpec(
        hypothesis_id=hypothesis.id,
        feature_name=parsed["feature_name"],
        module_source=parsed["source_code"],
        declared_input_columns=hypothesis.input_columns,
        declared_input_tables=hypothesis.input_tables,
        output_dtype=parsed["output_dtype"],
        codegen_model=tier.model,
        codegen_attempt=attempt,
    )
    return spec, updated_messages, response

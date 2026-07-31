"""The adversarial leakage-reviewer agent: an independent outer-graph gate
(NOT part of the hypothesis/codegen subgraph), consistent with its
adversarial role. Reviews a whole batch of specs that survived static
checks in one call, rather than per-spec, since it runs far less often than
codegen and can afford the larger per-call context.
"""

from __future__ import annotations

import json
from pathlib import Path

from feature_generator.agents.llm_client import LLMClient, LLMResponse
from feature_generator.config import ModelTierEntry
from feature_generator.schemas import FeatureHypothesis, FeatureSpec, LeakageReviewVerdict

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "leakage_review_system.md").read_text()
OUTPUT_SCHEMA = json.loads((PROMPTS_DIR / "leakage_review_output_schema.json").read_text())


def build_user_turn(
    batch: list[tuple[FeatureHypothesis, FeatureSpec]], rejected_patterns_excerpt: str
) -> str:
    lines = [
        f"Known previously-rejected patterns:\n{rejected_patterns_excerpt}\n",
        "Review this batch of candidate features:",
    ]
    for hypothesis, spec in batch:
        lines.append(f"\n--- Feature: {spec.feature_name} ---")
        lines.append(f"Rationale: {hypothesis.rationale}")
        lines.append(f"requires_target_in_fit: {hypothesis.requires_target_in_fit}")
        lines.append("Source code:")
        lines.append(spec.module_source)
    return "\n".join(lines)


def adversarial_review(
    llm_client: LLMClient,
    tier: ModelTierEntry,
    *,
    messages: list[dict],
    batch: list[tuple[FeatureHypothesis, FeatureSpec]],
    rejected_patterns_excerpt: str = "(none yet)",
) -> tuple[dict[str, LeakageReviewVerdict], list[dict], LLMResponse | None]:
    if not batch:
        return {}, messages, None

    user_text = build_user_turn(batch, rejected_patterns_excerpt)
    request_messages = [*messages, {"role": "user", "content": user_text}]

    response = llm_client.call(
        tier,
        system_prompt=SYSTEM_PROMPT,
        messages=request_messages,
        output_schema=OUTPUT_SCHEMA,
        max_tokens=tier.max_output_tokens,
    )

    updated_messages = [*request_messages, {"role": "assistant", "content": response.raw_content}]
    verdicts = {
        v["feature_name"]: LeakageReviewVerdict(
            reviewer_model=tier.model,
            verdict=v["verdict"],
            concerns=v.get("concerns", []),
            confidence=v.get("confidence", 0.0),
        )
        for v in response.parsed["verdicts"]
    }
    return verdicts, updated_messages, response

"""Cheap Haiku-backed utility calls: re-summarizing an over-long knowledge-
base excerpt so it fits the hypothesis agent's context comfortably. No
structured output schema needed -- plain text in, plain text out.
"""

from __future__ import annotations

from feature_generator.agents.llm_client import LLMClient
from feature_generator.config import ModelTierEntry

SUMMARIZE_SYSTEM_PROMPT = (
    "You compress a running feature-engineering catalog to fit a target size "
    "while preserving every feature name, its final status, and its key metrics. "
    "Never invent information that is not present in the input. Output only the "
    "compressed catalog, no preamble."
)


def summarize_kb_excerpt(
    llm_client: LLMClient, tier: ModelTierEntry, *, excerpt: str, target_chars: int
) -> str:
    user_text = f"Compress the following to under {target_chars} characters:\n\n{excerpt}"
    response = llm_client.call(
        tier,
        system_prompt=SUMMARIZE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
        output_schema=None,
        max_tokens=2000,
    )
    return response.text

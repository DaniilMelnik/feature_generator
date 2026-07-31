"""Run configuration: everything that controls one pipeline run, loaded from a
YAML file (see ``configs/spaceship_titanic.yaml`` / ``configs/ieee_fraud.yaml``).

Model tiers, sandbox resource limits, retry bounds, and the time/token budget
are all config-driven here rather than hardcoded anywhere else in the
codebase, per the design's explicit "never hardcode model choice" rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Effort = Literal["low", "medium", "high", "xhigh", "max"]
Thinking = Literal["adaptive", "disabled"]


class ModelTierEntry(BaseModel):
    model: str
    effort: Effort = "high"
    thinking: Thinking = "adaptive"
    use_compaction: bool = False
    use_context_editing: bool = False
    # Escalation ladder: if set, `budget.BudgetTracker.should_escalate()` can
    # switch this role to the escalated model/effort after N stalled
    # iterations (see BudgetConfig.stalled_iterations_before_escalation).
    escalate_to_model: str | None = None
    escalate_to_effort: Effort | None = None
    # Response-length cap passed explicitly to LLMClient.call() by every agent
    # (was previously an implicit client-side default) -- lets a non-Anthropic
    # backend with a tight tokens-per-minute rate limit (e.g. a free tier) use
    # a much smaller cap than Anthropic's generous default.
    max_output_tokens: int = 32000

    def escalated(self) -> "ModelTierEntry":
        """Returns a new tier with the escalation-ladder overrides applied
        (model and/or effort), or an unchanged copy if none are configured.
        """
        return self.model_copy(
            update={
                "model": self.escalate_to_model or self.model,
                "effort": self.escalate_to_effort or self.effort,
            }
        )


class ModelTiersConfig(BaseModel):
    hypothesis: ModelTierEntry = ModelTierEntry(
        model="claude-sonnet-5",
        effort="high",
        use_compaction=True,
        escalate_to_model="claude-opus-5",
        escalate_to_effort="xhigh",
    )
    codegen: ModelTierEntry = ModelTierEntry(
        model="claude-sonnet-5",
        effort="xhigh",
        use_context_editing=True,
    )
    leakage_reviewer: ModelTierEntry = ModelTierEntry(
        model="claude-opus-5",
        effort="medium",
        escalate_to_effort="high",
    )
    utility: ModelTierEntry = ModelTierEntry(
        model="claude-haiku-4-5",
        effort="low",
        thinking="disabled",
    )


class LLMProviderConfig(BaseModel):
    """Which LLM backend build_dependencies() should construct. "anthropic" (default)
    keeps today's behavior unchanged. "openai_compatible" talks to any server exposing
    an OpenAI-compatible /chat/completions endpoint (Groq, a local Ollama/vLLM server,
    etc.) -- see agents/openai_compatible_client.py for the translation layer that lets
    the same hypothesize/codegen/leakage_review agent code run against either backend
    unmodified.
    """

    backend: Literal["anthropic", "openai_compatible"] = "anthropic"
    base_url: str | None = None  # required when backend == "openai_compatible"
    api_key_env: str = "GROQ_API_KEY"


class RetryConfig(BaseModel):
    max_codegen_retries: int = 3
    max_leakage_revision_retries: int = 2
    max_sandbox_retries: int = 2


class SandboxConfig(BaseModel):
    backend: Literal["docker", "subprocess"] = "docker"
    image_tag: str = "feature-gen-sandbox:latest"
    memory_limit: str = "1g"
    cpus: float = 1.0
    pids_limit: int = 64
    timeout_seconds: int = 60
    # batch calls (compute_oof_batch, transform_batch) intentionally do far
    # more work in one container invocation than a single fit/transform call
    # -- they need a distinctly larger timeout, kept separate so a genuine
    # single-call hang is still caught quickly by `timeout_seconds`.
    batch_timeout_seconds: int = 300
    scratch_dir: str = ".feature_gen_scratch"


class BudgetConfig(BaseModel):
    max_wall_clock_minutes: float | None = 60.0
    max_iterations: int | None = None
    max_total_tokens: int | None = None
    stalled_iterations_before_escalation: int = 3


class TableConfig(BaseModel):
    name: str
    path: str
    join_key: str | None = None
    role: Literal["primary", "secondary"] = "primary"


class DatasetConfig(BaseModel):
    name: str
    tables: list[TableConfig]
    target_column: str
    target_table: str
    id_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    task_type: Literal["binary_classification"] = "binary_classification"
    known_leakage_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_primary_table(self) -> "DatasetConfig":
        primaries = [t for t in self.tables if t.role == "primary"]
        if len(primaries) != 1:
            raise ValueError(
                f"dataset '{self.name}' must have exactly one primary table, "
                f"found {len(primaries)}"
            )
        return self


class StabilityConfig(BaseModel):
    method: Literal["temporal_oot", "bootstrap_resampling"] = "bootstrap_resampling"
    n_windows: int = 5
    bootstrap_iterations: int = 20
    psi_bins: int = 10


class FeatureSelectionConfig(BaseModel):
    max_evals: int = 50
    correlation_threshold: float = 0.95
    single_feature_auc_lift_min: float = 0.001
    single_feature_auc_leak_threshold: float = 0.90


class HoldoutConfig(BaseModel):
    # A slice of rows carved off once at run start that the CV loop, feature
    # selection, and the LLM never see -- a check against overfitting to the
    # CV procedure itself across many iterations. "temporal" (chronologically
    # last `fraction`) is the correct choice once a dataset has a real time
    # axis (e.g. IEEE-CIS); "random_stratified" for iid datasets.
    enabled: bool = True
    fraction: float = 0.15
    method: Literal["random_stratified", "temporal"] = "random_stratified"


class OutputConfig(BaseModel):
    run_dir: str = "runs"
    knowledge_base_path: str = "runs/knowledge_base.duckdb"
    results_store_path: str = "runs/results_store.duckdb"
    feature_store_dir: str = "runs/feature_store"
    checkpoint_sqlite_path: str = "runs/checkpoints.sqlite"


class RunConfig(BaseModel):
    dataset: DatasetConfig
    llm_provider: LLMProviderConfig = LLMProviderConfig()
    model_tiers: ModelTiersConfig = ModelTiersConfig()
    retries: RetryConfig = RetryConfig()
    sandbox: SandboxConfig = SandboxConfig()
    budget: BudgetConfig = BudgetConfig()
    stability: StabilityConfig = StabilityConfig()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    holdout: HoldoutConfig = HoldoutConfig()
    output: OutputConfig = OutputConfig()
    cv_folds: int = 5
    random_seed: int = 42
    hypotheses_per_iteration: int = 3


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)

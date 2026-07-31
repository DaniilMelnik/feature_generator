"""CLI entrypoints: ``feature-gen run|inspect``.

Wires the real dependencies (Anthropic-backed LLMClient, Docker-backed
sandbox, DuckDB knowledge base / results store) into ``orchestration.graph``
and drives ``run_pipeline`` until the configured budget is exhausted.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import click
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from feature_generator.agents.llm_client import LLMClient
from feature_generator.agents.openai_compatible_client import OpenAICompatibleLLMClient
from feature_generator.config import RunConfig, load_config
from feature_generator.dataset.builder import DatasetBuilder, build_raw_feature_frame
from feature_generator.dataset.feature_store import FeatureStore
from feature_generator.dataset.joiner import join_tables
from feature_generator.knowledge_base.db import KnowledgeBase
from feature_generator.knowledge_base.results_store import ResultsStore
from feature_generator.modeling.train import evaluate_holdout, train_catboost_cv
from feature_generator.orchestration.budget import BudgetTracker
from feature_generator.orchestration.graph import GraphDependencies, run_pipeline
from feature_generator.profiling.profiler import binarize_target, profile_tables
from feature_generator.reporting.report import build_run_report
from feature_generator.sandbox.contract import FeatureComputer
from feature_generator.sandbox.docker_runner import (
    DockerSandboxRunner,
    SandboxedFeatureComputer,
    docker_available,
)
from feature_generator.schemas import FeatureSpec


def make_feature_computer_factory(config: RunConfig):
    """LLM-generated code always runs sandboxed -- there is no in-process
    escape hatch for untrusted specs in the CLI path (tests use
    InProcessFeatureComputer directly with trusted, test-authored sources).
    """
    runner = DockerSandboxRunner(config.sandbox)

    def factory(spec: FeatureSpec, allow_target_in_fit: bool) -> FeatureComputer:
        return SandboxedFeatureComputer(
            spec.feature_name,
            spec.module_source,
            spec.declared_input_columns,
            runner,
            allow_target_in_fit=allow_target_in_fit,
        )

    return factory


def build_llm_client(config: RunConfig) -> LLMClient | OpenAICompatibleLLMClient:
    provider = config.llm_provider
    if provider.backend == "anthropic":
        return LLMClient()
    if provider.backend == "openai_compatible":
        if not provider.base_url:
            raise click.ClickException(
                "llm_provider.backend is 'openai_compatible' but llm_provider.base_url is not set."
            )
        api_key = os.environ.get(provider.api_key_env)
        if not api_key:
            raise click.ClickException(
                f"llm_provider.backend is 'openai_compatible' but the '{provider.api_key_env}' "
                "environment variable is not set. Add it to .env."
            )
        return OpenAICompatibleLLMClient(base_url=provider.base_url, api_key=api_key)
    raise click.ClickException(f"Unknown llm_provider.backend: {provider.backend!r}")


def llm_backend_summary(config: RunConfig) -> str:
    """Short, human-readable record of which backend/models produced a run --
    stored in run_metadata so runs made with different backends (e.g. the
    Anthropic-backed default vs. a free/local openai_compatible run) stay
    distinguishable when browsing the knowledge base later."""
    provider = config.llm_provider
    if provider.backend == "anthropic":
        tiers = config.model_tiers
        return (
            f"anthropic:hypothesis={tiers.hypothesis.model},codegen={tiers.codegen.model},"
            f"leakage_reviewer={tiers.leakage_reviewer.model}"
        )
    return f"{provider.backend}:{config.model_tiers.codegen.model}"


def build_dependencies(config: RunConfig) -> GraphDependencies:
    if config.sandbox.backend == "docker" and not docker_available():
        raise click.ClickException(
            "sandbox.backend is 'docker' but the Docker daemon is not reachable. "
            "Install and start Docker Desktop before running the pipeline."
        )

    data_profile = profile_tables(config.dataset)
    # join_tables() is a no-op pass-through for single-table datasets (see
    # dataset.joiner), so this is safe for both Spaceship Titanic and
    # multi-table datasets like IEEE-CIS Fraud Detection alike.
    base_df = join_tables(config.dataset)
    y = binarize_target(base_df[config.dataset.target_column])
    raw_feature_frame, raw_cat_features = build_raw_feature_frame(base_df, data_profile)

    Path(config.output.run_dir).mkdir(parents=True, exist_ok=True)
    knowledge_base = KnowledgeBase(config.output.knowledge_base_path)
    results_store = ResultsStore(config.output.results_store_path)
    feature_store = FeatureStore(config.output.feature_store_dir)
    dataset_builder = DatasetBuilder(cv_folds=config.cv_folds, random_seed=config.random_seed)
    llm_client = build_llm_client(config)

    if config.holdout.enabled:
        time_values = base_df[config.dataset.time_column] if config.holdout.method == "temporal" else None
        dev_index, holdout_index = dataset_builder.split_dev_holdout(
            y, config.holdout.fraction, method=config.holdout.method, time_values=time_values
        )
    else:
        dev_index, holdout_index = np.arange(len(y)), np.array([], dtype=int)

    return GraphDependencies(
        run_config=config,
        data_profile=data_profile,
        llm_client=llm_client,
        knowledge_base=knowledge_base,
        results_store=results_store,
        feature_store=feature_store,
        dataset_builder=dataset_builder,
        base_df=base_df,
        y=y,
        raw_feature_frame=raw_feature_frame,
        raw_cat_features=raw_cat_features,
        dev_index=dev_index,
        holdout_index=holdout_index,
        feature_computer_factory=make_feature_computer_factory(config),
        hypothesis_tier=config.model_tiers.hypothesis,
        codegen_tier=config.model_tiers.codegen,
        leakage_reviewer_tier=config.model_tiers.leakage_reviewer,
    )


@click.group()
def main() -> None:
    """Autonomous LLM feature-engineering pipeline for binary classification."""


@main.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False),
    help="Path to a run config YAML file (see configs/*.yaml).",
)
@click.option("--run-id", default=None, help="Reuse this run ID (e.g. to continue accumulating in the same knowledge base) instead of generating a fresh one.")
def run(config_path: str, run_id: str | None) -> None:
    """Run the pipeline until its configured time/token budget is exhausted."""
    config = load_config(config_path)
    deps = build_dependencies(config)
    resolved_run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
    tracker = BudgetTracker(config.budget)

    click.echo(f"Starting run '{resolved_run_id}' on dataset '{config.dataset.name}'")
    baseline_result = train_catboost_cv(
        deps.raw_feature_frame, deps.y, deps.folds, cat_features=deps.raw_cat_features
    )
    baseline_auc, _ = baseline_result.metric_mean_std("auc")
    baseline_line = (
        f"Features-off control baseline (raw columns only, {len(deps.raw_feature_frame.columns)} "
        f"column(s)): CV AUC = {baseline_auc:.4f}"
    )
    if len(deps.holdout_index):
        baseline_holdout = evaluate_holdout(
            deps.raw_feature_frame.iloc[deps.dev_index], deps.y.iloc[deps.dev_index],
            deps.raw_feature_frame.iloc[deps.holdout_index], deps.y.iloc[deps.holdout_index],
            cat_features=deps.raw_cat_features, random_seed=config.random_seed,
        )
        baseline_line += f", holdout AUC = {baseline_holdout['auc']:.4f}"
    click.echo(baseline_line + " -- this is the yardstick engineered features must beat")
    deps.knowledge_base.set_run_metadata(
        resolved_run_id,
        dataset_name=config.dataset.name,
        baseline_auc=baseline_auc,
        raw_feature_columns=list(deps.raw_feature_frame.columns),
        llm_backend=llm_backend_summary(config),
    )
    if config.budget.max_wall_clock_minutes:
        click.echo(f"Time budget: {config.budget.max_wall_clock_minutes} minute(s)")
    elif config.budget.max_iterations:
        click.echo(f"Iteration budget: {config.budget.max_iterations}")
    else:
        click.echo("No budget configured -- this will run until interrupted.")

    result = run_pipeline(deps, resolved_run_id, budget_tracker=tracker)

    click.echo(
        f"\nRun '{resolved_run_id}' stopped: {result.stopped_reason} "
        f"after {result.iterations_completed} iteration(s)"
    )
    best = result.final_state.get("best_metric_so_far")
    click.echo(f"Best CV AUC: {best:.4f}" if best is not None and best != float("-inf") else "Best CV AUC: n/a")
    best_metrics = deps.knowledge_base.get_best_training_metrics(resolved_run_id)
    if best_metrics:
        holdout_str = f"{best_metrics.holdout_auc:.4f}" if best_metrics.holdout_auc is not None else "n/a"
        click.echo(
            f"Best model breakdown -- train AUC: {best_metrics.train_auc_mean:.4f}, "
            f"CV AUC: {best_metrics.auc_mean:.4f} +/- {best_metrics.auc_std:.4f}, "
            f"holdout AUC: {holdout_str}"
        )
    promoted = deps.results_store.list_promoted_features(resolved_run_id)
    click.echo(f"Promoted features ({len(promoted)}): {[p['feature_name'] for p in promoted]}")
    click.echo(f"\nInspect further with: feature-gen inspect --config {config_path} --run-id {resolved_run_id}")


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--run-id", required=True)
@click.option("--show", type=click.Choice(["knowledge-base", "results"]), default="knowledge-base")
def inspect(config_path: str, run_id: str, show: str) -> None:
    """Inspect a run's knowledge base (all outcomes) or its promoted results."""
    config = load_config(config_path)
    if show == "knowledge-base":
        kb = KnowledgeBase(config.output.knowledge_base_path)
        results = kb.list_validation_results(run_id)
        by_status: dict[str, int] = {}
        for r in results:
            by_status[r.final_status] = by_status.get(r.final_status, 0) + 1
        click.echo(f"Run '{run_id}': {len(results)} feature outcome(s)")
        for status, count in sorted(by_status.items()):
            click.echo(f"  {status}: {count}")
        kb.close()
    else:
        store = ResultsStore(config.output.results_store_path)
        rows = store.list_promoted_features(run_id)
        if not rows:
            click.echo(f"No promoted features yet for run '{run_id}'")
        for row in rows:
            click.echo(
                f"  {row['feature_name']}  auc={row['auc_mean']}  "
                f"stability={row['stability_flag']}  promoted_at={row['promoted_at']}"
            )
        store.close()


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--run-id", required=True)
@click.option("--output", "output_path", default=None, type=click.Path(dir_okay=False), help="Defaults to <run_dir>/report_<run-id>.html")
def report(config_path: str, run_id: str, output_path: str | None) -> None:
    """Generate a self-contained HTML report: promoted features (rationale,
    code, stability, lift over baseline) and the feature-selection tool's
    per-iteration combination rounds.
    """
    config = load_config(config_path)
    html = build_run_report(config, run_id)

    resolved_output = Path(output_path) if output_path else Path(config.output.run_dir) / f"report_{run_id}.html"
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(html)

    click.echo(f"Report written to {resolved_output}")
    click.echo(f"Open with: open {resolved_output}")


if __name__ == "__main__":
    main()

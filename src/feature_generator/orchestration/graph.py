"""LangGraph wiring for the autonomous feature-engineering pipeline.

Architectural note -- a deliberate, EMPIRICALLY VERIFIED refinement of the
original design (kept here so future readers don't "fix" this back into a
broken shape): a single continuously-looping graph cannot give per-iteration
fan-in fields (``current_feature_specs`` etc.) a clean reset, because
LangGraph's ``operator.add`` reducer channels accumulate for the lifetime of
a thread -- looping the whole run through one ``.invoke()`` either
duplicates fan-in results (when the per-hypothesis retry subgraph is nested
inside the loop) or raises ``InvalidUpdateError`` (when it's flattened and
concurrent Send branches collide on private per-branch fields). Both failure
modes were reproduced and confirmed against the installed LangGraph version
before adopting this design.

The fix: the compiled graph here represents exactly ONE iteration
(hypothesize -> codegen+static-review retry -> leakage review (+bounded
revision retry) -> sandbox execution (+bounded retry) -> dataset assembly ->
feature selection -> training -> serving-parity -> stability -> knowledge
base -> promotion). ``run_pipeline()`` is a plain Python loop that invokes
this graph once per iteration, threads the persistent fields (budget,
accepted features, escalation tier, agent conversation history, ...)
between calls explicitly, and checks the budget in Python between
iterations. Each iteration still gets full LangGraph checkpointing
(resumable if the process dies mid-iteration); resuming a run that died
*between* iterations is a CLI-level concern (Step 7) that replays the
carry-forward fields from the knowledge base.

The hypothesize -> codegen fan-out (one Send per hypothesis) IS a genuine
nested subgraph (``codegen_branch``), because per-branch private state
(``hypothesis``, ``attempt``) must not collide across parallel branches --
this part of the original design was correct and is unchanged.

Holdout note: ``GraphDependencies.holdout_index`` (see ``dataset.builder.
split_dev_holdout``) marks rows carved out once at run start that CV folds,
feature engineering, and feature selection never see. ``train_model_node``
scores a dev-only-fit model against it purely as a human-facing diagnostic
(``TrainingMetrics.holdout_auc``, surfaced in the CLI and HTML report only).
It must never be threaded into ``kb_excerpt``, ``feature_selection_summary``,
or any agent prompt -- doing so would let the LLM adapt to it, destroying the
one check this project has against overfitting to the validation procedure
itself across many iterations.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Callable, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from feature_generator.agents import codegen_agent, hypothesis_agent, leakage_reviewer_agent
from feature_generator.agents.llm_client import LLMClient
from feature_generator.config import ModelTierEntry, RunConfig
from feature_generator.dataset.builder import DatasetBuilder, FoldSplit, build_raw_feature_frame
from feature_generator.dataset.feature_store import FeatureStore
from feature_generator.knowledge_base.db import KnowledgeBase
from feature_generator.knowledge_base.results_store import ResultsStore
from feature_generator.modeling.feature_selection import run_feature_selection
from feature_generator.modeling.metrics import compute_single_feature_auc, flag_single_feature_auc
from feature_generator.modeling.shap_utils import compute_shap_importance
from feature_generator.modeling.train import evaluate_holdout, train_catboost_cv
from feature_generator.orchestration.budget import BudgetTracker
from feature_generator.orchestration.state import BudgetState
from feature_generator.profiling.schemas import DataProfile
from feature_generator.sandbox.contract import FeatureComputer
from feature_generator.sandbox.static_checks import run_static_checks
from feature_generator.schemas import (
    DynamicCheckResult,
    FeatureHypothesis,
    FeatureSpec,
    FeatureStability,
    StabilityReport,
    StaticCheckResult,
    TrainingMetrics,
    ValidationResult,
)
from feature_generator.stability.csi_psi import classify_psi, compute_bootstrap_stability, compute_csi
from feature_generator.serving_parity.replay_simulator import run_iid, run_temporal


# --- Dependency injection ---------------------------------------------------


@dataclass
class GraphDependencies:
    """Everything a node closure needs. Large/non-checkpoint-friendly working
    data (dataframes, fitted params, fold assignments) lives here as a
    process-lifetime scratch space rather than in `IterationState` -- exactly
    like `feature_store`/`knowledge_base` already do; this just extends that
    pattern to in-memory working data.
    """

    run_config: RunConfig
    data_profile: DataProfile
    llm_client: LLMClient
    knowledge_base: KnowledgeBase
    results_store: ResultsStore
    feature_store: FeatureStore
    dataset_builder: DatasetBuilder
    base_df: pd.DataFrame
    y: pd.Series
    feature_computer_factory: Callable[[FeatureSpec, bool], FeatureComputer]

    hypothesis_tier: ModelTierEntry
    codegen_tier: ModelTierEntry
    leakage_reviewer_tier: ModelTierEntry

    # Overridable role functions -- default to the real agents; tests inject
    # stubs with the same signature to exercise control flow with zero API calls.
    hypothesize_fn: Callable = hypothesis_agent.propose_hypotheses
    codegen_fn: Callable = codegen_agent.generate_feature_module
    leakage_review_fn: Callable = leakage_reviewer_agent.adversarial_review

    folds: list[FoldSplit] = field(default_factory=list)

    # The profiler-identified raw dataset columns (see
    # dataset.builder.build_raw_feature_frame) -- the permanent foundation
    # every iteration's model trains on. Left empty by default so tests that
    # construct GraphDependencies directly (no real data_profile/base_df
    # correspondence) keep their existing all-engineered-features behavior;
    # the CLI always populates both from the real data profile.
    raw_feature_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_cat_features: list[str] = field(default_factory=list)

    # A one-time holdout carve-out (see dataset.builder.split_dev_holdout):
    # `dev_index` is what CV folds/feature engineering/feature selection ever
    # see; `holdout_index` is evaluated once per iteration purely as a
    # human-facing diagnostic (CLI/report) and MUST NEVER be threaded into
    # anything the hypothesis/feature-selection logic reads (kb_excerpt,
    # feature_selection_summary, agent prompts) -- see train_model_node.
    # Left as "no holdout" by default so tests that construct
    # GraphDependencies directly are unaffected; the CLI always populates
    # both from config.holdout.
    dev_index: np.ndarray | None = None
    holdout_index: np.ndarray | None = None

    # process-lifetime scratch space, written by earlier nodes, read by later
    # ones within the SAME iteration (never checkpointed, never carried
    # across iterations except via what's explicitly re-derived from
    # feature_store/knowledge_base)
    working_X_baseline: pd.DataFrame | None = field(default=None, repr=False)
    working_candidates: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    working_selection: object | None = field(default=None, repr=False)
    working_final_features: list[str] = field(default_factory=list, repr=False)
    working_stability: StabilityReport | None = field(default=None, repr=False)
    feature_specs_by_name: dict[str, FeatureSpec] = field(default_factory=dict, repr=False)
    feature_full_fit_params: dict[str, dict] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.dev_index is None:
            self.dev_index = np.arange(len(self.y))
        if self.holdout_index is None:
            self.holdout_index = np.array([], dtype=int)
        if not self.folds:
            self.folds = self.dataset_builder.make_folds(self.y, restrict_to=self.dev_index)
        if len(self.raw_feature_frame.columns) == 0:
            self.raw_feature_frame = pd.DataFrame(index=self.base_df.index)
        else:
            self.raw_feature_frame = self.raw_feature_frame.reindex(self.base_df.index)


def _failed_codegen_placeholder(
    deps: GraphDependencies, hypothesis: FeatureHypothesis, attempt: int, error: str
) -> tuple[FeatureSpec, StaticCheckResult]:
    """A synthetic spec + failing StaticCheckResult -- lets a codegen call
    that raised (LLMResponseError from a truncated/malformed response, or
    any other transient failure) be treated exactly like a genuine static-
    check violation by the existing retry/give-up routing, instead of
    crashing the whole iteration.
    """
    placeholder = FeatureSpec(
        hypothesis_id=hypothesis.id,
        feature_name=f"codegen_failed_{hypothesis.id}",
        module_source="",
        declared_input_columns=hypothesis.input_columns,
        declared_input_tables=hypothesis.input_tables,
        output_dtype="float64",
        codegen_model=deps.codegen_tier.model,
        codegen_attempt=attempt,
    )
    return placeholder, StaticCheckResult(passed=False, violations=[f"codegen call failed: {error}"])


def _regenerate_with_feedback(
    deps: GraphDependencies, hypothesis: FeatureHypothesis, attempt: int, feedback: str
) -> tuple[FeatureSpec, StaticCheckResult]:
    try:
        new_spec, _, _ = deps.codegen_fn(
            deps.llm_client,
            deps.codegen_tier,
            messages=[],
            hypothesis=hypothesis,
            data_profile=deps.data_profile,
            attempt=attempt,
            feedback=feedback,
        )
    except Exception as exc:  # noqa: BLE001 -- any codegen-call failure (LLMResponseError or otherwise), never crash the run
        return _failed_codegen_placeholder(deps, hypothesis, attempt, str(exc))

    static_result = run_static_checks(
        new_spec.module_source,
        forbidden_target_column=deps.run_config.dataset.target_column,
        declared_input_columns=new_spec.declared_input_columns,
        allow_target_in_fit=hypothesis.requires_target_in_fit,
    )
    return new_spec, static_result


def _lift_for_feature(selection, feature_name: str) -> float | None:
    """How much CV AUC this specific feature added at the point the greedy
    stepwise search added it -- `selection.auc_trace` is an ordered list of
    (feature_name, resulting_auc); the lift is the delta from the previous
    step's AUC (or from `baseline_auc` if it's the first step).
    """
    if selection is None:
        return None
    previous_auc = selection.baseline_auc
    for name, auc in selection.auc_trace:
        if name == feature_name:
            return auc - previous_auc
        previous_auc = auc
    return None


def _assemble_holdout_frame(deps: GraphDependencies, final_features: list[str]) -> pd.DataFrame:
    """Reconstruct `final_features` on holdout rows the leakage-safe way: raw
    columns need no computation; engineered columns go through the same
    `transform_new_data` path used for serving-parity checks, using fit
    params that only ever saw dev rows (see `compute_oof_feature`'s
    `dev_index`) applied fresh to rows nothing in the iterative loop has
    touched.
    """
    holdout_df = deps.base_df.iloc[deps.holdout_index]
    raw_columns = set(deps.raw_feature_frame.columns)
    engineered_names = [name for name in final_features if name not in raw_columns]
    computers = [deps.feature_computer_factory(deps.feature_specs_by_name[name], False) for name in engineered_names]
    engineered = deps.dataset_builder.transform_new_data(holdout_df, computers, deps.feature_full_fit_params)
    combined = pd.concat([deps.raw_feature_frame.iloc[deps.holdout_index], engineered], axis=1)
    return combined[final_features]


def _cat_features_for(deps: GraphDependencies, columns: list[str]) -> list[str]:
    """Raw categorical columns plus any LLM-generated feature whose declared
    ``output_dtype`` is "category" -- restricted to columns actually present
    in the frame being trained on right now.
    """
    generated = [name for name, spec in deps.feature_specs_by_name.items() if spec.output_dtype == "category"]
    return [c for c in [*deps.raw_cat_features, *generated] if c in columns]


def _merge_validation_results(results: list[dict]) -> list[dict]:
    """Later entries for the same feature_name override only their non-null
    fields -- e.g. a later serving-parity failure overrides `final_status`
    without discarding the earlier dynamic-check details.
    """
    merged: dict[str, dict] = {}
    for r in results:
        name = r["feature_name"]
        if name not in merged:
            merged[name] = dict(r)
        else:
            for key, value in r.items():
                if value is not None:
                    merged[name][key] = value
    return list(merged.values())


# --- Per-hypothesis codegen+static-review retry subgraph --------------------


class CodegenBranchState(TypedDict, total=False):
    hypothesis: dict
    attempt: int
    feedback: str | None
    messages: list[dict]
    current_spec: dict | None
    current_static_result: dict | None
    current_feature_specs: Annotated[list[dict], operator.add]
    current_validation_results: Annotated[list[dict], operator.add]


def build_codegen_branch_subgraph(deps: GraphDependencies) -> CompiledStateGraph:
    max_retries = deps.run_config.retries.max_codegen_retries

    def codegen_node(state: CodegenBranchState) -> dict:
        hypothesis = FeatureHypothesis.model_validate(state["hypothesis"])
        attempt = state.get("attempt", 0)
        try:
            spec, messages, _response = deps.codegen_fn(
                deps.llm_client,
                deps.codegen_tier,
                messages=state.get("messages", []),
                hypothesis=hypothesis,
                data_profile=deps.data_profile,
                attempt=attempt,
                feedback=state.get("feedback"),
            )
        except Exception as exc:  # noqa: BLE001 -- any codegen-call failure, routed through the
            # existing static-check retry/give-up machinery below instead of crashing the run
            spec, static_result = _failed_codegen_placeholder(deps, hypothesis, attempt, str(exc))
            return {
                "hypothesis": state["hypothesis"],
                "attempt": attempt,
                "messages": state.get("messages", []),
                "current_spec": spec.model_dump(),
                "current_static_result": static_result.model_dump(),
            }
        return {
            "hypothesis": state["hypothesis"],
            "attempt": attempt,
            "messages": messages,
            "current_spec": spec.model_dump(),
        }

    def static_review_node(state: CodegenBranchState) -> dict:
        spec = state["current_spec"]
        if not spec["module_source"]:
            # codegen_node already produced a failing static_result for this
            # branch (the codegen call itself raised, e.g. LLMResponseError)
            # -- nothing to statically check on an empty module; pass that
            # result through rather than overwriting it with a fresh (and
            # trivially-passing, since an empty AST has nothing to deny) one.
            return {"current_static_result": state["current_static_result"]}
        hypothesis = FeatureHypothesis.model_validate(state["hypothesis"])
        result = run_static_checks(
            spec["module_source"],
            forbidden_target_column=deps.run_config.dataset.target_column,
            declared_input_columns=spec["declared_input_columns"],
            allow_target_in_fit=hypothesis.requires_target_in_fit,
        )
        return {"current_static_result": result.model_dump()}

    def route_after_static_review(state: CodegenBranchState) -> str:
        if state["current_static_result"]["passed"]:
            return "success"
        if state.get("attempt", 0) < max_retries:
            return "retry"
        return "failure"

    def retry_bump_node(state: CodegenBranchState) -> dict:
        violations = state["current_static_result"]["violations"]
        feedback = "Static check violations:\n" + "\n".join(violations)
        return {"hypothesis": state["hypothesis"], "attempt": state.get("attempt", 0) + 1, "feedback": feedback}

    def emit_success_node(state: CodegenBranchState) -> dict:
        return {"current_feature_specs": [state["current_spec"]]}

    def emit_failure_node(state: CodegenBranchState) -> dict:
        hypothesis = state["hypothesis"]
        validation = ValidationResult(
            feature_name=state["current_spec"]["feature_name"],
            hypothesis_id=hypothesis["id"],
            static=StaticCheckResult.model_validate(state["current_static_result"]),
            final_status="static_failed",
        )
        return {"current_validation_results": [validation.model_dump()]}

    graph = StateGraph(CodegenBranchState)
    graph.add_node("codegen", codegen_node)
    graph.add_node("static_review", static_review_node)
    graph.add_node("retry_bump", retry_bump_node)
    graph.add_node("emit_success", emit_success_node)
    graph.add_node("emit_failure", emit_failure_node)
    graph.add_edge(START, "codegen")
    graph.add_edge("codegen", "static_review")
    graph.add_conditional_edges(
        "static_review",
        route_after_static_review,
        {"success": "emit_success", "retry": "retry_bump", "failure": "emit_failure"},
    )
    graph.add_edge("retry_bump", "codegen")
    graph.add_edge("emit_success", END)
    graph.add_edge("emit_failure", END)
    return graph.compile()


# --- One-iteration graph -----------------------------------------------------


class IterationState(TypedDict, total=False):
    run_id: str
    iteration: int
    kb_excerpt: str
    feature_selection_summary: str
    hypotheses_per_iteration: int
    iterations_since_improvement: int
    escalation_tier: str
    hypothesis_agent_messages: list[dict]
    accepted_feature_names: list[str]
    best_metric_so_far: float

    current_hypotheses: list[dict]
    current_feature_specs: Annotated[list[dict], operator.add]
    approved_feature_specs: list[dict]
    newly_validated_feature_names: list[str]
    current_validation_results: Annotated[list[dict], operator.add]
    current_training_metrics: dict | None
    current_stability_report: dict | None


def _hypothesis_tier_for(deps: GraphDependencies, state: IterationState) -> ModelTierEntry:
    return deps.hypothesis_tier.escalated() if state.get("escalation_tier") == "escalated" else deps.hypothesis_tier


def build_iteration_graph(deps: GraphDependencies) -> CompiledStateGraph:
    codegen_branch = build_codegen_branch_subgraph(deps)

    # --- hypothesize + fan-out --------------------------------------------

    def hypothesize_node(state: IterationState) -> dict:
        tier = _hypothesis_tier_for(deps, state)
        hypotheses, messages, _response = deps.hypothesize_fn(
            deps.llm_client,
            tier,
            messages=state.get("hypothesis_agent_messages", []),
            data_profile=deps.data_profile,
            kb_excerpt=state.get("kb_excerpt", ""),
            feature_selection_summary=state.get("feature_selection_summary", ""),
            iteration=state.get("iteration", 0),
            iterations_since_improvement=state.get("iterations_since_improvement", 0),
            hypotheses_per_iteration=state.get("hypotheses_per_iteration", 3),
        )
        return {
            "hypothesis_agent_messages": messages,
            "current_hypotheses": [h.model_dump() for h in hypotheses],
        }

    def fan_out_to_codegen(state: IterationState) -> list[Send]:
        hypotheses = state.get("current_hypotheses", [])
        if not hypotheses:
            # No hypotheses this iteration (e.g. a stubbed/degenerate case in
            # tests) -- an empty Send list would dead-end the graph right
            # here (verified: LangGraph does not fall through to any static
            # edge in that case), so route straight to leakage_review with an
            # empty candidate set instead, ensuring promote/budget bookkeeping
            # for this iteration still runs.
            return [Send("leakage_review", {**state, "current_feature_specs": []})]
        return [
            Send("codegen_branch", {"hypothesis": h, "attempt": 0, "feedback": None, "messages": []})
            for h in hypotheses
        ]

    # --- leakage review (+ bounded revision retry, sequential) ------------

    def leakage_review_node(state: IterationState) -> dict:
        max_retries = deps.run_config.retries.max_leakage_revision_retries
        hypotheses_by_id = {h["id"]: FeatureHypothesis.model_validate(h) for h in state.get("current_hypotheses", [])}
        pending = {s["feature_name"]: s for s in state.get("current_feature_specs", [])}
        attempts = {name: 0 for name in pending}
        approved: list[dict] = []
        validation_results: list[dict] = []
        rejected_patterns_excerpt = deps.knowledge_base.render_kb_excerpt(state["run_id"], max_chars=2000)

        while pending:
            batch = [
                (hypotheses_by_id[spec["hypothesis_id"]], FeatureSpec.model_validate(spec))
                for spec in pending.values()
            ]
            verdicts, _messages, _response = deps.leakage_review_fn(
                deps.llm_client,
                deps.leakage_reviewer_tier,
                messages=[],
                batch=batch,
                rejected_patterns_excerpt=rejected_patterns_excerpt,
            )

            next_pending: dict[str, dict] = {}
            for name, spec_dict in pending.items():
                hypothesis = hypotheses_by_id[spec_dict["hypothesis_id"]]
                verdict = verdicts.get(name)

                if verdict is None or verdict.verdict == "approve":
                    approved.append(spec_dict)
                    continue

                if verdict.verdict == "reject" or attempts[name] >= max_retries:
                    validation_results.append(
                        ValidationResult(
                            feature_name=name,
                            hypothesis_id=hypothesis.id,
                            static=StaticCheckResult(passed=True),
                            leakage_review=verdict,
                            final_status="leakage_rejected",
                        ).model_dump()
                    )
                    continue

                feedback = "Leakage reviewer concerns:\n" + "\n".join(verdict.concerns)
                new_spec, static_result = _regenerate_with_feedback(
                    deps, hypothesis, attempts[name] + 1, feedback
                )
                new_attempt = attempts.pop(name) + 1
                if not static_result.passed:
                    validation_results.append(
                        ValidationResult(
                            feature_name=new_spec.feature_name,
                            hypothesis_id=hypothesis.id,
                            static=static_result,
                            final_status="static_failed",
                        ).model_dump()
                    )
                    continue
                attempts[new_spec.feature_name] = new_attempt
                next_pending[new_spec.feature_name] = new_spec.model_dump()
            pending = next_pending

        return {"approved_feature_specs": approved, "current_validation_results": validation_results}

    # --- sandbox execution (+ bounded retry, sequential) ------------------

    def sandbox_execute_node(state: IterationState) -> dict:
        max_retries = deps.run_config.retries.max_sandbox_retries
        hypotheses_by_id = {h["id"]: FeatureHypothesis.model_validate(h) for h in state.get("current_hypotheses", [])}
        pending = {s["feature_name"]: s for s in state.get("approved_feature_specs", [])}
        attempts = {name: 0 for name in pending}
        validated_names: list[str] = []
        validation_results: list[dict] = []
        leak_threshold = deps.run_config.feature_selection.single_feature_auc_leak_threshold

        while pending:
            next_pending: dict[str, dict] = {}
            for name, spec_dict in pending.items():
                spec = FeatureSpec.model_validate(spec_dict)
                hypothesis = hypotheses_by_id[spec.hypothesis_id]
                computer = deps.feature_computer_factory(spec, hypothesis.requires_target_in_fit)

                try:
                    oof_result = deps.dataset_builder.compute_oof_feature(
                        deps.base_df,
                        computer,
                        deps.folds,
                        target_column=deps.run_config.dataset.target_column,
                        time_column=deps.run_config.dataset.time_column,
                        id_columns=deps.run_config.dataset.id_columns,
                        dev_index=deps.dev_index,
                    )
                except Exception as exc:  # noqa: BLE001 -- deliberately broad: any sandboxed-code failure
                    if attempts[name] < max_retries:
                        new_spec, static_result = _regenerate_with_feedback(
                            deps, hypothesis, attempts[name] + 1, f"Sandbox execution raised: {exc}"
                        )
                        new_attempt = attempts.pop(name) + 1
                        if static_result.passed:
                            attempts[new_spec.feature_name] = new_attempt
                            next_pending[new_spec.feature_name] = new_spec.model_dump()
                        else:
                            validation_results.append(
                                ValidationResult(
                                    feature_name=new_spec.feature_name,
                                    hypothesis_id=hypothesis.id,
                                    static=static_result,
                                    final_status="static_failed",
                                ).model_dump()
                            )
                        continue
                    validation_results.append(
                        ValidationResult(
                            feature_name=name,
                            hypothesis_id=hypothesis.id,
                            static=StaticCheckResult(passed=True),
                            dynamic=DynamicCheckResult(fold_fit_transform_ok=False, exceptions=[str(exc)]),
                            final_status="dynamic_failed",
                        ).model_dump()
                    )
                    continue

                deps.feature_store.save(state["run_id"], spec.feature_name, oof_result.oof_values)
                deps.feature_full_fit_params[spec.feature_name] = oof_result.full_fit_params
                deps.feature_specs_by_name[spec.feature_name] = spec

                single_auc = compute_single_feature_auc(oof_result.oof_values, deps.y)
                dynamic_result = DynamicCheckResult(
                    fold_fit_transform_ok=True,
                    single_feature_auc=single_auc,
                    single_feature_auc_flag=flag_single_feature_auc(single_auc, leak_threshold),
                )
                validation_results.append(
                    ValidationResult(
                        feature_name=spec.feature_name,
                        hypothesis_id=hypothesis.id,
                        static=StaticCheckResult(passed=True),
                        dynamic=dynamic_result,
                        final_status="validated",
                    ).model_dump()
                )
                validated_names.append(spec.feature_name)
            pending = next_pending

        return {"newly_validated_feature_names": validated_names, "current_validation_results": validation_results}

    # --- dataset assembly + feature selection + training ------------------

    def assemble_dataset_node(state: IterationState) -> dict:
        run_id = state["run_id"]
        X_baseline = deps.raw_feature_frame.copy()
        for name in state.get("accepted_feature_names", []):
            X_baseline[name] = deps.feature_store.load(run_id, name)

        deps.working_X_baseline = X_baseline
        deps.working_candidates = {
            name: deps.feature_store.load(run_id, name) for name in state.get("newly_validated_feature_names", [])
        }
        return {}

    def feature_selection_node(state: IterationState) -> dict:
        if not deps.working_candidates:
            deps.working_selection = None
            return {"feature_selection_summary": "No new validated candidates this iteration."}

        fs_config = deps.run_config.feature_selection
        cat_features = _cat_features_for(
            deps, [*deps.working_X_baseline.columns, *deps.working_candidates.keys()]
        )
        output = run_feature_selection(
            deps.working_X_baseline,
            deps.working_candidates,
            deps.y,
            deps.folds,
            cat_features=cat_features,
            correlation_threshold=fs_config.correlation_threshold,
            min_lift=fs_config.single_feature_auc_lift_min,
            max_evals=fs_config.max_evals,
        )
        deps.working_selection = output
        deps.knowledge_base.add_feature_selection_round(state["run_id"], state["iteration"], output)
        summary = (
            f"Lift-screen accepted: {output.accepted_candidates}; rejected: {output.rejected_by_lift}; "
            f"final selected set: {output.final_selected} "
            f"(baseline AUC {output.baseline_auc:.4f} -> final AUC {output.final_auc:.4f})"
        )
        return {"feature_selection_summary": summary}

    def train_model_node(state: IterationState) -> dict:
        selection = deps.working_selection
        final_features = list(selection.final_selected) if selection else list(deps.working_X_baseline.columns)
        deps.working_final_features = final_features
        if not final_features:
            return {"current_training_metrics": None}

        X = pd.DataFrame(index=deps.base_df.index)
        for name in final_features:
            X[name] = (
                deps.working_X_baseline[name]
                if name in deps.working_X_baseline.columns
                else deps.working_candidates[name]
            )
        deps.working_X = X

        cat_features = _cat_features_for(deps, list(X.columns))
        cv_result = train_catboost_cv(X, deps.y, deps.folds, cat_features=cat_features)
        auc_mean, auc_std = cv_result.metric_mean_std("auc")
        train_auc_mean, train_auc_std = cv_result.train_metric_mean_std("auc")
        logloss_mean, _ = cv_result.metric_mean_std("logloss")
        pr_auc_mean, _ = cv_result.metric_mean_std("pr_auc")
        ks_mean, _ = cv_result.metric_mean_std("ks")
        brier_mean, _ = cv_result.metric_mean_std("brier")
        shap_importance = compute_shap_importance(cv_result.models, X, deps.folds, cat_features=cat_features)

        # Human-facing-only diagnostic: a model fit on dev rows, scored once
        # against rows the CV loop above never touched. Must never be
        # threaded into kb_excerpt, feature_selection_summary, or any agent
        # prompt -- see the module docstring's holdout note.
        holdout_auc = None
        if len(deps.holdout_index):
            X_holdout = _assemble_holdout_frame(deps, final_features)
            holdout_metrics = evaluate_holdout(
                X.iloc[deps.dev_index],
                deps.y.iloc[deps.dev_index],
                X_holdout,
                deps.y.iloc[deps.holdout_index],
                cat_features=cat_features,
                random_seed=deps.run_config.random_seed,
            )
            holdout_auc = holdout_metrics["auc"]

        metrics = TrainingMetrics(
            run_id=state["run_id"],
            feature_set_id="|".join(sorted(final_features)),
            iteration=state["iteration"],
            cv_folds=len(deps.folds),
            auc_mean=auc_mean,
            auc_std=auc_std,
            train_auc_mean=train_auc_mean,
            train_auc_std=train_auc_std,
            holdout_auc=holdout_auc,
            logloss_mean=logloss_mean,
            pr_auc_mean=pr_auc_mean,
            ks_stat_mean=ks_mean,
            brier_score_mean=brier_mean,
            shap_importance=shap_importance,
            train_rows=len(X),
            val_rows=len(X) // max(len(deps.folds), 1),
        )
        return {"current_training_metrics": metrics.model_dump()}

    # --- serving parity + stability ----------------------------------------

    def serving_parity_check_node(state: IterationState) -> dict:
        results: list[dict] = []
        time_column = deps.run_config.dataset.time_column
        for name in state.get("newly_validated_feature_names", []):
            spec = deps.feature_specs_by_name.get(name)
            if spec is None:
                continue
            computer = deps.feature_computer_factory(spec, False)
            params = deps.feature_full_fit_params[name]
            parity = (
                run_temporal(computer, params, deps.base_df, time_column)
                if time_column
                else run_iid(computer, params, deps.base_df)
            )
            if not parity.matched:
                hypothesis_id = spec.hypothesis_id
                results.append(
                    ValidationResult(
                        feature_name=name,
                        hypothesis_id=hypothesis_id,
                        static=StaticCheckResult(passed=True),
                        serving_parity=parity,
                        final_status="serving_mismatch",
                    ).model_dump()
                )
        return {"current_validation_results": results}

    def stability_check_node(state: IterationState) -> dict:
        names = state.get("newly_validated_feature_names", [])
        if not names:
            deps.working_stability = None
            return {"current_stability_report": None}

        method = deps.run_config.stability.method
        feature_stabilities: list[FeatureStability] = []
        for name in names:
            series = deps.feature_store.load(state["run_id"], name)
            if method == "temporal_oot" and deps.run_config.dataset.time_column:
                time_series = deps.base_df[deps.run_config.dataset.time_column]
                order = time_series.sort_values().index
                windows = np.array_split(order, deps.run_config.stability.n_windows)
                reference = series.loc[windows[0]]
                psi_over_time = [
                    (f"window_{i}", compute_csi(reference, series.loc[w], bins=deps.run_config.stability.psi_bins))
                    for i, w in enumerate(windows[1:], start=1)
                ]
            else:
                psi_over_time = compute_bootstrap_stability(
                    series, deps.run_config.stability.bootstrap_iterations, bins=deps.run_config.stability.psi_bins
                )
            csi = max((v for _, v in psi_over_time), default=0.0)
            feature_stabilities.append(
                FeatureStability(feature_name=name, csi=csi, psi_over_time=psi_over_time, method=method)
            )

        worst = max((f.csi or 0.0 for f in feature_stabilities), default=0.0)
        report = StabilityReport(
            feature_set_id="|".join(sorted(names)),
            iteration=state["iteration"],
            features=feature_stabilities,
            overall_stability_flag=classify_psi(worst),
            method_note=(
                "bootstrap stand-in: no time axis" if method == "bootstrap_resampling" else "temporal OOT windows"
            ),
        )
        deps.working_stability = report
        return {"current_stability_report": report.model_dump()}

    # --- knowledge base + promotion -----------------------------------------

    def update_knowledge_base_node(state: IterationState) -> dict:
        run_id = state["run_id"]
        for h in state.get("current_hypotheses", []):
            deps.knowledge_base.add_hypothesis(run_id, FeatureHypothesis.model_validate(h))
        seen_specs: set[str] = set()
        for s in [*state.get("current_feature_specs", []), *state.get("approved_feature_specs", [])]:
            key = f"{s['hypothesis_id']}:{s['codegen_attempt']}"
            if key in seen_specs:
                continue
            seen_specs.add(key)
            deps.knowledge_base.add_feature_spec(FeatureSpec.model_validate(s))
        for v in _merge_validation_results(state.get("current_validation_results", [])):
            deps.knowledge_base.add_validation_result(run_id, ValidationResult.model_validate(v))
        if state.get("current_training_metrics"):
            deps.knowledge_base.add_training_metrics(run_id, TrainingMetrics.model_validate(state["current_training_metrics"]))
        if state.get("current_stability_report"):
            deps.knowledge_base.add_stability_report(run_id, StabilityReport.model_validate(state["current_stability_report"]))

        return {"kb_excerpt": deps.knowledge_base.render_kb_excerpt(run_id)}

    def promote_node(state: IterationState) -> dict:
        metrics_dict = state.get("current_training_metrics")
        best_so_far = state.get("best_metric_so_far", float("-inf"))
        if not metrics_dict:
            return {"iterations_since_improvement": state.get("iterations_since_improvement", 0) + 1}

        metrics = TrainingMetrics.model_validate(metrics_dict)
        improved = metrics.auc_mean > best_so_far
        stability = deps.working_stability
        stability_ok = stability is None or stability.overall_stability_flag != "unstable"

        merged = {
            r["feature_name"]: ValidationResult.model_validate(r)
            for r in _merge_validation_results(state.get("current_validation_results", []))
        }

        new_accepted = list(state.get("accepted_feature_names", []))
        for name in state.get("newly_validated_feature_names", []):
            result = merged.get(name)
            if result and result.final_status == "validated" and name not in new_accepted:
                new_accepted.append(name)

        if improved and stability_ok:
            for name in state.get("newly_validated_feature_names", []):
                result = merged.get(name)
                if result and result.final_status == "validated" and name in deps.working_final_features:
                    spec = deps.feature_specs_by_name[name]
                    if not deps.results_store.is_promoted(name):
                        deps.results_store.promote(
                            state["run_id"],
                            spec,
                            result,
                            metrics,
                            stability,
                            iteration=state["iteration"],
                            lift_over_baseline=_lift_for_feature(deps.working_selection, name),
                        )

        return {
            "accepted_feature_names": new_accepted,
            "best_metric_so_far": max(metrics.auc_mean, best_so_far),
            "iterations_since_improvement": 0 if improved else state.get("iterations_since_improvement", 0) + 1,
        }

    graph = StateGraph(IterationState)
    graph.add_node("hypothesize", hypothesize_node)
    graph.add_node("codegen_branch", codegen_branch)
    graph.add_node("leakage_review", leakage_review_node)
    graph.add_node("sandbox_execute", sandbox_execute_node)
    graph.add_node("assemble_dataset", assemble_dataset_node)
    graph.add_node("feature_selection", feature_selection_node)
    graph.add_node("train_model", train_model_node)
    graph.add_node("serving_parity_check", serving_parity_check_node)
    graph.add_node("stability_check", stability_check_node)
    graph.add_node("update_knowledge_base", update_knowledge_base_node)
    graph.add_node("promote", promote_node)

    graph.add_edge(START, "hypothesize")
    graph.add_conditional_edges("hypothesize", fan_out_to_codegen, ["codegen_branch", "leakage_review"])
    graph.add_edge("codegen_branch", "leakage_review")
    graph.add_edge("leakage_review", "sandbox_execute")
    graph.add_edge("sandbox_execute", "assemble_dataset")
    graph.add_edge("assemble_dataset", "feature_selection")
    graph.add_edge("feature_selection", "train_model")
    graph.add_edge("train_model", "serving_parity_check")
    graph.add_edge("serving_parity_check", "stability_check")
    graph.add_edge("stability_check", "update_knowledge_base")
    graph.add_edge("update_knowledge_base", "promote")
    graph.add_edge("promote", END)

    return graph.compile()


# --- Python-level run loop ---------------------------------------------------


@dataclass
class RunResult:
    run_id: str
    iterations_completed: int
    final_state: dict
    stopped_reason: str


def run_pipeline(
    deps: GraphDependencies,
    run_id: str,
    *,
    budget_tracker: BudgetTracker,
    checkpointer=None,
    max_iterations_hard_cap: int = 100_000,
) -> RunResult:
    """The actual "loop until the operator's time/token budget runs out"
    behavior -- implemented in Python, not as a graph self-loop (see module
    docstring for why). Each iteration is one full `.invoke()` of the
    compiled iteration graph, checkpointed under its own thread_id so a
    crash mid-iteration can resume that specific iteration.
    """
    iteration_graph = build_iteration_graph(deps)

    budget: BudgetState = budget_tracker.new_state()
    kb_excerpt = deps.knowledge_base.render_kb_excerpt(run_id)
    carry: dict = {
        "run_id": run_id,
        "iteration": 0,
        "kb_excerpt": kb_excerpt,
        "feature_selection_summary": "",
        "hypotheses_per_iteration": deps.run_config.hypotheses_per_iteration,
        "iterations_since_improvement": 0,
        "escalation_tier": "base",
        "hypothesis_agent_messages": [],
        "accepted_feature_names": [],
        "best_metric_so_far": float("-inf"),
    }

    stopped_reason = "max_iterations_hard_cap"
    final_state: dict = dict(carry)

    for _ in range(max_iterations_hard_cap):
        if budget_tracker.is_exhausted(budget, iteration=carry["iteration"]):
            stopped_reason = "budget_exhausted"
            break

        usage_before = _snapshot_usage(deps.llm_client)
        config = {"configurable": {"thread_id": f"{run_id}-iter-{carry['iteration']}"}}
        invoke_kwargs = {"config": config} if checkpointer is None else {"config": config}
        result = iteration_graph.invoke(dict(carry), **invoke_kwargs)
        usage_after = _snapshot_usage(deps.llm_client)
        budget = budget_tracker.record_usage(budget, _usage_delta(usage_before, usage_after))

        next_iteration = carry["iteration"] + 1
        iterations_since_improvement = result.get("iterations_since_improvement", carry["iterations_since_improvement"])
        escalation_tier = "escalated" if budget_tracker.should_escalate(iterations_since_improvement) else "base"

        carry = {
            "run_id": run_id,
            "iteration": next_iteration,
            "kb_excerpt": result.get("kb_excerpt", carry["kb_excerpt"]),
            "feature_selection_summary": result.get("feature_selection_summary", ""),
            "hypotheses_per_iteration": deps.run_config.hypotheses_per_iteration,
            "iterations_since_improvement": iterations_since_improvement,
            "escalation_tier": escalation_tier,
            "hypothesis_agent_messages": result.get("hypothesis_agent_messages", carry["hypothesis_agent_messages"]),
            "accepted_feature_names": result.get("accepted_feature_names", carry["accepted_feature_names"]),
            "best_metric_so_far": result.get("best_metric_so_far", carry["best_metric_so_far"]),
        }
        final_state = dict(carry)
        final_state.update(result)
    else:
        stopped_reason = "max_iterations_hard_cap"

    return RunResult(
        run_id=run_id,
        iterations_completed=carry["iteration"],
        final_state=final_state,
        stopped_reason=stopped_reason,
    )


def _snapshot_usage(llm_client: LLMClient):
    from feature_generator.agents.llm_client import LLMUsage

    return LLMUsage(
        input_tokens=llm_client.total_usage.input_tokens,
        output_tokens=llm_client.total_usage.output_tokens,
        cache_read_tokens=llm_client.total_usage.cache_read_tokens,
        cache_creation_tokens=llm_client.total_usage.cache_creation_tokens,
    )


def _usage_delta(before, after):
    from feature_generator.agents.llm_client import LLMUsage

    return LLMUsage(
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
        cache_read_tokens=after.cache_read_tokens - before.cache_read_tokens,
        cache_creation_tokens=after.cache_creation_tokens - before.cache_creation_tokens,
    )

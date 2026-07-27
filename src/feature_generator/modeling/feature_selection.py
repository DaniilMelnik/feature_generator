"""Deterministic feature-selection tool -- no LLM calls inside it.

Pipeline: single-feature lift screening -> redundancy clustering/pruning ->
bounded greedy stepwise combination search. Returns ranked subsets + metrics
for the hypothesis agent to react to on its next turn, per the design's
requirement that combinatorial search be classical code, not LLM reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from feature_generator.dataset.builder import FoldSplit
from feature_generator.modeling.train import train_catboost_cv


def _safe_cv_auc(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[FoldSplit],
    cat_features: list[str],
    catboost_params: dict | None,
) -> float:
    if X.shape[1] == 0:
        return 0.5
    result = train_catboost_cv(
        X,
        y,
        folds,
        cat_features=[c for c in cat_features if c in X.columns],
        catboost_params=catboost_params,
    )
    return result.metric_mean_std("auc")[0]


def screen_single_feature_lift(
    X_baseline: pd.DataFrame,
    candidates: dict[str, pd.Series],
    y: pd.Series,
    folds: list[FoldSplit],
    *,
    cat_features: list[str] | None = None,
    catboost_params: dict | None = None,
) -> dict[str, float]:
    """For each candidate, independently measure CV-AUC lift over the
    baseline feature set from adding just that one feature.
    """
    cat_features = cat_features or []
    baseline_auc = _safe_cv_auc(X_baseline, y, folds, cat_features, catboost_params)

    lifts: dict[str, float] = {}
    for name, series in candidates.items():
        combined = X_baseline.copy()
        combined[name] = series
        auc = _safe_cv_auc(combined, y, folds, cat_features, catboost_params)
        lifts[name] = auc - baseline_auc
    return lifts


def cluster_redundant_features(
    X: pd.DataFrame, feature_names: list[str], correlation_threshold: float
) -> list[list[str]]:
    """Group numeric features whose pairwise |correlation| >= threshold into
    clusters (union-find). Non-numeric and singleton features are returned as
    their own one-element cluster.
    """
    numeric_cols = [
        c
        for c in feature_names
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
    ]
    non_numeric = [c for c in feature_names if c not in numeric_cols]

    if len(numeric_cols) < 2:
        return [[c] for c in feature_names]

    corr = X[numeric_cols].corr().abs()
    parent = {c: c for c in numeric_cols}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and value >= correlation_threshold:
                union(a, b)

    clusters_map: dict[str, list[str]] = {}
    for c in numeric_cols:
        clusters_map.setdefault(find(c), []).append(c)

    return list(clusters_map.values()) + [[c] for c in non_numeric]


def drop_redundant_features(clusters: list[list[str]], importance: dict[str, float]) -> list[str]:
    """From each redundancy cluster, keep only the single most-important
    member (ties broken by first occurrence).
    """
    kept: list[str] = []
    for cluster in clusters:
        if len(cluster) == 1:
            kept.append(cluster[0])
        else:
            kept.append(max(cluster, key=lambda c: importance.get(c, 0.0)))
    return kept


@dataclass
class GreedySearchResult:
    selected_features: list[str]
    auc_trace: list[tuple[str, float]]  # (feature_added, resulting_cv_auc), in selection order
    baseline_auc: float
    evals_used: int


def greedy_stepwise_search(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[FoldSplit],
    *,
    base_features: list[str],
    candidate_features: list[str],
    cat_features: list[str] | None = None,
    max_evals: int = 50,
    min_improvement: float = 1e-4,
    catboost_params: dict | None = None,
) -> GreedySearchResult:
    """Bounded greedy/hill-climbing forward selection: at each step, add the
    single remaining candidate that most improves CV AUC, stopping when no
    candidate improves by more than `min_improvement` or `max_evals` (total
    CatBoost trainings) is exhausted. Not exhaustive/brute-force by design.
    """
    cat_features = cat_features or []
    selected = list(base_features)
    remaining = [c for c in candidate_features if c not in selected]
    evals_used = 0

    def cv_auc(features: list[str]) -> float:
        nonlocal evals_used
        evals_used += 1
        return _safe_cv_auc(X[features], y, folds, cat_features, catboost_params)

    baseline_auc = cv_auc(selected) if selected else 0.5
    current_best_auc = baseline_auc
    auc_trace: list[tuple[str, float]] = []

    while remaining and evals_used < max_evals:
        best_candidate: str | None = None
        best_auc = current_best_auc
        for cand in remaining:
            if evals_used >= max_evals:
                break
            trial_auc = cv_auc(selected + [cand])
            if trial_auc > best_auc + min_improvement:
                best_auc = trial_auc
                best_candidate = cand
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_best_auc = best_auc
        auc_trace.append((best_candidate, best_auc))

    return GreedySearchResult(
        selected_features=selected,
        auc_trace=auc_trace,
        baseline_auc=baseline_auc,
        evals_used=evals_used,
    )


@dataclass
class FeatureSelectionOutput:
    accepted_candidates: list[str]  # passed single-feature lift screen
    rejected_by_lift: list[str]
    redundancy_clusters: list[list[str]] = field(default_factory=list)
    kept_after_redundancy: list[str] = field(default_factory=list)
    final_selected: list[str] = field(default_factory=list)
    auc_trace: list[tuple[str, float]] = field(default_factory=list)
    baseline_auc: float = 0.5
    final_auc: float = 0.5


def run_feature_selection(
    X_baseline: pd.DataFrame,
    candidate_series: dict[str, pd.Series],
    y: pd.Series,
    folds: list[FoldSplit],
    *,
    cat_features: list[str] | None = None,
    correlation_threshold: float = 0.95,
    min_lift: float = 0.001,
    max_evals: int = 50,
    catboost_params: dict | None = None,
) -> FeatureSelectionOutput:
    cat_features = cat_features or []

    lifts = screen_single_feature_lift(
        X_baseline, candidate_series, y, folds, cat_features=cat_features, catboost_params=catboost_params
    )
    accepted = [name for name, lift in lifts.items() if lift >= min_lift]
    rejected = [name for name in candidate_series if name not in accepted]

    if not accepted:
        baseline_auc = _safe_cv_auc(X_baseline, y, folds, cat_features, catboost_params)
        return FeatureSelectionOutput(
            accepted_candidates=[],
            rejected_by_lift=rejected,
            kept_after_redundancy=list(X_baseline.columns),
            final_selected=list(X_baseline.columns),
            baseline_auc=baseline_auc,
            final_auc=baseline_auc,
        )

    combined_X = X_baseline.copy()
    for name in accepted:
        combined_X[name] = candidate_series[name]

    all_feature_names = list(X_baseline.columns) + accepted
    clusters = cluster_redundant_features(combined_X, all_feature_names, correlation_threshold)

    # Tie-break redundancy pruning by single-feature lift for candidates;
    # pre-existing baseline features default to a high proxy so they are
    # preferred to survive over a newly-added, merely-as-good candidate.
    importance_proxy = {name: lifts.get(name, 1.0) for name in all_feature_names}
    kept = drop_redundant_features(clusters, importance_proxy)

    base_kept = [c for c in kept if c in X_baseline.columns]
    candidate_kept = [c for c in kept if c in accepted]

    search_result = greedy_stepwise_search(
        combined_X,
        y,
        folds,
        base_features=base_kept,
        candidate_features=candidate_kept,
        cat_features=cat_features,
        max_evals=max_evals,
        catboost_params=catboost_params,
    )

    return FeatureSelectionOutput(
        accepted_candidates=accepted,
        rejected_by_lift=rejected,
        redundancy_clusters=clusters,
        kept_after_redundancy=kept,
        final_selected=search_result.selected_features,
        auc_trace=search_result.auc_trace,
        baseline_auc=search_result.baseline_auc,
        final_auc=search_result.auc_trace[-1][1] if search_result.auc_trace else search_result.baseline_auc,
    )

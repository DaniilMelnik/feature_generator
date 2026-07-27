import numpy as np
import pandas as pd
import pytest

from feature_generator.dataset.builder import DatasetBuilder
from feature_generator.modeling.feature_selection import (
    cluster_redundant_features,
    drop_redundant_features,
    greedy_stepwise_search,
    run_feature_selection,
    screen_single_feature_lift,
)

FAST_PARAMS = {"iterations": 60, "depth": 3}


@pytest.fixture
def selection_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    n = 250
    informative1 = rng.normal(0, 1, size=n)
    informative2 = rng.normal(0, 1, size=n)
    noise1 = rng.normal(0, 1, size=n)
    noise2 = rng.normal(0, 1, size=n)

    logit = 1.1 * informative1 + 1.1 * informative2 - 0.2
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="target")

    X = pd.DataFrame(
        {
            "informative1": informative1,
            "informative2": informative2,
            "noise1": noise1,
            "noise2": noise2,
        }
    )
    return X, y


def test_screen_single_feature_lift_identifies_informative_vs_noise(selection_dataset) -> None:
    X, y = selection_dataset
    builder = DatasetBuilder(cv_folds=4, random_seed=1)
    folds = builder.make_folds(y)
    empty_baseline = pd.DataFrame(index=X.index)

    candidates = {"informative1": X["informative1"], "noise1": X["noise1"]}
    lifts = screen_single_feature_lift(
        empty_baseline, candidates, y, folds, catboost_params=FAST_PARAMS
    )

    assert lifts["informative1"] > lifts["noise1"]
    assert lifts["informative1"] > 0.05


def test_cluster_redundant_features_groups_near_duplicates() -> None:
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(0, 1, size=n)
    a_duplicate = a + rng.normal(0, 0.001, size=n)  # near-perfectly correlated with a
    b = rng.normal(0, 1, size=n)  # independent

    X = pd.DataFrame({"a": a, "a_duplicate": a_duplicate, "b": b})
    clusters = cluster_redundant_features(X, ["a", "a_duplicate", "b"], correlation_threshold=0.95)

    cluster_sets = [set(c) for c in clusters]
    assert {"a", "a_duplicate"} in cluster_sets
    assert {"b"} in cluster_sets
    assert len(clusters) == 2


def test_cluster_redundant_features_treats_independent_columns_as_singletons() -> None:
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    clusters = cluster_redundant_features(X, ["x", "y", "z"], correlation_threshold=0.95)
    assert sorted(clusters) == [["x"], ["y"], ["z"]]


def test_drop_redundant_features_keeps_highest_importance_member() -> None:
    clusters = [["a", "a_duplicate"], ["b"]]
    importance = {"a": 0.1, "a_duplicate": 0.5, "b": 0.2}
    kept = drop_redundant_features(clusters, importance)
    assert set(kept) == {"a_duplicate", "b"}


def test_greedy_stepwise_search_prefers_informative_features(selection_dataset) -> None:
    X, y = selection_dataset
    builder = DatasetBuilder(cv_folds=4, random_seed=1)
    folds = builder.make_folds(y)

    result = greedy_stepwise_search(
        X,
        y,
        folds,
        base_features=[],
        candidate_features=list(X.columns),
        max_evals=30,
        catboost_params=FAST_PARAMS,
    )

    assert "informative1" in result.selected_features
    assert "informative2" in result.selected_features
    assert result.auc_trace, "expected at least one improving step to be recorded"
    assert result.auc_trace[-1][1] > result.baseline_auc


def test_greedy_stepwise_search_respects_max_evals_budget(selection_dataset) -> None:
    X, y = selection_dataset
    builder = DatasetBuilder(cv_folds=4, random_seed=1)
    folds = builder.make_folds(y)

    result = greedy_stepwise_search(
        X,
        y,
        folds,
        base_features=[],
        candidate_features=list(X.columns),
        max_evals=2,
        catboost_params=FAST_PARAMS,
    )
    assert result.evals_used <= 2


def test_run_feature_selection_end_to_end_drops_redundant_and_noise(selection_dataset) -> None:
    X, y = selection_dataset
    builder = DatasetBuilder(cv_folds=4, random_seed=1)
    folds = builder.make_folds(y)

    empty_baseline = pd.DataFrame(index=X.index)
    candidate_series = {
        "informative1": X["informative1"],
        "informative1_dup": X["informative1"] + np.random.default_rng(2).normal(0, 0.001, len(X)),
        "noise1": X["noise1"],
    }

    output = run_feature_selection(
        empty_baseline,
        candidate_series,
        y,
        folds,
        min_lift=0.01,
        correlation_threshold=0.95,
        max_evals=20,
        catboost_params=FAST_PARAMS,
    )

    # the near-duplicate pair must not BOTH survive redundancy pruning
    assert not {"informative1", "informative1_dup"}.issubset(set(output.final_selected))
    assert output.final_auc >= output.baseline_auc

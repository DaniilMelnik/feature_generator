import numpy as np
import pandas as pd
import pytest

from feature_generator.modeling.metrics import (
    compute_binary_classification_metrics,
    compute_ks_statistic,
    compute_single_feature_auc,
)


def test_perfect_predictions_give_auc_near_one_and_low_logloss() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.98, 0.97, 0.99])
    metrics = compute_binary_classification_metrics(y_true, y_proba)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["logloss"] < 0.1
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["brier"] < 0.01


def test_random_predictions_give_auc_near_half() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    y_proba = rng.uniform(size=2000)  # independent of y_true
    metrics = compute_binary_classification_metrics(y_true, y_proba)
    assert 0.45 < metrics["auc"] < 0.55


def test_ks_statistic_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(1)
    same = rng.uniform(size=500)
    y_true = np.array([0] * 250 + [1] * 250)
    ks = compute_ks_statistic(y_true, same)
    assert ks < 0.15  # small but not necessarily exactly 0 due to sampling


def test_ks_statistic_is_high_for_well_separated_distributions() -> None:
    y_true = np.array([0] * 100 + [1] * 100)
    y_proba = np.array([0.1] * 100 + [0.9] * 100)
    ks = compute_ks_statistic(y_true, y_proba)
    assert ks == pytest.approx(1.0)


def test_metrics_clip_extreme_probabilities_to_avoid_infinite_logloss() -> None:
    y_true = np.array([0, 1])
    y_proba = np.array([0.0, 1.0])  # would otherwise blow up log_loss
    metrics = compute_binary_classification_metrics(y_true, y_proba)
    assert np.isfinite(metrics["logloss"])


def test_single_feature_auc_is_none_for_categorical_output() -> None:
    # a categorical-output feature's own labels aren't a ranking score --
    # must return None rather than crash inside roc_auc_score
    values = pd.Series(["A", "B", "A", "B"])
    y = pd.Series([0, 1, 0, 1])
    assert compute_single_feature_auc(values, y) is None

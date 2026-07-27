"""Binary-classification metric computations, shared by CV training and the
stability/decile-report modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

DEFAULT_SINGLE_FEATURE_AUC_LEAK_THRESHOLD = 0.90


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    positive = y_proba[y_true == 1]
    negative = y_proba[y_true == 0]
    if len(positive) == 0 or len(negative) == 0:
        return 0.0
    stat, _ = ks_2samp(positive, negative)
    return float(stat)


def compute_binary_classification_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "logloss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "ks": compute_ks_statistic(y_true, y_proba),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def compute_single_feature_auc(values: pd.Series, y: pd.Series) -> float | None:
    """Treat a single (typically out-of-fold) feature's own values directly
    as a ranking score against the target -- no model fitting involved. A
    near-1.0 result on an engineered feature is a leakage smell worth
    flagging for review, independent of any full-model metric.

    Returns None when there isn't enough signal/variation to compute AUC
    (fewer than 2 classes, or a constant feature), or when the feature's
    values aren't numeric (e.g. a categorical-output feature) -- using its
    own category labels directly as a ranking score isn't meaningful.
    """
    if not pd.api.types.is_numeric_dtype(values):
        return None
    paired = pd.DataFrame({"v": values, "y": y}).dropna()
    if paired["y"].nunique() < 2 or paired["v"].nunique() < 2:
        return None
    auc = roc_auc_score(paired["y"], paired["v"])
    # A feature that's a strong NEGATIVE predictor is just as suspicious as a
    # strong positive one -- report the more extreme, sign-symmetric score.
    return float(max(auc, 1 - auc))


def flag_single_feature_auc(
    auc: float | None, threshold: float = DEFAULT_SINGLE_FEATURE_AUC_LEAK_THRESHOLD
) -> bool:
    return auc is not None and auc >= threshold

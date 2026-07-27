"""SHAP and permutation feature importance, computed out-of-fold (each row's
contribution comes from the model that did NOT see it during training).

Primary path uses the ``shap`` package's ``TreeExplainer``; if that raises
(shap/catboost API combinations have shifted across versions), falls back to
CatBoost's own native, exact SHAP-value computation -- both are genuine
Shapley-value computations, so either path is correct, not an approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.inspection import permutation_importance

from feature_generator.dataset.builder import FoldSplit
from feature_generator.modeling.train import prepare_categorical_columns


def _shap_values_for_positive_class(
    model: CatBoostClassifier, X: pd.DataFrame, cat_features: list[str]
) -> np.ndarray:
    pool = Pool(X, cat_features=cat_features or None)
    raw = None
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(pool)
    except Exception:
        raw = None

    if raw is None:
        # CatBoost's native exact SHAP computation. Shape is
        # (n_samples, n_features + 1); the trailing column is the expected
        # value / bias term, which we drop.
        native = model.get_feature_importance(pool, type="ShapValues")
        return np.asarray(native)[:, :-1]

    if isinstance(raw, list):
        raw = raw[-1] if len(raw) > 1 else raw[0]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[..., -1]
    return raw


def compute_shap_importance(
    models: list[CatBoostClassifier],
    X: pd.DataFrame,
    folds: list[FoldSplit],
    *,
    cat_features: list[str] | None = None,
) -> dict[str, float]:
    cat_features = cat_features or []
    X_prepared = prepare_categorical_columns(X, cat_features)

    abs_sum = np.zeros(X_prepared.shape[1])
    total_rows = 0
    for model, fold in zip(models, folds):
        X_val = X_prepared.iloc[fold.val_index]
        shap_values = _shap_values_for_positive_class(model, X_val, cat_features)
        abs_sum += np.abs(shap_values).sum(axis=0)
        total_rows += len(X_val)

    mean_abs = abs_sum / max(total_rows, 1)
    return dict(zip(X_prepared.columns, (float(v) for v in mean_abs)))


def compute_permutation_importance(
    models: list[CatBoostClassifier],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[FoldSplit],
    *,
    cat_features: list[str] | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    cat_features = cat_features or []
    X_prepared = prepare_categorical_columns(X, cat_features)

    importances = np.zeros(X_prepared.shape[1])
    for model, fold in zip(models, folds):
        X_val = X_prepared.iloc[fold.val_index]
        y_val = y.iloc[fold.val_index]
        result = permutation_importance(
            model,
            X_val,
            y_val,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
        )
        importances += result.importances_mean

    importances /= max(len(models), 1)
    return dict(zip(X_prepared.columns, (float(v) for v in importances)))

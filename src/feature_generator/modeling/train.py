"""CatBoost + cross-validation training. Hyperparameter search is explicitly
out of scope for this MVP (deferred to a future "model selection" phase) --
sane fixed defaults are used, overridable via ``catboost_params``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_generator.dataset.builder import FoldSplit
from feature_generator.modeling.metrics import compute_binary_classification_metrics

DEFAULT_CATBOOST_PARAMS: dict = {
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "allow_writing_files": False,
}

MISSING_CATEGORY_PLACEHOLDER = "__missing__"


def prepare_categorical_columns(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """CatBoost requires categorical columns to be a consistent non-float
    dtype; fill missing values with an explicit placeholder category rather
    than leaving NaN (which CatBoost rejects for categorical columns).
    """
    if not cat_features:
        return X
    X = X.copy()
    for col in cat_features:
        X[col] = X[col].astype("object").fillna(MISSING_CATEGORY_PLACEHOLDER).astype(str)
    return X


@dataclass
class CVResult:
    fold_metrics: list[dict[str, float]]
    oof_predictions: pd.Series
    models: list[CatBoostClassifier]
    cat_features: list[str] = field(default_factory=list)
    fold_train_metrics: list[dict[str, float]] = field(default_factory=list)

    def metric_mean_std(self, key: str) -> tuple[float, float]:
        values = [m[key] for m in self.fold_metrics]
        return float(np.mean(values)), float(np.std(values))

    def train_metric_mean_std(self, key: str) -> tuple[float, float]:
        values = [m[key] for m in self.fold_train_metrics]
        return float(np.mean(values)), float(np.std(values))


def train_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[FoldSplit],
    *,
    cat_features: list[str] | None = None,
    catboost_params: dict | None = None,
) -> CVResult:
    cat_features = cat_features or []
    params = {**DEFAULT_CATBOOST_PARAMS, **(catboost_params or {})}
    X_prepared = prepare_categorical_columns(X, cat_features)

    fold_metrics: list[dict[str, float]] = []
    fold_train_metrics: list[dict[str, float]] = []
    oof = pd.Series(index=X_prepared.index, dtype="float64")
    models: list[CatBoostClassifier] = []

    for fold in folds:
        X_train, X_val = X_prepared.iloc[fold.train_index], X_prepared.iloc[fold.val_index]
        y_train, y_val = y.iloc[fold.train_index], y.iloc[fold.val_index]

        model = CatBoostClassifier(**params, random_seed=fold.fold_id)
        model.fit(X_train, y_train, cat_features=cat_features or None)

        proba = model.predict_proba(X_val)[:, 1]
        oof.iloc[fold.val_index] = proba
        fold_metrics.append(compute_binary_classification_metrics(y_val.to_numpy(), proba))

        train_proba = model.predict_proba(X_train)[:, 1]
        fold_train_metrics.append(compute_binary_classification_metrics(y_train.to_numpy(), train_proba))

        models.append(model)

    return CVResult(
        fold_metrics=fold_metrics,
        oof_predictions=oof,
        models=models,
        cat_features=cat_features,
        fold_train_metrics=fold_train_metrics,
    )


def evaluate_holdout(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    *,
    cat_features: list[str] | None = None,
    catboost_params: dict | None = None,
    random_seed: int = 42,
) -> dict[str, float]:
    """Fit one model on dev rows only, evaluate once on rows the CV loop
    never touched -- the honest check k-fold CV alone can't provide against
    adaptive overfitting to the validation procedure across many iterations.
    Callers are responsible for assembling `X_holdout` correctly (e.g. via
    `DatasetBuilder.transform_new_data` for engineered features) -- this
    function does no leakage-safety work of its own beyond never fitting on
    holdout rows.
    """
    cat_features = cat_features or []
    model = train_full_model(
        X_dev, y_dev, cat_features=cat_features, catboost_params=catboost_params, random_seed=random_seed
    )
    X_holdout_prepared = prepare_categorical_columns(X_holdout, cat_features)
    proba = model.predict_proba(X_holdout_prepared)[:, 1]
    return compute_binary_classification_metrics(y_holdout.to_numpy(), proba)


def train_full_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cat_features: list[str] | None = None,
    catboost_params: dict | None = None,
    random_seed: int = 42,
) -> CatBoostClassifier:
    """Train one final model on ALL rows -- used for production inference,
    never for CV metric reporting (that would be optimistic/leaked).
    """
    cat_features = cat_features or []
    params = {**DEFAULT_CATBOOST_PARAMS, **(catboost_params or {})}
    X_prepared = prepare_categorical_columns(X, cat_features)
    model = CatBoostClassifier(**params, random_seed=random_seed)
    model.fit(X_prepared, y, cat_features=cat_features or None)
    return model

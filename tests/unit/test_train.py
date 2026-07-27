import numpy as np
import pandas as pd

from feature_generator.dataset.builder import DatasetBuilder
from feature_generator.modeling.train import (
    evaluate_holdout,
    prepare_categorical_columns,
    train_catboost_cv,
    train_full_model,
)

CAT_FEATURES = ["categorical_feature", "noise_categorical"]
FAST_PARAMS = {"iterations": 100}


def test_train_catboost_cv_beats_random_on_informative_data(synthetic_binary_dataset) -> None:
    X, y = synthetic_binary_dataset
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    folds = builder.make_folds(y)

    result = train_catboost_cv(X, y, folds, cat_features=CAT_FEATURES, catboost_params=FAST_PARAMS)

    auc_mean, auc_std = result.metric_mean_std("auc")
    assert auc_mean > 0.75, f"expected clear signal to be picked up, got AUC={auc_mean}"
    assert auc_std < 0.15
    assert len(result.models) == 5
    assert result.oof_predictions.isna().sum() == 0
    assert result.oof_predictions.between(0, 1).all()


def test_train_full_model_fits_on_all_rows_and_predicts(synthetic_binary_dataset) -> None:
    X, y = synthetic_binary_dataset
    model = train_full_model(X, y, cat_features=CAT_FEATURES, catboost_params=FAST_PARAMS)
    proba = model.predict_proba(prepare_categorical_columns(X, CAT_FEATURES))[:, 1]
    assert len(proba) == len(X)
    assert np.all((proba >= 0) & (proba <= 1))


def test_prepare_categorical_columns_fills_missing_with_placeholder() -> None:
    df = pd.DataFrame({"cat": ["a", None, "b", np.nan], "num": [1.0, 2.0, 3.0, 4.0]})
    prepared = prepare_categorical_columns(df, ["cat"])
    assert prepared["cat"].tolist() == ["a", "__missing__", "b", "__missing__"]
    assert prepared["num"].tolist() == [1.0, 2.0, 3.0, 4.0]  # untouched


def test_prepare_categorical_columns_is_noop_without_cat_features() -> None:
    df = pd.DataFrame({"num": [1.0, 2.0]})
    assert prepare_categorical_columns(df, []) is df


def test_train_catboost_cv_reports_train_metrics_alongside_val_metrics(synthetic_binary_dataset) -> None:
    X, y = synthetic_binary_dataset
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    folds = builder.make_folds(y)

    result = train_catboost_cv(X, y, folds, cat_features=CAT_FEATURES, catboost_params=FAST_PARAMS)

    assert len(result.fold_train_metrics) == 5
    train_auc_mean, _ = result.train_metric_mean_std("auc")
    val_auc_mean, _ = result.metric_mean_std("auc")
    # in-sample (train) performance should be at least as good as held-out
    # (val) performance -- the standard overfit-gap direction
    assert train_auc_mean >= val_auc_mean - 0.05


def test_evaluate_holdout_scores_a_dev_only_fit_model_on_holdout_rows(synthetic_binary_dataset) -> None:
    X, y = synthetic_binary_dataset
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    dev_idx, holdout_idx = builder.split_dev_holdout(y, 0.2)

    metrics = evaluate_holdout(
        X.iloc[dev_idx], y.iloc[dev_idx], X.iloc[holdout_idx], y.iloc[holdout_idx],
        cat_features=CAT_FEATURES, catboost_params=FAST_PARAMS,
    )

    assert metrics["auc"] > 0.7  # same informative signal as the CV test above
    assert 0.0 <= metrics["auc"] <= 1.0

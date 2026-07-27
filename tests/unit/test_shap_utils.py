from feature_generator.dataset.builder import DatasetBuilder
from feature_generator.modeling.shap_utils import (
    compute_permutation_importance,
    compute_shap_importance,
)
from feature_generator.modeling.train import train_catboost_cv

CAT_FEATURES = ["categorical_feature", "noise_categorical"]
FAST_PARAMS = {"iterations": 150, "depth": 4}


def _train(synthetic_binary_dataset):
    X, y = synthetic_binary_dataset
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    folds = builder.make_folds(y)
    result = train_catboost_cv(X, y, folds, cat_features=CAT_FEATURES, catboost_params=FAST_PARAMS)
    return X, y, folds, result


def test_shap_importance_ranks_informative_features_above_noise(synthetic_binary_dataset) -> None:
    X, y, folds, result = _train(synthetic_binary_dataset)

    importance = compute_shap_importance(result.models, X, folds, cat_features=CAT_FEATURES)

    assert set(importance.keys()) == set(X.columns)
    assert all(v >= 0 for v in importance.values())
    assert importance["informative_numeric"] > importance["noise_numeric"]
    assert importance["categorical_feature"] > importance["noise_categorical"]


def test_permutation_importance_ranks_informative_features_above_noise(synthetic_binary_dataset) -> None:
    X, y, folds, result = _train(synthetic_binary_dataset)

    importance = compute_permutation_importance(
        result.models, X, y, folds, cat_features=CAT_FEATURES, n_repeats=5
    )

    assert set(importance.keys()) == set(X.columns)
    assert importance["informative_numeric"] > importance["noise_numeric"]
    assert importance["categorical_feature"] > importance["noise_categorical"]

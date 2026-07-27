import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_binary_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """A small dataset with one informative numeric feature, one informative
    categorical feature, and two pure-noise features (one numeric, one
    categorical) -- used across modeling/shap/feature-selection tests to
    check that importance/selection logic actually distinguishes signal from
    noise, not just that the code runs.
    """
    rng = np.random.default_rng(123)
    n = 300

    informative_numeric = rng.normal(0, 1, size=n)
    categorical_feature = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])
    noise_numeric = rng.normal(0, 1, size=n)
    noise_categorical = rng.choice(["X", "Y"], size=n)

    cat_effect = np.where(categorical_feature == "C", 1.5, 0.0)
    logit = 1.2 * informative_numeric + cat_effect - 0.3
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="target")

    X = pd.DataFrame(
        {
            "informative_numeric": informative_numeric,
            "categorical_feature": categorical_feature,
            "noise_numeric": noise_numeric,
            "noise_categorical": noise_categorical,
        }
    )
    return X, y

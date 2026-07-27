"""End-to-end proof that the safety machinery is not a no-op: run the
deliberately-leaky target-encoding fixture through the real fold-wise
DatasetBuilder and confirm the single-feature-AUC sanity check flags it --
and that a properly fold-isolated encoding of the exact same (adversarial,
unique-per-row) column does NOT get flagged, proving the check is about the
memoization bug, not about target-encoding as a technique.

Static-check rejection of the malicious-import fixture is covered in
test_static_checks.py; the "Docker is never invoked for a statically-rejected
module" guarantee is tested in Step 4's Docker sandbox test suite once
docker_runner.py exists.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_generator.dataset.builder import DatasetBuilder
from feature_generator.modeling.metrics import compute_single_feature_auc, flag_single_feature_auc
from feature_generator.sandbox.contract import InProcessFeatureComputer, load_trusted_module_from_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(fixture_name: str, **kwargs) -> InProcessFeatureComputer:
    module = load_trusted_module_from_file(FIXTURES / fixture_name)
    return InProcessFeatureComputer(module, **kwargs)


def test_target_encoding_memoization_leak_inflates_single_feature_auc() -> None:
    n = 300
    # Every row is its own category: a properly fold-isolated target encoding
    # of a unique-per-row column carries no signal (each fold's val category
    # is unseen in that fold's own train split). The leaky, memoized version
    # instead reuses fold-0's memo -- and fold-0's train split contains most
    # other folds' validation rows, so their own labels leak back to them.
    category_id = pd.Series([f"cat_{i}" for i in range(n)])
    y = pd.Series(np.random.default_rng(0).integers(0, 2, size=n))  # pure noise, no real relationship
    df = pd.DataFrame({"category_id": category_id, "target": y})

    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    folds = builder.make_folds(y)

    leaky_computer = _load("leaky_feature_target_encoding.py", allow_target_in_fit=True)
    leaky_result = builder.compute_oof_feature(df, leaky_computer, folds, target_column="target")

    leaky_auc = compute_single_feature_auc(leaky_result.oof_values, y)
    assert leaky_auc is not None
    assert flag_single_feature_auc(leaky_auc), (
        f"expected the memoized target-encoding leak to trip the single-feature-AUC "
        f"flag, got AUC={leaky_auc}"
    )


def test_properly_isolated_target_encoding_of_unique_column_carries_no_signal() -> None:
    """Negative control: the same adversarial unique-per-row category column,
    encoded WITHOUT the memoization bug (fresh fit() each fold), must show
    ~no single-feature signal.
    """
    n = 300
    category_id = pd.Series([f"cat_{i}" for i in range(n)])
    y = pd.Series(np.random.default_rng(0).integers(0, 2, size=n))
    df = pd.DataFrame({"category_id": category_id, "target": y})

    class _ProperlyIsolatedModule:
        FEATURE_NAME = "proper_category_target_mean"
        REQUIRED_COLUMNS = ["category_id"]

        def fit(self, train_df, context):
            means = train_df.groupby("category_id")[context.target_column].mean().to_dict()
            return {"means": means}

        def transform(self, df, params):
            return df["category_id"].map(params["means"]).fillna(0.5)

    computer = InProcessFeatureComputer(_ProperlyIsolatedModule(), allow_target_in_fit=True)
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    folds = builder.make_folds(y)
    result = builder.compute_oof_feature(df, computer, folds, target_column="target")

    fixed_auc = compute_single_feature_auc(result.oof_values, y)
    # None means "no variation at all" (every val category was unseen in its
    # own fold's train split, so every row fell back to the constant 0.5) --
    # that IS the correct "no signal" outcome for this adversarial column.
    assert fixed_auc is None or fixed_auc < 0.65, (
        f"expected ~no signal from a properly-isolated encoding, got AUC={fixed_auc}"
    )
    assert fixed_auc is None or not flag_single_feature_auc(fixed_auc)


def test_compliant_fixture_is_not_flagged_by_single_feature_auc() -> None:
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "HomePlanet": rng.choice(["Earth", "Mars"], size=n),
            "RoomService": rng.exponential(50, size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))  # unrelated to HomePlanet/RoomService

    computer = _load("compliant_feature_groupby_mean.py")
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    folds = builder.make_folds(y)
    result = builder.compute_oof_feature(df, computer, folds, target_column="Transported")

    auc = compute_single_feature_auc(result.oof_values, y)
    assert auc is None or not flag_single_feature_auc(auc)

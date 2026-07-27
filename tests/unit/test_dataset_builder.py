from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_generator.dataset.builder import DatasetBuilder, build_raw_feature_frame
from feature_generator.profiling.schemas import (
    CategoricalStats,
    ColumnProfile,
    DataProfile,
    NumericStats,
    TableProfile,
)
from feature_generator.sandbox.contract import (
    FeatureComputer,
    FitContext,
    InProcessFeatureComputer,
    load_trusted_module_from_file,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _synthetic_df(n: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    home_planet = rng.choice(["Earth", "Mars"], size=n)
    room_service = rng.normal(50, 10, size=n)
    y = pd.Series(rng.choice([0, 1], size=n), name="Transported")
    df = pd.DataFrame({"HomePlanet": home_planet, "RoomService": room_service})
    return df, y


class SpyComputer(FeatureComputer):
    """Wraps another FeatureComputer and records exactly which row-indices
    were visible to fit()/transform() on every call, so tests can assert the
    fold-isolation invariant structurally rather than trusting it.
    """

    def __init__(self, inner: FeatureComputer) -> None:
        self._inner = inner
        self.feature_name = inner.feature_name
        self.required_columns = inner.required_columns
        self.fit_calls: list[np.ndarray] = []
        self.transform_calls: list[np.ndarray] = []

    def fit(self, train_df: pd.DataFrame, context: FitContext) -> dict:
        self.fit_calls.append(train_df.index.to_numpy().copy())
        return self._inner.fit(train_df, context)

    def transform(self, df: pd.DataFrame, params: dict) -> pd.Series:
        self.transform_calls.append(df.index.to_numpy().copy())
        return self._inner.transform(df, params)


def _make_spy() -> SpyComputer:
    module = load_trusted_module_from_file(FIXTURES / "compliant_feature_groupby_mean.py")
    return SpyComputer(InProcessFeatureComputer(module))


def test_fit_is_called_exactly_once_per_fold_plus_one_full_fit() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    spy = _make_spy()

    builder.build(df, y, [spy], target_column="Transported")

    # 5 per-fold fits + 1 final full-data fit
    assert len(spy.fit_calls) == 6


def test_fit_never_sees_validation_rows_for_its_own_fold() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    spy = _make_spy()

    built = builder.build(df, y, [spy], target_column="Transported")

    per_fold_fits = spy.fit_calls[:5]  # the final full-data fit is fit_calls[5]
    per_fold_transforms = spy.transform_calls  # 5 calls, one per fold's val slice

    for fold, fit_idx, transform_idx in zip(built.folds, per_fold_fits, per_fold_transforms):
        expected_train_idx = df.index[fold.train_index].to_numpy()
        expected_val_idx = df.index[fold.val_index].to_numpy()

        assert set(fit_idx) == set(expected_train_idx)
        assert set(transform_idx) == set(expected_val_idx)
        # the defining anti-leakage invariant: no overlap between what fit()
        # saw and what transform() was later evaluated on for the same fold
        assert set(fit_idx).isdisjoint(set(transform_idx))


def test_full_fit_call_sees_every_row() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    spy = _make_spy()

    builder.build(df, y, [spy], target_column="Transported")

    full_fit_idx = spy.fit_calls[-1]
    assert set(full_fit_idx) == set(df.index.to_numpy())


def test_full_fit_call_restricted_to_dev_index_when_given() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    spy = _make_spy()
    dev_index, holdout_index = builder.split_dev_holdout(y, 0.2)
    folds = builder.make_folds(y, restrict_to=dev_index)

    builder.compute_oof_feature(
        df, spy, folds, target_column="Transported", dev_index=dev_index
    )

    full_fit_idx = spy.fit_calls[-1]
    assert set(full_fit_idx) == set(df.index[dev_index].to_numpy())
    assert set(full_fit_idx).isdisjoint(set(df.index[holdout_index].to_numpy()))


def test_oof_feature_frame_has_no_gaps_and_right_shape() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    spy = _make_spy()

    built = builder.build(df, y, [spy], target_column="Transported")

    assert built.feature_frame.shape == (len(df), 1)
    assert "avg_room_service_by_home_planet" in built.feature_frame.columns
    assert built.feature_frame["avg_room_service_by_home_planet"].isna().sum() == 0


def test_folds_are_disjoint_and_cover_all_rows() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    folds = builder.make_folds(y)

    assert len(folds) == 5
    all_val_indices = np.concatenate([f.val_index for f in folds])
    assert sorted(all_val_indices.tolist()) == list(range(len(df)))
    for f in folds:
        assert set(f.train_index).isdisjoint(set(f.val_index))


def test_transform_new_data_uses_full_fit_params_without_refitting() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    spy = _make_spy()
    built = builder.build(df, y, [spy], target_column="Transported")

    n_fit_calls_before = len(spy.fit_calls)
    new_df, _ = _synthetic_df(n=8, seed=99)
    output = builder.transform_new_data(new_df, [spy], built.full_fit_params)

    assert len(spy.fit_calls) == n_fit_calls_before  # transform_new_data must never call fit()
    assert output.shape == (8, 1)
    assert output["avg_room_service_by_home_planet"].isna().sum() == 0


def test_cv_folds_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        DatasetBuilder(cv_folds=1, random_seed=0)


def test_split_dev_holdout_random_stratified_gives_disjoint_correctly_sized_sets() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series((rng.uniform(size=500) < 0.3).astype(int))
    builder = DatasetBuilder(cv_folds=5, random_seed=42)

    dev_idx, holdout_idx = builder.split_dev_holdout(y, 0.2)

    assert set(dev_idx).isdisjoint(set(holdout_idx))
    assert len(dev_idx) + len(holdout_idx) == len(y)
    assert len(holdout_idx) == pytest.approx(100, abs=2)
    # stratification: holdout's positive rate should track the overall rate
    holdout_rate = y.iloc[holdout_idx].mean()
    assert holdout_rate == pytest.approx(y.mean(), abs=0.05)


def test_split_dev_holdout_temporal_takes_chronologically_last_rows() -> None:
    y = pd.Series([0, 1] * 50)
    time_values = pd.Series(np.arange(100))  # already-sorted "time"
    builder = DatasetBuilder(cv_folds=5, random_seed=42)

    dev_idx, holdout_idx = builder.split_dev_holdout(y, 0.2, method="temporal", time_values=time_values)

    assert len(holdout_idx) == 20
    assert set(holdout_idx) == set(range(80, 100))  # the chronologically-last 20%
    assert set(dev_idx) == set(range(0, 80))


def test_split_dev_holdout_temporal_requires_time_values() -> None:
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    with pytest.raises(ValueError, match="time_values"):
        builder.split_dev_holdout(pd.Series([0, 1, 0, 1]), 0.5, method="temporal")


def test_make_folds_restrict_to_never_touches_excluded_positions() -> None:
    rng = np.random.default_rng(1)
    y = pd.Series((rng.uniform(size=200) < 0.4).astype(int))
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    dev_idx, holdout_idx = builder.split_dev_holdout(y, 0.15)

    folds = builder.make_folds(y, restrict_to=dev_idx)

    for fold in folds:
        assert set(fold.train_index).isdisjoint(set(holdout_idx))
        assert set(fold.val_index).isdisjoint(set(holdout_idx))
    all_val = np.concatenate([f.val_index for f in folds])
    assert set(all_val) == set(dev_idx)  # every dev row validated exactly once, no holdout rows


def test_make_folds_without_restrict_to_is_unchanged() -> None:
    rng = np.random.default_rng(2)
    y = pd.Series((rng.uniform(size=100) < 0.5).astype(int))
    builder = DatasetBuilder(cv_folds=5, random_seed=42)

    folds = builder.make_folds(y)

    all_val = np.concatenate([f.val_index for f in folds])
    assert sorted(all_val.tolist()) == list(range(len(y)))


class _StringFeatureComputer(FeatureComputer):
    """A compliant feature that legitimately returns categorical (string)
    values, e.g. a "CabinDeck" split -- not every feature is numeric.
    """

    feature_name = "home_planet_initial"
    required_columns = ["HomePlanet"]

    def fit(self, train_df: pd.DataFrame, context: FitContext) -> dict:
        return {}

    def transform(self, df: pd.DataFrame, params: dict) -> pd.Series:
        return df["HomePlanet"].str[0]


def test_categorical_output_feature_is_not_coerced_to_float64() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)

    built = builder.build(df, y, [_StringFeatureComputer()], target_column="Transported")

    values = built.feature_frame["home_planet_initial"]
    assert values.isna().sum() == 0
    assert set(values.unique()) <= {"E", "M"}
    assert not pd.api.types.is_numeric_dtype(values)


def _column(
    name: str,
    role: str,
    *,
    source_table: str = "passengers",
    numeric: bool = False,
    categorical: bool = False,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        source_table=source_table,
        dtype="float64" if numeric else "object",
        role=role,
        n_missing=0,
        pct_missing=0.0,
        n_unique=5,
        is_constant=False,
        numeric_stats=(
            NumericStats(min=0, max=1, mean=0.5, std=0.1, skew=0.0, kurtosis=0.0, quantiles={})
            if numeric
            else None
        ),
        categorical_stats=CategoricalStats(cardinality=2) if categorical else None,
    )


def test_build_raw_feature_frame_selects_feature_role_columns_and_flags_categoricals() -> None:
    df = pd.DataFrame(
        {
            "PassengerId": ["1", "2"],
            "HomePlanet": ["Earth", "Mars"],
            "Age": [10.0, 20.0],
            "Transported": [True, False],
            "Name": ["a", "b"],
        }
    )
    profile = DataProfile(
        dataset_name="ds",
        tables=[
            TableProfile(
                table_name="passengers",
                file_path="x.csv",
                n_rows=2,
                n_cols=5,
                role="primary",
                columns=[
                    _column("PassengerId", "id"),
                    _column("HomePlanet", "feature", categorical=True),
                    _column("Age", "feature", numeric=True),
                    _column("Transported", "target"),
                    _column("Name", "ignore", categorical=True),
                ],
            )
        ],
        target_column="Transported",
        target_table="passengers",
        id_columns=["PassengerId"],
        positive_rate=0.5,
        known_leakage_columns=["Name"],
        generated_at=datetime.now(timezone.utc),
    )

    frame, cat_features = build_raw_feature_frame(df, profile)

    assert set(frame.columns) == {"HomePlanet", "Age"}
    assert cat_features == ["HomePlanet"]


def test_build_raw_feature_frame_follows_join_collision_rename() -> None:
    # mirrors dataset.joiner's rename of a secondary table's colliding column
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "ProductCD": ["W", "C"],  # primary table's own column, unprefixed
            "identity_ProductCD": ["A", "B"],  # secondary table's, renamed by the joiner
        }
    )
    profile = DataProfile(
        dataset_name="ds",
        tables=[
            TableProfile(
                table_name="transaction",
                file_path="t.csv",
                n_rows=2,
                n_cols=2,
                role="primary",
                columns=[_column("ProductCD", "feature", source_table="transaction", categorical=True)],
            ),
            TableProfile(
                table_name="identity",
                file_path="i.csv",
                n_rows=2,
                n_cols=1,
                role="secondary",
                columns=[_column("ProductCD", "feature", source_table="identity", categorical=True)],
            ),
        ],
        target_column="isFraud",
        target_table="transaction",
        id_columns=["TransactionID"],
        positive_rate=0.5,
        generated_at=datetime.now(timezone.utc),
    )

    frame, cat_features = build_raw_feature_frame(df, profile)

    assert list(frame["ProductCD"]) == ["W", "C"]
    assert list(frame["identity_ProductCD"]) == ["A", "B"]
    assert set(cat_features) == {"ProductCD", "identity_ProductCD"}


class _ComputeOofOverridingComputer(FeatureComputer):
    """A computer that overrides `compute_oof` entirely (mirroring
    `SandboxedFeatureComputer`'s batching override) -- proves
    `DatasetBuilder.compute_oof_feature` delegates to it instead of running
    its own fold loop, and never calls `fit`/`transform` directly itself.
    """

    feature_name = "batched"
    required_columns = ["RoomService"]

    def __init__(self) -> None:
        self.compute_oof_calls: list[dict] = []
        self.fit_calls: list[object] = []
        self.transform_calls: list[object] = []

    def fit(self, train_df: pd.DataFrame, context: FitContext) -> dict:
        self.fit_calls.append(train_df)
        raise AssertionError("fit() must not be called directly when compute_oof is overridden")

    def transform(self, df: pd.DataFrame, params: dict) -> pd.Series:
        self.transform_calls.append(df)
        raise AssertionError("transform() must not be called directly when compute_oof is overridden")

    def compute_oof(self, base_df, folds, *, target_column, time_column=None, id_columns=None, dev_index=None):
        self.compute_oof_calls.append({"n_folds": len(folds), "dev_index": dev_index})
        return pd.Series(9.0, index=base_df.index), {"batched": True}


def test_compute_oof_feature_delegates_to_computer_override() -> None:
    df, y = _synthetic_df()
    builder = DatasetBuilder(cv_folds=5, random_seed=42)
    folds = builder.make_folds(y)
    computer = _ComputeOofOverridingComputer()

    result = builder.compute_oof_feature(df, computer, folds, target_column="Transported")

    assert len(computer.compute_oof_calls) == 1
    assert computer.compute_oof_calls[0]["n_folds"] == 5
    assert computer.fit_calls == []
    assert computer.transform_calls == []
    assert (result.oof_values == 9.0).all()
    assert result.full_fit_params == {"batched": True}
    assert len(result.per_fold_train_index) == 5

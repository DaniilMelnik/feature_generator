"""Fold-aware fit/transform orchestration -- the core anti-leakage mechanism
in code form.

For every feature, ``DatasetBuilder`` computes an out-of-fold (OOF) column
for cross-validation (each row's value comes only from a ``fit()`` performed
on the *other* folds), plus a separate "full fit" of params on the entire
training set for use at inference/serving time. This module knows nothing
about *how* fit/transform execute -- it only depends on the
``sandbox.contract.FeatureComputer`` interface, so it works identically for
trusted, hand-written computers (tests, fixtures) and for
``sandbox.docker_runner.SandboxedFeatureComputer`` (LLM-generated code) alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_generator.profiling.schemas import DataProfile
from feature_generator.sandbox.contract import FeatureComputer


def build_raw_feature_frame(base_df: pd.DataFrame, data_profile: DataProfile) -> tuple[pd.DataFrame, list[str]]:
    """The profiler-identified raw columns (role == "feature"), ready to feed
    CatBoost directly. This is the permanent foundation every run's model is
    trained on -- LLM-generated features are additive on top of it, never a
    replacement for it, and running this alone (zero engineered features) is
    the "features-off" control baseline the pipeline compares itself against.
    """
    # dataset.joiner never renames the primary table's own columns -- only a
    # secondary table's column that collides with one already in the primary
    # table gets the f"{table}_{col}" prefix. Mirror that exact rule here
    # rather than probing base_df.columns, since a secondary-table column can
    # share a name with a PRIMARY-table column without being that column
    # (e.g. IEEE-CIS's "ProductCD" appears on both `transaction` and
    # `identity`) -- a presence check alone would silently pick the wrong one.
    primary_table = next((t for t in data_profile.tables if t.role == "primary"), None)
    primary_column_names = {c.name for c in primary_table.columns} if primary_table else set()

    columns: dict[str, pd.Series] = {}
    cat_features: list[str] = []

    for col in data_profile.all_feature_columns():
        is_primary_column = primary_table is not None and col.source_table == primary_table.table_name
        if is_primary_column or col.name not in primary_column_names:
            source_name = col.name
        else:
            source_name = f"{col.source_table}_{col.name}"

        if source_name not in base_df.columns:
            continue

        columns[source_name] = base_df[source_name]
        if col.categorical_stats is not None:
            cat_features.append(source_name)

    return pd.DataFrame(columns, index=base_df.index), cat_features


@dataclass
class FoldSplit:
    fold_id: int
    train_index: np.ndarray
    val_index: np.ndarray


@dataclass
class OofFeatureResult:
    feature_name: str
    oof_values: pd.Series  # aligned to base_df.index, out-of-fold (leakage-safe for CV)
    full_fit_params: dict  # fit on ALL training rows -- for use at inference time
    per_fold_train_index: list[np.ndarray] = field(default_factory=list)  # for tests/audit


@dataclass
class BuiltDataset:
    folds: list[FoldSplit]
    feature_frame: pd.DataFrame  # one column per feature, OOF values, index-aligned to base_df
    full_fit_params: dict[str, dict]  # feature_name -> params fit on all rows


class DatasetBuilder:
    def __init__(self, cv_folds: int, random_seed: int) -> None:
        if cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        self.cv_folds = cv_folds
        self.random_seed = random_seed

    def split_dev_holdout(
        self,
        y: pd.Series,
        fraction: float,
        *,
        method: str = "random_stratified",
        time_values: pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-time split into a "dev" index (everything the iterative loop
        ever sees -- CV folds, feature engineering, feature selection) and a
        "holdout" index that stays completely untouched until final
        evaluation. Positions into `y`, not labels -- usable directly with
        `.iloc`.

        `method="temporal"` takes the chronologically-last `fraction` of rows
        as holdout (the correct choice once a real time axis exists, e.g.
        IEEE-CIS -- a random holdout would leak future rows into "dev").
        """
        n = len(y)
        if method == "temporal":
            if time_values is None:
                raise ValueError("method='temporal' requires time_values")
            order = np.argsort(time_values.to_numpy(), kind="stable")
            cutoff = int(round(n * (1 - fraction)))
            return order[:cutoff], order[cutoff:]

        dev_idx, holdout_idx = train_test_split(
            np.arange(n), test_size=fraction, stratify=y.to_numpy(), random_state=self.random_seed
        )
        return np.sort(dev_idx), np.sort(holdout_idx)

    def make_folds(self, y: pd.Series, restrict_to: np.ndarray | None = None) -> list[FoldSplit]:
        if restrict_to is None:
            restrict_to = np.arange(len(y))
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        y_subset = y.iloc[restrict_to].to_numpy()
        splits = skf.split(np.zeros(len(y_subset)), y_subset)
        return [
            FoldSplit(fold_id=i, train_index=restrict_to[train_idx], val_index=restrict_to[val_idx])
            for i, (train_idx, val_idx) in enumerate(splits)
        ]

    def compute_oof_feature(
        self,
        base_df: pd.DataFrame,
        computer: FeatureComputer,
        folds: list[FoldSplit],
        *,
        target_column: str,
        time_column: str | None = None,
        id_columns: list[str] | None = None,
        dev_index: np.ndarray | None = None,
    ) -> OofFeatureResult:
        """Compute one feature's out-of-fold values plus its full-data fit.

        `fit()` is called exactly once per fold, on that fold's train rows
        only, and once more on the full dataset for the final/serving params.
        It is NEVER called on validation rows, and `transform()` on a fold's
        validation rows only ever uses that fold's own fit params.

        `dev_index`, when given, restricts the "full fit" step to those rows
        instead of every row in `base_df` -- otherwise a feature's
        serving-time params would already have seen the holdout rows,
        silently invalidating a holdout evaluation done against them.

        The actual fit/transform orchestration lives on the computer itself
        (`FeatureComputer.compute_oof`) -- this method is a thin wrapper that
        adds the audit-only `per_fold_train_index` (pure index lookups, no
        execution) and wraps the result. This is what lets
        `SandboxedFeatureComputer` batch every fold's fit/transform calls
        into a single container invocation without `DatasetBuilder` needing
        to know or care -- see `sandbox.contract.FeatureComputer.compute_oof`.
        """
        id_columns = id_columns or []
        per_fold_train_index = [base_df.index[fold.train_index].to_numpy() for fold in folds]

        oof, full_fit_params = computer.compute_oof(
            base_df,
            folds,
            target_column=target_column,
            time_column=time_column,
            id_columns=id_columns,
            dev_index=dev_index,
        )

        return OofFeatureResult(
            feature_name=computer.feature_name,
            oof_values=oof,
            full_fit_params=full_fit_params,
            per_fold_train_index=per_fold_train_index,
        )

    def build(
        self,
        base_df: pd.DataFrame,
        y: pd.Series,
        computers: list[FeatureComputer],
        *,
        target_column: str,
        time_column: str | None = None,
        id_columns: list[str] | None = None,
        folds: list[FoldSplit] | None = None,
    ) -> BuiltDataset:
        folds = folds if folds is not None else self.make_folds(y)
        feature_columns: dict[str, pd.Series] = {}
        full_fit_params: dict[str, dict] = {}

        for computer in computers:
            result = self.compute_oof_feature(
                base_df,
                computer,
                folds,
                target_column=target_column,
                time_column=time_column,
                id_columns=id_columns,
            )
            feature_columns[result.feature_name] = result.oof_values
            full_fit_params[result.feature_name] = result.full_fit_params

        feature_frame = pd.DataFrame(feature_columns, index=base_df.index)
        return BuiltDataset(folds=folds, feature_frame=feature_frame, full_fit_params=full_fit_params)

    def transform_new_data(
        self,
        new_df: pd.DataFrame,
        computers: list[FeatureComputer],
        full_fit_params: dict[str, dict],
    ) -> pd.DataFrame:
        """Apply already-fit (full-training-set) params to brand-new rows --
        i.e. what happens at inference/serving time, never re-fitting.
        """
        columns = {
            computer.feature_name: computer.transform(new_df, full_fit_params[computer.feature_name])
            for computer in computers
        }
        return pd.DataFrame(columns, index=new_df.index)

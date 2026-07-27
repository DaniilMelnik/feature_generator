"""Deterministic data profiling: pandas/numpy/scipy only, no LLM involved.

This produces the structured ``DataProfile`` that the hypothesis agent reads
and reasons over. Keeping "analysis" entirely in classical, reproducible code
(rather than giving the LLM live code execution over raw rows) keeps this
phase cheap, deterministic, and outside the sandbox/safety surface.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from feature_generator.config import DatasetConfig
from feature_generator.profiling.schemas import (
    CategoricalStats,
    ColumnProfile,
    ColumnRole,
    DataProfile,
    DatetimeStats,
    NumericStats,
    TableProfile,
)

QUANTILE_LEVELS = {"p1": 0.01, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p99": 0.99}
TOP_VALUES_LIMIT = 20
MIN_ROWS_FOR_CORRELATION = 10


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def binarize_target(series: pd.Series) -> pd.Series:
    """Map an arbitrary 2-class target column to {0, 1} ints."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        uniques = series.dropna().unique()
        if set(uniques.tolist()) <= {0, 1}:
            return series.astype(int)
        raise ValueError(
            f"Target column is numeric but not already binary 0/1: "
            f"unique values include {sorted(uniques.tolist())[:5]}"
        )
    uniques = sorted(series.dropna().unique().tolist(), key=str)
    if len(uniques) != 2:
        raise ValueError(
            f"Target column must have exactly 2 classes for binary classification; "
            f"found {len(uniques)}: {uniques}"
        )
    mapping = {uniques[0]: 0, uniques[1]: 1}
    return series.map(mapping).astype(int)


def _column_role(column: str, *, table_name: str, config: DatasetConfig) -> ColumnRole:
    if table_name == config.target_table and column == config.target_column:
        return "target"
    if column in config.id_columns:
        return "id"
    if config.time_column is not None and column == config.time_column:
        return "time"
    if column in config.known_leakage_columns:
        return "ignore"
    return "feature"


def _is_effectively_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _numeric_stats(series: pd.Series) -> NumericStats:
    clean = series.dropna().astype(float)
    if len(clean) == 0:
        zero = {k: 0.0 for k in QUANTILE_LEVELS}
        return NumericStats(min=0.0, max=0.0, mean=0.0, std=0.0, skew=0.0, kurtosis=0.0, quantiles=zero)
    quantiles = {k: float(clean.quantile(q)) for k, q in QUANTILE_LEVELS.items()}
    return NumericStats(
        min=float(clean.min()),
        max=float(clean.max()),
        mean=float(clean.mean()),
        std=float(clean.std()) if len(clean) > 1 else 0.0,
        skew=float(scipy_stats.skew(clean)) if len(clean) > 2 else 0.0,
        kurtosis=float(scipy_stats.kurtosis(clean)) if len(clean) > 3 else 0.0,
        quantiles=quantiles,
    )


def _categorical_stats(series: pd.Series) -> CategoricalStats:
    value_counts = series.astype(str).value_counts().head(TOP_VALUES_LIMIT)
    return CategoricalStats(
        cardinality=int(series.nunique(dropna=True)),
        top_values=[(str(v), int(c)) for v, c in value_counts.items()],
    )


def _datetime_stats(series: pd.Series) -> DatetimeStats:
    clean = series.dropna()
    min_ts, max_ts = clean.min(), clean.max()
    span_days = (max_ts - min_ts).total_seconds() / 86400.0
    return DatetimeStats(
        min_ts=min_ts.to_pydatetime(), max_ts=max_ts.to_pydatetime(), span_days=span_days
    )


def _cramers_v(contingency: pd.DataFrame) -> float | None:
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    chi2, _, _, _ = scipy_stats.chi2_contingency(contingency, correction=False)
    n = contingency.to_numpy().sum()
    if n == 0:
        return None
    r, k = contingency.shape
    denom = min(r - 1, k - 1)
    if denom <= 0:
        return 0.0
    v = float(np.sqrt((chi2 / n) / denom))
    return v if np.isfinite(v) else None


def _correlation_with_target(feature: pd.Series, target: pd.Series) -> float | None:
    paired = pd.DataFrame({"f": feature, "t": target}).dropna()
    if len(paired) < MIN_ROWS_FOR_CORRELATION or paired["t"].nunique() < 2:
        return None
    if _is_effectively_numeric(paired["f"]):
        if paired["f"].nunique() < 2:
            return None
        corr, _ = scipy_stats.pointbiserialr(paired["t"], paired["f"])
        return float(corr) if np.isfinite(corr) else None
    return _cramers_v(pd.crosstab(paired["f"], paired["t"]))


def _profile_column(
    series: pd.Series,
    *,
    name: str,
    table_name: str,
    role: ColumnRole,
    target: pd.Series | None,
) -> ColumnProfile:
    n_missing = int(series.isna().sum())
    n_total = len(series)
    n_unique = int(series.nunique(dropna=True))
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    is_numeric = _is_effectively_numeric(series)

    numeric_stats = categorical_stats = datetime_stats = None
    if is_datetime:
        datetime_stats = _datetime_stats(series)
    elif is_numeric:
        numeric_stats = _numeric_stats(series)
    else:
        categorical_stats = _categorical_stats(series)

    correlation = None
    if role == "feature" and target is not None:
        correlation = _correlation_with_target(series, target)

    return ColumnProfile(
        name=name,
        source_table=table_name,
        dtype=str(series.dtype),
        role=role,
        n_missing=n_missing,
        pct_missing=(n_missing / n_total) if n_total else 0.0,
        n_unique=n_unique,
        is_constant=n_unique <= 1,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        datetime_stats=datetime_stats,
        correlation_with_target=correlation,
    )


def profile_tables(config: DatasetConfig) -> DataProfile:
    """Load every table in ``config`` and compute a structured ``DataProfile``.

    Correlation-with-target is only computed for feature columns that live in
    the target table -- secondary tables need a join (``dataset.joiner``)
    before their columns can be correlated, which happens later in the
    pipeline, not during profiling.
    """
    frames: dict[str, pd.DataFrame] = {t.name: read_table(t.path) for t in config.tables}
    target_series = binarize_target(frames[config.target_table][config.target_column])

    notes: list[str] = []
    table_profiles: list[TableProfile] = []
    for table_cfg in config.tables:
        df = frames[table_cfg.name]
        target_for_table = target_series if table_cfg.name == config.target_table else None
        columns = [
            _profile_column(
                df[col],
                name=col,
                table_name=table_cfg.name,
                role=_column_role(col, table_name=table_cfg.name, config=config),
                target=target_for_table,
            )
            for col in df.columns
        ]
        if table_cfg.role == "secondary":
            notes.append(
                f"Table '{table_cfg.name}' is secondary and joins to the primary table on "
                f"'{table_cfg.join_key}'; its columns' correlation_with_target is null here "
                f"because computing it requires joining to the target first."
            )
        table_profiles.append(
            TableProfile(
                table_name=table_cfg.name,
                file_path=table_cfg.path,
                n_rows=len(df),
                n_cols=len(df.columns),
                join_key=table_cfg.join_key,
                role=table_cfg.role,
                columns=columns,
            )
        )

    if config.known_leakage_columns:
        notes.append(
            f"Columns {config.known_leakage_columns} are flagged known_leakage_columns and "
            "marked role='ignore' -- do not use them directly as feature inputs."
        )

    return DataProfile(
        dataset_name=config.name,
        tables=table_profiles,
        target_column=config.target_column,
        target_table=config.target_table,
        id_columns=config.id_columns,
        time_column=config.time_column,
        positive_rate=float(target_series.mean()),
        known_leakage_columns=config.known_leakage_columns,
        profiling_notes=notes,
        generated_at=datetime.utcnow(),
    )

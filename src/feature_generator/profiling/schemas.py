"""Pydantic schemas for the deterministic data-profiling output.

These are computed by ``profiling.profiler`` (pandas/scipy/numpy only, no LLM)
and are the *only* view of the raw data the hypothesis agent ever sees.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ColumnRole = Literal["feature", "target", "id", "time", "ignore"]


class NumericStats(BaseModel):
    min: float
    max: float
    mean: float
    std: float
    skew: float
    kurtosis: float
    quantiles: dict[str, float]  # keys: "p1", "p25", "p50", "p75", "p99"


class CategoricalStats(BaseModel):
    cardinality: int
    top_values: list[tuple[str, int]] = Field(default_factory=list)  # (value, count), top 20


class DatetimeStats(BaseModel):
    min_ts: datetime
    max_ts: datetime
    span_days: float


class ColumnProfile(BaseModel):
    name: str
    source_table: str
    dtype: str
    role: ColumnRole
    n_missing: int
    pct_missing: float
    n_unique: int
    is_constant: bool
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None
    # point-biserial correlation (numeric) or Cramer's V (categorical) with the
    # target, computed on the TRAIN split only. None for id/time/ignore columns.
    correlation_with_target: float | None = None


class TableProfile(BaseModel):
    table_name: str
    file_path: str
    n_rows: int
    n_cols: int
    join_key: str | None = None
    role: Literal["primary", "secondary"] = "primary"
    columns: list[ColumnProfile]


class DataProfile(BaseModel):
    dataset_name: str
    tables: list[TableProfile]
    target_column: str
    target_table: str
    id_columns: list[str]
    time_column: str | None = None
    positive_rate: float
    task_type: Literal["binary_classification"] = "binary_classification"
    known_leakage_columns: list[str] = Field(default_factory=list)
    profiling_notes: list[str] = Field(default_factory=list)
    generated_at: datetime

    def primary_table(self) -> TableProfile:
        for t in self.tables:
            if t.role == "primary":
                return t
        raise ValueError("DataProfile has no primary table")

    def all_feature_columns(self) -> list[ColumnProfile]:
        return [c for t in self.tables for c in t.columns if c.role == "feature"]

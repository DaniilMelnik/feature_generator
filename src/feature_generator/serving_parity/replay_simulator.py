"""The "online test bed": replays raw rows as if arriving one-at-a-time (or in
small batches) through production, using the exact same `transform()` code
path as the offline batch pipeline, and diffs the result against the
offline-computed values.

Invariant being checked: `transform(df, params)` must be a PURE, ROW-WISE
function of its own input rows and `params` -- no dependence on which other
rows happen to be present in the same call, and no dependence on rows that
are chronologically later. This is exactly what the fit/transform contract
(sandbox.contract) asks feature code to uphold; this module verifies it
rather than trusting compliance. Any mismatch is a hard block on promotion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_generator.sandbox.contract import FeatureComputer
from feature_generator.schemas import ServingParityMismatch, ServingParityResult

DEFAULT_IID_BATCH_SIZES = [1, 5, 37]
DEFAULT_TEMPORAL_BATCH_SIZES = [1, 50, 500]
DEFAULT_MAX_ROWS_FOR_TRUNCATED_CHECK = 200
MAX_MISMATCH_EXAMPLES = 10


def _to_jsonable(value: object) -> float | int | bool | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _values_match(offline_value: object, online_value: object) -> tuple[bool, float | None]:
    offline_na, online_na = pd.isna(offline_value), pd.isna(online_value)
    if offline_na or online_na:
        return (bool(offline_na) and bool(online_na)), None

    is_numeric = isinstance(offline_value, (float, np.floating)) or isinstance(
        online_value, (float, np.floating)
    )
    if is_numeric:
        try:
            a, b = float(offline_value), float(online_value)
        except (TypeError, ValueError):
            return offline_value == online_value, None
        return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12)), abs(a - b)

    return offline_value == online_value, None


def _replay_in_batches(computer: FeatureComputer, df: pd.DataFrame, params: dict, batch_size: int) -> pd.Series:
    """Chunk `df` into `batch_size`-row groups and transform() each -- via
    ONE `transform_many` call (many jobs, one container for a Docker-backed
    computer) rather than one call per chunk.
    """
    if len(df) == 0:
        return pd.Series(dtype="float64")
    n = len(df)
    jobs = [list(range(start, min(start + batch_size, n))) for start in range(0, n, batch_size)]
    results = computer.transform_many(df, params, jobs)
    chunks = [
        pd.Series(np.asarray(result)).set_axis(df.iloc[job].index) for job, result in zip(jobs, results)
    ]
    return pd.concat(chunks)


class _MismatchAccumulator:
    def __init__(self) -> None:
        self.mismatches: list[ServingParityMismatch] = []
        self.max_abs_diff: float | None = None
        self.n_checked = 0

    def compare(self, offline: pd.Series, online: pd.Series) -> None:
        for idx in online.index:
            offline_value = offline.loc[idx]
            online_value = online.loc[idx]
            matched, diff = _values_match(offline_value, online_value)
            self.n_checked += 1
            if diff is not None:
                self.max_abs_diff = diff if self.max_abs_diff is None else max(self.max_abs_diff, diff)
            if not matched:
                self.mismatches.append(
                    ServingParityMismatch(
                        row_id=str(idx),
                        offline_value=_to_jsonable(offline_value),
                        online_value=_to_jsonable(online_value),
                        abs_diff=diff,
                    )
                )

    def compare_scalar(self, row_id: object, offline_value: object, online_value: object) -> None:
        matched, diff = _values_match(offline_value, online_value)
        self.n_checked += 1
        if diff is not None:
            self.max_abs_diff = diff if self.max_abs_diff is None else max(self.max_abs_diff, diff)
        if not matched:
            self.mismatches.append(
                ServingParityMismatch(
                    row_id=str(row_id),
                    offline_value=_to_jsonable(offline_value),
                    online_value=_to_jsonable(online_value),
                    abs_diff=diff,
                )
            )

    def to_result(self, mode: str) -> ServingParityResult:
        return ServingParityResult(
            ran=True,
            matched=len(self.mismatches) == 0,
            mode=mode,
            n_rows_checked=self.n_checked,
            n_mismatches=len(self.mismatches),
            max_abs_diff=self.max_abs_diff,
            mismatch_examples=self.mismatches[:MAX_MISMATCH_EXAMPLES],
        )


def run_iid(
    computer: FeatureComputer,
    params: dict,
    df: pd.DataFrame,
    *,
    batch_sizes: list[int] | None = None,
    seed: int = 0,
) -> ServingParityResult:
    """Replay iid data (no meaningful row order) in shuffled order and
    varying batch sizes -- catches order/positional-index bugs such as an
    internal `reset_index()` or `.rank()` inside `transform()`.
    """
    batch_sizes = batch_sizes or DEFAULT_IID_BATCH_SIZES
    offline_values = pd.Series(np.asarray(computer.transform(df, params))).set_axis(df.index)

    rng = np.random.default_rng(seed)
    shuffled_index = df.index.to_numpy().copy()
    rng.shuffle(shuffled_index)
    shuffled_df = df.loc[shuffled_index]

    acc = _MismatchAccumulator()
    for batch_size in batch_sizes:
        online_values = _replay_in_batches(computer, shuffled_df, params, batch_size)
        acc.compare(offline_values, online_values)

    return acc.to_result("iid_shuffle")


def _sample_positions(n_rows: int, max_rows: int) -> list[int]:
    if n_rows <= max_rows:
        return list(range(n_rows))
    step = n_rows / max_rows
    return sorted({int(i * step) for i in range(max_rows)})


def run_temporal(
    computer: FeatureComputer,
    params: dict,
    df: pd.DataFrame,
    time_column: str,
    *,
    batch_sizes: list[int] | None = None,
    max_rows_for_truncated_check: int = DEFAULT_MAX_ROWS_FOR_TRUNCATED_CHECK,
) -> ServingParityResult:
    """Replay data in true chronological order.

    Two checks: (1) batch replay in chronologically-ordered chunks (same
    order/batch-size bugs as `run_iid`); (2) truncated-visibility replay --
    for a sample of rows at time T, call `transform()` with input restricted
    to rows where time <= T and compare to the full-window offline value.
    A feature that aggregates across its own batch instead of using a
    `fit()`-precomputed lookup diverges here even if it passed the batch
    replay, because withholding "future" same-batch rows changes its result.
    """
    batch_sizes = batch_sizes or DEFAULT_TEMPORAL_BATCH_SIZES
    sorted_df = df.sort_values(time_column, kind="stable")
    offline_values = pd.Series(np.asarray(computer.transform(sorted_df, params))).set_axis(sorted_df.index)

    acc = _MismatchAccumulator()

    for batch_size in batch_sizes:
        online_values = _replay_in_batches(computer, sorted_df, params, batch_size)
        acc.compare(offline_values, online_values)

    # each "window" job is rows [0..pos] -- withholding rows after `pos`
    # tests exactly what a feature only sees at genuine serving time. All
    # windows go through ONE transform_many call (one container for a
    # Docker-backed computer) rather than one call per sampled row.
    sample_positions = _sample_positions(len(sorted_df), max_rows_for_truncated_check)
    window_jobs = [list(range(pos + 1)) for pos in sample_positions]
    window_results = computer.transform_many(sorted_df, params, window_jobs)
    for pos, result in zip(sample_positions, window_results):
        row_id = sorted_df.index[pos]
        online_value = pd.Series(np.asarray(result)).iloc[-1]  # the window's last row is `pos` itself
        acc.compare_scalar(row_id, offline_values.loc[row_id], online_value)

    return acc.to_result("temporal_replay")

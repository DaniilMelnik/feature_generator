"""Population/Characteristic Stability Index (PSI/CSI) computation.

PSI is conventionally applied to the model *score* distribution, CSI to an
individual *feature*'s distribution -- both are the same statistic, so both
are implemented on top of one shared ``stability_index_from_proportions``.

Industry-standard thresholds: < 0.1 stable, 0.1-0.25 watch, > 0.25 unstable.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

STABLE_THRESHOLD = 0.1
UNSTABLE_THRESHOLD = 0.25
EPSILON = 1e-4


def stability_index_from_proportions(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    """The PSI/CSI formula applied directly to two proportion vectors that
    already sum to ~1 and are aligned bin-for-bin.
    """
    expected_pct = np.clip(np.asarray(expected_pct, dtype=float), EPSILON, None)
    actual_pct = np.clip(np.asarray(actual_pct, dtype=float), EPSILON, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def classify_psi(value: float) -> Literal["stable", "watch", "unstable"]:
    if value < STABLE_THRESHOLD:
        return "stable"
    if value < UNSTABLE_THRESHOLD:
        return "watch"
    return "unstable"


def _quantile_bin_edges(reference: pd.Series, bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        # degenerate (constant) reference distribution -- fall back to one bin
        lo = float(reference.min()) if len(reference) else 0.0
        edges = np.array([lo - 1e-9, lo + 1e-9])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI between a numeric reference (`expected`) and current (`actual`)
    distribution, using quantile bin edges derived from `expected`.
    """
    expected_clean = expected.dropna().astype(float)
    actual_clean = actual.dropna().astype(float)
    edges = _quantile_bin_edges(expected_clean, bins)

    expected_counts, _ = np.histogram(expected_clean, bins=edges)
    actual_counts, _ = np.histogram(actual_clean, bins=edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)
    return stability_index_from_proportions(expected_pct, actual_pct)


def compute_csi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """CSI for one feature. Numeric features use the same quantile-binned PSI
    computation; categorical features bin by category value directly (a
    category present in one side but not the other still counts, at a
    smoothed near-zero proportion via `EPSILON`).
    """
    is_numeric = pd.api.types.is_numeric_dtype(expected) and not pd.api.types.is_bool_dtype(expected)
    if is_numeric:
        return compute_psi(expected, actual, bins=bins)

    expected_counts_s = expected.dropna().astype(str).value_counts()
    actual_counts_s = actual.dropna().astype(str).value_counts()
    categories = sorted(set(expected_counts_s.index) | set(actual_counts_s.index))
    if not categories:
        return 0.0

    expected_counts = np.array([expected_counts_s.get(c, 0) for c in categories], dtype=float)
    actual_counts = np.array([actual_counts_s.get(c, 0) for c in categories], dtype=float)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)
    return stability_index_from_proportions(expected_pct, actual_pct)


def bootstrap_windows(n_rows: int, n_iterations: int, sample_frac: float = 0.5, seed: int = 42) -> list[np.ndarray]:
    """Index arrays for repeated random subsamples of a population -- used as
    a stand-in "current window" against the full population as "reference"
    when a dataset has no time axis (e.g. Spaceship Titanic).
    """
    rng = np.random.default_rng(seed)
    sample_size = max(1, int(n_rows * sample_frac))
    return [rng.choice(n_rows, size=sample_size, replace=False) for _ in range(n_iterations)]


def compute_bootstrap_stability(
    series: pd.Series, n_iterations: int, *, sample_frac: float = 0.5, seed: int = 42, bins: int = 10
) -> list[tuple[str, float]]:
    """PSI/CSI between the full series (reference) and each of several random
    subsamples (stand-in "current" windows). A genuinely stable feature
    should show small values here even with no real time axis to test against.
    """
    windows = bootstrap_windows(len(series), n_iterations, sample_frac=sample_frac, seed=seed)
    results: list[tuple[str, float]] = []
    for i, idx in enumerate(windows):
        subset = series.iloc[idx]
        value = compute_csi(series, subset, bins=bins)
        results.append((f"bootstrap_{i}", value))
    return results

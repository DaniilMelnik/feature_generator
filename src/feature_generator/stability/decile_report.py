"""Score-decile stability and target-rate-by-decile-over-time reporting.

These reuse the PSI machinery from ``stability.csi_psi`` to answer a more
specific question than a raw feature/score PSI: not just "did the score
distribution shift" but "did the population *and* the target rate within
each decile of that distribution stay stable".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_generator.stability.csi_psi import classify_psi, compute_psi

N_DECILES_DEFAULT = 10


def _decile_edges(reference_scores: pd.Series, n_deciles: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_deciles + 1)
    edges = np.unique(reference_scores.quantile(quantiles).to_numpy()).astype(float)
    if len(edges) < 2:
        lo = float(reference_scores.min()) if len(reference_scores) else 0.0
        edges = np.array([lo - 1e-9, lo + 1e-9])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _assign_deciles(scores: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = list(range(len(edges) - 1))
    return pd.cut(scores, bins=edges, labels=labels, include_lowest=True).astype("Int64")


def score_decile_stability(
    reference_scores: pd.Series, current_scores: pd.Series, n_deciles: int = N_DECILES_DEFAULT
) -> dict:
    """Compare the proportion of population landing in each decile (deciles
    defined by `reference_scores`) between the reference and current windows.
    """
    edges = _decile_edges(reference_scores, n_deciles)
    ref_deciles = _assign_deciles(reference_scores, edges)
    cur_deciles = _assign_deciles(current_scores, edges)

    ref_pct = ref_deciles.value_counts(normalize=True).sort_index()
    cur_pct = cur_deciles.value_counts(normalize=True).sort_index()
    all_labels = sorted(set(ref_pct.index) | set(cur_pct.index))

    per_decile = {
        int(label): {
            "reference_pct": float(ref_pct.get(label, 0.0)),
            "current_pct": float(cur_pct.get(label, 0.0)),
        }
        for label in all_labels
    }
    psi = compute_psi(reference_scores, current_scores, bins=n_deciles)

    return {
        "n_deciles": n_deciles,
        "per_decile": per_decile,
        "psi": psi,
        "stability_flag": classify_psi(psi),
    }


def target_rate_by_decile_over_time(
    scores: pd.Series,
    y: pd.Series,
    *,
    time_bucket: pd.Series | None = None,
    n_deciles: int = N_DECILES_DEFAULT,
) -> dict:
    """Target rate within each decile of `scores` (deciles defined on the
    full `scores` series), optionally broken down by `time_bucket` so a
    monotonic decile/target relationship can be checked for stability across
    windows rather than just in aggregate.
    """
    edges = _decile_edges(scores, n_deciles)
    deciles = _assign_deciles(scores, edges)

    overall = pd.DataFrame({"decile": deciles, "target": y.to_numpy()}).groupby("decile", observed=True)[
        "target"
    ].mean()
    overall_rates = {int(k): float(v) for k, v in overall.items()}

    is_monotonic = _is_non_decreasing([overall_rates[d] for d in sorted(overall_rates)])

    result: dict = {
        "n_deciles": n_deciles,
        "overall_target_rate_by_decile": overall_rates,
        "is_monotonic_overall": is_monotonic,
    }

    if time_bucket is not None:
        frame = pd.DataFrame({"decile": deciles, "target": y.to_numpy(), "bucket": time_bucket.to_numpy()})
        grouped = frame.groupby(["bucket", "decile"], observed=True)["target"].mean()
        by_bucket: dict[str, dict[int, float]] = {}
        for (bucket, decile), rate in grouped.items():
            by_bucket.setdefault(str(bucket), {})[int(decile)] = float(rate)
        result["target_rate_by_decile_by_bucket"] = by_bucket

    return result


def _is_non_decreasing(values: list[float]) -> bool:
    return all(a <= b + 1e-9 for a, b in zip(values, values[1:]))

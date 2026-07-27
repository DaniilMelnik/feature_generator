import numpy as np
import pandas as pd
import pytest

from feature_generator.stability.csi_psi import (
    classify_psi,
    compute_bootstrap_stability,
    compute_csi,
    compute_psi,
    stability_index_from_proportions,
)


def test_stability_index_matches_hand_computed_reference() -> None:
    # PSI = (0.6-0.5)*ln(0.6/0.5) + (0.4-0.5)*ln(0.4/0.5) = 0.04054651081...
    value = stability_index_from_proportions([0.5, 0.5], [0.6, 0.4])
    assert value == pytest.approx(0.04054651081, abs=1e-9)


def test_stability_index_is_zero_for_identical_proportions() -> None:
    value = stability_index_from_proportions([0.3, 0.3, 0.4], [0.3, 0.3, 0.4])
    assert value == pytest.approx(0.0, abs=1e-12)


def test_classify_psi_thresholds() -> None:
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "watch"
    assert classify_psi(0.30) == "unstable"
    assert classify_psi(0.0999) == "stable"
    assert classify_psi(0.1) == "watch"
    assert classify_psi(0.25) == "unstable"


def test_compute_psi_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, size=5000))
    current = pd.Series(rng.normal(0, 1, size=5000))
    psi = compute_psi(reference, current, bins=10)
    assert psi < 0.05
    assert classify_psi(psi) == "stable"


def test_compute_psi_is_high_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, size=5000))
    current = pd.Series(rng.normal(2.5, 1, size=5000))  # large mean shift
    psi = compute_psi(reference, current, bins=10)
    assert psi > 0.25
    assert classify_psi(psi) == "unstable"


def test_compute_csi_numeric_matches_compute_psi() -> None:
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(0, 1, size=1000))
    current = pd.Series(rng.normal(0.5, 1, size=1000))
    assert compute_csi(reference, current, bins=10) == pytest.approx(
        compute_psi(reference, current, bins=10)
    )


def test_compute_csi_categorical_hand_computed() -> None:
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current = pd.Series(["A"] * 900 + ["B"] * 100)
    # proportions: expected [0.5, 0.5], actual [0.9, 0.1]
    expected_value = stability_index_from_proportions([0.5, 0.5], [0.9, 0.1])
    assert compute_csi(reference, current) == pytest.approx(expected_value)


def test_compute_csi_categorical_handles_unseen_category() -> None:
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current = pd.Series(["A"] * 500 + ["C"] * 500)  # B disappeared, C appeared
    psi = compute_csi(reference, current)
    assert psi > 0.25  # a full category swap is a dramatic shift


def test_compute_psi_handles_constant_reference_distribution() -> None:
    reference = pd.Series([5.0] * 100)
    current = pd.Series([5.0] * 90 + [6.0] * 10)
    # must not raise despite a degenerate (single-value) reference distribution
    psi = compute_psi(reference, current, bins=10)
    assert psi >= 0.0


def test_bootstrap_stability_is_low_for_a_stable_feature() -> None:
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0, 1, size=2000))
    results = compute_bootstrap_stability(series, n_iterations=5, sample_frac=0.5, seed=3)
    assert len(results) == 5
    labels = [label for label, _ in results]
    assert labels == [f"bootstrap_{i}" for i in range(5)]
    assert all(value < 0.1 for _, value in results)

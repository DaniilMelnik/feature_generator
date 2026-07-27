import numpy as np
import pandas as pd
import pytest

from feature_generator.stability.decile_report import (
    score_decile_stability,
    target_rate_by_decile_over_time,
)


def test_score_decile_stability_near_zero_for_identical_windows() -> None:
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.uniform(0, 1, size=2000))
    report = score_decile_stability(scores, scores, n_deciles=10)

    assert report["n_deciles"] == 10
    assert len(report["per_decile"]) == 10
    assert report["psi"] < 0.05
    assert report["stability_flag"] == "stable"
    for decile_stats in report["per_decile"].values():
        assert decile_stats["reference_pct"] == pytest.approx(decile_stats["current_pct"], abs=0.02)
        assert decile_stats["reference_pct"] == pytest.approx(0.1, abs=0.02)


def test_score_decile_stability_detects_population_shift_to_top_deciles() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.uniform(0, 1, size=2000))
    # current window is heavily skewed toward high scores
    current = pd.Series(rng.uniform(0.7, 1.0, size=2000))

    report = score_decile_stability(reference, current, n_deciles=10)
    assert report["psi"] > 0.25
    assert report["stability_flag"] == "unstable"
    # top decile (9) should now hold much more than 10% of the current population
    assert report["per_decile"][9]["current_pct"] > 0.3


def test_target_rate_by_decile_is_monotonic_for_a_well_ranking_score() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    scores = pd.Series(rng.uniform(0, 1, size=n))
    # target probability increases with score -> should give a monotonic decile lift
    y = pd.Series((rng.uniform(size=n) < scores).astype(int))

    report = target_rate_by_decile_over_time(scores, y, n_deciles=10)
    assert report["is_monotonic_overall"] is True
    rates = [report["overall_target_rate_by_decile"][d] for d in range(10)]
    assert rates[0] < rates[-1]


def test_target_rate_by_decile_flags_non_monotonic_score() -> None:
    n = 500
    scores = pd.Series(np.linspace(0, 1, n))
    # target rate is high at both extremes and low in the middle -- not monotonic
    y = pd.Series(((scores < 0.1) | (scores > 0.9)).astype(int))

    report = target_rate_by_decile_over_time(scores, y, n_deciles=10)
    assert report["is_monotonic_overall"] is False


def test_target_rate_by_decile_with_time_buckets_breaks_down_per_bucket() -> None:
    rng = np.random.default_rng(2)
    n = 1000
    scores = pd.Series(rng.uniform(0, 1, size=n))
    y = pd.Series((rng.uniform(size=n) < scores).astype(int))
    bucket = pd.Series(np.where(np.arange(n) < n // 2, "window_1", "window_2"))

    report = target_rate_by_decile_over_time(scores, y, time_bucket=bucket, n_deciles=5)

    assert "target_rate_by_decile_by_bucket" in report
    by_bucket = report["target_rate_by_decile_by_bucket"]
    assert set(by_bucket.keys()) == {"window_1", "window_2"}
    for bucket_rates in by_bucket.values():
        assert len(bucket_rates) <= 5

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_generator.sandbox.contract import (
    FitContext,
    InProcessFeatureComputer,
    load_trusted_module_from_file,
)
from feature_generator.serving_parity.replay_simulator import run_iid, run_temporal

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(fixture_name: str, **kwargs) -> InProcessFeatureComputer:
    module = load_trusted_module_from_file(FIXTURES / fixture_name)
    return InProcessFeatureComputer(module, **kwargs)


def _spend_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "HomePlanet": rng.choice(["Earth", "Mars", "Europa"], size=n),
            "RoomService": rng.exponential(50, size=n).round(2),
        }
    )


def test_compliant_fixture_passes_iid_replay() -> None:
    computer = _load("compliant_feature_groupby_mean.py")
    df = _spend_df()
    params = computer.fit(df, FitContext(target_column="Transported"))

    result = run_iid(computer, params, df, seed=1)

    assert result.ran is True
    assert result.matched is True
    assert result.n_mismatches == 0


def test_order_dependent_fixture_fails_iid_replay() -> None:
    computer = _load("leaky_feature_order_dependent.py")
    df = _spend_df()
    params = computer.fit(df, FitContext(target_column="Transported"))

    result = run_iid(computer, params, df, seed=1)

    assert result.matched is False
    assert result.n_mismatches > 0
    assert result.mode == "iid_shuffle"
    assert len(result.mismatch_examples) > 0


def test_compliant_fixture_passes_temporal_replay() -> None:
    computer = _load("compliant_feature_groupby_mean.py")
    df = _spend_df()
    df["TransactionDT"] = np.arange(len(df))  # already-monotonic time column
    params = computer.fit(df, FitContext(target_column="Transported"))

    result = run_temporal(computer, params, df, time_column="TransactionDT")

    assert result.matched is True
    assert result.mode == "temporal_replay"


def test_temporal_cumsum_fixture_fails_temporal_replay() -> None:
    computer = _load("leaky_feature_temporal_cumsum.py")
    df = _spend_df()
    df["TransactionDT"] = np.arange(len(df))
    params = computer.fit(df, FitContext(target_column="Transported"))

    result = run_temporal(computer, params, df, time_column="TransactionDT")

    assert result.matched is False
    assert result.n_mismatches > 0


def test_temporal_cumsum_fixture_also_fails_iid_replay() -> None:
    # a feature that aggregates across its own batch is unsafe regardless of
    # whether the dataset has a time axis at all
    computer = _load("leaky_feature_temporal_cumsum.py")
    df = _spend_df()
    params = computer.fit(df, FitContext(target_column="Transported"))

    result = run_iid(computer, params, df, seed=2)
    assert result.matched is False


def test_run_iid_reports_max_abs_diff_for_float_mismatches() -> None:
    computer = _load("leaky_feature_order_dependent.py")
    df = _spend_df()
    params = computer.fit(df, FitContext(target_column="Transported"))

    result = run_iid(computer, params, df, seed=1)
    assert result.max_abs_diff is not None
    assert result.max_abs_diff > 0


def test_run_temporal_truncated_visibility_uses_sampling_for_large_data() -> None:
    computer = _load("compliant_feature_groupby_mean.py")
    df = _spend_df(n=500)
    df["TransactionDT"] = np.arange(len(df))
    params = computer.fit(df, FitContext(target_column="Transported"))

    result = run_temporal(
        computer, params, df, time_column="TransactionDT", max_rows_for_truncated_check=50
    )
    assert result.matched is True
    # batch replay checks every row across 3 batch sizes + <=50 truncated-visibility rows
    assert result.n_rows_checked >= len(df) * 3

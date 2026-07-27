from pathlib import Path

import pandas as pd
import pytest

from feature_generator.dataset.feature_store import FeatureStore


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    series = pd.Series([1.0, 2.0, 3.0], name="ignored_name")

    store.save("run-1", "avg_spend", series)
    loaded = store.load("run-1", "avg_spend")

    assert list(loaded) == [1.0, 2.0, 3.0]
    assert loaded.name == "avg_spend"


def test_exists_reflects_saved_state(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    assert store.exists("run-1", "avg_spend") is False
    store.save("run-1", "avg_spend", pd.Series([1.0]))
    assert store.exists("run-1", "avg_spend") is True


def test_delete_removes_file(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    store.save("run-1", "avg_spend", pd.Series([1.0]))
    store.delete("run-1", "avg_spend")
    assert store.exists("run-1", "avg_spend") is False
    store.delete("run-1", "avg_spend")  # deleting a missing feature is a no-op, not an error


def test_list_features_returns_sorted_names(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    store.save("run-1", "zeta", pd.Series([1.0]))
    store.save("run-1", "alpha", pd.Series([1.0]))
    assert store.list_features("run-1") == ["alpha", "zeta"]
    assert store.list_features("run-does-not-exist") == []


def test_unsafe_feature_name_characters_are_sanitized(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    weird_name = "avg spend/per household!"
    store.save("run 1", weird_name, pd.Series([5.0]))
    assert store.exists("run 1", weird_name)
    loaded = store.load("run 1", weird_name)
    assert list(loaded) == [5.0]


def test_different_runs_are_isolated(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    store.save("run-a", "f", pd.Series([1.0]))
    store.save("run-b", "f", pd.Series([2.0]))
    assert store.load("run-a", "f").iloc[0] == 1.0
    assert store.load("run-b", "f").iloc[0] == 2.0

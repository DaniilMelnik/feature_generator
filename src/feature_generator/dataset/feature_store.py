"""Parquet-backed cache of computed feature columns, keyed by (run_id,
feature_name). Avoids recomputing an expensive fold-wise fit/transform for a
feature that has already been validated earlier in the same run (e.g. across
a checkpoint resume).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_name(feature_name: str) -> str:
    return _UNSAFE_CHARS.sub("_", feature_name)


class FeatureStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str, feature_name: str) -> Path:
        return self.base_dir / _safe_name(run_id) / f"{_safe_name(feature_name)}.parquet"

    def exists(self, run_id: str, feature_name: str) -> bool:
        return self._path(run_id, feature_name).exists()

    def save(self, run_id: str, feature_name: str, values: pd.Series) -> Path:
        path = self._path(run_id, feature_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        values.rename(feature_name).to_frame().to_parquet(path)
        return path

    def load(self, run_id: str, feature_name: str) -> pd.Series:
        path = self._path(run_id, feature_name)
        return pd.read_parquet(path)[feature_name]

    def delete(self, run_id: str, feature_name: str) -> None:
        path = self._path(run_id, feature_name)
        if path.exists():
            path.unlink()

    def list_features(self, run_id: str) -> list[str]:
        run_dir = self.base_dir / _safe_name(run_id)
        if not run_dir.exists():
            return []
        return sorted(p.stem for p in run_dir.glob("*.parquet"))

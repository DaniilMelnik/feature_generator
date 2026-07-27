"""The feature-module contract: the core anti-leakage mechanism.

Every feature (whether hand-written for tests/fixtures or LLM-generated) is a
small Python module exposing:

    FEATURE_NAME: str
    REQUIRED_COLUMNS: list[str]
    def fit(train_df: pd.DataFrame, context: FitContext) -> dict          # json-serializable
    def transform(df: pd.DataFrame, params: dict) -> pd.Series            # ROW-WISE ONLY

``fit`` is only ever called on the current training fold's rows; ``transform``
is called on every split (train/val/test/OOT) using the params produced by
``fit``. ``transform`` must be a pure, row-wise function of its own input rows
and ``params`` -- any cross-row aggregation (groupby mean, rolling window,
rank, cumulative sum) MUST be precomputed once inside ``fit`` and stored in
``params``. This is both what prevents fold-to-fold statistical leakage and
what the serving-parity simulator (``serving_parity.replay_simulator``)
actually verifies.

This module defines the *interface* (``FeatureComputer``) that
``dataset.builder.DatasetBuilder`` programs against, plus an in-process
implementation for trusted, hand-written modules (test fixtures, and any
human-authored feature). LLM-generated code never goes through
``InProcessFeatureComputer`` -- it is always executed via
``sandbox.docker_runner.SandboxedFeatureComputer``, which implements the same
interface. This split is what keeps "dataset assembly" and "untrusted code
execution" architecturally separate, as required by the design.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    from feature_generator.dataset.builder import FoldSplit


class FitContext(BaseModel):
    """Everything a feature module's ``fit`` is allowed to know about the run."""

    target_column: str
    time_column: str | None = None
    id_columns: list[str] = []
    fold_id: int | None = None


class ContractViolationError(Exception):
    """Raised when a feature module does not satisfy the fit/transform contract."""

    def __init__(self, feature_name: str, violations: list[str]) -> None:
        self.feature_name = feature_name
        self.violations = violations
        super().__init__(f"{feature_name}: {'; '.join(violations)}")


REQUIRED_ATTRS = ("FEATURE_NAME", "REQUIRED_COLUMNS", "fit", "transform")


@runtime_checkable
class FeatureModuleLike(Protocol):
    """Structural shape a loaded module (or module-like namespace) must have."""

    FEATURE_NAME: str
    REQUIRED_COLUMNS: list[str]

    def fit(self, train_df: pd.DataFrame, context: FitContext) -> dict: ...

    def transform(self, df: pd.DataFrame, params: dict) -> pd.Series: ...


def validate_module_structure(module: types.ModuleType | object) -> list[str]:
    """Structural (post-load) contract check. Only ever run on code that has
    already been executed inside an isolated environment (the Docker
    container) or that is trusted (hand-written fixtures) -- this function
    itself does no code execution beyond attribute access.
    """
    violations: list[str] = []
    for attr in REQUIRED_ATTRS:
        if not hasattr(module, attr):
            violations.append(f"missing required attribute '{attr}'")
    if hasattr(module, "FEATURE_NAME") and not isinstance(module.FEATURE_NAME, str):
        violations.append("FEATURE_NAME must be a str")
    if hasattr(module, "REQUIRED_COLUMNS"):
        cols = module.REQUIRED_COLUMNS
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            violations.append("REQUIRED_COLUMNS must be a list[str]")
    for fn_name in ("fit", "transform"):
        if hasattr(module, fn_name) and not callable(getattr(module, fn_name)):
            violations.append(f"'{fn_name}' must be callable")
    return violations


class FeatureComputer(ABC):
    """Interface ``dataset.builder.DatasetBuilder`` programs against.

    Implementations decide *how* fit/transform actually execute (in-process
    for trusted code, sandboxed-in-Docker for LLM-generated code) -- the
    builder does not know or care which.
    """

    feature_name: str
    required_columns: list[str]

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, context: FitContext) -> dict:
        """Fit on the given (already fold-restricted) training rows only."""

    @abstractmethod
    def transform(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Row-wise transform of ``df`` using previously-fit ``params``."""

    def compute_oof(
        self,
        base_df: pd.DataFrame,
        folds: list["FoldSplit"],
        *,
        target_column: str,
        time_column: str | None = None,
        id_columns: list[str] | None = None,
        dev_index: np.ndarray | None = None,
    ) -> tuple[pd.Series, dict]:
        """Compute this feature's out-of-fold values plus its full-data fit
        params. Default: the fold-by-fold loop, calling ``self.fit()``/
        ``self.transform()`` once per fold through ``self`` -- so a subclass
        (or a test double wrapping one, e.g. a fold-isolation spy) that only
        overrides ``fit``/``transform`` keeps observing every call exactly as
        before. ``SandboxedFeatureComputer`` overrides this method entirely
        to batch all per-fold calls into a single container invocation --
        see its docstring for why that's safe to do without weakening
        isolation.

        ``dev_index``, when given, restricts the full-data fit to those rows
        instead of every row in ``base_df`` -- otherwise a feature's
        serving-time params would already have seen held-out rows, silently
        invalidating a holdout evaluation done against them.
        """
        id_columns = id_columns or []
        oof = pd.Series(index=base_df.index, dtype=object)

        for fold in folds:
            train_rows = base_df.iloc[fold.train_index]
            val_rows = base_df.iloc[fold.val_index]
            ctx = FitContext(
                target_column=target_column, time_column=time_column,
                id_columns=id_columns, fold_id=fold.fold_id,
            )
            params = self.fit(train_rows, ctx)
            values = self.transform(val_rows, params)
            oof.iloc[fold.val_index] = pd.Series(values).to_numpy()

        try:
            oof = oof.astype("float64")
        except (TypeError, ValueError):
            pass  # genuinely categorical/string output -- leave as object

        full_ctx = FitContext(
            target_column=target_column, time_column=time_column,
            id_columns=id_columns, fold_id=None,
        )
        full_fit_df = base_df.iloc[dev_index] if dev_index is not None else base_df
        full_fit_params = self.fit(full_fit_df, full_ctx)

        return oof, full_fit_params

    def transform_many(self, df: pd.DataFrame, params: dict, jobs: list[list[int]]) -> list[pd.Series]:
        """Run ``transform()`` once per job, where each job is a list of
        positions into ``df`` (its own isolated row subset). Default: a
        plain Python loop calling ``self.transform()`` per job.
        ``SandboxedFeatureComputer`` overrides this to run every job inside
        ONE container invocation instead of one container per job -- see its
        docstring. Used by ``serving_parity.replay_simulator`` for
        many-small-batch replay checks.
        """
        return [self.transform(df.iloc[positions], params) for positions in jobs]


class InProcessFeatureComputer(FeatureComputer):
    """Executes a trusted, already-imported module's fit/transform directly.

    Used for hand-written fixtures and unit tests only. Never point this at
    LLM-generated source -- that must go through the Docker sandbox.

    ``allow_target_in_fit`` mirrors ``FeatureHypothesis.requires_target_in_fit``:
    when True, ``fit()`` additionally receives the target column appended to
    its (still REQUIRED_COLUMNS-sliced) dataframe -- e.g. for target/likelihood
    encoding. ``transform()`` NEVER receives the target column, regardless of
    this flag, since it also runs at inference time when no label exists;
    this is what makes serving-parity checking meaningful for such features.
    """

    def __init__(self, module: types.ModuleType, allow_target_in_fit: bool = False) -> None:
        violations = validate_module_structure(module)
        if violations:
            name = getattr(module, "FEATURE_NAME", getattr(module, "__name__", "<unknown>"))
            raise ContractViolationError(name, violations)
        self._module = module
        self.feature_name = module.FEATURE_NAME
        self.required_columns = list(module.REQUIRED_COLUMNS)
        self.allow_target_in_fit = allow_target_in_fit

    def fit(self, train_df: pd.DataFrame, context: FitContext) -> dict:
        columns = list(self.required_columns)
        if self.allow_target_in_fit and context.target_column not in columns:
            columns = [*columns, context.target_column]
        return self._module.fit(train_df[columns], context)

    def transform(self, df: pd.DataFrame, params: dict) -> pd.Series:
        return self._module.transform(df[self.required_columns], params)


def load_trusted_module_from_file(path: str | Path) -> types.ModuleType:
    """Import a hand-written, trusted feature module from disk.

    NOT for LLM-generated source -- this executes top-level module code with
    full host privileges. Only ever call this on files under our own control
    (``tests/fixtures/*.py``, or a future first-party feature library).
    """
    path = Path(path)
    module_name = f"feature_generator._trusted_features.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

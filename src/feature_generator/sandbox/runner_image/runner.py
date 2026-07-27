#!/usr/bin/env python3
"""In-container harness for sandboxed feature-module execution.

Fully self-contained -- no dependency on the `feature_generator` package --
so the sandbox image only needs pandas/numpy/scipy/scikit-learn, matching the
import allowlist that `sandbox.static_checks` already vetted the module
against before it ever reached here.

Protocol (mounted host volume at IO_DIR, default /sandbox/io). Four call
types, all sharing `module.py` + `manifest.json` + `input.parquet`:

  "fit"            -- manifest: {"call": "fit", "context": {...}}
                      out: params_out.json (the dict fit() returned)
  "transform"      -- manifest: {"call": "transform", "context": {...}}
                      in:  params.json
                      out: output.parquet (single "value" column)
  "oof_batch"       -- manifest: {"call": "oof_batch",
                          "folds": [{"fold_id", "train_positions",
                                     "val_positions", "context"}, ...]}
                      Runs fit(train)+transform(val) for every fold in ONE
                      container invocation instead of one container per
                      fold -- see sandbox.contract.FeatureComputer.compute_oof.
                      out: oof_output.parquet ("position", "value" -- one row
                      per val position across all folds; disjoint by
                      construction of k-fold, so no position repeats).
  "transform_batch" -- manifest: {"call": "transform_batch", "jobs":
                          [[pos, pos, ...], ...]}
                      in:  params.json (one shared params dict for every job)
                      Runs transform() once per job (each job an independent
                      row subset) in ONE container invocation -- see
                      sandbox.contract.FeatureComputer.transform_many.
                      out: transform_batch_output.parquet ("job_id",
                      "position_in_job", "value").

  All calls: out: status.json -- always {success, error, traceback,
  runtime_seconds}.

Exit codes: 0 ok, 1 feature code raised, 2 contract violation,
3 output validation failure. (124 is reserved for the host-side timeout kill
via `docker kill` -- this script never sets it itself.)

For the batch call types, one fold/job raising or returning malformed output
fails the whole invocation (matching the all-or-nothing retry semantics the
host already applies to a single fit/transform failure) -- the error message
names which fold_id/job_id failed.

SANDBOX_IO_DIR may override the IO directory (used by host-side tests that
exercise this protocol directly, without a container, against a temp dir).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

IO_DIR = Path(os.environ.get("SANDBOX_IO_DIR", "/sandbox/io"))
REQUIRED_ATTRS = ("FEATURE_NAME", "REQUIRED_COLUMNS", "fit", "transform")
CALL_TYPES = ("fit", "transform", "oof_batch", "transform_batch")

EXIT_OK = 0
EXIT_FEATURE_RAISED = 1
EXIT_CONTRACT_VIOLATION = 2
EXIT_OUTPUT_INVALID = 3


def _write_status(
    success: bool,
    *,
    error: str | None = None,
    tb: str | None = None,
    runtime_seconds: float = 0.0,
) -> None:
    (IO_DIR / "status.json").write_text(
        json.dumps(
            {"success": success, "error": error, "traceback": tb, "runtime_seconds": runtime_seconds}
        )
    )


class _Context:
    """Plain attribute-access shim -- avoids importing the main package's
    pydantic FitContext inside the minimal sandbox image.
    """

    def __init__(self, data: dict) -> None:
        self.target_column = data.get("target_column")
        self.time_column = data.get("time_column")
        self.id_columns = data.get("id_columns", [])
        self.fold_id = data.get("fold_id")


def _load_module():
    spec = importlib.util.spec_from_file_location("sandboxed_feature_module", IO_DIR / "module.py")
    if spec is None or spec.loader is None:
        raise ImportError("could not build module spec for module.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_value_list(result, expected_len: int, pd) -> list:
    """Validate a transform() result's length and return it as a plain list,
    aligned positionally to the input rows it was computed from.
    """
    series = pd.Series(result)
    if len(series) != expected_len:
        raise ValueError(f"transform() returned {len(series)} values for {expected_len} input rows")
    return series.to_numpy().tolist()


def _run_fit(module, input_df, manifest: dict) -> int:
    context = _Context(manifest.get("context", {}))
    start = time.monotonic()
    try:
        params = module.fit(input_df, context)
    except Exception as exc:
        _write_status(False, error=str(exc), tb=traceback.format_exc(), runtime_seconds=time.monotonic() - start)
        return EXIT_FEATURE_RAISED

    try:
        serialized = json.dumps(params)
    except (TypeError, ValueError) as exc:
        _write_status(False, error=f"fit() returned non-JSON-serializable params: {exc}")
        return EXIT_OUTPUT_INVALID

    (IO_DIR / "params_out.json").write_text(serialized)
    _write_status(True, runtime_seconds=time.monotonic() - start)
    return EXIT_OK


def _run_transform(module, input_df, manifest: dict, pd) -> int:
    start = time.monotonic()
    try:
        params = json.loads((IO_DIR / "params.json").read_text())
    except Exception as exc:
        _write_status(False, error=f"could not read params.json: {exc}", tb=traceback.format_exc())
        return EXIT_CONTRACT_VIOLATION

    try:
        result = module.transform(input_df, params)
    except Exception as exc:
        _write_status(False, error=str(exc), tb=traceback.format_exc(), runtime_seconds=time.monotonic() - start)
        return EXIT_FEATURE_RAISED

    try:
        result_series = pd.Series(result)
        if len(result_series) != len(input_df):
            raise ValueError(
                f"transform() returned {len(result_series)} values for {len(input_df)} input rows"
            )
        result_series.index = input_df.index
    except Exception as exc:
        _write_status(False, error=f"invalid transform() output: {exc}", tb=traceback.format_exc())
        return EXIT_OUTPUT_INVALID

    result_series.rename("value").to_frame().to_parquet(IO_DIR / "output.parquet")
    _write_status(True, runtime_seconds=time.monotonic() - start)
    return EXIT_OK


def _run_oof_batch(module, input_df, manifest: dict, pd) -> int:
    start = time.monotonic()
    folds = manifest.get("folds", [])

    positions: list[int] = []
    values: list = []
    for fold_spec in folds:
        fold_id = fold_spec.get("fold_id")
        context = _Context(fold_spec.get("context", {}))
        train_positions = fold_spec["train_positions"]
        val_positions = fold_spec["val_positions"]

        try:
            train_df = input_df.iloc[train_positions]
            params = module.fit(train_df, context)
            val_df = input_df.iloc[val_positions]
            # transform() must NEVER see the target column, even though the
            # shared input frame may include it for fit()'s benefit
            # (allow_target_in_fit) -- it also runs at inference time when no
            # label exists, and this is what makes serving-parity checking
            # meaningful for such features.
            if context.target_column and context.target_column in val_df.columns:
                val_df = val_df.drop(columns=[context.target_column])
            result = module.transform(val_df, params)
            fold_values = _as_value_list(result, len(val_positions), pd)
        except Exception as exc:
            _write_status(
                False,
                error=f"fold {fold_id}: {exc}",
                tb=traceback.format_exc(),
                runtime_seconds=time.monotonic() - start,
            )
            return EXIT_FEATURE_RAISED

        positions.extend(val_positions)
        values.extend(fold_values)

    pd.DataFrame({"position": positions, "value": values}).to_parquet(IO_DIR / "oof_output.parquet")
    _write_status(True, runtime_seconds=time.monotonic() - start)
    return EXIT_OK


def _run_transform_batch(module, input_df, manifest: dict, pd) -> int:
    start = time.monotonic()
    try:
        params = json.loads((IO_DIR / "params.json").read_text())
    except Exception as exc:
        _write_status(False, error=f"could not read params.json: {exc}", tb=traceback.format_exc())
        return EXIT_CONTRACT_VIOLATION

    jobs = manifest.get("jobs", [])
    job_ids: list[int] = []
    positions_in_job: list[int] = []
    values: list = []

    for job_id, job_positions in enumerate(jobs):
        try:
            chunk = input_df.iloc[job_positions]
            result = module.transform(chunk, params)
            job_values = _as_value_list(result, len(job_positions), pd)
        except Exception as exc:
            _write_status(
                False,
                error=f"job {job_id}: {exc}",
                tb=traceback.format_exc(),
                runtime_seconds=time.monotonic() - start,
            )
            return EXIT_FEATURE_RAISED

        job_ids.extend([job_id] * len(job_positions))
        positions_in_job.extend(range(len(job_positions)))
        values.extend(job_values)

    pd.DataFrame(
        {"job_id": job_ids, "position_in_job": positions_in_job, "value": values}
    ).to_parquet(IO_DIR / "transform_batch_output.parquet")
    _write_status(True, runtime_seconds=time.monotonic() - start)
    return EXIT_OK


def main() -> int:
    try:
        manifest = json.loads((IO_DIR / "manifest.json").read_text())
    except Exception as exc:
        _write_status(False, error=f"could not read manifest.json: {exc}", tb=traceback.format_exc())
        return EXIT_CONTRACT_VIOLATION

    call = manifest.get("call")
    if call not in CALL_TYPES:
        _write_status(False, error=f"manifest.json 'call' must be one of {CALL_TYPES}, got {call!r}")
        return EXIT_CONTRACT_VIOLATION

    try:
        module = _load_module()
    except Exception as exc:
        _write_status(False, error=f"failed to import module.py: {exc}", tb=traceback.format_exc())
        return EXIT_CONTRACT_VIOLATION

    for attr in REQUIRED_ATTRS:
        if not hasattr(module, attr):
            _write_status(False, error=f"module.py missing required attribute '{attr}'")
            return EXIT_CONTRACT_VIOLATION

    import pandas as pd  # deferred: only needed past the cheap contract checks above

    try:
        input_df = pd.read_parquet(IO_DIR / "input.parquet")
    except Exception as exc:
        _write_status(False, error=f"could not read input.parquet: {exc}", tb=traceback.format_exc())
        return EXIT_CONTRACT_VIOLATION

    if call == "fit":
        return _run_fit(module, input_df, manifest)
    if call == "transform":
        return _run_transform(module, input_df, manifest, pd)
    if call == "oof_batch":
        return _run_oof_batch(module, input_df, manifest, pd)
    return _run_transform_batch(module, input_df, manifest, pd)


if __name__ == "__main__":
    sys.exit(main())

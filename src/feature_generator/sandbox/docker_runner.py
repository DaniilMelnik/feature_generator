"""Host-side Docker sandbox orchestration for executing LLM-generated feature
modules -- tier 2 of the two-tier defense (tier 1 is the AST-based
``sandbox.static_checks`` pass, which MUST already have succeeded before any
module reaches here).

``SandboxedFeatureComputer`` implements the exact same ``FeatureComputer``
interface as ``sandbox.contract.InProcessFeatureComputer``, so
``dataset.builder.DatasetBuilder`` works identically regardless of which one
it is given -- this is what keeps "dataset assembly" and "untrusted code
execution" architecturally separate.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd

from feature_generator.config import SandboxConfig
from feature_generator.sandbox.contract import FeatureComputer, FitContext

if TYPE_CHECKING:
    from feature_generator.dataset.builder import FoldSplit

DOCKER_VERSION_TIMEOUT_SECONDS = 10
# `docker kill`/`docker rm` are meant to be fast local daemon calls, but if
# the daemon itself is degraded they can hang just as easily as `docker run`/
# `docker wait` -- a real failure mode observed in practice (a container
# already gone, yet the CLI call against it never returned). Every docker
# subprocess call in this module gets an explicit timeout for that reason;
# none are allowed to block the host process indefinitely.
DOCKER_CLEANUP_TIMEOUT_SECONDS = 20


class SandboxExecutionError(Exception):
    def __init__(self, message: str, exit_code: int | None = None, traceback_text: str | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.traceback_text = traceback_text


def docker_available() -> bool:
    """Runtime probe -- gate all Docker-dependent code paths on this rather
    than assuming Docker Desktop is installed.
    """
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=DOCKER_VERSION_TIMEOUT_SECONDS,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _best_effort_docker(cmd: list[str]) -> None:
    """Run a cleanup-only docker command (kill/rm) with a hard timeout and
    swallow any failure -- these exist to free resources, not to determine
    success/failure of a sandbox run, and must never themselves be able to
    hang the host process if the daemon is degraded.
    """
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=DOCKER_CLEANUP_TIMEOUT_SECONDS)
    except (subprocess.TimeoutExpired, OSError):
        pass


@dataclass
class SandboxRunResult:
    success: bool
    error: str | None
    traceback_text: str | None
    runtime_seconds: float
    exit_code: int


class DockerSandboxRunner:
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    def _docker_run_cmd(self, scratch_dir: Path, container_name: str) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--memory",
            self.config.memory_limit,
            "--memory-swap",
            self.config.memory_limit,
            "--cpus",
            str(self.config.cpus),
            "--pids-limit",
            str(self.config.pids_limit),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=256m",
            "-v",
            f"{scratch_dir}:/sandbox/io:rw",
            "--user",
            "1000:1000",
            "-d",
            self.config.image_tag,
        ]

    def _run_container(self, scratch_dir: Path, *, timeout_seconds: int | None = None) -> SandboxRunResult:
        timeout_seconds = self.config.timeout_seconds if timeout_seconds is None else timeout_seconds
        container_name = f"feature-gen-sandbox-{uuid4().hex[:12]}"
        start = time.monotonic()

        try:
            launch = subprocess.run(
                self._docker_run_cmd(scratch_dir, container_name),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # the daemon itself is unresponsive -- best-effort cleanup by
            # name (no container_id was ever returned), then surface a clear
            # error instead of hanging the whole pipeline.
            _best_effort_docker(["docker", "rm", "-f", container_name])
            return SandboxRunResult(
                success=False,
                error=f"docker run did not respond within {timeout_seconds}s (daemon unresponsive?)",
                traceback_text=None,
                runtime_seconds=time.monotonic() - start,
                exit_code=124,
            )
        if launch.returncode != 0:
            return SandboxRunResult(
                success=False,
                error=f"failed to launch sandbox container: {launch.stderr.strip()}",
                traceback_text=None,
                runtime_seconds=time.monotonic() - start,
                exit_code=-1,
            )
        container_id = launch.stdout.strip()

        try:
            # `docker wait` blocks until the container exits and prints its exit
            # code -- a Python-side subprocess timeout here does NOT stop the
            # container itself, so an explicit `docker kill` follows on timeout.
            wait = subprocess.run(
                ["docker", "wait", container_id],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            exit_code = int(wait.stdout.strip()) if wait.returncode == 0 else -1
        except subprocess.TimeoutExpired:
            _best_effort_docker(["docker", "kill", container_id])
            return SandboxRunResult(
                success=False,
                error=f"sandbox execution exceeded {timeout_seconds}s timeout",
                traceback_text=None,
                runtime_seconds=float(timeout_seconds),
                exit_code=124,
            )
        finally:
            # bounded, best-effort -- a degraded daemon must never hang this
            # cleanup step indefinitely (see DOCKER_CLEANUP_TIMEOUT_SECONDS).
            _best_effort_docker(["docker", "rm", "-f", container_id])

        runtime = time.monotonic() - start
        status_path = scratch_dir / "status.json"
        if not status_path.exists():
            return SandboxRunResult(
                success=False,
                error=f"container exited (code {exit_code}) without writing status.json",
                traceback_text=None,
                runtime_seconds=runtime,
                exit_code=exit_code,
            )
        status = json.loads(status_path.read_text())
        return SandboxRunResult(
            success=bool(status.get("success", False)),
            error=status.get("error"),
            traceback_text=status.get("traceback"),
            runtime_seconds=float(status.get("runtime_seconds", runtime)),
            exit_code=exit_code,
        )

    def fit(self, module_source: str, train_df: pd.DataFrame, context: FitContext) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            (scratch / "module.py").write_text(module_source)
            (scratch / "manifest.json").write_text(
                json.dumps({"call": "fit", "context": context.model_dump()})
            )
            train_df.to_parquet(scratch / "input.parquet")

            result = self._run_container(scratch)
            if not result.success:
                raise SandboxExecutionError(
                    result.error or "fit() failed in sandbox", result.exit_code, result.traceback_text
                )
            return json.loads((scratch / "params_out.json").read_text())

    def transform(self, module_source: str, df: pd.DataFrame, params: dict) -> pd.Series:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            (scratch / "module.py").write_text(module_source)
            (scratch / "manifest.json").write_text(json.dumps({"call": "transform", "context": {}}))
            df.to_parquet(scratch / "input.parquet")
            (scratch / "params.json").write_text(json.dumps(params))

            result = self._run_container(scratch)
            if not result.success:
                raise SandboxExecutionError(
                    result.error or "transform() failed in sandbox", result.exit_code, result.traceback_text
                )
            output = pd.read_parquet(scratch / "output.parquet")
            return output["value"]

    def compute_oof_batch(
        self,
        module_source: str,
        input_df: pd.DataFrame,
        folds: list["FoldSplit"],
        *,
        target_column: str,
        time_column: str | None,
        id_columns: list[str],
    ) -> pd.Series:
        """Run every fold's fit()+transform() in ONE container invocation
        instead of one container per fold -- see
        ``sandbox.contract.FeatureComputer.compute_oof``. ``input_df`` must
        already be sliced to exactly the columns fit()/transform() should
        see (including the target column when the caller needs it in fit();
        the runner strips it before every transform() call regardless, since
        transform() must never see the target column).
        """
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            (scratch / "module.py").write_text(module_source)
            fold_specs = [
                {
                    "fold_id": fold.fold_id,
                    "train_positions": fold.train_index.tolist(),
                    "val_positions": fold.val_index.tolist(),
                    "context": {
                        "target_column": target_column,
                        "time_column": time_column,
                        "id_columns": id_columns,
                        "fold_id": fold.fold_id,
                    },
                }
                for fold in folds
            ]
            (scratch / "manifest.json").write_text(json.dumps({"call": "oof_batch", "folds": fold_specs}))
            input_df.to_parquet(scratch / "input.parquet")

            result = self._run_container(scratch, timeout_seconds=self.config.batch_timeout_seconds)
            if not result.success:
                raise SandboxExecutionError(
                    result.error or "oof_batch failed in sandbox", result.exit_code, result.traceback_text
                )
            output = pd.read_parquet(scratch / "oof_output.parquet")
            oof = pd.Series(index=input_df.index, dtype=object)
            oof.iloc[output["position"].to_numpy()] = output["value"].to_numpy()
            try:
                oof = oof.astype("float64")
            except (TypeError, ValueError):
                pass  # genuinely categorical/string output -- leave as object
            return oof

    def transform_batch(
        self, module_source: str, df: pd.DataFrame, params: dict, jobs: list[list[int]]
    ) -> list[pd.Series]:
        """Run every job's transform() in ONE container invocation instead of
        one container per job -- see
        ``sandbox.contract.FeatureComputer.transform_many``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            (scratch / "module.py").write_text(module_source)
            (scratch / "manifest.json").write_text(json.dumps({"call": "transform_batch", "jobs": jobs}))
            df.to_parquet(scratch / "input.parquet")
            (scratch / "params.json").write_text(json.dumps(params))

            result = self._run_container(scratch, timeout_seconds=self.config.batch_timeout_seconds)
            if not result.success:
                raise SandboxExecutionError(
                    result.error or "transform_batch failed in sandbox", result.exit_code, result.traceback_text
                )
            output = pd.read_parquet(scratch / "transform_batch_output.parquet")
            results: list[pd.Series] = []
            for job_id, job_positions in enumerate(jobs):
                job_rows = output[output["job_id"] == job_id].sort_values("position_in_job")
                results.append(pd.Series(job_rows["value"].to_numpy(), index=df.index[job_positions]))
            return results


class SandboxedFeatureComputer(FeatureComputer):
    """The LLM-generated-code counterpart to ``InProcessFeatureComputer`` --
    every fit()/transform() call is executed inside an isolated container.
    """

    def __init__(
        self,
        feature_name: str,
        module_source: str,
        required_columns: list[str],
        runner: DockerSandboxRunner,
        allow_target_in_fit: bool = False,
    ) -> None:
        self.feature_name = feature_name
        self.required_columns = required_columns
        self.allow_target_in_fit = allow_target_in_fit
        self._module_source = module_source
        self._runner = runner

    def fit(self, train_df: pd.DataFrame, context: FitContext) -> dict:
        columns = list(self.required_columns)
        if self.allow_target_in_fit and context.target_column not in columns:
            columns = [*columns, context.target_column]
        return self._runner.fit(self._module_source, train_df[columns], context)

    def transform(self, df: pd.DataFrame, params: dict) -> pd.Series:
        return self._runner.transform(self._module_source, df[self.required_columns], params)

    def compute_oof(
        self,
        base_df: pd.DataFrame,
        folds: list["FoldSplit"],
        *,
        target_column: str,
        time_column: str | None = None,
        id_columns: list[str] | None = None,
        dev_index=None,
    ) -> tuple[pd.Series, dict]:
        """Batches every fold's fit()+transform() into one container
        invocation (see ``DockerSandboxRunner.compute_oof_batch``); the
        final full-data fit stays a plain, separate ``self.fit()`` call --
        one extra container launch is a negligible cost next to the ~10x
        reduction from batching the folds, and reusing ``self.fit()`` here
        keeps the `allow_target_in_fit` column logic in exactly one place.
        """
        id_columns = id_columns or []
        columns = list(self.required_columns)
        if self.allow_target_in_fit and target_column not in columns:
            columns = [*columns, target_column]

        oof = self._runner.compute_oof_batch(
            self._module_source,
            base_df[columns],
            folds,
            target_column=target_column,
            time_column=time_column,
            id_columns=id_columns,
        )

        full_ctx = FitContext(
            target_column=target_column, time_column=time_column, id_columns=id_columns, fold_id=None,
        )
        full_fit_df = base_df.iloc[dev_index] if dev_index is not None else base_df
        full_fit_params = self.fit(full_fit_df, full_ctx)

        return oof, full_fit_params

    def transform_many(self, df: pd.DataFrame, params: dict, jobs: list[list[int]]) -> list[pd.Series]:
        """Batches every job's transform() into one container invocation
        instead of one per job -- see ``DockerSandboxRunner.transform_batch``.
        """
        return self._runner.transform_batch(self._module_source, df[self.required_columns], params, jobs)

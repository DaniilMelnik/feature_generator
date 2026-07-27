"""Docker-mocked unit tests: verify command construction, timeout handling,
and status parsing WITHOUT needing a running Docker daemon. The genuine
live-container integration test lives in test_docker_sandbox_integration.py,
marked `@pytest.mark.docker` and skipped until Docker Desktop is installed.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from feature_generator.config import SandboxConfig
from feature_generator.dataset.builder import FoldSplit
from feature_generator.sandbox.contract import FitContext
from feature_generator.sandbox.docker_runner import (
    DockerSandboxRunner,
    SandboxedFeatureComputer,
    SandboxExecutionError,
    _best_effort_docker,
    docker_available,
)


def test_docker_available_returns_false_on_missing_binary(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise FileNotFoundError("no such file: docker")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert docker_available() is False


def test_docker_available_returns_true_when_server_responds(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: MagicMock(returncode=0, stdout="27.0.0")
    )
    assert docker_available() is True


def _config(**overrides) -> SandboxConfig:
    defaults = dict(
        backend="docker",
        image_tag="feature-gen-sandbox:test",
        memory_limit="512m",
        cpus=1.0,
        pids_limit=32,
        timeout_seconds=5,
    )
    defaults.update(overrides)
    return SandboxConfig(**defaults)


def test_docker_run_cmd_includes_all_required_isolation_flags(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(_config())
    cmd = runner._docker_run_cmd(tmp_path, "container-123")

    assert cmd[:2] == ["docker", "run"]
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--memory" in cmd and cmd[cmd.index("--memory") + 1] == "512m"
    assert "--cpus" in cmd and cmd[cmd.index("--cpus") + 1] == "1.0"
    assert "--pids-limit" in cmd and cmd[cmd.index("--pids-limit") + 1] == "32"
    assert "--read-only" in cmd
    assert "--user" in cmd and cmd[cmd.index("--user") + 1] == "1000:1000"
    assert any(str(tmp_path) in arg and "/sandbox/io:rw" in arg for arg in cmd)
    assert cmd[-1] == "feature-gen-sandbox:test"


def test_run_container_reports_launch_failure_without_calling_wait(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(_config())

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="no such image", stdout="")
        result = runner._run_container(tmp_path)

    assert result.success is False
    assert "no such image" in result.error
    assert result.exit_code == -1
    mock_run.assert_called_once()  # only the launch attempt -- never reached `docker wait`


def test_run_container_kills_and_reports_timeout(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(_config(timeout_seconds=2))

    launch_result = MagicMock(returncode=0, stdout="container-abc\n", stderr="")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "run":
            return launch_result
        if cmd[1] == "wait":
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=2)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        result = runner._run_container(tmp_path)

    assert result.success is False
    assert result.exit_code == 124
    assert "timeout" in result.error
    kill_calls = [c for c in calls if c[1] == "kill"]
    assert len(kill_calls) == 1
    assert kill_calls[0][2] == "container-abc"


def test_run_container_reports_launch_timeout_without_hanging(tmp_path: Path) -> None:
    # a degraded daemon that never responds to `docker run -d` itself --
    # previously unbounded (the real bug this hardening fixes), must now
    # surface a clear error rather than block the caller indefinitely.
    runner = DockerSandboxRunner(_config(timeout_seconds=2))
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "run":
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=2)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        result = runner._run_container(tmp_path)

    assert result.success is False
    assert result.exit_code == 124
    assert "unresponsive" in result.error
    wait_calls = [c for c in calls if c[1] == "wait"]
    assert wait_calls == []  # never reached -- launch itself never returned a container id
    rm_calls = [c for c in calls if c[1] == "rm"]
    assert len(rm_calls) == 1  # best-effort cleanup by name still attempted


def test_best_effort_docker_swallows_timeout_and_os_error(monkeypatch) -> None:
    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["docker"], timeout=1)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    _best_effort_docker(["docker", "rm", "-f", "whatever"])  # must not raise

    def _raise_os_error(*a, **k):
        raise OSError("daemon socket gone")

    monkeypatch.setattr(subprocess, "run", _raise_os_error)
    _best_effort_docker(["docker", "kill", "whatever"])  # must not raise


def test_run_container_parses_status_json_on_success(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text(
        json.dumps({"success": True, "error": None, "traceback": None, "runtime_seconds": 1.23})
    )
    runner = DockerSandboxRunner(_config())

    def _fake_run(cmd, **kwargs):
        if cmd[1] == "run":
            return MagicMock(returncode=0, stdout="container-abc\n", stderr="")
        if cmd[1] == "wait":
            return MagicMock(returncode=0, stdout="0\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        result = runner._run_container(tmp_path)

    assert result.success is True
    assert result.runtime_seconds == pytest.approx(1.23)
    assert result.exit_code == 0


def test_run_container_reports_missing_status_json(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(_config())

    def _fake_run(cmd, **kwargs):
        if cmd[1] == "run":
            return MagicMock(returncode=0, stdout="container-abc\n", stderr="")
        if cmd[1] == "wait":
            return MagicMock(returncode=0, stdout="137\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        result = runner._run_container(tmp_path)

    assert result.success is False
    assert "without writing status.json" in result.error
    assert result.exit_code == 137


def _scratch_dir_from_cmd(cmd: list[str]) -> Path:
    volume_arg = cmd[cmd.index("-v") + 1]  # "{scratch}:/sandbox/io:rw"
    return Path(volume_arg.split(":")[0])


def test_compute_oof_batch_writes_fold_manifest_and_uses_batch_timeout(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(_config(timeout_seconds=5, batch_timeout_seconds=111))
    df = pd.DataFrame({"Age": [10, 20, 30, 40]})
    folds = [
        FoldSplit(fold_id=0, train_index=np.array([2, 3]), val_index=np.array([0, 1])),
        FoldSplit(fold_id=1, train_index=np.array([0, 1]), val_index=np.array([2, 3])),
    ]
    seen_timeouts: list[int] = []
    scratch_box: list[Path] = []

    def _fake_run(cmd, **kwargs):
        if cmd[1] == "run":
            scratch = _scratch_dir_from_cmd(cmd)
            scratch_box.append(scratch)
            manifest = json.loads((scratch / "manifest.json").read_text())
            assert manifest["call"] == "oof_batch"
            assert len(manifest["folds"]) == 2
            assert manifest["folds"][0]["train_positions"] == [2, 3]
            assert manifest["folds"][0]["val_positions"] == [0, 1]
            return MagicMock(returncode=0, stdout="container-abc\n", stderr="")
        if cmd[1] == "wait":
            seen_timeouts.append(kwargs.get("timeout"))
            scratch = scratch_box[0]
            pd.DataFrame({"position": [0, 1, 2, 3], "value": [1.0, 2.0, 3.0, 4.0]}).to_parquet(
                scratch / "oof_output.parquet"
            )
            (scratch / "status.json").write_text(json.dumps({"success": True, "runtime_seconds": 0.1}))
            return MagicMock(returncode=0, stdout="0\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        oof = runner.compute_oof_batch(
            "SOURCE", df, folds, target_column="y", time_column=None, id_columns=[]
        )

    assert seen_timeouts == [111]  # batch_timeout_seconds, not timeout_seconds
    assert oof.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_compute_oof_batch_raises_on_sandbox_failure(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(_config())
    df = pd.DataFrame({"Age": [1, 2]})
    folds = [FoldSplit(fold_id=0, train_index=np.array([1]), val_index=np.array([0]))]
    scratch_box: list[Path] = []

    def _fake_run(cmd, **kwargs):
        if cmd[1] == "run":
            scratch_box.append(_scratch_dir_from_cmd(cmd))
            return MagicMock(returncode=0, stdout="container-abc\n", stderr="")
        if cmd[1] == "wait":
            scratch = scratch_box[0]
            (scratch / "status.json").write_text(
                json.dumps({"success": False, "error": "fold 0: boom", "runtime_seconds": 0.1})
            )
            return MagicMock(returncode=0, stdout="1\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SandboxExecutionError, match="fold 0"):
        runner.compute_oof_batch("SOURCE", df, folds, target_column="y", time_column=None, id_columns=[])


def test_transform_batch_writes_job_manifest_and_returns_aligned_series(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(_config(timeout_seconds=5, batch_timeout_seconds=222))
    df = pd.DataFrame({"Age": [10, 20, 30]})
    seen_timeouts: list[int] = []
    scratch_box: list[Path] = []

    def _fake_run(cmd, **kwargs):
        if cmd[1] == "run":
            scratch = _scratch_dir_from_cmd(cmd)
            scratch_box.append(scratch)
            manifest = json.loads((scratch / "manifest.json").read_text())
            assert manifest["call"] == "transform_batch"
            assert manifest["jobs"] == [[0, 1], [2]]
            return MagicMock(returncode=0, stdout="container-abc\n", stderr="")
        if cmd[1] == "wait":
            seen_timeouts.append(kwargs.get("timeout"))
            scratch = scratch_box[0]
            pd.DataFrame(
                {"job_id": [0, 0, 1], "position_in_job": [0, 1, 0], "value": [100.0, 200.0, 300.0]}
            ).to_parquet(scratch / "transform_batch_output.parquet")
            (scratch / "status.json").write_text(json.dumps({"success": True, "runtime_seconds": 0.1}))
            return MagicMock(returncode=0, stdout="0\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        results = runner.transform_batch("SOURCE", df, {}, jobs=[[0, 1], [2]])

    assert seen_timeouts == [222]
    assert results[0].tolist() == [100.0, 200.0]
    assert results[1].tolist() == [300.0]


def test_transform_batch_raises_on_sandbox_failure(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(_config())
    df = pd.DataFrame({"Age": [1, 2]})
    scratch_box: list[Path] = []

    def _fake_run(cmd, **kwargs):
        if cmd[1] == "run":
            scratch_box.append(_scratch_dir_from_cmd(cmd))
            return MagicMock(returncode=0, stdout="container-abc\n", stderr="")
        if cmd[1] == "wait":
            scratch = scratch_box[0]
            (scratch / "status.json").write_text(
                json.dumps({"success": False, "error": "job 0: boom", "runtime_seconds": 0.1})
            )
            return MagicMock(returncode=0, stdout="1\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SandboxExecutionError, match="job 0"):
        runner.transform_batch("SOURCE", df, {}, jobs=[[0], [1]])


class _FakeRunner:
    """Test double standing in for DockerSandboxRunner -- lets
    SandboxedFeatureComputer be tested without touching subprocess at all.
    """

    def __init__(self) -> None:
        self.fit_calls: list[tuple[str, list[str]]] = []
        self.fit_call_indexes: list[list] = []
        self.transform_calls: list[tuple[str, list[str]]] = []
        self.compute_oof_batch_calls: list[dict] = []
        self.transform_batch_calls: list[dict] = []

    def fit(self, module_source, train_df, context):
        self.fit_calls.append((module_source, list(train_df.columns)))
        self.fit_call_indexes.append(list(train_df.index))
        return {"mean": 1.0}

    def transform(self, module_source, df, params):
        self.transform_calls.append((module_source, list(df.columns)))
        return pd.Series([params["mean"]] * len(df), index=df.index)

    def compute_oof_batch(self, module_source, input_df, folds, *, target_column, time_column, id_columns):
        self.compute_oof_batch_calls.append(
            {"columns": list(input_df.columns), "n_folds": len(folds), "target_column": target_column}
        )
        return pd.Series(2.0, index=input_df.index)

    def transform_batch(self, module_source, df, params, jobs):
        self.transform_batch_calls.append({"columns": list(df.columns), "jobs": jobs})
        return [pd.Series([params["mean"]] * len(job), index=df.index[job]) for job in jobs]


def test_sandboxed_feature_computer_slices_to_required_columns() -> None:
    fake = _FakeRunner()
    computer = SandboxedFeatureComputer("f", "SOURCE", ["Age"], fake)
    df = pd.DataFrame({"Age": [1, 2], "Transported": [True, False]})

    params = computer.fit(df, FitContext(target_column="Transported"))
    computer.transform(df, params)

    assert fake.fit_calls[0][1] == ["Age"]
    assert fake.transform_calls[0][1] == ["Age"]


def test_sandboxed_feature_computer_allow_target_in_fit_appends_target_for_fit_only() -> None:
    fake = _FakeRunner()
    computer = SandboxedFeatureComputer("f", "SOURCE", ["Age"], fake, allow_target_in_fit=True)
    df = pd.DataFrame({"Age": [1, 2], "Transported": [True, False]})

    params = computer.fit(df, FitContext(target_column="Transported"))
    computer.transform(df, params)

    assert set(fake.fit_calls[0][1]) == {"Age", "Transported"}
    assert fake.transform_calls[0][1] == ["Age"]  # transform never sees the target


def test_sandboxed_feature_computer_compute_oof_delegates_to_batched_runner_call() -> None:
    fake = _FakeRunner()
    computer = SandboxedFeatureComputer("f", "SOURCE", ["Age"], fake)
    df = pd.DataFrame({"Age": [1, 2, 3, 4], "Transported": [True, False, True, False]})
    folds = [
        FoldSplit(fold_id=0, train_index=np.array([2, 3]), val_index=np.array([0, 1])),
        FoldSplit(fold_id=1, train_index=np.array([0, 1]), val_index=np.array([2, 3])),
    ]

    oof, full_fit_params = computer.compute_oof(df, folds, target_column="Transported")

    # exactly one batched call for all folds -- never per-fold fit/transform
    assert len(fake.compute_oof_batch_calls) == 1
    assert fake.compute_oof_batch_calls[0]["n_folds"] == 2
    assert fake.compute_oof_batch_calls[0]["columns"] == ["Age"]  # target not included by default
    # exactly one fit() call recorded -- the separate full-data fit; no
    # per-fold fit() calls, since those are inside the batched runner call
    assert len(fake.fit_calls) == 1
    assert full_fit_params == {"mean": 1.0}
    assert (oof == 2.0).all()


def test_sandboxed_feature_computer_compute_oof_includes_target_column_when_allowed() -> None:
    fake = _FakeRunner()
    computer = SandboxedFeatureComputer("f", "SOURCE", ["Age"], fake, allow_target_in_fit=True)
    df = pd.DataFrame({"Age": [1, 2], "Transported": [True, False]})
    folds = [FoldSplit(fold_id=0, train_index=np.array([0]), val_index=np.array([1]))]

    computer.compute_oof(df, folds, target_column="Transported")

    assert set(fake.compute_oof_batch_calls[0]["columns"]) == {"Age", "Transported"}


def test_sandboxed_feature_computer_compute_oof_restricts_full_fit_to_dev_index() -> None:
    fake = _FakeRunner()
    computer = SandboxedFeatureComputer("f", "SOURCE", ["Age"], fake)
    df = pd.DataFrame({"Age": [1, 2, 3], "Transported": [True, False, True]})
    folds = [FoldSplit(fold_id=0, train_index=np.array([1, 2]), val_index=np.array([0]))]

    computer.compute_oof(df, folds, target_column="Transported", dev_index=np.array([0, 1]))

    assert fake.fit_calls[0][1] == ["Age"]  # the full-data fit call
    assert fake.fit_call_indexes[0] == [0, 1]  # restricted to dev_index positions only


def test_sandboxed_feature_computer_transform_many_delegates_to_batched_runner_call() -> None:
    fake = _FakeRunner()
    computer = SandboxedFeatureComputer("f", "SOURCE", ["Age"], fake)
    df = pd.DataFrame({"Age": [10, 20, 30], "Transported": [True, False, True]})
    params = {"mean": 5.0}

    results = computer.transform_many(df, params, jobs=[[0, 1], [2]])

    assert len(fake.transform_batch_calls) == 1  # one call, not one per job
    assert fake.transform_batch_calls[0]["columns"] == ["Age"]  # sliced to required_columns
    assert fake.transform_batch_calls[0]["jobs"] == [[0, 1], [2]]
    assert len(results) == 2
    assert results[0].tolist() == [5.0, 5.0]
    assert results[1].tolist() == [5.0]


def test_sandbox_execution_error_carries_exit_code_and_traceback() -> None:
    err = SandboxExecutionError("boom", exit_code=1, traceback_text="Traceback...")
    assert err.exit_code == 1
    assert err.traceback_text == "Traceback..."
    assert str(err) == "boom"

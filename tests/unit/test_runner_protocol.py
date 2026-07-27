"""Exercises sandbox/runner_image/runner.py's I/O contract directly as a
subprocess (SANDBOX_IO_DIR overridden to a temp dir) -- no Docker needed.
This validates the harness protocol itself (manifest/module/input in,
status/params/output out, exit codes) independent of container isolation,
which is covered separately by the mocked docker_runner tests and the
(deferred, Docker-marked) live integration test.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "feature_generator"
    / "sandbox"
    / "runner_image"
    / "runner.py"
)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

EXIT_OK = 0
EXIT_FEATURE_RAISED = 1
EXIT_CONTRACT_VIOLATION = 2
EXIT_OUTPUT_INVALID = 3


def _run_runner(io_dir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "SANDBOX_IO_DIR": str(io_dir)}
    return subprocess.run([sys.executable, str(RUNNER_PATH)], env=env, capture_output=True, text=True)


def _status(io_dir: Path) -> dict:
    return json.loads((io_dir / "status.json").read_text())


def test_fit_call_succeeds_and_writes_params(tmp_path: Path) -> None:
    io_dir = tmp_path
    module_source = (FIXTURES / "compliant_feature_groupby_mean.py").read_text()
    (io_dir / "module.py").write_text(module_source)
    (io_dir / "manifest.json").write_text(json.dumps({"call": "fit", "context": {"target_column": "y"}}))
    pd.DataFrame({"HomePlanet": ["Earth", "Mars", "Earth"], "RoomService": [10.0, 20.0, 30.0]}).to_parquet(
        io_dir / "input.parquet"
    )

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_OK, proc.stderr
    status = _status(io_dir)
    assert status["success"] is True
    params = json.loads((io_dir / "params_out.json").read_text())
    assert params["means"]["Earth"] == pytest.approx(20.0)


def test_transform_call_succeeds_and_writes_output(tmp_path: Path) -> None:
    io_dir = tmp_path
    module_source = (FIXTURES / "compliant_feature_groupby_mean.py").read_text()
    (io_dir / "module.py").write_text(module_source)
    (io_dir / "manifest.json").write_text(json.dumps({"call": "transform", "context": {}}))
    pd.DataFrame({"HomePlanet": ["Earth", "Mars"], "RoomService": [0.0, 0.0]}).to_parquet(
        io_dir / "input.parquet"
    )
    (io_dir / "params.json").write_text(json.dumps({"means": {"Earth": 20.0, "Mars": 5.0}, "global_mean": 10.0}))

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_OK, proc.stderr
    output = pd.read_parquet(io_dir / "output.parquet")
    assert output["value"].tolist() == [20.0, 5.0]


def test_feature_code_exception_is_reported_with_traceback(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "module.py").write_text(
        "FEATURE_NAME = 'x'\n"
        "REQUIRED_COLUMNS = ['Age']\n"
        "def fit(train_df, context):\n"
        "    raise ValueError('deliberate failure')\n"
        "def transform(df, params):\n"
        "    return df['Age']\n"
    )
    (io_dir / "manifest.json").write_text(json.dumps({"call": "fit", "context": {}}))
    pd.DataFrame({"Age": [1, 2]}).to_parquet(io_dir / "input.parquet")

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_FEATURE_RAISED
    status = _status(io_dir)
    assert status["success"] is False
    assert "deliberate failure" in status["error"]
    assert "Traceback" in status["traceback"]


def test_missing_required_attribute_is_contract_violation(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "module.py").write_text("FEATURE_NAME = 'x'\n")  # missing REQUIRED_COLUMNS/fit/transform
    (io_dir / "manifest.json").write_text(json.dumps({"call": "fit", "context": {}}))
    pd.DataFrame({"Age": [1]}).to_parquet(io_dir / "input.parquet")

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_CONTRACT_VIOLATION
    status = _status(io_dir)
    assert "missing required attribute" in status["error"]


def test_non_json_serializable_fit_output_is_output_invalid(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "module.py").write_text(
        "FEATURE_NAME = 'x'\n"
        "REQUIRED_COLUMNS = ['Age']\n"
        "def fit(train_df, context):\n"
        "    return {'v': train_df}\n"  # a DataFrame isn't JSON-serializable
        "def transform(df, params):\n"
        "    return df['Age']\n"
    )
    (io_dir / "manifest.json").write_text(json.dumps({"call": "fit", "context": {}}))
    pd.DataFrame({"Age": [1, 2]}).to_parquet(io_dir / "input.parquet")

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_OUTPUT_INVALID


def test_transform_output_length_mismatch_is_output_invalid(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "module.py").write_text(
        "FEATURE_NAME = 'x'\n"
        "REQUIRED_COLUMNS = ['Age']\n"
        "def fit(train_df, context):\n"
        "    return {}\n"
        "def transform(df, params):\n"
        "    return df['Age'].iloc[:1]\n"  # returns fewer rows than input
    )
    (io_dir / "manifest.json").write_text(json.dumps({"call": "transform", "context": {}}))
    pd.DataFrame({"Age": [1, 2, 3]}).to_parquet(io_dir / "input.parquet")
    (io_dir / "params.json").write_text("{}")

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_OUTPUT_INVALID


def test_malformed_manifest_is_contract_violation(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "manifest.json").write_text("not valid json")

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_CONTRACT_VIOLATION


def test_unknown_call_type_is_contract_violation(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "module.py").write_text(
        "FEATURE_NAME = 'x'\nREQUIRED_COLUMNS = []\n"
        "def fit(train_df, context):\n    return {}\n"
        "def transform(df, params):\n    return df\n"
    )
    (io_dir / "manifest.json").write_text(json.dumps({"call": "predict", "context": {}}))

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_CONTRACT_VIOLATION
    status = _status(io_dir)
    assert "'call'" in status["error"]


def test_oof_batch_call_runs_every_fold_in_one_invocation(tmp_path: Path) -> None:
    io_dir = tmp_path
    module_source = (FIXTURES / "compliant_feature_groupby_mean.py").read_text()
    (io_dir / "module.py").write_text(module_source)
    pd.DataFrame(
        {
            "HomePlanet": ["Earth", "Mars", "Earth", "Mars", "Earth", "Mars"],
            "RoomService": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    ).to_parquet(io_dir / "input.parquet")
    manifest = {
        "call": "oof_batch",
        "folds": [
            {"fold_id": 0, "train_positions": [2, 3, 4, 5], "val_positions": [0, 1], "context": {}},
            {"fold_id": 1, "train_positions": [0, 1, 4, 5], "val_positions": [2, 3], "context": {}},
            {"fold_id": 2, "train_positions": [0, 1, 2, 3], "val_positions": [4, 5], "context": {}},
        ],
    }
    (io_dir / "manifest.json").write_text(json.dumps(manifest))

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_OK, proc.stderr
    output = pd.read_parquet(io_dir / "oof_output.parquet").set_index("position")["value"]
    assert output.loc[0] == pytest.approx(40.0)
    assert output.loc[1] == pytest.approx(50.0)
    assert output.loc[2] == pytest.approx(30.0)
    assert output.loc[3] == pytest.approx(40.0)
    assert output.loc[4] == pytest.approx(20.0)
    assert output.loc[5] == pytest.approx(30.0)


def test_oof_batch_reports_which_fold_failed(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "module.py").write_text(
        "FEATURE_NAME = 'x'\nREQUIRED_COLUMNS = ['Age']\n"
        "def fit(train_df, context):\n"
        "    if context.fold_id == 1:\n"
        "        raise ValueError('deliberate failure on fold 1')\n"
        "    return {}\n"
        "def transform(df, params):\n    return df['Age']\n"
    )
    pd.DataFrame({"Age": [1, 2, 3, 4]}).to_parquet(io_dir / "input.parquet")
    manifest = {
        "call": "oof_batch",
        "folds": [
            {"fold_id": 0, "train_positions": [2, 3], "val_positions": [0, 1], "context": {"fold_id": 0}},
            {"fold_id": 1, "train_positions": [0, 1], "val_positions": [2, 3], "context": {"fold_id": 1}},
        ],
    }
    (io_dir / "manifest.json").write_text(json.dumps(manifest))

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_FEATURE_RAISED
    status = _status(io_dir)
    assert "fold 1" in status["error"]
    assert "deliberate failure on fold 1" in status["error"]


def test_oof_batch_never_exposes_target_column_to_transform(tmp_path: Path) -> None:
    # the shared input frame may include the target column for fit()'s
    # benefit (allow_target_in_fit) -- transform() must never see it, even
    # though it reads from that same frame.
    io_dir = tmp_path
    (io_dir / "module.py").write_text(
        "FEATURE_NAME = 'x'\nREQUIRED_COLUMNS = ['category']\n"
        "def fit(train_df, context):\n"
        "    means = train_df.groupby('category')[context.target_column].mean().to_dict()\n"
        "    return {'means': means}\n"
        "def transform(df, params):\n"
        "    if 'target' in df.columns:\n"
        "        raise AssertionError('transform() must never see the target column')\n"
        "    return df['category'].map(params['means']).fillna(0.5)\n"
    )
    pd.DataFrame({"category": ["a", "b", "a", "b"], "target": [1, 0, 0, 1]}).to_parquet(
        io_dir / "input.parquet"
    )
    manifest = {
        "call": "oof_batch",
        "folds": [
            {
                "fold_id": 0, "train_positions": [2, 3], "val_positions": [0, 1],
                "context": {"target_column": "target", "fold_id": 0},
            },
            {
                "fold_id": 1, "train_positions": [0, 1], "val_positions": [2, 3],
                "context": {"target_column": "target", "fold_id": 1},
            },
        ],
    }
    (io_dir / "manifest.json").write_text(json.dumps(manifest))

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_OK, proc.stderr


def test_transform_batch_call_runs_every_job_in_one_invocation(tmp_path: Path) -> None:
    io_dir = tmp_path
    module_source = (FIXTURES / "compliant_feature_groupby_mean.py").read_text()
    (io_dir / "module.py").write_text(module_source)
    pd.DataFrame(
        {"HomePlanet": ["Earth", "Mars", "Earth"], "RoomService": [0.0, 0.0, 0.0]}
    ).to_parquet(io_dir / "input.parquet")
    (io_dir / "params.json").write_text(json.dumps({"means": {"Earth": 20.0, "Mars": 5.0}, "global_mean": 10.0}))
    manifest = {"call": "transform_batch", "jobs": [[0, 2], [1], [0, 1, 2]]}
    (io_dir / "manifest.json").write_text(json.dumps(manifest))

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_OK, proc.stderr
    output = pd.read_parquet(io_dir / "transform_batch_output.parquet")
    job0 = output[output["job_id"] == 0].sort_values("position_in_job")["value"].tolist()
    job1 = output[output["job_id"] == 1].sort_values("position_in_job")["value"].tolist()
    job2 = output[output["job_id"] == 2].sort_values("position_in_job")["value"].tolist()
    assert job0 == [20.0, 20.0]
    assert job1 == [5.0]
    assert job2 == [20.0, 5.0, 20.0]


def test_transform_batch_reports_which_job_failed(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "module.py").write_text(
        "FEATURE_NAME = 'x'\nREQUIRED_COLUMNS = ['Age']\n"
        "def fit(train_df, context):\n    return {}\n"
        "def transform(df, params):\n"
        "    if len(df) == 1:\n"
        "        raise ValueError('deliberate failure on a singleton batch')\n"
        "    return df['Age']\n"
    )
    pd.DataFrame({"Age": [1, 2, 3]}).to_parquet(io_dir / "input.parquet")
    (io_dir / "params.json").write_text("{}")
    manifest = {"call": "transform_batch", "jobs": [[0, 1], [2]]}
    (io_dir / "manifest.json").write_text(json.dumps(manifest))

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_FEATURE_RAISED
    status = _status(io_dir)
    assert "job 1" in status["error"]
    assert "deliberate failure on a singleton batch" in status["error"]


def test_unknown_call_type_still_lists_all_four_call_types_in_error(tmp_path: Path) -> None:
    io_dir = tmp_path
    (io_dir / "manifest.json").write_text(json.dumps({"call": "predict"}))

    proc = _run_runner(io_dir)

    assert proc.returncode == EXIT_CONTRACT_VIOLATION
    status = _status(io_dir)
    for call_type in ("fit", "transform", "oof_batch", "transform_batch"):
        assert call_type in status["error"]


def test_malicious_module_still_runs_when_invoked_directly_demonstrating_need_for_docker(
    tmp_path: Path,
) -> None:
    """This is NOT a security test -- it documents *why* tier-2 (Docker)
    exists: running runner.py directly (as this test does, and as would never
    happen in production -- static_checks must pass first, and even then
    execution always goes through Docker) gives malicious code full host
    access. The real defenses are (1) static_checks rejects this module
    before it ever reaches a runner, and (2) production always invokes this
    script inside a `--network none`, resource-limited, non-root container.
    """
    io_dir = tmp_path
    marker = tmp_path / "proof_of_host_access.txt"
    (io_dir / "module.py").write_text(
        "import os\n"
        "FEATURE_NAME = 'x'\nREQUIRED_COLUMNS = ['Age']\n"
        "def fit(train_df, context):\n"
        f"    os.environ['SANDBOX_ESCAPE_TEST'] = 'yes'\n"
        f"    open({str(marker)!r}, 'w').write('escaped')\n"
        "    return {}\n"
        "def transform(df, params):\n    return df['Age']\n"
    )
    (io_dir / "manifest.json").write_text(json.dumps({"call": "fit", "context": {}}))
    pd.DataFrame({"Age": [1]}).to_parquet(io_dir / "input.parquet")

    _run_runner(io_dir)

    # Confirms the premise: without static_checks + Docker, nothing in this
    # harness alone stops arbitrary host access -- both layers are required.
    assert marker.exists()

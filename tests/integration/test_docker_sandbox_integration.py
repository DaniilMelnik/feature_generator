"""Live-Docker integration tests -- DEFERRED until Docker Desktop is
installed on the dev machine (not present as of this writing). Skips
gracefully via `docker_available()` rather than failing when Docker is
absent; run with `pytest -m docker` once it's installed.

Everything these exercise (command construction, timeout/kill handling,
status.json parsing, the runner.py I/O protocol) is already covered without
Docker in test_docker_runner.py and test_runner_protocol.py -- these tests
add the one thing those can't: proof that the actual container isolation
(--network none, resource limits, non-root) holds in a real container.
"""

import subprocess
from pathlib import Path

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
    docker_available,
)

pytestmark = pytest.mark.docker

IMAGE_TAG = "feature-gen-sandbox:test"
RUNNER_IMAGE_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "feature_generator" / "sandbox" / "runner_image"
)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def built_image() -> str:
    if not docker_available():
        pytest.skip("Docker is not available on this machine")
    subprocess.run(
        ["docker", "build", "-t", IMAGE_TAG, str(RUNNER_IMAGE_DIR)],
        check=True,
        capture_output=True,
        text=True,
    )
    return IMAGE_TAG


def test_compliant_feature_runs_end_to_end_in_real_container(built_image: str) -> None:
    config = SandboxConfig(backend="docker", image_tag=built_image, timeout_seconds=30)
    runner = DockerSandboxRunner(config)
    module_source = (FIXTURES / "compliant_feature_groupby_mean.py").read_text()
    computer = SandboxedFeatureComputer(
        "avg_room_service_by_home_planet", module_source, ["HomePlanet", "RoomService"], runner
    )

    df = pd.DataFrame({"HomePlanet": ["Earth", "Mars", "Earth"], "RoomService": [10.0, 20.0, 30.0]})
    params = computer.fit(df, FitContext(target_column="Transported"))
    output = computer.transform(df, params)

    assert output.tolist() == pytest.approx([20.0, 20.0, 20.0])


def test_compute_oof_batches_every_fold_into_one_real_container(built_image: str) -> None:
    # the whole point of this optimization: proves the batched protocol
    # actually works against the real daemon (mocked tests can't catch a
    # protocol mismatch between docker_runner.py and the real runner.py).
    config = SandboxConfig(backend="docker", image_tag=built_image, timeout_seconds=10, batch_timeout_seconds=30)
    runner = DockerSandboxRunner(config)
    module_source = (FIXTURES / "compliant_feature_groupby_mean.py").read_text()
    computer = SandboxedFeatureComputer(
        "avg_room_service_by_home_planet", module_source, ["HomePlanet", "RoomService"], runner
    )
    df = pd.DataFrame(
        {
            "HomePlanet": ["Earth", "Mars", "Earth", "Mars", "Earth", "Mars"],
            "RoomService": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    folds = [
        FoldSplit(fold_id=0, train_index=np.array([2, 3, 4, 5]), val_index=np.array([0, 1])),
        FoldSplit(fold_id=1, train_index=np.array([0, 1, 4, 5]), val_index=np.array([2, 3])),
        FoldSplit(fold_id=2, train_index=np.array([0, 1, 2, 3]), val_index=np.array([4, 5])),
    ]

    oof, full_fit_params = computer.compute_oof(df, folds, target_column="Transported")

    assert oof.tolist() == pytest.approx([40.0, 50.0, 30.0, 40.0, 20.0, 30.0])
    assert "means" in full_fit_params


def test_transform_many_batches_every_job_into_one_real_container(built_image: str) -> None:
    config = SandboxConfig(backend="docker", image_tag=built_image, timeout_seconds=10, batch_timeout_seconds=30)
    runner = DockerSandboxRunner(config)
    module_source = (FIXTURES / "compliant_feature_groupby_mean.py").read_text()
    computer = SandboxedFeatureComputer(
        "avg_room_service_by_home_planet", module_source, ["HomePlanet", "RoomService"], runner
    )
    df = pd.DataFrame({"HomePlanet": ["Earth", "Mars", "Earth"], "RoomService": [0.0, 0.0, 0.0]})
    params = {"means": {"Earth": 20.0, "Mars": 5.0}, "global_mean": 10.0}

    results = computer.transform_many(df, params, jobs=[[0], [1, 2]])

    assert results[0].tolist() == pytest.approx([20.0])
    assert results[1].tolist() == pytest.approx([5.0, 20.0])


def test_container_has_no_network_access(built_image: str) -> None:
    config = SandboxConfig(backend="docker", image_tag=built_image, timeout_seconds=15)
    runner = DockerSandboxRunner(config)
    malicious_source = (
        "import socket\n"
        "FEATURE_NAME = 'x'\n"
        "REQUIRED_COLUMNS = ['Age']\n"
        "def fit(train_df, context):\n"
        "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
        "    return {}\n"
        "def transform(df, params):\n"
        "    return df['Age']\n"
    )
    computer = SandboxedFeatureComputer("x", malicious_source, ["Age"], runner)
    df = pd.DataFrame({"Age": [1, 2]})

    with pytest.raises(SandboxExecutionError):
        computer.fit(df, FitContext(target_column="y"))


def test_container_enforces_timeout_on_infinite_loop(built_image: str) -> None:
    config = SandboxConfig(backend="docker", image_tag=built_image, timeout_seconds=3)
    runner = DockerSandboxRunner(config)
    infinite_loop_source = (
        "FEATURE_NAME = 'x'\n"
        "REQUIRED_COLUMNS = ['Age']\n"
        "def fit(train_df, context):\n"
        "    while True:\n"
        "        pass\n"
        "def transform(df, params):\n"
        "    return df['Age']\n"
    )
    computer = SandboxedFeatureComputer("x", infinite_loop_source, ["Age"], runner)
    df = pd.DataFrame({"Age": [1]})

    with pytest.raises(SandboxExecutionError) as exc_info:
        computer.fit(df, FitContext(target_column="y"))
    assert exc_info.value.exit_code == 124

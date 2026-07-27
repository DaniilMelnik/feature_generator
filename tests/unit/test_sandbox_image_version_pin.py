"""The sandbox image's requirements.txt must stay pinned to EXACTLY the
host's data-science library versions -- this reproducibility is required for
the serving-parity diff (serving_parity.replay_simulator) to be meaningful.
Fails if the pin drifts from what's actually installed.
"""

from importlib.metadata import version
from pathlib import Path

REQUIREMENTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "feature_generator"
    / "sandbox"
    / "runner_image"
    / "requirements.txt"
)
PINNED_PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "pyarrow"]


def _parse_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, pinned_version = line.split("==", 1)
        pins[name.strip()] = pinned_version.strip()
    return pins


def test_requirements_file_pins_match_host_installed_versions() -> None:
    pins = _parse_requirements(REQUIREMENTS_PATH)
    for package in PINNED_PACKAGES:
        assert package in pins, f"{package} must be pinned in {REQUIREMENTS_PATH.name}"
        installed = version(package)
        assert pins[package] == installed, (
            f"{package} pin in {REQUIREMENTS_PATH.name} ({pins[package]}) has drifted from "
            f"the host's installed version ({installed}) -- update requirements.txt"
        )

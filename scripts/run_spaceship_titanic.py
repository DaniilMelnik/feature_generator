#!/usr/bin/env python3
"""Thin wrapper: run the pipeline against the Spaceship Titanic config.

Prerequisites (see README): Docker Desktop running, ANTHROPIC_API_KEY set,
and the dataset downloaded (`python scripts/download_data.py spaceship_titanic`).

    python scripts/run_spaceship_titanic.py --run-id my-run

Equivalent to: feature-gen run --config configs/spaceship_titanic.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "spaceship_titanic.yaml"


def main() -> int:
    from feature_generator.cli import main as cli_main

    argv = ["run", "--config", str(CONFIG_PATH), *sys.argv[1:]]
    cli_main(args=argv, standalone_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

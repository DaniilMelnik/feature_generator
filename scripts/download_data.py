#!/usr/bin/env python3
"""Download a Kaggle competition's data into data/<dataset>/ via the Kaggle
CLI (`pip install kaggle`, with ~/.kaggle/kaggle.json credentials configured
-- see https://www.kaggle.com/docs/api). Not run automatically by anything
else in this repo; a one-off convenience wrapper.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITIONS = {
    "spaceship_titanic": "spaceship-titanic",
    "ieee_fraud": "ieee-fraud-detection",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(COMPETITIONS), help="Which dataset to download")
    args = parser.parse_args()

    competition_slug = COMPETITIONS[args.dataset]
    dest_dir = Path(__file__).resolve().parents[1] / "data" / args.dataset
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition_slug, "-p", str(dest_dir)],
    )
    if result.returncode != 0:
        print(
            "Kaggle CLI download failed -- confirm `pip install kaggle`, that you have joined "
            f"the '{competition_slug}' competition on kaggle.com, and that ~/.kaggle/kaggle.json "
            "holds a valid API token.",
            file=sys.stderr,
        )
        return result.returncode

    for zip_path in dest_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()

    print(f"Downloaded and extracted into {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

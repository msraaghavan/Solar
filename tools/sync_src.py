"""Publish the local ``src/`` package as a new version of the Kaggle dataset.

Kaggle kernels cannot read a git checkout, so the training code is shipped as a
dataset that the kernels attach.  This script is the single place that keeps the
two in step - editing ``src/`` and forgetting to re-publish silently trains the
*previous* version of the code, which is a hard failure to notice from a log.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
PKG = os.path.join(REPO, "kernels", "srcpkg")
SLUG = "raaghavanms/filament-src"


def fingerprint(directory: str) -> str:
    """Hash of every .py file, so the version message identifies the code."""
    digest = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(directory, "*.py"))):
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default=None)
    args = parser.parse_args()

    os.makedirs(PKG, exist_ok=True)
    for stale in glob.glob(os.path.join(PKG, "*.py")):
        os.remove(stale)
    for path in glob.glob(os.path.join(SRC, "*.py")):
        shutil.copy2(path, PKG)

    with open(os.path.join(PKG, "dataset-metadata.json"), "w") as fh:
        json.dump(
            {"title": "filament-src", "id": SLUG, "licenses": [{"name": "CC0-1.0"}]},
            fh,
            indent=2,
        )

    tag = fingerprint(PKG)
    message = args.message or f"src {tag}"
    print(f"publishing {len(glob.glob(os.path.join(PKG, '*.py')))} modules as {tag}")

    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "version", "-p", PKG, "-m", message, "-r", "zip"],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

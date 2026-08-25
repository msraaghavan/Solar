"""Publish ``src/`` and launch one of the standing kernels in ``kernels/``.

``push_train.py`` does this for training runs, which it also generates.  The
calibration and prediction kernels are hand-written and live in the repo, so
they need the same discipline without the code generation: re-publish ``src/``,
*wait for Kaggle to finish processing it*, and only then push the kernel.

The waiting is the whole point.  Dataset versions are processed asynchronously,
so a kernel pushed immediately after ``sync_src.py`` attaches whatever version
happens to be current - usually the previous one.  It then runs to completion,
reports a plausible number, and that number describes code that is not the code
in the working tree.  There is nothing in the log that says so.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "raaghavanms"


def run(*command: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", *command], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def dataset_updated_at() -> str:
    for line in run("datasets", "list", "-m", "-s", "filament-src").splitlines():
        if line.startswith(f"{USER}/filament-src"):
            parts = line.split()
            return f"{parts[3]} {parts[4]}"
    return ""


def wait_for_src(previous: str, timeout: float = 900.0) -> None:
    """Block until the publication timestamp advances past ``previous``.

    The timestamp, not file sizes: a module untouched this round already has its
    expected size in the previous version, so a size check passes instantly
    while the modules that did change are still uploading.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = dataset_updated_at()
        if current and current != previous:
            print(f"  source dataset published at {current}")
            return
        time.sleep(15)
    raise SystemExit("timed out waiting for the source dataset to publish")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="directory under kernels/, e.g. calib")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    folder = os.path.join(REPO, "kernels", args.kernel)
    if not os.path.isdir(folder):
        raise SystemExit(f"no such kernel directory: {folder}")

    if not args.no_sync:
        previous = dataset_updated_at()
        print(f"publishing src/ (previous version {previous or 'none'}) ...")
        subprocess.run([sys.executable, os.path.join(REPO, "tools", "sync_src.py")], check=True)
        wait_for_src(previous)

    print(run("kernels", "push", "-p", folder).strip())

    import json

    with open(os.path.join(folder, "kernel-metadata.json")) as fh:
        slug = json.load(fh)["id"]
    print(f"\n  status: python -m kaggle kernels status {slug}")
    print(f"  output: python -m kaggle kernels output {slug} -p kernels/_runs/out_{args.kernel}")


if __name__ == "__main__":
    main()

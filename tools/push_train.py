"""Launch a training run on Kaggle with a given configuration.

Each run gets its own kernel slug so that variants can run concurrently and
their logs stay separately addressable, e.g.

    python tools/push_train.py --name b4-f0 --encoder tf_efficientnet_b4 --epochs 30
    python tools/push_train.py --name b4-t768 --tile 768 --batch 4

The source package is re-published and *waited on* before the kernel is pushed:
Kaggle dataset versions are processed asynchronously, and launching against a
half-written version silently trains the previous revision of the code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(REPO, "kernels", "train")
USER = "raaghavanms"


def run(*command: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", *command], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def dataset_updated_at() -> str:
    """Publication timestamp of the source dataset, as Kaggle reports it."""
    for line in run("datasets", "list", "-m", "-s", "filament-src").splitlines():
        if line.startswith(f"{USER}/filament-src"):
            # ref  title  size  lastUpdated(date time)  ...
            parts = line.split()
            return f"{parts[3]} {parts[4]}"
    return ""


def wait_for_src(previous: str, timeout: float = 900.0) -> None:
    """Block until Kaggle reports a *newer* version than ``previous``.

    Comparing file sizes is not enough: a module that did not change this round
    already has the expected size in the previous version, so the check passes
    instantly while other modules are still uploading.  That is precisely the
    failure that silently trains stale code, so the gate is the publication
    timestamp advancing instead.
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
    parser.add_argument("--name", required=True, help="short run name, e.g. b4-f0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--encoder", default="tf_efficientnet_b4")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tiles-per-sample", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--val-files", type=int, default=30)
    parser.add_argument("--lr", default="3e-4")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="soften targets; the task is label-noise-limited, so this is worth a fold",
    )
    parser.add_argument(
        "--spine-weight",
        type=float,
        default=0.0,
        help="weight on the auxiliary spine head (0 disables it)",
    )
    parser.add_argument("--no-sync", action="store_true", help="skip re-publishing src/")
    args = parser.parse_args()

    if not args.no_sync:
        previous = dataset_updated_at()
        print(f"publishing src/ (previous version {previous or 'none'}) ...")
        subprocess.run([sys.executable, os.path.join(REPO, "tools", "sync_src.py")], check=True)
        wait_for_src(previous)

    slug = f"filament-{args.name}"
    stage = os.path.join(REPO, "kernels", "_runs", slug)
    os.makedirs(stage, exist_ok=True)

    config = {
        "FOLD": args.fold,
        "EPOCHS": args.epochs,
        "ENCODER": args.encoder,
        "TILE": args.tile,
        "BATCH": args.batch,
        "TILES_PER_SAMPLE": args.tiles_per_sample,
        "VAL_EVERY": args.val_every,
        "VAL_FILES": args.val_files,
        "LR": args.lr,
        "LABEL_SMOOTHING": args.label_smoothing,
        "SPINE_WEIGHT": args.spine_weight,
    }

    source = open(os.path.join(TEMPLATE_DIR, "train_kernel.py")).read()
    patched, n = re.subn(
        r"CONFIG = \{.*?\n\}", "CONFIG = " + json.dumps(config, indent=4), source, count=1, flags=re.S
    )
    if n != 1:
        raise SystemExit("could not substitute CONFIG block in train_kernel.py")
    with open(os.path.join(stage, "train_kernel.py"), "w") as fh:
        fh.write(patched)

    metadata = json.load(open(os.path.join(TEMPLATE_DIR, "kernel-metadata.json")))
    metadata["id"] = f"{USER}/{slug}"
    metadata["title"] = slug
    with open(os.path.join(stage, "kernel-metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"pushing {slug}: {json.dumps(config)}")
    print(run("kernels", "push", "-p", stage).strip())
    print(f"\n  status: python -m kaggle kernels status {USER}/{slug}")
    print(f"  output: python -m kaggle kernels output {USER}/{slug} -p out")


if __name__ == "__main__":
    main()

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


def wait_for_src(timeout: float = 600.0) -> None:
    """Block until the published dataset matches the local train.py byte-for-byte."""
    local = os.path.getsize(os.path.join(REPO, "src", "train.py"))
    deadline = time.time() + timeout
    while time.time() < deadline:
        for line in run("datasets", "files", f"{USER}/filament-src").splitlines():
            parts = line.split()
            if parts and parts[0] == "train.py" and parts[1].isdigit():
                if int(parts[1]) == local:
                    print(f"  source dataset live ({local} bytes)")
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
    parser.add_argument("--no-sync", action="store_true", help="skip re-publishing src/")
    args = parser.parse_args()

    if not args.no_sync:
        print("publishing src/ ...")
        subprocess.run([sys.executable, os.path.join(REPO, "tools", "sync_src.py")], check=True)
        wait_for_src()

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

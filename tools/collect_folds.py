"""Gather five per-fold pod checkpoints into one Kaggle dataset for the ensemble.

    python tools/collect_folds.py --tag b0bnd3 fold0tag fold1tag fold2tag fold3tag fold4tag

Each pod trains one fold and publishes its checkpoint to its own dataset, so the
ensemble arrives in five pieces under five names the prediction kernel's globs do
not recognise.  This downloads them, renames each to ``fold{N}_best.pt`` (which
is what ``predict_test.py`` and ``evaluate_oof.py`` look for), and republishes
them as a single dataset that can be attached by ``dataset_sources``.

**The check that matters is that all five folds are the same model.**  An
ensemble assembled from, say, four boundary-weighted folds and one baseline fold
is not an ablation of anything: it would score somewhere between the two and the
number would describe no configuration that exists.  Nothing downstream can
detect that - the checkpoints load fine and the run completes - so it is checked
here, against the training arguments each checkpoint carries, and refused rather
than warned about.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

USER = "raaghavanms"

# Training arguments that must agree across folds for the ensemble to mean one
# thing.  Deliberately not `fold`, `seed` or anything about run length.
# A checkpoint trained before a flag existed simply has no key for it, and that
# is not a difference - it means the default, which is the behaviour that flag
# was added to leave untouched.  Comparing raw `.get()` results would read None
# against 0.0 and refuse a perfectly uniform ensemble.
DEFAULTS = {
    "spine_weight": 0.0,
    "boundary_weight": 0.0,
    "boundary_radius": 2,
    "stem_skip": False,
    "tile_size": 512,
    "tiles_per_sample": 8,
}
ARCHITECTURE = ("encoder", "spine_weight", "boundary_weight", "boundary_radius",
                "stem_skip", "tile_size", "tiles_per_sample")


def setting(args: dict, key: str):
    """One architecture setting, with a missing key read as its default."""
    value = args.get(key, DEFAULTS.get(key))
    return DEFAULTS.get(key) if value is None else value


def kaggle(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kaggle", *args],
                          capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+", help="pod tags, one per fold")
    parser.add_argument("--tag", required=True, help="name for the collected dataset")
    parser.add_argument("--keep", action="store_true", help="leave the staging dir")
    args = parser.parse_args()

    work = tempfile.mkdtemp(prefix="folds-")
    stage = os.path.join(work, "stage")
    os.makedirs(stage)
    found: dict[int, dict] = {}

    for tag in args.tags:
        into = os.path.join(work, tag)
        os.makedirs(into, exist_ok=True)
        result = kaggle("datasets", "download", "-d", f"{USER}/filament-pod-{tag}",
                        "-p", into, "--unzip", "-q")
        checkpoints = [f for f in os.listdir(into) if f.endswith(".pt")] if os.path.isdir(into) else []
        # fold*_last.pt is the end-of-run state; the ensemble wants the
        # checkpoint selected on validation PQ.
        checkpoints = [f for f in checkpoints if not f.endswith("_last.pt")]
        if not checkpoints:
            raise SystemExit(f"{tag}: no best checkpoint published "
                             f"({result.stderr.strip()[:80]})")
        if len(checkpoints) > 1:
            raise SystemExit(
                f"{tag}: {len(checkpoints)} checkpoints published ({checkpoints}).  "
                f"That pod ran more than one job, so which of them belongs in the "
                f"ensemble is a choice, not something to guess at."
            )
        path = os.path.join(into, checkpoints[0])
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        fold = ckpt["args"]["fold"]
        if fold in found:
            raise SystemExit(f"two pods claim fold {fold}: {found[fold]['tag']} and {tag}")
        found[fold] = {"tag": tag, "path": path, "args": ckpt["args"],
                       "pq": ckpt.get("pq"), "epoch": ckpt.get("epoch")}

    reference = found[min(found)]["args"]
    for fold, info in sorted(found.items()):
        differing = {
            k: (setting(reference, k), setting(info["args"], k))
            for k in ARCHITECTURE
            if setting(reference, k) != setting(info["args"], k)
        }
        if differing:
            raise SystemExit(
                f"fold {fold} ({info['tag']}) was trained with a different model than "
                f"fold {min(found)}: {differing}.  An ensemble mixing configurations "
                f"scores somewhere between them and describes none of them; nothing "
                f"downstream can detect this, so it is refused here."
            )

    expected = reference.get("n_folds", 5)
    missing = sorted(set(range(expected)) - set(found))
    if missing:
        raise SystemExit(f"folds {missing} are missing; the ensemble would be partial")

    print(f"{len(found)} folds, all trained with:")
    for key in ARCHITECTURE:
        print(f"    {key:<18}{setting(reference, key)}")
    print()
    for fold, info in sorted(found.items()):
        target = os.path.join(stage, f"fold{fold}_best.pt")
        shutil.copy2(info["path"], target)
        size = os.path.getsize(target) / 1e6
        print(f"  fold {fold}  <- {info['tag']:<14}val PQ {info['pq']:.4f}  "
              f"epoch {info['epoch']}  {size:.0f} MB")

    slug = f"{USER}/filament-folds-{args.tag}"
    with open(os.path.join(stage, "dataset-metadata.json"), "w") as fh:
        json.dump({"title": f"filament-folds-{args.tag}", "id": slug,
                   "licenses": [{"name": "CC0-1.0"}]}, fh, indent=2)

    print(f"\npublishing {slug} ...")
    out = kaggle("datasets", "create", "-p", stage, "-r", "zip", "-q")
    text = (out.stdout or "") + (out.stderr or "")
    # `create` prints its refusal and exits 0, so the message decides, not status.
    if "already in use" in text or "already exists" in text:
        out = kaggle("datasets", "version", "-p", stage, "-r", "zip",
                     "-m", f"folds for {args.tag}", "-q")
        text = (out.stdout or "") + (out.stderr or "")
    print(text.strip()[:400])
    print(f"\nattach it with dataset_sources: [\"{slug}\"]")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

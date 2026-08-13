"""Kaggle driver: build caches, then train one fold.

Kept deliberately thin.  All logic lives in the ``filament-src`` dataset, which
is the same ``src/`` package as the public repository, so a run here and a run
from a git checkout execute identical code.

Configuration comes from environment variables so that a single committed
kernel can be re-pushed for different folds and encoders without editing code.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"


def locate(marker: str, what: str) -> str:
    """Find the directory containing ``marker`` anywhere under /kaggle/input.

    Kaggle's mount layout is not stable across environments - datasets have
    appeared at both ``/kaggle/input/<slug>`` and
    ``/kaggle/input/datasets/<owner>/<slug>`` - so the paths are discovered
    rather than assumed.
    """
    import glob as _glob

    hits = _glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not hits:
        print(f"could not find {what} ({marker}). /kaggle/input contains:", flush=True)
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            if dirpath.count(os.sep) <= 5:
                print(f"  {dirpath}/ dirs={dirnames[:8]} files={filenames[:8]}", flush=True)
        raise SystemExit(f"missing {what}")
    return os.path.dirname(sorted(hits)[0])


SRC = locate("train.py", "source package")
DATA_ROOT = os.path.dirname(
    locate("MAGFiLO_1.0_Annotations_kaggle2026_train.json", "competition annotations")
)

sys.path.insert(0, SRC)
print("SRC       =", SRC, flush=True)
print("DATA_ROOT =", DATA_ROOT, flush=True)
print("modules   =", sorted(f for f in os.listdir(SRC) if f.endswith(".py")), flush=True)

# --- run configuration (rewritten by tools/push_train.py per run) ---
CONFIG = {
    "FOLD": 0,
    "EPOCHS": 10,
    "ENCODER": "tf_efficientnet_b0",
    "TILE": 512,
    "BATCH": 8,
    "TILES_PER_SAMPLE": 4,
    "VAL_EVERY": 2,
    "VAL_FILES": 12,
    "LR": "3e-4"
}

FOLD = int(os.environ.get("FOLD", CONFIG["FOLD"]))
EPOCHS = int(os.environ.get("EPOCHS", CONFIG["EPOCHS"]))
ENCODER = os.environ.get("ENCODER", CONFIG["ENCODER"])
TILE = int(os.environ.get("TILE", CONFIG["TILE"]))
BATCH = int(os.environ.get("BATCH", CONFIG["BATCH"]))
TILES_PER_SAMPLE = int(os.environ.get("TILES_PER_SAMPLE", CONFIG["TILES_PER_SAMPLE"]))
VAL_EVERY = int(os.environ.get("VAL_EVERY", CONFIG["VAL_EVERY"]))
VAL_FILES = int(os.environ.get("VAL_FILES", CONFIG["VAL_FILES"]))
LR = os.environ.get("LR", CONFIG["LR"])
# Optional, and read with .get so that a CONFIG block written before these
# existed still runs.  0.0 leaves each feature off, which is what every fold
# trained so far used, so runs stay comparable unless a variant asks otherwise.
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", CONFIG.get("LABEL_SMOOTHING", 0.0)))
SPINE_WEIGHT = float(os.environ.get("SPINE_WEIGHT", CONFIG.get("SPINE_WEIGHT", 0.0)))
print("CONFIG:", json.dumps(CONFIG), flush=True)

t0 = time.time()

# /kaggle/input is a network-backed FUSE mount.  Training re-reads every 2048x2048
# JPEG once per sampled tile, so random reads there dominate the step time.  One
# sequential copy to instance-local disk turns that into local I/O and pays for
# itself within the first epoch.
STAGE = os.environ.get("STAGE_LOCAL", "1") == "1"
if STAGE:
    import shutil

    local_root = "/tmp/magfilo"
    if not os.path.exists(local_root):
        print(f"staging data locally: {DATA_ROOT} -> {local_root}", flush=True)
        shutil.copytree(DATA_ROOT, local_root)
    n_train = len(os.listdir(f"{local_root}/train/train_images"))
    print(f"staged {n_train} train images in {time.time() - t0:.0f}s", flush=True)
    DATA_ROOT = local_root

# Disk fitting (~130 s) and limb profiles (~200 s) depend only on the images, so
# they are computed once and attached as a dataset.  Both are regenerated
# automatically if that dataset is not present, so the kernel still stands alone.
cached = glob.glob("/kaggle/input/**/disk_cache.json", recursive=True)
cached_contexts = glob.glob("/kaggle/input/**/contexts.npz", recursive=True)

disk_cache = f"{WORK}/disk_cache.json"
if cached:
    shutil.copy(cached[0], disk_cache)
    print(f"disk cache from {cached[0]}", flush=True)
else:
    import build_disk_cache

    print("--- fitting solar disks ---", flush=True)
    build_disk_cache.build(DATA_ROOT, disk_cache)

context_cache = f"{WORK}/contexts.npz"
if cached_contexts:
    shutil.copy(cached_contexts[0], context_cache)
    print(f"context cache from {cached_contexts[0]}", flush=True)
print(f"caches ready ({time.time() - t0:.0f}s)", flush=True)

print("--- training ---", flush=True)
command = [
    sys.executable,
    f"{SRC}/train.py",
    "--data-root", DATA_ROOT,
    "--disk-cache", disk_cache,
    "--context-cache", context_cache,
    "--out-dir", WORK,
    "--fold", str(FOLD),
    "--encoder", ENCODER,
    "--tile-size", str(TILE),
    "--tiles-per-sample", str(TILES_PER_SAMPLE),
    "--batch-size", str(BATCH),
    "--epochs", str(EPOCHS),
    "--lr", LR,
    "--val-every", str(VAL_EVERY),
    "--val-files", str(VAL_FILES),
    "--workers", os.environ.get("WORKERS", "4"),
]
if LABEL_SMOOTHING:
    command += ["--label-smoothing", str(LABEL_SMOOTHING)]
if SPINE_WEIGHT:
    command += ["--spine-weight", str(SPINE_WEIGHT)]
print(" ".join(command), flush=True)

# Stream the child's output so progress is visible in the kernel log rather
# than arriving in one block at the end.
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env={**os.environ, "PYTHONPATH": SRC, "PYTHONUNBUFFERED": "1"},
)
for line in process.stdout:
    print(line.rstrip(), flush=True)
code = process.wait()

print(f"\ntraining exited with code {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
if code != 0:
    print("outputs:", sorted(os.listdir(WORK)), flush=True)
    raise SystemExit(code)

# Fit the post-processing decision parameters in the same session.  The
# probability maps for a whole validation fold are ~2.3 GB, so tuning here
# avoids shipping them anywhere, and the tuned config lands next to the
# checkpoint that produced it.
checkpoint = f"{WORK}/fold{FOLD}_best.pt"
if os.path.exists(checkpoint):
    print("\n--- fitting post-processing against PQ ---", flush=True)
    evaluate = subprocess.Popen(
        [
            sys.executable, f"{SRC}/evaluate_fold.py",
            "--data-root", DATA_ROOT,
            "--context-cache", context_cache,
            "--checkpoint", checkpoint,
            "--fold", str(FOLD),
            "--tta", os.environ.get("EVAL_TTA", "4"),
            "--out", f"{WORK}/fold{FOLD}_tuned.json",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env={**os.environ, "PYTHONPATH": SRC, "PYTHONUNBUFFERED": "1"},
    )
    for line in evaluate.stdout:
        print(line.rstrip(), flush=True)
    print(f"evaluation exited {evaluate.wait()}", flush=True)

print("outputs:", sorted(os.listdir(WORK)), flush=True)

"""Kaggle driver: out-of-fold scoring across every fold at once.

Attaches all five training kernels through ``kernel_sources`` and predicts each
training observation with the one model that held it out, so a single
post-processing configuration can be fitted honestly over the whole training set
rather than five noisy ones over a fold apiece.

Set ``MAX_FILES`` to a small number for a pilot: the full run holds 707 maps in
memory and spends most of its time in the tuning sweep, so it is worth proving
the wiring on 40 observations first.
"""

import glob
import json
import os
import subprocess
import sys
import time

CONFIG = {
    "MAX_FILES": 40,
    "TTA": 4,
    "ROUNDS": 2,
    "DIAGNOSE_ENSEMBLE": 0,
}

WORK = "/kaggle/working"
MAX_FILES = int(CONFIG["MAX_FILES"])
TTA = int(CONFIG["TTA"])
ROUNDS = int(CONFIG["ROUNDS"])


def locate(marker: str, what: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not hits:
        print(f"could not find {what} ({marker}). /kaggle/input contains:", flush=True)
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            if dirpath.count(os.sep) <= 5:
                print(f"  {dirpath}/ dirs={dirnames[:8]} files={filenames[:8]}", flush=True)
        raise SystemExit(f"missing {what}")
    return os.path.dirname(sorted(hits)[0])


SRC = locate("evaluate_oof.py", "source package")
DATA_ROOT = os.path.dirname(
    locate("MAGFiLO_1.0_Annotations_kaggle2026_train.json", "competition annotations")
)
CONTEXTS = sorted(glob.glob("/kaggle/input/**/contexts.npz", recursive=True))[0]

checkpoints = sorted(glob.glob("/kaggle/input/**/fold*_best.pt", recursive=True))
if not checkpoints:
    raise SystemExit("no checkpoints found among attached sources")

# requirements.txt claims the versions these results were produced on.  Nothing
# checked that until now, so record what the runtime actually provides.
print("--- runtime ---", flush=True)
for module in ("torch", "timm", "numpy", "cv2", "pycocotools", "scipy"):
    try:
        loaded = __import__(module)
        print(f"  {module:14s} {getattr(loaded, '__version__', 'unknown')}", flush=True)
    except ImportError:
        print(f"  {module:14s} NOT INSTALLED", flush=True)
try:
    import torch

    print(f"  cuda           {torch.version.cuda}  device={torch.cuda.get_device_name(0)}", flush=True)
except Exception as exc:  # noqa: BLE001 - diagnostic only
    print(f"  cuda           unavailable: {exc}", flush=True)

print("SRC         =", SRC, flush=True)
print("DATA_ROOT   =", DATA_ROOT, flush=True)
print("contexts    =", CONTEXTS, flush=True)
print("checkpoints =", flush=True)
for path in checkpoints:
    print("   ", path, flush=True)

# Copy each fold's tuned configuration next to its checkpoint so that
# evaluate_oof can score every fold's fitted operating point on the shared
# out-of-fold half.  /kaggle/input is read-only, so they are staged here.
staged = os.path.join(WORK, "checkpoints")
os.makedirs(staged, exist_ok=True)
local = []
for path in checkpoints:
    target = os.path.join(staged, os.path.basename(path))
    if not os.path.exists(target):
        os.symlink(path, target)
    local.append(target)
for tuned in glob.glob("/kaggle/input/**/fold*_tuned.json", recursive=True):
    target = os.path.join(staged, os.path.basename(tuned))
    if not os.path.exists(target):
        os.symlink(tuned, target)

command = [
    sys.executable,
    f"{SRC}/evaluate_oof.py",
    "--data-root", DATA_ROOT,
    "--context-cache", CONTEXTS,
    "--checkpoints", *local,
    "--tta", str(TTA),
    "--rounds", str(ROUNDS),
    "--out", f"{WORK}/oof_tuned.json",
]
if MAX_FILES:
    command += ["--max-files", str(MAX_FILES)]
if int(CONFIG["DIAGNOSE_ENSEMBLE"]):
    command += ["--diagnose-ensemble"]

print("\n" + " ".join(command) + "\n", flush=True)

t0 = time.time()
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    env={**os.environ, "PYTHONPATH": SRC, "PYTHONUNBUFFERED": "1"},
)
for line in process.stdout:
    print(line.rstrip(), flush=True)
code = process.wait()
print(f"\nout-of-fold evaluation exited {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
if code != 0:
    raise SystemExit(code)

print("outputs:", sorted(os.listdir(WORK)), flush=True)

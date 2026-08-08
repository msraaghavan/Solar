"""Kaggle driver: score a trained fold and fit the post-processing parameters.

Runs against the checkpoint produced by a training kernel, attached through
``kernel_sources``.  Kept separate from training so the decision parameters can
be re-fitted (a few minutes) without repeating a multi-hour run.
"""

import glob
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
FOLD = int(os.environ.get("FOLD", "0"))
TTA = int(os.environ.get("TTA", "4"))
MAX_FILES = int(os.environ.get("MAX_FILES", "0"))


def locate(marker: str, what: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not hits:
        print(f"could not find {what} ({marker}). /kaggle/input contains:", flush=True)
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            if dirpath.count(os.sep) <= 5:
                print(f"  {dirpath}/ dirs={dirnames[:8]} files={filenames[:8]}", flush=True)
        raise SystemExit(f"missing {what}")
    return os.path.dirname(sorted(hits)[0])


SRC = locate("evaluate_fold.py", "source package")
DATA_ROOT = os.path.dirname(
    locate("MAGFiLO_1.0_Annotations_kaggle2026_train.json", "competition annotations")
)
CONTEXTS = sorted(glob.glob("/kaggle/input/**/contexts.npz", recursive=True))[0]
checkpoints = sorted(glob.glob("/kaggle/input/**/fold*_best.pt", recursive=True))
if not checkpoints:
    raise SystemExit("no checkpoint found among attached sources")

print("SRC        =", SRC, flush=True)
print("DATA_ROOT  =", DATA_ROOT, flush=True)
print("contexts   =", CONTEXTS, flush=True)
print("checkpoint =", checkpoints[0], flush=True)

command = [
    sys.executable,
    f"{SRC}/evaluate_fold.py",
    "--data-root", DATA_ROOT,
    "--context-cache", CONTEXTS,
    "--checkpoint", checkpoints[0],
    "--fold", str(FOLD),
    "--tta", str(TTA),
    "--out", f"{WORK}/fold{FOLD}_tuned.json",
]
if MAX_FILES:
    command += ["--max-files", str(MAX_FILES)]

t0 = time.time()
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    env={**os.environ, "PYTHONPATH": SRC, "PYTHONUNBUFFERED": "1"},
)
for line in process.stdout:
    print(line.rstrip(), flush=True)
code = process.wait()
print(f"\nevaluation exited {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
if code != 0:
    raise SystemExit(code)

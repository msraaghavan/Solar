"""Kaggle driver: ensemble the trained folds over the test set and emit the CSV.

Model weights arrive as the *output of the training kernels*, attached through
``kernel_sources``, so there is no manual upload step between training and
submission and the provenance of every checkpoint stays visible in the kernel
graph.
"""

import glob
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"


def locate(marker: str, what: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not hits:
        print(f"could not find {what} ({marker}). /kaggle/input contains:", flush=True)
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            if dirpath.count(os.sep) <= 5:
                print(f"  {dirpath}/ dirs={dirnames[:8]} files={filenames[:8]}", flush=True)
        raise SystemExit(f"missing {what}")
    return os.path.dirname(sorted(hits)[0])


SRC = locate("predict_test.py", "source package")
DATA_ROOT = os.path.dirname(
    locate("MAGFiLO_1.0_Annotations_kaggle2026_train.json", "competition annotations")
)
sys.path.insert(0, SRC)

checkpoints = sorted(glob.glob("/kaggle/input/**/*_best.pt", recursive=True))
if not checkpoints:
    checkpoints = sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))
if not checkpoints:
    raise SystemExit("no checkpoints found among attached sources")

candidates = sorted(glob.glob("/kaggle/input/**/*_tuned.json", recursive=True))

# Choosing the operating point by sort order is how the wrong one gets used:
# "fold0_tuned.json" precedes "oof_tuned.json" alphabetically, so taking the
# first match would silently submit a per-fold configuration - fitted on 143
# images against a grid that was censored at its own ceiling - in place of the
# pooled one.  Prefer the out-of-fold fit explicitly, and refuse to guess when
# it is absent but several per-fold configurations are attached.
oof = [c for c in candidates if os.path.basename(c) == "oof_tuned.json"]
if oof:
    configs = oof
elif len(candidates) <= 1:
    configs = candidates
else:
    raise SystemExit(
        "no oof_tuned.json among the attached sources, and several per-fold "
        f"configurations to choose between: {[os.path.basename(c) for c in candidates]}. "
        "Attach the out-of-fold kernel, or attach exactly one configuration."
    )

print("SRC        =", SRC, flush=True)
print("DATA_ROOT  =", DATA_ROOT, flush=True)
print("checkpoints=", checkpoints, flush=True)
print("config     =", configs[0] if configs else "(defaults)", flush=True)

TTA = int(os.environ.get("TTA", "4"))

command = [
    sys.executable,
    f"{SRC}/predict_test.py",
    "--data-root", DATA_ROOT,
    "--checkpoints", *checkpoints,
    "--tta", str(TTA),
    "--out", f"{WORK}/submission.csv",
]
if configs:
    command += ["--config", configs[0]]

t0 = time.time()
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
print(f"\nprediction exited {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
print("outputs:", sorted(os.listdir(WORK)), flush=True)
if code != 0:
    raise SystemExit(code)

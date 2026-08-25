"""Kaggle driver: transfer the out-of-fold operating point onto the ensemble.

Needs the five fold checkpoints and the out-of-fold configuration, both arriving
as kernel outputs through ``kernel_sources``.  Uses no ground truth - it only
compares probability distributions - so it cannot leak.
"""

import glob
import os
import subprocess
import sys
import time

CONFIG = {
    "IMAGES": 150,
    "TTA": 4,
    # Measure both histograms on the *test* images.  On train images four of the
    # five models have seen each one, so the ensemble map there is sharper than
    # it will ever be at test time and the correction comes out too small - which
    # is the leading explanation for why the stored shift (0.95 -> 0.9255) looked
    # too small to account for a -0.047 CV-to-LB gap.  Reading test pixels uses
    # no test labels and is not the MAGFiLO leak.
    "ON": "test",
}

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


SRC = locate("calibrate_ensemble.py", "source package")
DATA_ROOT = os.path.dirname(
    locate("MAGFiLO_1.0_Annotations_kaggle2026_train.json", "competition annotations")
)
CONTEXTS = sorted(glob.glob("/kaggle/input/**/contexts.npz", recursive=True))[0]
checkpoints = sorted(glob.glob("/kaggle/input/**/fold*_best.pt", recursive=True))
oof = glob.glob("/kaggle/input/**/oof_tuned.json", recursive=True)
if not oof:
    raise SystemExit("attach the out-of-fold kernel; its oof_tuned.json is the input")
if len(checkpoints) < 2:
    raise SystemExit(f"need the fold ensemble, found {checkpoints}")

print("checkpoints:", *checkpoints, sep="\n  ", flush=True)
print("config     :", oof[0], flush=True)

command = [
    sys.executable,
    f"{SRC}/calibrate_ensemble.py",
    "--data-root", DATA_ROOT,
    "--context-cache", CONTEXTS,
    "--checkpoints", *checkpoints,
    "--config", oof[0],
    "--images", str(CONFIG["IMAGES"]),
    "--tta", str(CONFIG["TTA"]),
    "--on", CONFIG["ON"],
    "--out", f"{WORK}/ensemble_config.json",
]

t0 = time.time()
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    env={**os.environ, "PYTHONPATH": SRC, "PYTHONUNBUFFERED": "1"},
)
for line in process.stdout:
    print(line.rstrip(), flush=True)
code = process.wait()
print(f"\ncalibration exited {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
if code != 0:
    raise SystemExit(code)
print("outputs:", sorted(os.listdir(WORK)), flush=True)

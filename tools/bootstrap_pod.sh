#!/usr/bin/env bash
# Bring a rented GPU pod to the point where it can train a fold.
#
# Usage, from the pod's shell:
#     bash bootstrap_pod.sh
#
# Expects ~/.kaggle/kaggle.json to exist already.  Upload it yourself; do not
# paste the token into a shell command, because it lands in the pod's shell
# history and RunPod pods are not private machines.
#
# The one setting that matters here is --workers.  Training on the T4 was
# measured input-bound, not compute-bound, which is why halving the data
# pipeline's cost bought more than a faster accelerator would have.  A rented
# 4090 will idle behind a 4-worker loader, so workers is set from the pod's
# actual core count.
set -euo pipefail

REPO=${REPO:-https://github.com/msraaghavan/Solar.git}
WORK=${WORK:-/workspace}
DATA="$WORK/Solar/data"

echo "=== pod ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "vCPUs: $(nproc)   RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
if [ "$(nproc)" -lt 8 ]; then
    echo "WARNING: fewer than 8 vCPUs. Training is input-bound; the GPU will idle."
fi

cd "$WORK"
[ -d Solar ] || git clone "$REPO"
cd Solar

python -m pip install -q --upgrade pip
# requirements.txt pins the Kaggle T4 runtime the submitted results came from.
# A pod usually ships its own torch; reinstalling the pinned build over a
# different CUDA runtime is the classic way to end up with a torch that imports
# and then fails on the first kernel launch.  Keep the pod's torch, pin the rest.
python - <<'PY'
import re
keep = []
for line in open("requirements.txt"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if re.match(r"^(torch|torchvision)==", line):
        continue
    keep.append(line)
open("/tmp/reqs-nopod-torch.txt", "w").write("\n".join(keep) + "\n")
print("installing:", ", ".join(keep))
PY
python -m pip install -q -r /tmp/reqs-nopod-torch.txt

# The Kaggle CLI is deliberately not in requirements.txt: that file documents the
# runtime the submitted results were produced *on*, and the competition image
# already provides it.  A rented pod provides nothing, and needs it twice - to
# fetch the data below, and to publish results and checkpoints afterwards, which
# is the only way anything survives an ephemeral pod.
python -m pip install -q kaggle

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
assert torch.cuda.is_available(), "no CUDA device visible"
# The Kaggle P100 failure mode is silent: is_available() is True and every
# kernel launch then fails.  Force one real launch before trusting the pod.
(torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
print("a real kernel launched successfully")
PY

if [ ! -d "$DATA/MAGFiLO_1.0_Kaggle_2026" ]; then
    echo "=== fetching competition data (751 MB) ==="
    mkdir -p "$DATA"
    # The CLI also reads KAGGLE_USERNAME/KAGGLE_KEY straight from the
    # environment, which is how a pod normally receives them, so the file is
    # optional here.  Under `set -e` an unconditional chmod on a file that was
    # never written would end the run before it started.
    [ -f ~/.kaggle/kaggle.json ] && chmod 600 ~/.kaggle/kaggle.json
    python -m kaggle competitions download -c filament-segmentation-2026 -p "$DATA"
    unzip -q -d "$DATA" "$DATA"/*.zip && rm -f "$DATA"/*.zip
fi

python src/build_disk_cache.py --root "$DATA/MAGFiLO_1.0_Kaggle_2026"
python tests/test_pipeline.py
python tests/test_official_metric.py

WORKERS=$(( $(nproc) > 16 ? 16 : $(nproc) ))
cat <<EOF

=== ready ===
Train a fold (workers set from this pod's $(nproc) cores):

  python src/train.py --fold 0 --encoder tf_efficientnet_b0 \\
      --tile-size 512 --batch-size 8 --tiles-per-sample 8 \\
      --epochs 20 --val-every 3 --val-files 40 --workers $WORKERS

Then fit the operating point and score it honestly:

  python src/evaluate_fold.py --checkpoint artifacts/fold0_best.pt --fold 0

Watch the first epoch time. The T4 did 233 s/epoch input-bound; if this pod is
not much faster, raise --workers or --tiles-per-sample rather than the GPU.
EOF

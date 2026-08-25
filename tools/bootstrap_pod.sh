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

# What the code actually imports, across all of src/ and tests/: cv2, numpy,
# pycocotools, torch and timm.  scipy, scikit-image, pandas and matplotlib are in
# requirements.txt because that file documents the *Kaggle* runtime the submitted
# results came from, where they are already present - nothing here imports them.
#
# Installing them anyway is not free.  Every extra package is another chance for
# the resolver to move torch, and moving torch is fatal in a way that is not
# obvious from the error: pip pulled torch 2.13.0+cu130 from PyPI over the
# image's 2.9.1+cu129, it imported perfectly, and then reported *no CUDA device*
# because the host driver is 12.9.  Stripping the "torch==" lines out of
# requirements.txt did not prevent that, because timm *depends* on torch and pip
# is free to satisfy that by upgrading it.
#
# So: keep the pod's torch, install timm without its dependencies, and add back
# only the ones that are not torch.
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
echo "pod torch: $TORCH_BEFORE  (this must not change)"

python -m pip install -q --no-deps timm==1.0.26
# timm's runtime dependencies other than torch/torchvision.  huggingface_hub and
# safetensors are what fetch and load the pretrained encoder weights, so
# --no-deps without these gets you a timm that fails at create_model(pretrained).
python -m pip install -q pyyaml huggingface_hub safetensors
python -m pip install -q opencv-python-headless==4.13.0.90 pycocotools==2.0.10

# The Kaggle CLI is deliberately not in requirements.txt: that file documents the
# runtime the submitted results were produced *on*, and the competition image
# already provides it.  A rented pod provides nothing, and needs it twice - to
# fetch the data below, and to publish results and checkpoints afterwards, which
# is the only way anything survives an ephemeral pod.
python -m pip install -q kaggle

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    echo "!!! torch moved from $TORCH_BEFORE to $TORCH_AFTER while installing deps."
    echo "!!! That is the exact failure the --no-deps above exists to prevent:"
    echo "!!! a PyPI build compiled against a newer CUDA than this host's driver"
    echo "!!! imports cleanly and then sees no GPU at all."
    exit 1
fi
echo "torch unchanged at $TORCH_AFTER"

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

WORKERS=$(( $(nproc) > 32 ? 32 : $(nproc) ))
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

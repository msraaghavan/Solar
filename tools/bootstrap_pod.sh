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

# These images are bare: torch and torchaudio, but no numpy and no torchvision.
# Both are needed - everything here is numpy-first, and timm imports torchvision
# during create_model - and both have to arrive without letting pip move torch.
#
# Install only what is actually missing.  Reinstalling a package the image
# already has is how you downgrade numpy underneath a torch that was compiled
# against it, which fails at import with a C-level ABI error rather than
# anything that mentions numpy.
python -c "import numpy" 2>/dev/null || python -m pip install -q "numpy==2.0.2"

if ! python -c "import torchvision" 2>/dev/null; then
    # torchvision's minor version tracks torch's at a fixed offset of +15 for
    # the 2.x line: torch 2.8/2.9/2.10/2.13 pair with torchvision
    # 0.23/0.24/0.25/0.28.  Checked against two independent points - this
    # machine's torch 2.13.0 with torchvision 0.28.0, and requirements.txt's
    # torch 2.10.0 with torchvision 0.25.0.
    #
    # The build must also come from the same CUDA index as the installed torch,
    # or the two disagree at the C++ ABI.  --no-deps so it cannot drag a
    # different torch in behind it, which is exactly what timm did on pod one.
    TV_SPEC=$(python -c "import torch;m=int(torch.__version__.split('+')[0].split('.')[1]);print('torchvision==0.%d.*' % (m + 15))")
    TV_INDEX=$(python -c "import torch;v=torch.__version__;print('https://download.pytorch.org/whl/' + v.split('+')[1] if '+' in v else '')")
    echo "installing $TV_SPEC from ${TV_INDEX:-pypi}"
    if [ -n "$TV_INDEX" ]; then
        python -m pip install -q --no-deps "$TV_SPEC" --index-url "$TV_INDEX" ||
        python -m pip install -q --no-deps "$TV_SPEC"
    else
        python -m pip install -q --no-deps "$TV_SPEC"
    fi
fi

python -m pip install -q --no-deps timm==1.0.26
# timm's remaining runtime dependencies.  huggingface_hub and safetensors are
# what fetch and load the pretrained encoder weights, so --no-deps without these
# gets a timm that fails at create_model(pretrained=True).
python -m pip install -q pyyaml huggingface_hub safetensors
python -m pip install -q opencv-python-headless==4.13.0.90 pycocotools==2.0.10

# The Kaggle CLI is deliberately not in requirements.txt: that file documents the
# runtime the submitted results were produced *on*, where it is already present.
# A rented pod provides nothing, and needs it twice - to fetch the data below,
# and to publish results and checkpoints afterwards, which is the only way
# anything survives an ephemeral pod.
python -m pip install -q kaggle

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    echo "!!! torch moved from $TORCH_BEFORE to $TORCH_AFTER while installing deps."
    echo "!!! That is the exact failure the --no-deps flags above exist to prevent:"
    echo "!!! a PyPI build compiled against a newer CUDA than this host's driver"
    echo "!!! imports cleanly and then sees no GPU at all."
    exit 1
fi
python -c "import numpy, torchvision, timm, cv2, pycocotools; print('imports ok: numpy', numpy.__version__, '| torchvision', torchvision.__version__, '| timm', timm.__version__)"
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

# The disk geometry and the per-image contexts are pure functions of the images,
# and both are already published as raaghavanms/filament-cache (348 KB total).
# Fetching them costs seconds; rebuilding the contexts costs 8.7 minutes of a
# 4090 sitting idle, measured on the first successful pod - per pod, every time.
mkdir -p artifacts
python -m kaggle datasets download -d raaghavanms/filament-cache -p artifacts --unzip -q ||
    echo "cache download failed; will rebuild from the images"

# Trust it only if it actually describes this dataset.  A stale or partial cache
# would hand training the wrong disk geometry for every image, silently.
python - <<'CHECK'
import glob, json, os, sys
sys.path.insert(0, "src")
names = {os.path.basename(p) for p in glob.glob(
    "data/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpeg")}
ok = True
if os.path.exists("artifacts/disk_cache.json"):
    cache = json.load(open("artifacts/disk_cache.json"))
    missing = names - set(cache)
    if missing:
        print(f"disk cache is missing {len(missing)} of {len(names)} images; rebuilding")
        os.remove("artifacts/disk_cache.json")
        ok = False
    else:
        print(f"disk cache covers all {len(names)} train images")
if ok and os.path.exists("artifacts/contexts.npz"):
    from data import load_contexts
    contexts = load_contexts("artifacts/contexts.npz")
    if names - set(contexts):
        print(f"context cache covers only {len(contexts)}; rebuilding")
        os.remove("artifacts/contexts.npz")
    else:
        print(f"context cache covers all {len(contexts)} train images")
CHECK

[ -f artifacts/disk_cache.json ] ||
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

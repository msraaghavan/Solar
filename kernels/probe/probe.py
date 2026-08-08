"""Probe the Kaggle runtime: GPU, competition data, and available packages.

Run once before committing to the remote-training path so that any missing
dependency shows up here rather than half-way through a training run.
"""

import glob
import importlib
import os
import subprocess

print("=== GPU ===", flush=True)
try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    print("nvidia-smi not on PATH (does not by itself prove there is no GPU)")

print("\n=== PACKAGES ===", flush=True)
for name in [
    "torch",
    "torchvision",
    "timm",
    "segmentation_models_pytorch",
    "cv2",
    "pycocotools",
    "albumentations",
    "numpy",
    "scipy",
    "skimage",
]:
    try:
        module = importlib.import_module(name)
        print(f"{name:32s} {getattr(module, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:32s} MISSING ({type(exc).__name__})")

print("\n=== TORCH / CUDA ===", flush=True)
import torch  # noqa: E402

print("torch", torch.__version__, "cuda avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name}  {props.total_memory / 1e9:.1f} GB")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    print("matmul ok:", (a @ a).float().mean().item() is not None)

print("\n=== COMPETITION DATA ===", flush=True)
for root in ["/kaggle/input"]:
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - root.count(os.sep)
        if depth <= 3:
            print(f"{dirpath}  ({len(filenames)} files)")

train = glob.glob("/kaggle/input/**/train_images/*.jpeg", recursive=True)
test = glob.glob("/kaggle/input/**/test_images/*.jpeg", recursive=True)
ann = glob.glob("/kaggle/input/**/*.json", recursive=True)
print(f"\ntrain images {len(train)}  test images {len(test)}  json {len(ann)}")
print("ann files:", ann)

print("\n=== WRITE TEST ===", flush=True)
os.makedirs("/kaggle/working", exist_ok=True)
with open("/kaggle/working/probe_ok.txt", "w") as fh:
    fh.write("ok\n")
print("wrote /kaggle/working/probe_ok.txt")
print("\nPROBE COMPLETE")

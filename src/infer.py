"""Full-disk inference by overlapped tiling.

The network sees 512-pixel tiles but the metric is defined on whole 2048-pixel
observations, so tiles are stitched back into a single probability map before
instances are extracted.  Two details matter:

*Cosine blending.*  A network is least certain near a tile border, where it has
no context on one side.  Averaging overlapping tiles under a raised-cosine
window suppresses the seams that would otherwise appear as spurious instance
boundaries - and a filament split by a seam is expensive under PQ.

*Dihedral test-time augmentation.*  The eight square symmetries are exact pixel
permutations, so averaging over them costs no sharpness, unlike TTA over scales
or rotations.  It is the natural counterpart of the training augmentation.
"""

from __future__ import annotations

import numpy as np
import torch

from data import ImageContext

IMAGE_SIZE = 2048

AMP_CHOICES = ("auto", "bf16", "fp16", "fp32")


def choose_amp_dtype(
    preference: str, device: str, capability: tuple[int, int] | None
) -> "torch.dtype | None":
    """Pick the autocast dtype; ``None`` means run in full precision.

    Every autocast site here used to take torch's default, which on CUDA is
    fp16.  That is the right choice on the Kaggle T4 and the wrong one on
    anything newer, and the difference is not cosmetic: EfficientNet-B4
    overflowed fp16 on the T4, and the resulting non-finite gradients were
    *silently* dropped by GradScaler.  A B4 run can therefore be badly
    undertrained while its loss curve looks healthy - which is exactly the
    caveat recorded against the measured B4 result.

    bf16 carries fp32's exponent range, so that failure mode does not exist on
    hardware that supports it natively.  ``auto`` selects it from compute
    capability >= 8.0 (Ampere and later) rather than from
    ``torch.cuda.is_bf16_supported()``, which reports True for emulated bf16 on
    older cards and would quietly select a slow path on the very T4 the current
    results were measured on.
    """
    if preference not in AMP_CHOICES:
        raise ValueError(f"amp dtype must be one of {AMP_CHOICES}, got {preference!r}")
    if preference == "fp32" or not device.startswith("cuda"):
        return None
    if preference == "bf16":
        return torch.bfloat16
    if preference == "fp16":
        return torch.float16
    native_bf16 = capability is not None and capability[0] >= 8
    return torch.bfloat16 if native_bf16 else torch.float16


def amp_dtype_for(preference: str, device: str) -> "torch.dtype | None":
    """:func:`choose_amp_dtype` against the live device."""
    capability = (
        torch.cuda.get_device_capability() if device.startswith("cuda") else None
    )
    return choose_amp_dtype(preference, device, capability)


def _cosine_window(size: int) -> np.ndarray:
    """Separable raised-cosine, floored so tile centres keep full weight."""
    ramp = np.hanning(size + 2)[1:-1].astype(np.float32)
    ramp = np.maximum(ramp, 1e-3)
    return np.outer(ramp, ramp)


def _dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply one of the 8 square symmetries (k in 0..7)."""
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


def _dihedral_inverse(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.rot90(x, -(k % 4), dims=[-2, -1])
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def predict_full(
    model: torch.nn.Module,
    image: np.ndarray,
    context: ImageContext,
    tile_size: int = 512,
    stride: int = 384,
    tta: int = 4,
    batch_size: int = 8,
    device: str = "cuda",
    amp: bool = True,
    amp_dtype: str = "auto",
) -> np.ndarray:
    """Probability map for one full 2048x2048 observation.

    ``tta`` selects how many of the eight symmetries to average (1 disables it,
    4 uses the rotations, 8 uses rotations and reflections).

    ``amp_dtype`` follows :func:`choose_amp_dtype`; ``amp=False`` overrides it
    and runs in full precision.
    """
    dtype = amp_dtype_for(amp_dtype, device) if amp else None
    model.eval()
    accumulator = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    weights = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    window = _cosine_window(tile_size)

    origins = _tile_origins(context, tile_size, stride)
    nonfinite_batches = [0]

    for start in range(0, len(origins), batch_size):
        chunk = origins[start : start + batch_size]
        features = np.stack(
            [context.tile_features(image, y0, x0, tile_size) for y0, x0 in chunk]
        )
        batch = torch.from_numpy(features).to(device)

        probs = torch.zeros(
            (len(chunk), 1, tile_size, tile_size), dtype=torch.float32, device=device
        )
        for k in range(tta):
            view = _dihedral(batch, k)
            with torch.autocast(
                device_type=device.split(":")[0],
                dtype=dtype or torch.float32,
                enabled=dtype is not None,
            ):
                logits = model(view)
            # A model trained with the auxiliary spine head emits two channels;
            # only channel 0 is the filament mask.  Slicing here rather than
            # assuming one channel keeps spine and non-spine checkpoints on the
            # same inference path.
            logits = logits[:, :1].float()

            # Half precision overflows for larger encoders: EfficientNet-B4
            # produced non-finite activations on a T4 where B0 never did.  A NaN
            # is silently *lost* downstream rather than raised - every comparison
            # against it is False, so the pixel drops out as background and the
            # mask is quietly corrupted.  Recompute the offending batch in full
            # precision instead of letting that through.  The guard stays under
            # bf16, which has fp32's range and should never reach it: if it ever
            # fires there the cause is not overflow and is worth knowing about.
            if not torch.isfinite(logits).all():
                nonfinite_batches[0] += 1
                with torch.autocast(device_type=device.split(":")[0], enabled=False):
                    logits = model(view.float())[:, :1].float()
                logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

            probs += torch.sigmoid(_dihedral_inverse(logits, k))
        probs /= tta

        for (y0, x0), prob in zip(chunk, probs.cpu().numpy()[:, 0]):
            accumulator[y0 : y0 + tile_size, x0 : x0 + tile_size] += prob * window
            weights[y0 : y0 + tile_size, x0 : x0 + tile_size] += window

    if nonfinite_batches[0]:
        print(
            f"    [infer] recomputed {nonfinite_batches[0]} tile batch(es) in fp32 "
            f"after non-finite half-precision output",
            flush=True,
        )

    probability = np.divide(
        accumulator, weights, out=np.zeros_like(accumulator), where=weights > 0
    )
    if not np.isfinite(probability).all():
        raise RuntimeError("non-finite probability map survived the fp32 fallback")
    if probability.min() < 0.0 or probability.max() > 1.0 + 1e-5:
        raise RuntimeError(
            f"probability map outside [0,1]: [{probability.min()}, {probability.max()}]"
        )
    return probability


def _tile_origins(
    context: ImageContext, tile_size: int, stride: int
) -> list[tuple[int, int]]:
    """Tile origins covering the solar disk, skipping fully off-disk tiles."""
    limit = IMAGE_SIZE - tile_size
    positions = list(range(0, limit + 1, stride))
    if positions[-1] != limit:
        positions.append(limit)

    disk = context.disk
    origins = []
    for y0 in positions:
        for x0 in positions:
            # Closest point of the tile to the disk centre.
            dx = max(disk.cx - (x0 + tile_size), 0.0, x0 - disk.cx)
            dy = max(disk.cy - (y0 + tile_size), 0.0, y0 - disk.cy)
            if dx * dx + dy * dy <= disk.r * disk.r:
                origins.append((y0, x0))
    return origins


def disk_mask_for(context: ImageContext, shrink: float = 0.995) -> np.ndarray:
    """On-disk mask; the slight shrink drops the noisy one-pixel limb ring."""
    return context.disk.mask(IMAGE_SIZE, shrink=shrink)

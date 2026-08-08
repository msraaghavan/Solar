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
) -> np.ndarray:
    """Probability map for one full 2048x2048 observation.

    ``tta`` selects how many of the eight symmetries to average (1 disables it,
    4 uses the rotations, 8 uses rotations and reflections).
    """
    model.eval()
    accumulator = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    weights = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    window = _cosine_window(tile_size)

    origins = _tile_origins(context, tile_size, stride)

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
            with torch.autocast(device_type=device.split(":")[0], enabled=amp):
                logits = model(_dihedral(batch, k))
            # A model trained with the auxiliary spine head emits two channels;
            # only channel 0 is the filament mask.  Slicing here rather than
            # assuming one channel keeps spine and non-spine checkpoints on the
            # same inference path.
            logits = logits[:, :1]
            probs += torch.sigmoid(_dihedral_inverse(logits, k).float())
        probs /= tta

        for (y0, x0), prob in zip(chunk, probs.cpu().numpy()[:, 0]):
            accumulator[y0 : y0 + tile_size, x0 : x0 + tile_size] += prob * window
            weights[y0 : y0 + tile_size, x0 : x0 + tile_size] += window

    return np.divide(accumulator, weights, out=np.zeros_like(accumulator), where=weights > 0)


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

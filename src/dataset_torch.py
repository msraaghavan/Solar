"""Torch dataset that cuts training tiles from full-resolution observations.

Sampling is deliberately biased towards filaments.  A 512x512 tile placed
uniformly on the disk contains no filament about 80% of the time, and training
on that distribution spends almost all of its gradient budget confirming that
quiet Sun is quiet.  Tiles are instead centred on a randomly chosen filament
three quarters of the time, with the remainder placed uniformly to keep a
realistic supply of hard negatives (sunspots, plage boundaries, seeing
artefacts).

Geometric augmentation is restricted to the eight square symmetries.  These are
exact permutations of the pixel grid, so unlike rotation or scale jitter they
add no interpolation blur - which matters because measurements showed thin
filament barbs are the first thing lost to resampling.  The Sun is radially
symmetric, so the transformed tile remains physically plausible, provided the
radius plane is transformed with it.
"""

from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from data import ImageContext, Sample

IMAGE_SIZE = 2048


class FilamentTiles(Dataset):
    """Random tiles from (image, annotator) samples.

    One epoch is defined as ``tiles_per_sample`` tiles from every sample; the
    positions are redrawn every epoch, so the effective dataset is much larger
    than the 1154 annotations.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        image_dir: str,
        contexts: dict[str, ImageContext],
        tile_size: int = 512,
        tiles_per_sample: int = 8,
        positive_fraction: float = 0.75,
        augment: bool = True,
        seed: int = 0,
    ):
        self.samples = list(samples)
        self.image_dir = image_dir
        self.contexts = contexts
        self.tile_size = tile_size
        self.tiles_per_sample = tiles_per_sample
        self.positive_fraction = positive_fraction
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples) * self.tiles_per_sample

    def set_epoch(self, epoch: int) -> None:
        """Re-seed tile positions so each epoch sees different crops."""
        self.epoch = epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self.epoch, index))

    def __getitem__(self, index: int):
        rng = self._rng(index)
        sample = self.samples[index // self.tiles_per_sample]
        context = self.contexts[sample.file_name]

        image = cv2.imread(
            os.path.join(self.image_dir, sample.file_name), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            raise FileNotFoundError(sample.file_name)

        mask = (
            mask_utils.decode(sample.semantic_rle)
            if sample.instances
            else np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        )

        y0, x0 = self._choose_origin(rng, sample, context)
        features = context.tile_features(image, y0, x0, self.tile_size)
        target = mask[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size].astype(np.float32)

        # Only on-disk pixels carry information; off-disk is trivially negative
        # and would otherwise dominate tiles that straddle the limb.  Taken from
        # the geometry, not recovered from the (standardised) radius plane.
        weight = context.tile_disk_mask(y0, x0, self.tile_size)

        if self.augment:
            features, target, weight = self._augment(rng, features, target, weight)

        return (
            torch.from_numpy(np.ascontiguousarray(features)),
            torch.from_numpy(np.ascontiguousarray(target))[None],
            torch.from_numpy(np.ascontiguousarray(weight))[None],
        )

    def _choose_origin(
        self, rng: np.random.Generator, sample: Sample, context: ImageContext
    ) -> tuple[int, int]:
        size = self.tile_size
        limit = IMAGE_SIZE - size

        if sample.instances and rng.random() < self.positive_fraction:
            instance = sample.instances[rng.integers(len(sample.instances))]
            x, y, w, h = mask_utils.toBbox(instance)
            cy, cx = y + h / 2.0, x + w / 2.0
            # Jitter so the filament is not always dead centre, which would let
            # the network key on tile position.
            jitter = size // 3
            cy += rng.integers(-jitter, jitter + 1)
            cx += rng.integers(-jitter, jitter + 1)
        else:
            disk = context.disk
            angle = rng.uniform(0, 2 * np.pi)
            radius = disk.r * np.sqrt(rng.uniform(0, 1)) * 0.98
            cy = disk.cy + radius * np.sin(angle)
            cx = disk.cx + radius * np.cos(angle)

        y0 = int(np.clip(cy - size / 2, 0, limit))
        x0 = int(np.clip(cx - size / 2, 0, limit))
        return y0, x0

    def _augment(
        self,
        rng: np.random.Generator,
        features: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # --- dihedral group: exact, interpolation-free ---
        k = int(rng.integers(4))
        if k:
            features = np.rot90(features, k, axes=(1, 2))
            target = np.rot90(target, k)
            weight = np.rot90(weight, k)
        if rng.random() < 0.5:
            features = features[:, :, ::-1]
            target = target[:, ::-1]
            weight = weight[:, ::-1]

        features = features.copy()

        # --- photometric: applied only to the two intensity planes, never to
        # the radius plane, which encodes fixed geometry ---
        if rng.random() < 0.8:
            gain = rng.uniform(0.85, 1.18)
            offset = rng.uniform(-0.06, 0.06)
            features[:2] = features[:2] * gain + offset
        if rng.random() < 0.3:
            sigma = rng.uniform(0.5, 1.3)  # variable seeing
            for c in range(2):
                features[c] = cv2.GaussianBlur(features[c], (0, 0), sigma)
        if rng.random() < 0.3:
            features[:2] += rng.normal(0, rng.uniform(0.01, 0.04), features[:2].shape)

        np.clip(features[:2], -1.5, 1.5, out=features[:2])
        return features, target.copy(), weight.copy()

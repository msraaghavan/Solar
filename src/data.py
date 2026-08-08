"""Dataset construction for filament segmentation.

Design decisions worth stating up front, because they follow from measurements
in ``notebooks/01_eda.md`` rather than from convention:

*Native resolution.*  Round-trip tests show that warping masks through a
1024-pixel canonical frame costs ~0.11 IoU, against a segmentation-quality
ceiling of only ~0.65 set by inter-annotator disagreement.  Downsampling is
therefore unaffordable, and the model is trained on tiles cut from the original
2048x2048 grid.

*One sample per annotator.*  707 images carry 1154 independent annotations.
Rather than fusing them into a consensus target, every (image, annotator) pair
is a separate training example, so a pixel marked by two of three annotators
converges to a target of ~2/3 under a proper loss.  The network therefore learns
the *annotator posterior*, which is what the operating-point analysis needs.

*Grouping.*  Folds are grouped by file name so that the several annotations of
one observation never straddle a split, and stratified by GONG site.
"""

from __future__ import annotations

import collections
import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from preprocess import Disk, limb_profile

IMAGE_SIZE = 2048
N_PROFILE_BINS = 256

# Per-plane standardisation, measured over the training corpus (see
# ``tools/analysis.py``).  Limb-corrected contrast is a narrow quantity - its
# standard deviation is only ~0.09 - and feeding that straight into an
# ImageNet-pretrained stem wastes most of the first layer's dynamic range.  The
# constants are fixed rather than computed per tile on purpose: per-tile
# normalisation would erase the absolute-brightness cue that separates sunspots
# from filaments, and would behave differently at test time.
FEATURE_MEAN = np.array([-0.005, -0.098, -0.083], dtype=np.float32)
FEATURE_STD = np.array([0.093, 0.292, 0.256], dtype=np.float32)


@dataclass
class Sample:
    """One annotator's reading of one observation."""

    image_id: str        # e.g. "010401-20160920230134Lh" (annotator batch + name)
    file_name: str       # e.g. "20160920230134Lh.jpeg"
    instances: list[dict]  # per-filament COCO RLEs, at 2048x2048

    @property
    def site(self) -> str:
        """Two-letter GONG site code, e.g. 'Bh' for Big Bear."""
        return self.file_name[14:16]

    @property
    def semantic_rle(self) -> dict:
        """Union of every filament in this reading."""
        if not self.instances:
            empty = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            return mask_utils.encode(np.asfortranarray(empty))
        return mask_utils.merge(self.instances)


def load_samples(annotation_path: str) -> list[Sample]:
    """Parse the COCO annotation file into per-(image, annotator) samples."""
    with open(annotation_path) as fh:
        coco = json.load(fh)

    by_image: dict[str, list[dict]] = collections.defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    samples = []
    for image in coco["images"]:
        instances = [
            mask_utils.merge(
                mask_utils.frPyObjects(ann["segmentation"], IMAGE_SIZE, IMAGE_SIZE)
            )
            for ann in by_image[image["id"]]
        ]
        samples.append(
            Sample(
                image_id=image["id"],
                file_name=image["file_name"],
                instances=instances,
            )
        )
    return samples


def make_folds(
    samples: Sequence[Sample], n_folds: int = 5, seed: int = 42
) -> dict[str, int]:
    """Assign each *file* to a fold, stratified by GONG site.

    Returns a mapping ``file_name -> fold``.  Grouping by file (not by sample)
    is essential: the same observation read by three annotators would otherwise
    appear on both sides of the split and inflate validation scores.
    """
    rng = np.random.default_rng(seed)
    by_site: dict[str, list[str]] = collections.defaultdict(list)
    for file_name in sorted({s.file_name for s in samples}):
        by_site[file_name[14:16]].append(file_name)

    folds: dict[str, int] = {}
    for site in sorted(by_site):
        files = np.array(sorted(by_site[site]))
        rng.shuffle(files)
        for i, file_name in enumerate(files):
            folds[str(file_name)] = i % n_folds
    return folds


class ImageContext:
    """Per-observation geometry and photometry, computed once and reused.

    Holds the fitted disk and the radial intensity profile so that flat-fielding
    a tile is a table lookup rather than a full-frame reduction.
    """

    def __init__(self, disk: Disk, profile: np.ndarray):
        self.disk = disk
        self.profile = profile

    @classmethod
    def build(cls, image: np.ndarray, disk: Disk) -> "ImageContext":
        return cls(disk, limb_profile(image, disk, N_PROFILE_BINS))

    def tile_radius(self, y0: int, x0: int, size: int) -> np.ndarray:
        """Normalised radius r/R over a tile, in original-frame coordinates."""
        yy, xx = np.mgrid[y0 : y0 + size, x0 : x0 + size].astype(np.float32)
        return np.sqrt((xx - self.disk.cx) ** 2 + (yy - self.disk.cy) ** 2) / self.disk.r

    def tile_disk_mask(self, y0: int, x0: int, size: int) -> np.ndarray:
        """On-disk indicator for a tile.

        Derived from the geometry directly rather than by inverting the
        standardised radius plane, so changing the feature normalisation cannot
        silently corrupt the loss mask.
        """
        return (self.tile_radius(y0, x0, size) <= 1.0).astype(np.float32)

    def tile_features(
        self, image: np.ndarray, y0: int, x0: int, size: int
    ) -> np.ndarray:
        """Three input planes for one tile, as float32 in roughly [-1, 1].

        ``0`` limb-corrected contrast - the filament signal, flat across radius.
        ``1`` raw intensity - absolute brightness, which separates quiet Sun
              from plage and carries the residual the flat field removed.
        ``2`` normalised radius r/R - tells the network where the limb is, so it
              can learn that off-disk pixels and the extreme limb are never
              filaments.
        """
        tile = image[y0 : y0 + size, x0 : x0 + size].astype(np.float32)
        radius = self.tile_radius(y0, x0, size)

        bins = np.clip((radius * N_PROFILE_BINS).astype(np.int32), 0, N_PROFILE_BINS - 1)
        expected = np.maximum(self.profile[bins], 1.0)
        contrast = np.clip(tile / expected - 1.0, -1.0, 1.0)
        contrast[radius > 1.0] = 0.0

        planes = np.stack(
            [
                contrast,
                tile / 127.5 - 1.0,
                np.clip(radius, 0.0, 1.5) - 0.75,
            ]
        ).astype(np.float32)
        return (planes - FEATURE_MEAN[:, None, None]) / FEATURE_STD[:, None, None]


def build_contexts(
    image_dir: str, file_names: Sequence[str], disk_cache: dict[str, dict]
) -> dict[str, ImageContext]:
    """Compute an :class:`ImageContext` for each file (~0.05 s each)."""
    contexts = {}
    for file_name in file_names:
        image = cv2.imread(os.path.join(image_dir, file_name), cv2.IMREAD_GRAYSCALE)
        entry = disk_cache[file_name]
        disk = Disk(cx=entry["cx"], cy=entry["cy"], r=entry["r"])
        contexts[file_name] = ImageContext.build(image, disk)
    return contexts


def save_contexts(contexts: dict[str, ImageContext], path: str) -> None:
    np.savez_compressed(
        path,
        names=np.array(sorted(contexts)),
        disks=np.array(
            [
                [contexts[n].disk.cx, contexts[n].disk.cy, contexts[n].disk.r]
                for n in sorted(contexts)
            ]
        ),
        profiles=np.stack([contexts[n].profile for n in sorted(contexts)]),
    )


def load_contexts(path: str) -> dict[str, ImageContext]:
    data = np.load(path, allow_pickle=False)
    names = [str(n) for n in data["names"]]
    return {
        name: ImageContext(
            Disk(*(float(v) for v in data["disks"][i])), data["profiles"][i]
        )
        for i, name in enumerate(names)
    }

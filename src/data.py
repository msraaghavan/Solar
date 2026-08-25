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
    spines: list[list[float]] = field(default_factory=list)  # flat [x0,y0,x1,y1,...]

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
        annotations = by_image[image["id"]]
        instances = [
            mask_utils.merge(
                mask_utils.frPyObjects(ann["segmentation"], IMAGE_SIZE, IMAGE_SIZE)
            )
            for ann in annotations
        ]
        samples.append(
            Sample(
                image_id=image["id"],
                file_name=image["file_name"],
                instances=instances,
                spines=[list(ann.get("spine") or []) for ann in annotations],
            )
        )
    return samples


def spine_points(spine: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
    """Normalise one annotation's spine to an ``(N, 2)`` array of ``(x, y)`` points.

    COCO does not fix a convention for a polyline field, and the same polyline
    can arrive in three shapes: flat as ``[x0, y0, x1, y1, ...]``, wrapped once
    as ``[[x0, y0, x1, y1, ...]]`` the way ``segmentation`` is, or already paired
    as ``[[x0, y0], [x1, y1], ...]``.  Guessing wrong is *silent*: a wrapped
    spine has outer length 1, trips any ``len(spine) < 4`` guard, and rasterises
    to an all-zero target - so the auxiliary head trains on a blank image, the
    run looks healthy, and the experiment reports "spine does not help" having
    never tested it.  Accept all three instead, and reject anything else loudly.
    """
    if spine is None or len(spine) == 0:
        return np.empty((0, 2), dtype=np.float32)

    first = spine[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        parts = [np.asarray(p, dtype=np.float32).reshape(-1) for p in spine]
        # Pairs [[x0, y0], [x1, y1], ...] concatenate to the same flat vector as
        # a single wrapped run [[x0, y0, x1, y1, ...]], so both fall out here.
        flat = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    else:
        flat = np.asarray(spine, dtype=np.float32).reshape(-1)

    if flat.size % 2:
        raise ValueError(f"spine has an odd number of coordinates: {flat.size}")
    return flat.reshape(-1, 2)


def rasterise_spines(
    spines: Sequence[Sequence[float]], thickness: int = 3, size: int = IMAGE_SIZE
) -> np.ndarray:
    """Draw filament spines as a binary map.

    The spine is the annotated central axis of a filament: a connected polyline
    that carries its orientation and, through the barbs hanging off it, its
    magnetic chirality.  The competition scores masks only, so this annotation is
    normally discarded - but it is a strong auxiliary target.  Predicting it
    forces the network to represent a filament as one elongated object with an
    axis, which is exactly the distinction between a filament and the compact
    dark regions (sunspots, pores) that otherwise look identical locally.

    Thickness 3 is chosen from measurement: at that width 95.4% of spine pixels
    fall inside the corresponding filament mask while covering ~39% of its area,
    so the target is a genuine central core.  (At thickness 1 alignment is 99.3%,
    confirming the annotations agree; at 9 the "spine" is as large as the
    filament itself and stops being an axis.)

    The full frame is drawn and cropped, rather than rasterised per tile in a
    shifted frame, on measurement: the whole-frame draw costs 0.30 ms against
    7.4 ms for ``tile_planes`` and tens of milliseconds for the JPEG decode
    beside it, so tile-local drawing would buy ~0.2% of the data pipeline.  It
    would also cost correctness - ``cv2.polylines`` clips to integer canvas
    bounds, so a spine entering the tile from outside rasterises on a different
    Bresenham phase and lands up to a pixel off (measured: IoU 0.89 against the
    cropped full-frame draw, worst case 0.69).
    """
    canvas = np.zeros((size, size), dtype=np.uint8)
    for spine in spines:
        points = spine_points(spine)
        if len(points) < 2:
            continue
        cv2.polylines(
            canvas,
            [np.round(points).astype(np.int32)],
            isClosed=False,
            color=1,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
    return canvas


def spine_alignment(sample: "Sample", thickness: int = 3) -> tuple[float, float]:
    """``(spine pixels inside the mask, mask pixels covered)`` for one reading.

    The single measurement that decides whether the spine annotation has been
    understood at all.  Every way of misreading the field - a wrapped list that
    rasterises to nothing, or ``(row, column)`` order fed to ``cv2.polylines``,
    which expects ``(x, y)`` and would draw every spine transposed about the
    disk centre - lands the spine somewhere other than on its own filament.  The
    published alignment at thickness 3 is 95.4%, so anything near zero means the
    target is wrong, not merely noisy.
    """
    if not sample.instances:
        return (0.0, 0.0)
    mask = mask_utils.decode(sample.semantic_rle).astype(bool)
    # Size follows the mask rather than IMAGE_SIZE so the check works on the
    # small fixtures the tests use as well as on real 2048px readings.
    spine = rasterise_spines(
        sample.spines, thickness=thickness, size=mask.shape[0]
    ).astype(bool)
    if not spine.any():
        return (0.0, 0.0)
    inside = float((spine & mask).sum())
    return (inside / float(spine.sum()), inside / max(float(mask.sum()), 1.0))


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


def stride_split(file_names: Sequence[str]) -> tuple[set[str], set[str]]:
    """Halve a set of observations into (tune, report).

    Two properties matter and neither is free.  The split *strides* rather than
    cutting the sorted list in two, because file names sort by timestamp and a
    prefix would put one solar-cycle epoch on each side.  And it splits over
    observations rather than annotator readings: 42% of the training set is read
    by two or three annotators, so splitting the reading list puts readings of
    the same image on both sides - measured at 59.8% of the report half - and
    the tuner then scores itself on observations it has already fitted to.
    """
    ordered = sorted(set(file_names))
    return set(ordered[0::2]), set(ordered[1::2])


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
        """Normalised radius r/R over a tile, in original-frame coordinates.

        Built by broadcasting two 1-D offset vectors rather than with
        ``np.mgrid``.  Profiling put mgrid at 11 ms of a 17 ms call - it
        materialises two full size x size coordinate arrays before any
        arithmetic - and this is on the hot path of every training sample.
        """
        dy = (np.arange(y0, y0 + size, dtype=np.float32) - self.disk.cy)[:, None]
        dx = (np.arange(x0, x0 + size, dtype=np.float32) - self.disk.cx)[None, :]
        return np.sqrt(dy * dy + dx * dx, dtype=np.float32) / np.float32(self.disk.r)

    def tile_disk_mask(self, y0: int, x0: int, size: int) -> np.ndarray:
        """On-disk indicator for a tile.

        Derived from the geometry directly rather than by inverting the
        standardised radius plane, so changing the feature normalisation cannot
        silently corrupt the loss mask.
        """
        return (self.tile_radius(y0, x0, size) <= 1.0).astype(np.float32)

    def tile_planes(
        self, image: np.ndarray, y0: int, x0: int, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Features and on-disk mask together, sharing one radius computation.

        The training loop needs both, and computing the radius map twice was
        costing ~17 ms per sample - a sixth of the whole data pipeline - purely
        to derive two views of the same geometry.
        """
        radius = self.tile_radius(y0, x0, size)
        return (
            self._features_from_radius(image, radius, y0, x0, size),
            (radius <= 1.0).astype(np.float32),
        )

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
        return self._features_from_radius(
            image, self.tile_radius(y0, x0, size), y0, x0, size
        )

    def _features_from_radius(
        self, image: np.ndarray, radius: np.ndarray, y0: int, x0: int, size: int
    ) -> np.ndarray:
        tile = image[y0 : y0 + size, x0 : x0 + size].astype(np.float32)

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

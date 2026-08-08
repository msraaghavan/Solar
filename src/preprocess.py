"""Solar-disk geometry and photometric normalisation for GONG H-alpha images.

Two properties of full-disk H-alpha observations make raw pixels a poor input
for a segmentation network:

1.  *Limb darkening.*  Intensity falls off steeply towards the solar limb, so a
    "dark region" detector keyed on raw brightness fires on the limb annulus
    rather than on filaments.  Normalising each pixel by the median intensity of
    its radial shell flattens the disk and makes filament contrast comparable at
    disk centre and near the limb.

2.  *Framing.*  The disk does not fill the frame and its centre and radius drift
    between instruments and dates, so the same physical structure lands on
    different pixels in different images.

Both are handled here: :func:`detect_disk` recovers the disk circle and
:func:`normalise` produces a flat-fielded image in a canonical frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SIZE = 2048


@dataclass(frozen=True)
class Disk:
    """The fitted solar disk, in pixels of the original 2048x2048 frame."""

    cx: float
    cy: float
    r: float

    def mask(self, size: int = IMAGE_SIZE, shrink: float = 1.0) -> np.ndarray:
        """Boolean mask of the disk interior, optionally shrunk by ``shrink``."""
        yy, xx = np.ogrid[:size, :size]
        return ((xx - self.cx) ** 2 + (yy - self.cy) ** 2) <= (self.r * shrink) ** 2

    def radius_map(self, size: int = IMAGE_SIZE) -> np.ndarray:
        """Normalised radial coordinate r/R for every pixel."""
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        return np.sqrt((xx - self.cx) ** 2 + (yy - self.cy) ** 2) / self.r


def detect_disk(image: np.ndarray) -> Disk:
    """Fit the solar disk.

    The sky background in these JPEGs sits near zero while the disk is bright, so
    a coarse Otsu threshold separates them cleanly.  The centre comes from the
    image moments of the largest component and the radius from its area, which
    is far more stable against limb noise and prominences than a bounding box or
    a Hough circle would be.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {image.shape}")

    blur = cv2.GaussianBlur(image, (0, 0), 3)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close over filaments/sunspots so the disk is one solid component.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        raise RuntimeError("no solar disk found")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    # Fill interior holes before measuring the area, so dark features near the
    # centre do not shrink the radius estimate.
    component = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    area = float(filled.sum())
    cx, cy = centroids[largest]
    return Disk(cx=float(cx), cy=float(cy), r=float(np.sqrt(area / np.pi)))


def limb_profile(image: np.ndarray, disk: Disk, n_bins: int = 256) -> np.ndarray:
    """Median intensity per radial shell, from disk centre to the limb.

    The median (rather than the mean) is deliberate: filaments and sunspots are
    dark outliers, and a mean profile would be dragged down by them, partially
    "normalising away" the very signal we want to keep.
    """
    r = disk.radius_map(image.shape[0])
    inside = r <= 1.0
    bins = np.clip((r[inside] * n_bins).astype(np.int32), 0, n_bins - 1)
    values = image[inside].astype(np.float32)

    profile = np.zeros(n_bins, dtype=np.float32)
    order = np.argsort(bins, kind="stable")
    bins_sorted, values_sorted = bins[order], values[order]
    edges = np.searchsorted(bins_sorted, np.arange(n_bins + 1))
    for i in range(n_bins):
        segment = values_sorted[edges[i] : edges[i + 1]]
        profile[i] = np.median(segment) if segment.size else np.nan

    # Fill empty shells (possible in the outermost bins) by interpolation.
    valid = ~np.isnan(profile)
    if not valid.all():
        profile = np.interp(
            np.arange(n_bins), np.flatnonzero(valid), profile[valid]
        ).astype(np.float32)
    # Light smoothing: the profile is physically smooth, so this removes shot
    # noise without touching the trend.
    return cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).ravel()


def flat_field(image: np.ndarray, disk: Disk, n_bins: int = 256) -> np.ndarray:
    """Divide out limb darkening; returns contrast relative to the local shell.

    Output is centred on 0 (a pixel at its shell median maps to 0) with filaments
    negative, and is clipped to +/-1 to bound the effect of dead pixels and
    saturated plage.
    """
    profile = limb_profile(image, disk, n_bins)
    r = disk.radius_map(image.shape[0])
    bins = np.clip((r * n_bins).astype(np.int32), 0, n_bins - 1)
    expected = profile[bins]
    expected = np.maximum(expected, 1.0)  # guard against division by zero

    contrast = image.astype(np.float32) / expected - 1.0
    contrast[r > 1.0] = 0.0  # off-disk carries no information
    return np.clip(contrast, -1.0, 1.0)


def normalise(
    image: np.ndarray, size: int, disk: Disk | None = None, margin: float = 1.02
) -> tuple[np.ndarray, Disk]:
    """Flat-field, crop to the disk, and resize to a canonical ``size`` frame.

    Returns the processed float32 image in ``[-1, 1]`` and the fitted disk, so
    that predicted masks can be mapped back to original 2048x2048 coordinates.
    """
    disk = disk or detect_disk(image)
    contrast = flat_field(image, disk)
    return _to_canonical(contrast, disk, size, margin), disk


def _to_canonical(array: np.ndarray, disk: Disk, size: int, margin: float) -> np.ndarray:
    """Resample so the disk of radius ``r`` fills a ``size`` box with ``margin``."""
    scale = size / (2.0 * disk.r * margin)
    matrix = np.array(
        [
            [scale, 0.0, size / 2.0 - scale * disk.cx],
            [0.0, scale, size / 2.0 - scale * disk.cy],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        array, matrix, (size, size), flags=cv2.INTER_AREA, borderValue=0.0
    )


def canonical_matrix(disk: Disk, size: int, margin: float = 1.02) -> np.ndarray:
    """The 2x3 affine used by :func:`normalise`, for mapping masks back."""
    scale = size / (2.0 * disk.r * margin)
    return np.array(
        [
            [scale, 0.0, size / 2.0 - scale * disk.cx],
            [0.0, scale, size / 2.0 - scale * disk.cy],
        ],
        dtype=np.float32,
    )


def to_canonical_mask(mask: np.ndarray, disk: Disk, size: int, margin: float = 1.02) -> np.ndarray:
    """Warp a full-frame mask into the canonical frame (nearest-neighbour)."""
    matrix = canonical_matrix(disk, size, margin)
    return cv2.warpAffine(
        mask.astype(np.uint8), matrix, (size, size), flags=cv2.INTER_NEAREST, borderValue=0
    )


def from_canonical_mask(
    mask: np.ndarray, disk: Disk, out_size: int = IMAGE_SIZE, margin: float = 1.02
) -> np.ndarray:
    """Invert :func:`to_canonical_mask`, back to the original frame."""
    matrix = canonical_matrix(disk, mask.shape[0], margin)
    full = np.vstack([matrix, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(full)[:2]
    return cv2.warpAffine(
        mask.astype(np.uint8),
        inverse,
        (out_size, out_size),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )

"""Turning a probability map into filament instances, tuned against PQ.

Connected components are provably almost lossless here: running them on the
*ground-truth* union recovers the ground-truth instances at PQ 0.9967 over 300
images, because annotated filaments essentially never touch.  So the instance
step needs no learned grouping, and the whole problem reduces to choosing a good
binary mask plus a few morphological decisions.

Those decisions are not free parameters to be eyeballed.  Panoptic Quality gives
an exact rule for when to emit a marginal detection.  Writing the metric as

    PQ = sum(IoU over TP) / (|TP| + 0.5|FP| + 0.5|FN|)

consider adding one more predicted instance.  If it matches an as-yet-unmatched
ground-truth segment, that segment stops counting 0.5 as a false negative and
starts counting 1.0 as a true positive, so the denominator rises by 0.5 and the
numerator by its IoU.  If it matches nothing, it is a false positive: the
denominator rises by 0.5 and the numerator by 0.  Either way the denominator
rises by exactly 0.5, which makes the trade-off unusually clean.  Writing ``p``
for the probability that the candidate lands on a real filament and ``SQ`` for
the mean IoU of a match, emitting it raises PQ whenever

    p * SQ > 0.5 * PQ        i.e.        p > 0.5 * PQ / SQ

At the levels measured on this dataset (PQ ~ 0.34, SQ ~ 0.64) the bar is only
p > 0.27: it pays to emit any candidate with better than roughly a one-in-four
chance of being real.  The optimum is therefore markedly recall-biased.

Two consequences shape the design here.

*Hysteresis, not a single threshold.*  One threshold has to serve two different
jobs - deciding **which** blobs to emit, and deciding **how far** each one
extends - and the analysis above wants opposite settings for them: low, to emit
marginal detections; but a boundary chosen for overlap quality.  Splitting them
lets a component be admitted only if it contains a confident core
(``seed_threshold``) while its extent is taken from a looser level
(``mask_threshold``).

*Instance dilation.*  Annotators disagree about filament extent, and the union
of two readings predicts a third better than either reading alone (IoU 0.646 vs
0.637, measured).  A predicted boundary slightly larger than the modal one is
therefore closer to the average annotator.  Because matches require IoU > 0.5,
nudging a near-miss over that line converts a false-positive-plus-false-negative
pair into a true positive, so shape accuracy pays off super-linearly near the
threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Sequence

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from metrics import evaluate, evaluate_image


@dataclass(frozen=True)
class PostprocessConfig:
    """Decision parameters applied to a probability map.

    ``seed_threshold`` admits a component; ``mask_threshold`` (<= seed) sets its
    extent.  Setting them equal recovers plain single-threshold behaviour.
    """

    seed_threshold: float = 0.45
    mask_threshold: float = 0.30
    min_area: int = 150
    min_seed_area: int = 20
    close_radius: int = 0
    open_radius: int = 0
    dilate_radius: int = 0
    fill_holes: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def marginal_threshold(pq: float, sq: float) -> float:
    """The break-even hit probability derived in the module docstring."""
    return 0.5 * pq / sq if sq > 0 else 1.0


def _disc(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def extract_instances(
    probability: np.ndarray,
    config: PostprocessConfig,
    disk_mask: np.ndarray | None = None,
) -> list[dict]:
    """Probability map -> list of per-instance COCO RLEs."""
    mask_level = min(config.mask_threshold, config.seed_threshold)
    binary = (probability >= mask_level).astype(np.uint8)
    seeds = (probability >= config.seed_threshold).astype(np.uint8)

    if disk_mask is not None:
        on_disk = disk_mask.astype(np.uint8)
        binary &= on_disk
        seeds &= on_disk

    if config.close_radius > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _disc(config.close_radius))
    if config.open_radius > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _disc(config.open_radius))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n_labels <= 1:
        return []

    # Seed support per component, in one pass rather than per-label masking.
    seed_area = np.bincount(
        labels[seeds > 0].ravel(), minlength=n_labels
    ) if seeds.any() else np.zeros(n_labels, dtype=np.int64)

    kernel = _disc(config.dilate_radius) if config.dilate_radius > 0 else None

    instances = []
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] < config.min_area:
            continue
        if seed_area[label] < config.min_seed_area:
            continue
        component = (labels == label).astype(np.uint8)
        if config.fill_holes:
            component = _fill_holes(component)
        if kernel is not None:
            component = cv2.dilate(component, kernel)
            if disk_mask is not None:
                component &= disk_mask.astype(np.uint8)
        instances.append(mask_utils.encode(np.asfortranarray(component)))
    return instances


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close interior holes; annotated filaments are solid single polygons."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def score_config(
    predictions: Sequence[tuple[str, np.ndarray, np.ndarray | None]],
    ground_truth: dict[str, list[dict]],
    config: PostprocessConfig,
) -> dict:
    """Evaluate one configuration over a set of probability maps."""
    results = []
    for image_id, probability, disk_mask in predictions:
        instances = extract_instances(probability, config, disk_mask)
        results.append(evaluate_image(image_id, ground_truth[image_id], instances))
    return evaluate(results)


def tune(
    predictions: Sequence[tuple[str, np.ndarray, np.ndarray | None]],
    ground_truth: dict[str, list[dict]],
    verbose: bool = True,
    rounds: int = 2,
) -> tuple[PostprocessConfig, dict]:
    """Coordinate ascent over the decision parameters, maximising PQ.

    One axis at a time, repeated for ``rounds`` passes so that later axes can
    revise earlier ones.  A full grid over seven parameters is unaffordable -
    every evaluation is a pass over the whole validation fold - and the axes are
    close enough to separable that coordinate ascent recovers most of the gain.
    """
    # The seed grid reaches well below 0.5 on purpose.  The marginal-emission
    # analysis says a candidate pays whenever its hit probability exceeds
    # ~0.5*PQ/SQ - about 0.27 at the levels measured here - so the optimum is
    # expected to sit at a threshold that looks recklessly permissive by the
    # standards of a pixel-overlap metric.
    grids: dict[str, Iterable] = {
        "seed_threshold": (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7),
        "mask_threshold": (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
        "min_area": (0, 30, 60, 100, 150, 250, 400, 600),
        "min_seed_area": (0, 5, 10, 20, 50, 100),
        "close_radius": (0, 2, 4, 6),
        "dilate_radius": (0, 1, 2, 3, 5),
        "open_radius": (0, 1, 2),
    }

    best = PostprocessConfig()
    best_report = score_config(predictions, ground_truth, best)
    if verbose:
        print(f"  start PQ {best_report['pq_micro']:.4f}", flush=True)

    for round_index in range(rounds):
        improved = False
        for name, values in grids.items():
            for value in values:
                candidate = replace(best, **{name: value})
                if candidate == best:
                    continue
                report = score_config(predictions, ground_truth, candidate)
                if report["pq_micro"] > best_report["pq_micro"] + 1e-6:
                    best, best_report, improved = candidate, report, True
                    if verbose:
                        print(
                            f"  round {round_index} {name}={value} -> PQ "
                            f"{report['pq_micro']:.4f} (SQ {report['sq']:.3f} "
                            f"RQ {report['rq']:.3f})",
                            flush=True,
                        )
        if not improved:
            break

    return best, best_report

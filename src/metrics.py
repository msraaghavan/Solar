"""Evaluation metrics for the Solar Filament Segmentation Challenge 2026.

The competition's leaderboard metric is Panoptic Quality (PQ), defined on the
competition Evaluation page as

    PQ(Y, Y_hat) = sum_{(y, y_hat) in TP} IoU(y, y_hat)
                   / ( |TP| + 0.5 |FP| + 0.5 |FN| )

with TP / FP / FN the sets of true-positive, false-positive and false-negative
*instances*.  Following Kirillov et al. (CVPR 2019), a predicted segment and a
ground-truth segment form a true positive when their IoU exceeds 0.5; that
threshold makes the matching unique, so greedy matching is optimal.

Because the exact aggregation the hosts use is not published, both variants are
reported:

``pq_micro``
    One global PQ over instances pooled across every image (the convention in
    the original panoptic-segmentation paper).
``pq_macro``
    Per-image PQ averaged over images.

Masks are handled as COCO RLE dicts throughout; ``pycocotools`` computes IoU on
the compressed representation, which is far cheaper than materialising many
2048x2048 boolean arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from pycocotools import mask as mask_utils

# A mask is either a COCO RLE dict or a 2-D boolean/uint8 array.
Mask = dict | np.ndarray

IOU_MATCH_THRESHOLD = 0.5


def to_rle(mask: Mask) -> dict:
    """Normalise a mask to a COCO RLE dict with ``bytes`` counts."""
    if isinstance(mask, dict):
        rle = dict(mask)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return rle
    arr = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {arr.shape}")
    return mask_utils.encode(arr)


def rle_areas(rles: Sequence[dict]) -> np.ndarray:
    if not rles:
        return np.zeros(0, dtype=np.float64)
    return mask_utils.area(list(rles)).astype(np.float64)


def iou_matrix(gt: Sequence[dict], pred: Sequence[dict]) -> np.ndarray:
    """Pairwise IoU between ground-truth and predicted instances.

    Returns an array of shape ``(len(gt), len(pred))``.
    """
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    # iscrowd=0 for every gt -> plain IoU rather than the "ignore" variant.
    iscrowd = [0] * len(gt)
    ious = mask_utils.iou(list(pred), list(gt), iscrowd)
    # pycocotools returns (n_pred, n_gt); transpose to (n_gt, n_pred).
    return np.asarray(ious, dtype=np.float64).reshape(len(pred), len(gt)).T


def match_instances(
    ious: np.ndarray, threshold: float = IOU_MATCH_THRESHOLD
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match gt to predictions on IoU.

    With ``threshold > 0.5`` at most one prediction can exceed the threshold for
    a given ground-truth segment, so greedy matching by descending IoU is
    provably optimal and agrees with Hungarian matching.

    Returns ``(matches, unmatched_gt, unmatched_pred)`` where ``matches`` holds
    ``(gt_index, pred_index)`` pairs.
    """
    n_gt, n_pred = ious.shape
    matches: list[tuple[int, int]] = []
    gt_taken = np.zeros(n_gt, dtype=bool)
    pred_taken = np.zeros(n_pred, dtype=bool)

    if n_gt and n_pred:
        candidates = np.argwhere(ious > threshold)
        order = np.argsort(-ious[candidates[:, 0], candidates[:, 1]])
        for gt_i, pred_i in candidates[order]:
            if gt_taken[gt_i] or pred_taken[pred_i]:
                continue
            gt_taken[gt_i] = True
            pred_taken[pred_i] = True
            matches.append((int(gt_i), int(pred_i)))

    return (
        matches,
        [int(i) for i in np.flatnonzero(~gt_taken)],
        [int(i) for i in np.flatnonzero(~pred_taken)],
    )


def fragmentation_counts(
    ious: np.ndarray, overlap_threshold: float = 0.0
) -> tuple[int, int]:
    """Count one-to-many (fragmentation) and many-to-one (over-merge) relations.

    The rubric scores these directly, so the definition has to be the host's
    rather than a reasonable-looking one of our own.  Their self-evaluation
    notebook builds ``hit_matrix = iou_matrix > 0`` for this plot - *any* overlap
    at all, however slight - and counts a ground-truth segment as fragmented when
    it touches two or more predictions, a prediction as over-merged when it
    touches two or more ground-truth segments.

    This used to default to 0.1, which quietly reported better fragmentation than
    the judges will measure: a prediction clipping a neighbouring filament by a
    few pixels is invisible at 0.1 and counts against us at 0.  The threshold
    stays a parameter, because 0.1 remains the more informative number for
    deciding whether a split is real, but the default now matches the rubric.
    """
    if ious.size == 0:
        return 0, 0
    hits = ious > overlap_threshold
    one_to_many = int((hits.sum(axis=1) >= 2).sum())
    many_to_one = int((hits.sum(axis=0) >= 2).sum())
    return one_to_many, many_to_one


def dice_from_iou(iou: float) -> float:
    """Dice and IoU are monotonically related: Dice = 2*IoU / (1 + IoU)."""
    return 2.0 * iou / (1.0 + iou) if iou > 0 else 0.0


@dataclass
class ImageResult:
    """Per-image accumulator, kept so distributions can be reported."""

    image_id: str
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    iou_sum: float
    matched_ious: list[float] = field(default_factory=list)
    # Best IoU achieved against each ground-truth segment that went unmatched.
    # PQ has a hard cliff at 0.5, and crossing it is worth far more than the
    # overlap gained: a near-miss is simultaneously a false positive and a false
    # negative, so promoting one to a true positive moves the denominator by
    # +1 - 0.5 - 0.5 = 0 while adding its IoU to the numerator.  Improvement is
    # therefore free in the denominator, and the size of the prize is exactly
    # how much mass sits just below 0.5 - which is what this records.
    near_miss_ious: list[float] = field(default_factory=list)
    area_ratios: list[float] = field(default_factory=list)
    one_to_many: int = 0
    many_to_one: int = 0
    semantic_dice: float = 0.0

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denom if denom > 0 else 1.0


def evaluate_image(
    image_id: str,
    gt_masks: Sequence[Mask],
    pred_masks: Sequence[Mask],
    threshold: float = IOU_MATCH_THRESHOLD,
) -> ImageResult:
    gt = [to_rle(m) for m in gt_masks]
    pred = [to_rle(m) for m in pred_masks]

    ious = iou_matrix(gt, pred)
    matches, unmatched_gt, unmatched_pred = match_instances(ious, threshold)
    matched_ious = [float(ious[g, p]) for g, p in matches]
    one_to_many, many_to_one = fragmentation_counts(ious)

    near_miss = [float(ious[g].max()) for g in unmatched_gt] if ious.size else []

    # Predicted area over ground-truth area, for matched pairs only.  IoU says
    # how wrong a boundary is but not which way; this says whether the fix is to
    # grow instances or shrink them, which is the difference between a useful
    # dilation and a harmful one.
    gt_area = rle_areas(gt)
    pred_area = rle_areas(pred)
    area_ratios = [
        float(pred_area[p] / gt_area[g]) for g, p in matches if gt_area[g] > 0
    ]

    return ImageResult(
        image_id=image_id,
        n_gt=len(gt),
        n_pred=len(pred),
        tp=len(matches),
        fp=len(unmatched_pred),
        fn=len(unmatched_gt),
        iou_sum=float(sum(matched_ious)),
        matched_ious=matched_ious,
        near_miss_ious=near_miss,
        area_ratios=area_ratios,
        one_to_many=one_to_many,
        many_to_one=many_to_one,
        semantic_dice=semantic_dice(gt, pred),
    )


def semantic_dice(gt: Sequence[dict], pred: Sequence[dict]) -> float:
    """Dice over the union of all instances, i.e. the binary filament map.

    Reported alongside PQ because the rubric still lists a Dice distribution.
    An image with no filaments and no predictions scores 1.0.
    """
    gt_union = mask_utils.merge(list(gt)) if gt else None
    pred_union = mask_utils.merge(list(pred)) if pred else None
    if gt_union is None and pred_union is None:
        return 1.0
    if gt_union is None or pred_union is None:
        return 0.0
    inter = float(mask_utils.area(mask_utils.merge([gt_union, pred_union], intersect=True)))
    total = float(mask_utils.area(gt_union)) + float(mask_utils.area(pred_union))
    return 2.0 * inter / total if total > 0 else 1.0


def evaluate(
    results: Iterable[ImageResult],
) -> dict[str, float | list[float]]:
    """Aggregate per-image results into the full rubric report."""
    results = list(results)
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    iou_sum = sum(r.iou_sum for r in results)
    denom = tp + 0.5 * fp + 0.5 * fn

    matched_ious = [i for r in results for i in r.matched_ious]
    dices = [dice_from_iou(i) for i in matched_ious]
    near_miss = [i for r in results for i in r.near_miss_ious]

    # How much PQ is sitting just under the cliff.  Promoting a near miss is
    # denominator-neutral, so the gain is simply the IoU it would contribute
    # divided by the unchanged denominator - which makes this an upper bound on
    # what better boundaries alone are worth, with no change to detection.
    almost = [i for i in near_miss if 0.4 <= i <= 0.5]
    headroom = sum(almost) / denom if denom > 0 else 0.0

    ratios = [r for x in results for r in x.area_ratios]
    median_ratio = float(np.median(ratios)) if ratios else 0.0

    pq_micro = iou_sum / denom if denom > 0 else 0.0
    if not (0.0 <= pq_micro <= 1.0 + 1e-9):
        raise ValueError(f"PQ outside [0,1]: {pq_micro} (tp={tp} fp={fp} fn={fn})")

    return {
        # Primary leaderboard metric, both plausible aggregations.
        "pq_micro": pq_micro,
        "pq_macro": float(np.mean([r.pq for r in results])) if results else 0.0,
        # Panoptic decomposition: PQ = SQ * RQ.
        "sq": iou_sum / tp if tp else 0.0,
        "rq": tp / denom if denom > 0 else 0.0,
        # Detection counts.
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": sum(r.n_gt for r in results),
        "n_pred": sum(r.n_pred for r in results),
        "hit_rate": tp / (tp + fn) if (tp + fn) else 0.0,
        "miss_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        # Rubric extras.
        "one_to_many": sum(r.one_to_many for r in results),
        "many_to_one": sum(r.many_to_one for r in results),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_matched_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_semantic_dice": float(np.mean([r.semantic_dice for r in results]))
        if results
        else 0.0,
        # Where the next PQ comes from, rather than how much we already have.
        "near_miss_count": len(almost),
        "near_miss_headroom_pq": headroom,
        # >1 means predictions are larger than the truth they matched.
        "median_area_ratio": median_ratio,
        "iou_distribution": matched_ious,
        "dice_distribution": dices,
        "near_miss_distribution": near_miss,
    }


def format_report(report: dict) -> str:
    """One-screen summary of an ``evaluate`` result."""
    lines = [
        f"PQ (micro)          {report['pq_micro']:.4f}   <- primary metric",
        f"PQ (macro)          {report['pq_macro']:.4f}",
        f"  SQ (seg quality)  {report['sq']:.4f}",
        f"  RQ (recognition)  {report['rq']:.4f}",
        f"TP/FP/FN            {report['tp']}/{report['fp']}/{report['fn']}",
        f"GT/Pred instances   {report['n_gt']}/{report['n_pred']}",
        f"hit / miss rate     {report['hit_rate']:.4f} / {report['miss_rate']:.4f}",
        f"precision           {report['precision']:.4f}",
        f"mean matched IoU    {report['mean_matched_iou']:.4f}",
        f"mean matched Dice   {report['mean_matched_dice']:.4f}",
        f"mean semantic Dice  {report['mean_semantic_dice']:.4f}",
        f"one-to-many         {report['one_to_many']}  (fragmentation, any overlap)",
        f"many-to-one         {report['many_to_one']}  (over-merge, any overlap)",
        f"near misses         {report['near_miss_count']}  (unmatched GT with IoU 0.4-0.5)",
        f"  headroom          {report['near_miss_headroom_pq']:+.4f} PQ if all crossed 0.5",
        f"median area ratio   {report['median_area_ratio']:.3f}  (pred/GT; >1 = we over-segment)",
    ]
    return "\n".join(lines)

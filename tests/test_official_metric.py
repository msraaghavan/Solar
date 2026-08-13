"""Cross-check ``metrics.evaluate`` against the host's published scorer.

The organisers released a self-evaluation notebook on 9 Aug 2026 that defines
the leaderboard metric exactly:

    https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Everything downstream - the tuned thresholds, the checkpoint selection, the
reported CV - optimises whatever PQ *we* compute, so a definitional
disagreement with that notebook would silently invalidate all of it.  This
module re-derives the host's algorithm from its stated rules and asserts our
implementation agrees on cases chosen to separate the two.

The reference below is a reimplementation, not a copy: the host's code is
theirs, and this repository is MIT.  The rules it encodes, taken from the
notebook, are:

1. Ground truth is keyed ``"<annotator>-<image>_<n>"``; predictions are keyed
   ``"<image>_<n>"``.  The scorer iterates over *ground-truth* annotator-image
   entries and, for each, looks up predictions by image alone.  An observation
   read by three annotators is therefore scored three times against one and the
   same prediction set - which is why the pipeline predicts once per
   observation and never duplicates per annotator.
2. A pair is a true positive when ``IoU > 0.5``, strictly.  Every qualifying
   pair counts; there is no one-to-one assignment step.
3. A prediction overlapping no ground truth above the threshold is a false
   positive; a ground-truth segment overlapping no prediction is a false
   negative.
4. Counts are pooled over every entry and divided once - micro-averaging, not a
   mean of per-image scores.
5. Predictions for an image with no ground-truth entry are never looked at, so
   they are not penalised.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pycocotools import mask as mask_utils  # noqa: E402

from metrics import evaluate, evaluate_image  # noqa: E402

PASSED: list[str] = []
SIZE = 256


def check(name: str):
    def wrap(fn):
        fn()
        print(f"ok    {name}")
        PASSED.append(name)
        return fn

    return wrap


def reference_pq(gt: dict[str, list[dict]], pred: dict[str, list[dict]]) -> float:
    """The host's rules, applied directly. Keys are as described in the module docstring."""
    tp_ious: list[float] = []
    fp = 0
    fn = 0

    for annotator_image, gt_rles in gt.items():
        image = annotator_image.split("-", maxsplit=1)[1]
        pred_rles = pred.get(image, [])
        n_gt, n_pred = len(gt_rles), len(pred_rles)

        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue

        ious = np.zeros((n_gt, n_pred))
        for i, g in enumerate(gt_rles):
            for j, p in enumerate(pred_rles):
                inter = mask_utils.area(mask_utils.merge([g, p], intersect=True))
                union = mask_utils.area(mask_utils.merge([g, p], intersect=False))
                ious[i, j] = inter / union if union else 0.0

        hit = ious > 0.5
        tp_ious.extend(ious[hit].tolist())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())

    denominator = len(tp_ious) + 0.5 * fp + 0.5 * fn
    return sum(tp_ious) / denominator if denominator > 0 else 0.0


def blob(y0: int, y1: int, x0: int, x1: int) -> dict:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask_utils.encode(np.asfortranarray(mask))


def ours(gt: dict[str, list[dict]], pred: dict[str, list[dict]]) -> float:
    results = [
        evaluate_image(key, gt_rles, pred.get(key.split("-", maxsplit=1)[1], []))
        for key, gt_rles in gt.items()
    ]
    return evaluate(results)["pq_micro"]


@check("our PQ equals the host's on a mixed set of hits, misses and spurious blobs")
def _():
    gt = {
        "010401-imageA": [blob(10, 40, 10, 60), blob(100, 140, 100, 160)],
        "010401-imageB": [blob(10, 40, 10, 60)],
        "010401-imageC": [blob(10, 40, 10, 60)],
    }
    pred = {
        # near-perfect hit, plus a slightly shrunk hit, plus a spurious blob
        "imageA": [blob(10, 40, 10, 58), blob(102, 140, 100, 160), blob(200, 220, 200, 220)],
        # a miss: overlaps but well under 0.5 IoU
        "imageB": [blob(35, 60, 10, 60)],
        # imageC gets no predictions at all
    }
    assert abs(ours(gt, pred) - reference_pq(gt, pred)) < 1e-9


@check("our PQ equals the host's when one observation is read by three annotators")
def _():
    # The same prediction set is scored once per annotator.  Annotators disagree,
    # so the same prediction is a hit for one and a miss for another.
    gt = {
        "010401-imageA": [blob(10, 40, 10, 60)],
        "010402-imageA": [blob(12, 42, 10, 60)],
        "010403-imageA": [blob(60, 90, 10, 60)],
    }
    pred = {"imageA": [blob(10, 40, 10, 60)]}
    assert abs(ours(gt, pred) - reference_pq(gt, pred)) < 1e-9


@check("our PQ equals the host's when an image has no ground truth or no prediction")
def _():
    gt = {"010401-imageA": [], "010401-imageB": [blob(10, 40, 10, 60)]}
    pred = {"imageA": [blob(10, 40, 10, 60)], "imageB": []}
    assert abs(ours(gt, pred) - reference_pq(gt, pred)) < 1e-9


@check("predictions on an image absent from the ground truth are not penalised")
def _():
    # The host's loop is driven by ground-truth entries, so an unmatched image
    # never contributes a false positive.  Ours must not invent one.
    gt = {"010401-imageA": [blob(10, 40, 10, 60)]}
    pred = {"imageA": [blob(10, 40, 10, 60)], "imageZ": [blob(10, 40, 10, 60)]}
    assert abs(ours(gt, pred) - reference_pq(gt, pred)) < 1e-9
    assert abs(ours(gt, pred) - 1.0) < 1e-9


@check("near-miss headroom counts only unmatched ground truth just under the cliff")
def _():
    from metrics import evaluate, evaluate_image, to_rle

    # 100x100 GT. A prediction covering 60x100 of it scores IoU 0.6 -> matched.
    # One covering 45x100 scores 0.45 -> unmatched, and sits in the 0.4-0.5 band.
    # One nowhere near anything -> unmatched, but contributes no headroom.
    gt = [blob(0, 100, 0, 100), blob(0, 100, 120, 220), blob(150, 250, 0, 100)]
    pred = [blob(0, 60, 0, 100), blob(0, 45, 120, 220)]
    result = evaluate_image("010401-imageA", gt, pred)

    assert result.tp == 1 and result.fn == 2
    assert sorted(round(v, 2) for v in result.near_miss_ious) == [0.0, 0.45]

    report = evaluate([result])
    assert report["near_miss_count"] == 1, "only the 0.45 pair is recoverable"
    # denominator = 1 TP + 0.5*1 FP + 0.5*2 FN = 2.5; headroom = 0.45 / 2.5
    assert abs(report["near_miss_headroom_pq"] - 0.18) < 1e-6

    # and promoting it really is denominator-neutral: assert that directly
    promoted = evaluate_image("010401-imageA", gt, [blob(0, 60, 0, 100), blob(0, 90, 120, 220)])
    before = report["tp"] + 0.5 * report["fp"] + 0.5 * report["fn"]
    after = promoted.tp + 0.5 * promoted.fp + 0.5 * promoted.fn
    assert before == after, "crossing the cliff must not change the denominator"
    assert promoted.iou_sum > result.iou_sum, "but it must raise the numerator"


@check("fragmentation counts use the host's any-overlap rule, not a stricter one")
def _():
    # The rubric scores one-to-many and many-to-one directly, and the host's
    # notebook builds its matrix as `iou_matrix > 0` - any overlap at all.  A
    # stricter threshold flatters us: a prediction clipping a neighbour by a few
    # pixels is a real over-merge to the judges and invisible at 0.1.
    from metrics import fragmentation_counts, iou_matrix, to_rle

    gt = [to_rle(m) for m in (blob(10, 50, 10, 60), blob(52, 90, 10, 60))]
    # one prediction spanning both segments, overlapping the second only barely
    pred = [to_rle(blob(10, 54, 10, 60))]
    ious = iou_matrix(gt, pred)
    assert 0 < ious[1, 0] < 0.1, "fixture needs a sliver overlap to be meaningful"

    one_to_many, many_to_one = fragmentation_counts(ious)
    assert many_to_one == 1, "the host counts this prediction as an over-merge"
    assert fragmentation_counts(ious, overlap_threshold=0.1)[1] == 0, (
        "and the old 0.1 default is exactly what used to hide it"
    )


@check("greedy one-to-one and the host's all-pairs rule agree on disjoint masks")
def _():
    # Our matcher assigns at most one prediction per ground-truth segment; the
    # host sums every pair above 0.5.  These coincide whenever masks are
    # disjoint, because two disjoint predictions cannot each cover more than
    # half of one segment.  Connected components are disjoint by construction,
    # so the equivalence holds for everything this pipeline emits - but it is
    # asserted on a deliberately crowded case rather than assumed.
    rng = np.random.default_rng(0)
    for _trial in range(25):
        gt_rles, pred_rles = [], []
        for k in range(4):
            y = 10 + 60 * k
            gt_rles.append(blob(y, y + 40, 10, 70))
            shift = int(rng.integers(-14, 15))
            width = int(rng.integers(40, 80))
            pred_rles.append(blob(y + shift, y + 40 + shift, 10, 10 + width))
        gt = {"010401-imageA": gt_rles}
        pred = {"imageA": pred_rles}
        assert abs(ours(gt, pred) - reference_pq(gt, pred)) < 1e-9


if __name__ == "__main__":
    print(f"\n{len(PASSED)} checks passed")

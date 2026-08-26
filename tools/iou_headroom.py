"""What a uniform change in mask quality would be worth, from a stored OOF run.

The metric matches at **IoU > 0.5 strictly**, so mask quality does not trade off
smoothly against score: a predicted instance at IoU 0.49 is not "almost a hit",
it is scored as a false positive *and* a false negative, costing 1.0 in the
denominator and contributing nothing to the numerator.  That cliff means the
value of better boundaries is concentrated entirely in whatever sits just below
it, and the only way to know how much is there is to look at the distribution.

    python tools/iou_headroom.py kernels/_runs/out_ooffull/oof_tuned.json

Reads ``iou_distribution`` (the IoU of every matched pair) and
``near_miss_count`` (unmatched ground truth in 0.4-0.5) from the artefact, and
reports what a uniform shift in IoU would do to PQ.
"""

from __future__ import annotations

import json
import sys


def pq(tp_iou_sum: float, tp: int, fp: int, fn: int) -> float:
    denominator = tp + 0.5 * fp + 0.5 * fn
    return tp_iou_sum / denominator if denominator else 0.0


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "kernels/_runs/out_ooffull/oof_tuned.json"
    data = json.load(open(path))
    report = data["report"]
    ious = sorted(data["iou_distribution"])
    tp, fp, fn = report["tp"], report["fp"], report["fn"]
    # The artefact stores the individual near-miss IoUs, so their conversion can
    # be computed exactly rather than assumed uniform over the 0.4-0.5 band.
    near_ious = sorted(data.get("near_miss_distribution", []))
    near = report.get("near_miss_count", len(near_ious))

    print(f"{path}")
    print(f"  matched pairs {len(ious)} (tp {tp})   fp {fp}   fn {fn}")
    base = pq(sum(ious), tp, fp, fn)
    print(f"  PQ recomputed from the distribution: {base:.4f} "
          f"(reported {report['pq_micro']:.4f})\n")

    print("  matched-IoU distribution")
    edges = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
    lo = 0.5
    for hi in edges[1:]:
        n = sum(1 for v in ious if lo <= v < hi)
        bar = "#" * round(60 * n / max(len(ious), 1))
        print(f"    {lo:.2f}-{hi if hi <= 1 else 1.0:.2f}  {n:4d}  {bar}")
        lo = hi

    fragile = sum(1 for v in ious if v < 0.55)
    print(f"\n  {fragile} matched pairs ({fragile / len(ious):.1%}) sit below IoU 0.55 -")
    print("  they are one small regression away from becoming FP *and* FN.\n")

    print("  effect of a uniform shift in every instance's IoU")
    print("  (near misses cross the 0.5 cliff and convert FN->TP, removing one FP each)")
    print(f"    {'shift':>7}{'PQ':>9}{'delta':>9}   note")
    for shift in (-0.05, -0.02, 0.0, 0.02, 0.05, 0.10):
        moved = [min(v + shift, 1.0) for v in ious]
        # Matched pairs that fall below the cliff stop being matches at all.
        kept = [v for v in moved if v > 0.5]
        lost = len(moved) - len(kept)
        # Near misses are spread over 0.4-0.5; a shift of s converts the
        # fraction of them that clears 0.5.  Uniform is the honest assumption
        # without the individual values, which the artefact does not store.
        gained = [min(v + shift, 1.0) for v in near_ious if v + shift > 0.5]
        converted = len(gained)
        new_tp = len(kept) + converted
        new_fp = fp - converted + lost
        new_fn = fn - converted + lost
        value = pq(sum(kept) + sum(gained), new_tp, new_fp, new_fn)
        note = f"{converted} near misses convert" if converted else (
            f"{lost} matches fall below the cliff" if lost else "")
        print(f"    {shift:>+7.2f}{value:>9.4f}{value - base:>+9.4f}   {note}")

    print(f"\n  ceiling if every near miss converted at IoU 0.6: ", end="")
    conv = [0.6] * near
    print(f"{pq(sum(ious) + sum(conv), tp + near, fp - near, fn - near):.4f}")


if __name__ == "__main__":
    main()

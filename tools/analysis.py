"""Reproduce every measurement the pipeline design rests on.

Each function answers one design question with a number rather than an opinion,
and the results are written to ``artifacts/analysis.json`` so the technical
report quotes measurements that a reader can regenerate:

1. What does Panoptic Quality look like *between human annotators*?  This is the
   effective ceiling and it calibrates what a good score even is.
2. Does averaging annotators beat any single annotator?  Sets the segmentation-
   quality ceiling.
3. Is semantic segmentation plus connected components enough, or is a learned
   instance head needed?
4. How much mask fidelity does working at reduced resolution cost?
5. Does limb-darkening correction actually improve filament separability, and
   where?

Run with ``python tools/analysis.py`` from the repository root.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pycocotools import mask as mask_utils  # noqa: E402

from metrics import evaluate, evaluate_image, iou_matrix  # noqa: E402
from preprocess import (  # noqa: E402
    detect_disk,
    flat_field,
    from_canonical_mask,
    to_canonical_mask,
)

SIZE = 2048


def _load(root: str):
    path = os.path.join(root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    with open(path) as fh:
        coco = json.load(fh)
    by_image = collections.defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    by_file = collections.defaultdict(list)
    for image in coco["images"]:
        by_file[image["file_name"]].append(image["id"])
    return coco, by_image, by_file


def _rles(by_image, image_id) -> list[dict]:
    return [
        mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], SIZE, SIZE))
        for a in by_image[image_id]
    ]


def inter_annotator(by_image, by_file) -> dict:
    """PQ between independent readings of the same observation."""
    results = []
    for ids in by_file.values():
        if len(ids) < 2:
            continue
        for a, b in itertools.combinations(sorted(ids), 2):
            results.append(evaluate_image(f"{a}|{b}", _rles(by_image, a), _rles(by_image, b)))
    report = evaluate(results)
    return {
        "pairs": len(results),
        "pq": report["pq_micro"],
        "sq": report["sq"],
        "rq": report["rq"],
        "hit_rate": report["hit_rate"],
        "mean_matched_iou": report["mean_matched_iou"],
    }


def consensus_shape(by_image, by_file, limit: int = 151) -> dict:
    """Does the union of two readings predict a third better than one does?"""
    triples = [(f, sorted(ids)) for f, ids in by_file.items() if len(ids) == 3][:limit]
    single, union = [], []

    def iou(a, b):
        inter = np.logical_and(a, b).sum()
        uni = np.logical_or(a, b).sum()
        return inter / uni if uni else 1.0

    for _, ids in triples:
        readings = {i: _rles(by_image, i) for i in ids}
        for held in ids:
            others = [i for i in ids if i != held]
            A, B, H = readings[others[0]], readings[others[1]], readings[held]
            if not (A and B and H):
                continue
            ab = iou_matrix(A, B)
            for ai in range(len(A)):
                if ab.shape[1] == 0:
                    continue
                bi = int(np.argmax(ab[ai]))
                if ab[ai, bi] <= 0.5:
                    continue
                ah = iou_matrix([A[ai]], H)[0]
                hi = int(np.argmax(ah))
                if ah[hi] <= 0.5:
                    continue
                ma = mask_utils.decode(A[ai]).astype(bool)
                mb = mask_utils.decode(B[bi]).astype(bool)
                mh = mask_utils.decode(H[hi]).astype(bool)
                single.append(iou(ma, mh))
                union.append(iou(np.logical_or(ma, mb), mh))
    return {
        "matched_triples": len(single),
        "single_annotator_iou": float(np.mean(single)) if single else 0.0,
        "union_of_two_iou": float(np.mean(union)) if union else 0.0,
    }


def cc_oracle(by_image, limit: int = 300) -> dict:
    """Ceiling of semantic segmentation + connected components."""
    results = []
    for image_id in sorted(by_image)[:limit]:
        gt = _rles(by_image, image_id)
        union = np.zeros((SIZE, SIZE), dtype=np.uint8)
        for rle in gt:
            union |= mask_utils.decode(rle).astype(np.uint8)
        n, labels = cv2.connectedComponents(union, 8)
        preds = [
            mask_utils.encode(np.asfortranarray((labels == k).astype(np.uint8)))
            for k in range(1, n)
        ]
        results.append(evaluate_image(image_id, gt, preds))
    report = evaluate(results)
    return {
        "images": min(limit, len(by_image)),
        "pq": report["pq_micro"],
        "false_positives": report["fp"],
        "false_negatives": report["fn"],
        "n_gt": report["n_gt"],
    }


def resolution_cost(root, by_image, by_file, sizes=(768, 1024, 1536), limit: int = 8) -> dict:
    """Mask fidelity lost by round-tripping through a downsampled frame."""
    out = {}
    files = sorted(by_file)[:limit]
    for size in sizes:
        ious = []
        for file_name in files:
            image = cv2.imread(
                os.path.join(root, "train", "train_images", file_name), cv2.IMREAD_GRAYSCALE
            )
            disk = detect_disk(image)
            for rle in _rles(by_image, by_file[file_name][0]):
                m = mask_utils.decode(rle).astype(np.uint8)
                back = from_canonical_mask(to_canonical_mask(m, disk, size), disk, SIZE)
                inter = np.logical_and(m, back).sum()
                uni = np.logical_or(m, back).sum()
                ious.append(inter / uni if uni else 1.0)
        out[str(size)] = {"mean_iou": float(np.mean(ious)), "min_iou": float(np.min(ious))}
    return out


def flat_field_gain(root, by_image, by_file, limit: int = 40) -> dict:
    """Filament-vs-background separability by radial band, raw vs flat-fielded."""
    bands = [(0.0, 0.4), (0.4, 0.7), (0.7, 0.85), (0.85, 0.95)]
    acc = {b: {"raw": [], "flat": []} for b in bands}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

    for file_name in sorted(by_file)[:limit]:
        image = cv2.imread(
            os.path.join(root, "train", "train_images", file_name), cv2.IMREAD_GRAYSCALE
        )
        disk = detect_disk(image)
        flat = flat_field(image, disk)
        radius = disk.radius_map(SIZE)

        filament = np.zeros((SIZE, SIZE), dtype=bool)
        for rle in _rles(by_image, by_file[file_name][0]):
            filament |= mask_utils.decode(rle).astype(bool)
        if not filament.any():
            continue
        ring = cv2.dilate(filament.astype(np.uint8), kernel).astype(bool) & ~filament

        for band in bands:
            sel = (radius >= band[0]) & (radius < band[1])
            fg, bg = filament & sel, ring & sel
            if fg.sum() < 50 or bg.sum() < 50:
                continue
            for key, plane in (("raw", image.astype(np.float32)), ("flat", flat)):
                a, b = plane[fg], plane[bg]
                spread = np.sqrt((a.var() + b.var()) / 2) + 1e-6
                acc[band][key].append((b.mean() - a.mean()) / spread)

    return {
        f"{lo:.2f}-{hi:.2f}": {
            "raw": float(np.mean(v["raw"])) if v["raw"] else None,
            "flat_fielded": float(np.mean(v["flat"])) if v["flat"] else None,
        }
        for (lo, hi), v in acc.items()
    }


def corpus_stats(coco, by_image, by_file) -> dict:
    areas = np.array([a["area"] for a in coco["annotations"]])
    per_image = np.array([len(v) for v in by_image.values()])
    return {
        "observations": len(by_file),
        "annotations_readings": len(coco["images"]),
        "filaments": len(coco["annotations"]),
        "annotators_per_observation": dict(
            sorted(collections.Counter(len(v) for v in by_file.values()).items())
        ),
        "filaments_per_reading_mean": float(per_image.mean()),
        "filament_area_median_px": float(np.median(areas)),
        "filament_area_p1_px": float(np.percentile(areas, 1)),
        "positive_pixel_fraction": float(areas.sum() / len(coco["images"]) / (SIZE * SIZE)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--out", default="artifacts/analysis.json")
    parser.add_argument("--quick", action="store_true", help="smaller subsets")
    args = parser.parse_args()

    coco, by_image, by_file = _load(args.data_root)
    scale = 0.3 if args.quick else 1.0

    results = {}
    print("corpus statistics ...", flush=True)
    results["corpus"] = corpus_stats(coco, by_image, by_file)
    print("inter-annotator agreement ...", flush=True)
    results["inter_annotator"] = inter_annotator(by_image, by_file)
    print("consensus shape ...", flush=True)
    results["consensus_shape"] = consensus_shape(by_image, by_file, int(151 * scale))
    print("connected-components oracle ...", flush=True)
    results["cc_oracle"] = cc_oracle(by_image, int(300 * scale))
    print("resolution cost ...", flush=True)
    results["resolution_cost"] = resolution_cost(args.data_root, by_image, by_file)
    print("flat-field gain ...", flush=True)
    results["flat_field_gain"] = flat_field_gain(args.data_root, by_image, by_file, int(40 * scale))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Fit the solar disk for every image once and cache the result.

Disk fitting costs ~0.08 s per image; caching keeps it out of the training loop
and gives a single place to sanity-check the geometry across the whole corpus.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

from preprocess import detect_disk

DEFAULT_ROOT = "data/MAGFiLO_1.0_Kaggle_2026"


def build(root: str = DEFAULT_ROOT, out: str = "artifacts/disk_cache.json") -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cache: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    paths = sorted(glob.glob(f"{root}/train/train_images/*.jpeg")) + sorted(
        glob.glob(f"{root}/test/test_images/*.jpeg")
    )
    for i, path in enumerate(paths):
        name = os.path.basename(path)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            failures.append(f"{name}: unreadable or wrong shape")
            continue
        try:
            disk = detect_disk(image)
        except Exception as exc:  # noqa: BLE001 - want the name with the error
            failures.append(f"{name}: {exc}")
            continue
        cache[name] = {"cx": disk.cx, "cy": disk.cy, "r": disk.r}
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    with open(out, "w") as fh:
        json.dump(cache, fh)

    radii = np.array([v["r"] for v in cache.values()])
    cxs = np.array([v["cx"] for v in cache.values()])
    cys = np.array([v["cy"] for v in cache.values()])
    print(f"fitted {len(cache)}/{len(paths)} images -> {out}")
    print(
        f"r   mean {radii.mean():7.2f}  std {radii.std():6.2f}  "
        f"min {radii.min():7.2f}  max {radii.max():7.2f}"
    )
    print(f"cx  mean {cxs.mean():7.2f}  std {cxs.std():6.2f}  min {cxs.min():7.2f}  max {cxs.max():7.2f}")
    print(f"cy  mean {cys.mean():7.2f}  std {cys.std():6.2f}  min {cys.min():7.2f}  max {cys.max():7.2f}")

    # A disk that runs outside the frame, or a wildly off-median radius, means
    # the fit latched onto something other than the Sun.
    suspicious = [
        name
        for name, v in cache.items()
        if abs(v["r"] - np.median(radii)) > 6 * radii.std()
        or v["cx"] - v["r"] < -50
        or v["cx"] + v["r"] > 2098
        or v["cy"] - v["r"] < -50
        or v["cy"] + v["r"] > 2098
    ]
    print(f"suspicious fits: {len(suspicious)} {suspicious[:10]}")
    print(f"failures: {len(failures)} {failures[:10]}")
    if failures:
        raise RuntimeError(f"{len(failures)} images failed disk fitting")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default="artifacts/disk_cache.json")
    args = parser.parse_args()
    build(args.root, args.out)


if __name__ == "__main__":
    main()

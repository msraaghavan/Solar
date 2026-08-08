"""Build the competition CSV from predicted instances.

The expected format is one row per predicted filament:

    filament_id,segmentation_rle
    20150125172714Mh_1,<counts>

``filament_id`` is the observation id (the file name without extension) with a
uniquifying suffix; only the prefix is meaningful to the scorer, which matches
predictions to ground truth by overlap rather than by index.  ``segmentation_rle``
is the *counts* field of a COCO RLE at a fixed 2048x2048 size, with no size
header and no surrounding quotes.

COCO's counts encoding emits only bytes in the range 48..111 ('0'-'o'), so the
payload can never contain a comma or a double quote.  That is asserted rather
than assumed, because a stray delimiter would corrupt every subsequent row.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Sequence

VALID_RLE_BYTES = set(range(48, 112))


def rle_to_counts(rle: dict) -> str:
    """Extract the counts string, validating that it is CSV-safe."""
    counts = rle["counts"]
    if isinstance(counts, bytes):
        payload = counts
        text = counts.decode("ascii")
    else:
        text = counts
        payload = counts.encode("ascii")

    bad = sorted(set(payload) - VALID_RLE_BYTES)
    if bad:
        raise ValueError(
            f"RLE counts contain unexpected bytes {bad!r}; refusing to write CSV"
        )
    return text


def write_submission(
    predictions: Iterable[tuple[str, Sequence[dict]]],
    path: str,
) -> int:
    """Write ``predictions`` (observation id -> instance RLEs) to ``path``.

    Returns the number of rows written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = 0
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["filament_id", "segmentation_rle"])
        for image_id, instances in predictions:
            for index, rle in enumerate(instances, start=1):
                writer.writerow([f"{image_id}_{index}", rle_to_counts(rle)])
                rows += 1
    return rows


def validate_submission(path: str, expected_images: Sequence[str]) -> dict:
    """Re-read a written submission and check it against the test manifest.

    Catches the failure modes that silently score zero: ids that do not match a
    test observation, duplicated ids, masks that decode to nothing, and images
    for which no instance was emitted at all.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    expected = set(expected_images)
    seen_ids: set[str] = set()
    per_image: dict[str, int] = {name: 0 for name in expected}
    empty_masks = 0
    unknown: list[str] = []

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"unexpected header: {header}")
        for filament_id, counts in reader:
            if filament_id in seen_ids:
                raise ValueError(f"duplicate filament_id: {filament_id}")
            seen_ids.add(filament_id)

            image_id = filament_id.rsplit("_", 1)[0]
            if image_id not in expected:
                unknown.append(filament_id)
                continue
            per_image[image_id] += 1

            area = mask_utils.area(
                {"size": [2048, 2048], "counts": counts.encode("ascii")}
            )
            if float(area) == 0.0:
                empty_masks += 1

    areas = np.array(list(per_image.values()))
    return {
        "rows": len(seen_ids),
        "images_covered": int((areas > 0).sum()),
        "images_expected": len(expected),
        "images_without_predictions": sorted(
            name for name, count in per_image.items() if count == 0
        ),
        "unknown_ids": unknown,
        "empty_masks": empty_masks,
        "instances_per_image_mean": float(areas.mean()) if areas.size else 0.0,
        "instances_per_image_max": int(areas.max()) if areas.size else 0,
    }

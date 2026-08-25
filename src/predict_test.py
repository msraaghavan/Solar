"""Predict the test set and write the competition CSV.

Fold models are ensembled by averaging probability maps before instances are
extracted, rather than by merging each model's instances.  Averaging first is
the right order for Panoptic Quality: it lets a filament that only some folds
find survive at reduced confidence and be judged once by the tuned threshold,
whereas merging instances afterwards would either duplicate it (false positives)
or require an arbitrary voting rule.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

from data import ImageContext, load_contexts
from infer import AMP_CHOICES, disk_mask_for, predict_full
from model import FilamentNet
from postprocess import PostprocessConfig, extract_instances
from preprocess import Disk, detect_disk
from submit import validate_submission, write_submission


def load_models(paths: list[str], device: str) -> list[torch.nn.Module]:
    models = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = FilamentNet(
            encoder_name=checkpoint["args"]["encoder"],
            pretrained=False,
            out_channels=checkpoint["model"]["head.weight"].shape[0],
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
        print(
            f"  {os.path.basename(path)}  encoder={checkpoint['args']['encoder']}  "
            f"epoch={checkpoint.get('epoch')}  val PQ={checkpoint.get('pq')}",
            flush=True,
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config", default=None, help="tuned PostprocessConfig JSON")
    parser.add_argument("--tta", type=int, default=4)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=AMP_CHOICES, default="auto",
                        help="autocast precision; 'auto' picks bf16 on Ampere and later")
    args = parser.parse_args()

    test_dir = os.path.join(args.data_root, "test", "test_images")
    files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")))
    if not files:
        raise SystemExit(f"no test images under {test_dir}")
    print(f"{len(files)} test images", flush=True)

    if args.config:
        with open(args.config) as fh:
            payload = json.load(fh)
        config = PostprocessConfig(**payload.get("config", payload))
    else:
        config = PostprocessConfig()
    print("post-processing:", config.as_dict(), flush=True)

    print("models:", flush=True)
    models = load_models(args.checkpoints, args.device)

    predictions = []
    counts = []
    t0 = time.time()
    for i, path in enumerate(files):
        name = os.path.basename(path)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise RuntimeError(f"unexpected test image {name}: {None if image is None else image.shape}")

        # Test geometry is fitted here rather than read from a cache, so the
        # inference path stands alone and works on unseen observations.
        context = ImageContext.build(image, detect_disk(image))

        probability = np.zeros((2048, 2048), dtype=np.float32)
        for model in models:
            probability += predict_full(
                model, image, context, tta=args.tta, device=args.device,
                amp_dtype=args.amp_dtype
            )
        probability /= len(models)

        instances = extract_instances(probability, config, disk_mask_for(context))
        predictions.append((os.path.splitext(name)[0], instances))
        counts.append(len(instances))
        if (i + 1) % 20 == 0:
            print(
                f"  {i + 1}/{len(files)}  ({time.time() - t0:.0f}s, "
                f"{np.mean(counts):.1f} instances/image)",
                flush=True,
            )

    rows = write_submission(predictions, args.out)
    print(f"\nwrote {rows} rows to {args.out}", flush=True)

    report = validate_submission(
        args.out, [os.path.splitext(os.path.basename(f))[0] for f in files]
    )
    print(json.dumps({k: v for k, v in report.items() if k != "images_without_predictions"}, indent=2))
    missing = report["images_without_predictions"]
    print(f"images with no predictions: {len(missing)} {missing[:10]}")
    if report["unknown_ids"]:
        raise SystemExit(f"submission contains unknown ids: {report['unknown_ids'][:5]}")
    if report["empty_masks"]:
        raise SystemExit(f"submission contains {report['empty_masks']} empty masks")


if __name__ == "__main__":
    main()

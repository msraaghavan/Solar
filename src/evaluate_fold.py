"""Score a trained fold and fit the post-processing decision parameters.

Run after training, in the same environment, so that the probability maps never
have to leave the machine: a full validation fold at 2048x2048 float32 is about
2.3 GB, which is awkward to ship out of a Kaggle kernel but trivial to hold in
memory for the length of a threshold search.  Maps are kept as uint8, which
quantises probability to 1/255 - far finer than the threshold grid.

The output is a small JSON: the tuned :class:`PostprocessConfig` plus the full
rubric report, which is what the technical report and the inference kernel both
consume.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from data import load_contexts, load_samples, make_folds, stride_split
from infer import AMP_CHOICES, disk_mask_for, predict_full
from metrics import format_report
from model import FilamentNet
from postprocess import PostprocessConfig, marginal_threshold, score_config, tune


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--context-cache", default="artifacts/contexts.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--tta", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--out", default="artifacts/fold0_tuned.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=AMP_CHOICES, default="auto",
                        help="autocast precision; \'auto\' picks bf16 on Ampere and later")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train", "train_images")
    annotations = os.path.join(
        args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )

    samples = load_samples(annotations)
    folds = make_folds(samples, n_folds=args.n_folds)
    val_samples = [s for s in samples if folds[s.file_name] == args.fold]
    contexts = load_contexts(args.context_cache)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = checkpoint["args"]["encoder"]
    # from_checkpoint reads the architecture off the weights, so a spine head
    # (two output channels) or a stem skip is picked up without either being
    # recorded in args - which is what lets checkpoints predating a flag load.
    model = FilamentNet.from_checkpoint(checkpoint, args.device)
    # Infer at the tile size this model trained on, not a fixed 512.
    tile_size, stride = FilamentNet.tiling_for(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (encoder {encoder}, epoch {checkpoint.get('epoch')})", flush=True)

    file_names = sorted({s.file_name for s in val_samples})
    if args.max_files and args.max_files < len(file_names):
        # Stride, never a prefix: names sort by timestamp, so a prefix would
        # restrict evaluation to the earliest observations.
        step = len(file_names) / args.max_files
        file_names = [file_names[int(i * step)] for i in range(args.max_files)]

    # --- inference over the validation fold ---
    t0 = time.time()
    predictions = []
    ground_truth: dict[str, list[dict]] = {}
    for i, file_name in enumerate(file_names):
        image = cv2.imread(os.path.join(train_dir, file_name), cv2.IMREAD_GRAYSCALE)
        context = contexts[file_name]
        probability = predict_full(
            model, image, context, tta=args.tta, device=args.device,
            amp_dtype=args.amp_dtype, tile_size=tile_size, stride=stride,
        )
        mask = disk_mask_for(context)
        # An observation read by three annotators yields three scoring entries
        # that share one probability map.  These are references, not copies -
        # materialising a 16 MB float array per annotator would cost several GB
        # over a fold for no reason.
        for sample in (s for s in val_samples if s.file_name == file_name):
            predictions.append((sample.image_id, probability, mask, file_name))
            ground_truth[sample.image_id] = sample.instances
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(file_names)} images ({time.time() - t0:.0f}s)", flush=True)

    print(
        f"inference done in {time.time() - t0:.0f}s over {len(file_names)} images "
        f"({len(predictions)} annotator readings)",
        flush=True,
    )

    # --- tune the decision parameters against PQ ---
    #
    # Split first.  The tuner searches ~100 configurations, so scoring the winner
    # on the same data it was chosen from reports a selection-biased number.
    # Tuning uses one half and the headline figure comes from the other, which
    # the search never saw.  The split strides rather than cutting the list in
    # two, because file names sort by date.
    #
    # The split is over *observations*, not readings.  Interleaving the reading
    # list instead - which this script used to do - puts the two or three
    # annotator readings of one image on opposite sides of the split: measured
    # over the training set, 59.8% of the "held out" observations were also in
    # the tuning half, so the reported optimism was itself optimistic.
    tune_files, _ = stride_split(file_names)
    tune_set = [p[:3] for p in predictions if p[3] in tune_files]
    held_out = [p[:3] for p in predictions if p[3] not in tune_files]
    print(
        f"\n--- tuning post-processing on {len(tune_set)} readings, "
        f"reporting on {len(held_out)} held out ---",
        flush=True,
    )
    best, in_sample = tune(tune_set, ground_truth)
    report = score_config(held_out, ground_truth, best)

    print("\n--- tuned configuration ---")
    print(json.dumps(best.as_dict(), indent=2))
    print(
        f"\nPQ on the tuning half   {in_sample['pq_micro']:.4f}  (selection-biased)"
    )
    print(f"PQ on the held-out half {report['pq_micro']:.4f}  <- honest estimate")
    print(
        f"selection optimism      {in_sample['pq_micro'] - report['pq_micro']:+.4f}\n"
    )
    print(format_report(report))
    print(
        f"\nmarginal-emission threshold implied by this operating point: "
        f"p > {marginal_threshold(report['pq_micro'], report['sq']):.3f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {
                "fold": args.fold,
                "checkpoint": os.path.basename(args.checkpoint),
                "encoder": encoder,
                "tta": args.tta,
                "config": best.as_dict(),
                "report": {k: v for k, v in report.items() if not isinstance(v, list)},
                "iou_distribution": report["iou_distribution"],
                "dice_distribution": report["dice_distribution"],
            },
            fh,
            indent=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

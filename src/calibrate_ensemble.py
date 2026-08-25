"""Transfer the operating point from single-model maps to the ensemble, unsupervised.

The tuned configuration is fitted out-of-fold, where each observation is
predicted by the one model that did not train on it.  The submission is
predicted by all five averaged.  Those two map families are not on the same
scale: averaging pulls disputed pixels toward the middle and shaves the peaks,
so a threshold fitted on single-model maps admits *less* on ensemble maps.  The
first submission shows the symptom - 6.15 instances per test image against 6.8
per observation out-of-fold, at a fitted ``seed_threshold`` of 0.95.

Re-fitting the threshold on ensemble maps is not available: under k-fold no
training image is unseen by two models, so there is nothing honest to fit on.
But the operating point can be transferred without any ground truth at all.

A threshold is only a way of naming a point in the probability distribution.
Fix instead the *quantity admitted*: find the fraction of on-disk pixels that
the fitted threshold selects under single-model maps, then choose the ensemble
threshold selecting that same fraction.  This uses no labels, so it cannot leak,
and it is invariant to any monotone recalibration between the two families -
which is what averaging mostly is.

The fraction is pooled over pixels from all sampled images rather than averaged
over per-image thresholds, because the metric is micro-averaged too: an
observation with many filaments should carry more weight than an empty one.

Where to measure it (``--on``) changes the answer, and ``test`` is the better
estimator of the two:

``train``   Single-model maps come from the one model that held each image out,
            so they are honest; the ensemble maps are not, because the other
            four models trained on that image and are over-confident on it.
            The ensemble histogram is therefore sharper than it will be at test
            time and the correction comes out too small.

``test``    Neither family has seen a test image, so both histograms are honest.
            Measuring both on the *same* images matters for a second reason: the
            transfer is only supposed to correct the single-to-ensemble shift,
            and drawing the two histograms from different image populations
            would fold in any difference in filament abundance between them -
            correcting away a real difference in the sky rather than an artefact
            of averaging.  Still label-free: it reads test *pixels*, never test
            annotations.
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

from data import ImageContext, load_contexts, load_samples, make_folds
from infer import AMP_CHOICES, disk_mask_for, predict_full
from model import FilamentNet
from postprocess import PostprocessConfig
from preprocess import detect_disk


def load_fold_models(paths: list[str], device: str) -> dict[int, torch.nn.Module]:
    models: dict[int, torch.nn.Module] = {}
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = FilamentNet(
            encoder_name=checkpoint["args"]["encoder"],
            pretrained=False,
            out_channels=checkpoint["model"]["head.weight"].shape[0],
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[checkpoint["args"]["fold"]] = model
    return models


def quantile_of(counts: np.ndarray, total: int, threshold: float) -> float:
    """Fraction of pixels at or above ``threshold``, from a 256-bin histogram."""
    level = int(np.ceil(threshold * 255.0))
    return float(counts[level:].sum() / total) if total else 0.0


def threshold_for(counts: np.ndarray, total: int, fraction: float) -> float:
    """The threshold admitting as close to ``fraction`` of pixels as possible.

    Taking the first bin whose tail falls under the target is wrong: the tail is
    a step function, so whole ranges of thresholds admit exactly the same pixels,
    and the first of them can sit far below the mass it is naming.  On a
    distribution with all its mass at 0 and a spike at 210/255, a 1% target is
    met by every threshold from 1/255 to 210/255 - and 1/255 is a nonsense
    answer to "where does the top 1% begin".

    Pick the closest tail instead, and on a plateau take the strictest threshold
    admitting that mass, which is the canonical representative of the range.
    """
    if total == 0:
        return 1.0
    tail = np.cumsum(counts[::-1])[::-1] / total  # tail[i] = P(p >= i/255)
    distance = np.abs(tail - fraction)
    closest = np.flatnonzero(distance == distance.min())
    return float(closest[-1] / 255.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--context-cache", default="artifacts/contexts.npz")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config", required=True, help="the out-of-fold tuned JSON")
    parser.add_argument("--images", type=int, default=150, help="observations to sample")
    parser.add_argument("--tta", type=int, default=4)
    parser.add_argument(
        "--on",
        choices=("train", "test"),
        default="train",
        help="which images to measure the two histograms on; 'test' is the "
             "better estimator (see the module docstring) and uses no labels",
    )
    parser.add_argument("--out", default="artifacts/ensemble_config.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=AMP_CHOICES, default="auto",
                        help="autocast precision; 'auto' picks bf16 on Ampere and later")
    args = parser.parse_args()

    with open(args.config) as fh:
        fitted = PostprocessConfig(**json.load(fh)["config"])
    print("out-of-fold operating point:", json.dumps(fitted.as_dict()), flush=True)

    models = load_fold_models(args.checkpoints, args.device)
    print(f"{len(models)} models, folds {sorted(models)}", flush=True)

    if args.on == "test":
        image_dir = os.path.join(args.data_root, "test", "test_images")
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(image_dir, "*.jpeg")))
        if not names:
            raise SystemExit(f"no test images under {image_dir}")
        contexts, folds = None, None
    else:
        samples = load_samples(
            os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
        )
        folds = make_folds(samples, n_folds=len(models))
        contexts = load_contexts(args.context_cache)
        image_dir = os.path.join(args.data_root, "train", "train_images")
        names = sorted({s.file_name for s in samples})

    if args.images and args.images < len(names):
        step = len(names) / args.images
        names = [names[int(i * step)] for i in range(args.images)]
    print(f"measuring on {len(names)} {args.on} observations", flush=True)

    single = np.zeros(256, dtype=np.int64)
    ensemble = np.zeros(256, dtype=np.int64)
    single_total = 0
    ensemble_total = 0
    t0 = time.time()
    for i, name in enumerate(names):
        image = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_GRAYSCALE)
        # Test observations have no cached context; fit the geometry here, the
        # same way predict_test.py does, so the two agree pixel for pixel.
        context = contexts[name] if contexts else ImageContext.build(image, detect_disk(image))
        on_disk = disk_mask_for(context).astype(bool)
        n_pixels = int(on_disk.sum())

        maps = {
            fold: predict_full(model, image, context, tta=args.tta,
                                 device=args.device, amp_dtype=args.amp_dtype)
            for fold, model in models.items()
        }

        if args.on == "test":
            # No model has seen a test image, so every one of the five is an
            # honest draw from the single-model family; pool them all rather
            # than picking one arbitrarily.
            for probability in maps.values():
                single += np.bincount(
                    np.round(probability[on_disk] * 255).astype(np.uint8), minlength=256
                )
                single_total += n_pixels
        else:
            single += np.bincount(
                np.round(maps[folds[name]][on_disk] * 255).astype(np.uint8), minlength=256
            )
            single_total += n_pixels

        averaged = np.mean(list(maps.values()), axis=0)
        ensemble += np.bincount(
            np.round(averaged[on_disk] * 255).astype(np.uint8), minlength=256
        )
        ensemble_total += n_pixels
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)

    print(
        f"\nsampled {ensemble_total / 1e9:.2f} G on-disk pixels from "
        f"{len(names)} {args.on} observations\n"
    )

    transferred = {}
    print(f"{'threshold':<18}{'fitted':>9}{'admits':>12}{'ensemble':>10}{'admits':>12}")
    for field in ("seed_threshold", "mask_threshold"):
        value = getattr(fitted, field)
        fraction = quantile_of(single, single_total, value)
        moved = threshold_for(ensemble, ensemble_total, fraction)
        transferred[field] = moved
        print(
            f"{field:<18}{value:>9.3f}{fraction:>12.3e}{moved:>10.3f}"
            f"{quantile_of(ensemble, ensemble_total, moved):>12.3e}"
        )

    # Areas are pixel counts, and the transfer holds admitted pixel mass fixed,
    # so min_area and min_seed_area carry over untouched.  min_seed_fraction is
    # a ratio of two counts that both move with the transfer, so it does too.
    adjusted = PostprocessConfig(**{**fitted.as_dict(), **transferred})
    print("\nensemble operating point:", json.dumps(adjusted.as_dict()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {
                "config": adjusted.as_dict(),
                "derived_from": os.path.basename(args.config),
                "single_model_config": fitted.as_dict(),
                "measured_on": args.on,
                "observations_sampled": len(names),
                "on_disk_pixels": ensemble_total,
                "method": "quantile transfer of admitted pixel mass, unsupervised",
            },
            fh,
            indent=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

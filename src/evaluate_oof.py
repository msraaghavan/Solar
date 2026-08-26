"""Out-of-fold scoring: one honest number and one config for the whole train set.

Per-fold tuning fits a separate configuration on each fold's validation half -
about 115 readings apiece - and the three fitted so far disagree sharply
(``mask_threshold`` 0.35 / 0.40 / 0.50, ``min_area`` 250 / 400 / 600) while
scoring within 0.016 PQ of one another.  That is the signature of a flat optimum
being resolved by noise, and it leaves an unanswered question at submission
time: the test set is predicted by an *ensemble*, so which fold's configuration
should it use?

This script removes the question.  Every training observation is predicted by
the one model that did not train on it, so the pooled map set is honest over the
entire training set at once.  A single configuration is then fitted on half of
it and reported on the other half, giving both a lower-variance operating point
and a CV estimate on 707 images rather than 143.

Two limits are worth stating plainly rather than hiding:

*The maps are single-model, the submission is an ensemble.*  Under k-fold, every
image is unseen by exactly one model, so an honest k-model-ensemble estimate is
unobtainable without a nested holdout - no image exists that all five models
missed.  Averaging probability maps mainly moves disputed pixels toward the
middle, which is a calibration shift, so the configuration fitted here is
applied to the ensemble on the assumption that the shift is small.  ``--diagnose
-ensemble`` measures that assumption by re-fitting on all-five-model maps over
the same images; those maps are leaked (four of five models trained on each) so
its *PQ* is meaningless, but the *configuration it selects* is informative about
which way the operating point moves.

*``dilate_radius`` is pinned to 0.*  Off-disk pixels are zeroed at storage time,
which makes the disk mask redundant everywhere except the post-dilation clamp,
and carrying 707 full-resolution disk masks costs another 3 GB.  All three folds
fitted so far selected 0, so the axis is inert; freezing it buys the memory that
holding every map at once requires.
"""

from __future__ import annotations

import argparse
import collections
import glob
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
from postprocess import (
    TUNING_GRIDS,
    PostprocessConfig,
    extract_instances,
    marginal_threshold,
    score_config,
    tune,
)


def load_fold_models(paths: list[str], device: str) -> dict[int, torch.nn.Module]:
    """Load checkpoints and index them by the fold each one held out."""
    models: dict[int, torch.nn.Module] = {}
    n_folds = set()
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        fold = checkpoint["args"]["fold"]
        n_folds.add(checkpoint["args"]["n_folds"])
        if fold in models:
            raise SystemExit(f"two checkpoints claim fold {fold}; refusing to guess")
        model = FilamentNet.from_checkpoint(checkpoint, device)
        # Infer with the tiling this model trained under; every model in an
        # ensemble is free to differ, so it travels with the model.
        model.tiling = FilamentNet.tiling_for(checkpoint)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[fold] = model
        print(
            f"  fold {fold}  {os.path.basename(path)}  "
            f"encoder={checkpoint['args']['encoder']}  epoch={checkpoint.get('epoch')}  "
            f"val PQ={checkpoint.get('pq'):.4f}",
            flush=True,
        )
    if len(n_folds) != 1:
        raise SystemExit(f"checkpoints disagree about the number of folds: {n_folds}")
    expected = set(range(n_folds.pop()))
    if set(models) != expected:
        raise SystemExit(
            f"out-of-fold prediction needs every fold: have {sorted(models)}, "
            f"need {sorted(expected)}"
        )
    return models


def quantise(probability: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
    """uint8 map with off-disk pixels zeroed.

    Zeroing off-disk rather than carrying the mask separately is exact for every
    threshold the grid contains: the smallest is 0.05, so a zeroed pixel can
    never be admitted, which is precisely what the mask would have enforced.
    """
    quantised = np.round(probability * 255.0).astype(np.uint8)
    quantised[~disk_mask.astype(bool)] = 0
    return quantised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--context-cache", default="artifacts/contexts.npz")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tta", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0, help="stride a subset (pilot)")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--diagnose-ensemble",
        action="store_true",
        help="also fit on all-model maps to measure the calibration shift (leaked PQ)",
    )
    parser.add_argument("--out", default="artifacts/oof_tuned.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=AMP_CHOICES, default="auto",
                        help="autocast precision; \'auto\' picks bf16 on Ampere and later")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train", "train_images")
    annotations = os.path.join(
        args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )

    samples = load_samples(annotations)
    print("models:", flush=True)
    models = load_fold_models(args.checkpoints, args.device)
    folds = make_folds(samples, n_folds=len(models))
    contexts = load_contexts(args.context_cache)

    file_names = sorted({s.file_name for s in samples})
    if args.max_files and args.max_files < len(file_names):
        # Stride, never a prefix: names sort by timestamp, so a prefix would
        # restrict the pilot to the earliest observations.
        step = len(file_names) / args.max_files
        file_names = [file_names[int(i * step)] for i in range(args.max_files)]

    by_fold: dict[int, int] = {}
    for name in file_names:
        by_fold[folds[name]] = by_fold.get(folds[name], 0) + 1
    print(
        f"\n{len(file_names)} observations, out-of-fold counts "
        f"{ {k: by_fold[k] for k in sorted(by_fold)} }",
        flush=True,
    )

    # --- out-of-fold inference -------------------------------------------------
    t0 = time.time()
    maps: dict[str, np.ndarray] = {}
    ensemble_maps: dict[str, np.ndarray] = {}
    for i, file_name in enumerate(file_names):
        image = cv2.imread(os.path.join(train_dir, file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"could not read {file_name}")
        context = contexts[file_name]
        disk = disk_mask_for(context)

        held_out_by = folds[file_name]
        held_out_model = models[held_out_by]
        probability = predict_full(
            held_out_model, image, context, tta=args.tta, device=args.device,
            amp_dtype=args.amp_dtype,
            tile_size=held_out_model.tiling[0], stride=held_out_model.tiling[1],
        )
        maps[file_name] = quantise(probability, disk)

        if args.diagnose_ensemble:
            total = probability.copy()
            for fold, model in models.items():
                if fold == held_out_by:
                    continue
                total += predict_full(
                    model, image, context, tta=args.tta, device=args.device,
                    amp_dtype=args.amp_dtype,
                    tile_size=model.tiling[0], stride=model.tiling[1],
                )
            ensemble_maps[file_name] = quantise(total / len(models), disk)

        if (i + 1) % 25 == 0:
            gb = sum(m.nbytes for m in maps.values()) / 1e9
            gb += sum(m.nbytes for m in ensemble_maps.values()) / 1e9
            print(
                f"  {i + 1}/{len(file_names)} images ({time.time() - t0:.0f}s, "
                f"{gb:.2f} GB held)",
                flush=True,
            )
    print(f"out-of-fold inference done in {time.time() - t0:.0f}s", flush=True)

    ground_truth = {s.image_id: s.instances for s in samples if s.file_name in maps}

    def readings(source: dict[str, np.ndarray], keep: set[str] | None = None):
        # One probability map per observation, shared by reference across the
        # two or three annotator readings of it, exactly as evaluate_fold does:
        # score_config caches instance extraction on the map's identity, so the
        # 2048x2048 connected-components pass happens once per observation
        # rather than once per reading.
        return [
            (s.image_id, source[s.file_name], None)
            for s in samples
            if s.file_name in source and (keep is None or s.file_name in keep)
        ]

    # Off-disk pixels are already zero, so the disk mask is redundant and
    # dilation - the one step that would need it - is pinned off.
    grids = dict(TUNING_GRIDS, dilate_radius=(0,))

    tune_files, report_files = stride_split(file_names)

    def fit(source: dict[str, np.ndarray], label: str) -> tuple[PostprocessConfig, dict, dict]:
        tune_set = readings(source, tune_files)
        held_out = readings(source, report_files)
        print(
            f"\n--- {label}: tuning on {len(tune_set)} readings, "
            f"reporting on {len(held_out)} held out ---",
            flush=True,
        )
        best, in_sample = tune(tune_set, ground_truth, rounds=args.rounds, grids=grids)
        report = score_config(held_out, ground_truth, best)
        print(f"\n--- {label}: fitted configuration ---")
        print(json.dumps(best.as_dict(), indent=2))
        print(f"PQ on the tuning half   {in_sample['pq_micro']:.4f}  (selection-biased)")
        print(f"PQ on the held-out half {report['pq_micro']:.4f}")
        print(
            f"selection optimism      "
            f"{in_sample['pq_micro'] - report['pq_micro']:+.4f}\n"
        )
        return best, report, in_sample

    best, report, _ = fit(maps, "out-of-fold")
    print(format_report(report))

    # No reading in the training set has zero filaments - the minimum is 1 and
    # the mean is 7.1 - so an observation the post-processing empties is a
    # guaranteed loss: every ground-truth segment becomes a false negative and
    # nothing can offset it.  Count them, because the fix (relax until something
    # is emitted) is only worth building if the fitted config actually does it.
    empties = [
        name
        for name, probability in maps.items()
        if not extract_instances(probability, best, None)
    ]
    print(
        f"\nobservations left with no predicted instance: {len(empties)} of "
        f"{len(maps)} ({100 * len(empties) / len(maps):.1f}%)"
    )
    if empties:
        missed = sum(
            len(s.instances) for s in samples if s.file_name in set(empties)
        )
        print(
            f"  they carry {missed} ground-truth segments, all of which become "
            f"false negatives: {sorted(empties)[:5]}"
        )
    print(
        f"\nmarginal-emission threshold implied by this operating point: "
        f"p > {marginal_threshold(report['pq_micro'], report['sq']):.3f}"
    )

    # --- does agreement between annotators change the score? ------------------
    # Every reading is scored separately against the one prediction set for its
    # observation, so an image read by three annotators contributes three times
    # and drags the pooled figure toward whatever the disputed images score.  We
    # cannot see how many annotators read each *test* image, so knowing the gap
    # is what turns a leaderboard number into evidence rather than a surprise.
    reads = collections.Counter(s.file_name for s in samples)
    for label, keep in (
        ("single-annotator", {f for f in report_files if reads[f] == 1}),
        ("multi-annotator ", {f for f in report_files if reads[f] > 1}),
    ):
        subset = readings(maps, keep)
        if subset:
            sub = score_config(subset, ground_truth, best)
            print(
                f"  {label}  {len(keep):3d} observations, {len(subset):4d} readings"
                f"  PQ {sub['pq_micro']:.4f}  SQ {sub['sq']:.3f}  RQ {sub['rq']:.3f}"
            )

    # --- how much does each fold's own configuration cost on pooled maps? ------
    # If the surface is flat, any fold's configuration is nearly as good here and
    # the choice of operating point is not worth agonising over.  If it is not,
    # the pooled fit is doing real work.
    print("\n--- fitted configurations scored on the same held-out OOF half ---", flush=True)
    held_out = readings(maps, report_files)
    sensitivity = {}
    for path in args.checkpoints:
        for candidate in sorted(glob.glob(os.path.join(os.path.dirname(path), "fold*_tuned.json"))):
            name = os.path.basename(candidate)
            if name in sensitivity:
                continue
            config = PostprocessConfig(**json.load(open(candidate))["config"])
            pq = score_config(held_out, ground_truth, config)["pq_micro"]
            sensitivity[name] = pq
            print(f"  {name:24s} PQ {pq:.4f}  ({pq - report['pq_micro']:+.4f})", flush=True)

    payload = {
        "n_folds": len(models),
        "checkpoints": [os.path.basename(p) for p in args.checkpoints],
        "observations": len(file_names),
        "tta": args.tta,
        "config": best.as_dict(),
        "report": {k: v for k, v in report.items() if not isinstance(v, list)},
        "iou_distribution": report["iou_distribution"],
        "dice_distribution": report["dice_distribution"],
        # Where the next PQ is, not just how much we have: the rubric asks for
        # the IoU and Dice distributions anyway, and this one sizes the lever.
        "near_miss_distribution": report["near_miss_distribution"],
        "per_fold_config_pq_on_oof": sensitivity,
    }

    if args.diagnose_ensemble:
        ens_best, ens_report, ens_in = fit(ensemble_maps, "all-model (LEAKED)")
        print(
            "  NOTE: four of five models trained on each of these images, so the "
            "PQ above is meaningless.\n  Only the direction of the configuration "
            "shift is informative.",
            flush=True,
        )
        moved = {
            k: (getattr(best, k), v)
            for k, v in ens_best.as_dict().items()
            if v != getattr(best, k)
        }
        print(f"  configuration shift out-of-fold -> all-model: {moved or 'none'}")
        payload["ensemble_diagnostic"] = {
            "config": ens_best.as_dict(),
            "leaked_pq": ens_report["pq_micro"],
            "shift_from_oof": {k: list(v) for k, v in moved.items()},
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

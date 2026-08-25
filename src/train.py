"""Train one cross-validation fold.

Validation is deliberately expensive: rather than tracking a tile-level Dice
that correlates only loosely with the leaderboard, every few epochs the model
runs full-disk inference on a subset of held-out observations and is scored with
the real Panoptic Quality implementation, including instance extraction.  That
keeps model selection aligned with the thing being optimised.

Where an observation carries several independent annotations, PQ is computed
against each annotator separately and averaged.  A single fused "consensus"
target would flatter the model, since the test ground truth is itself the work
of individual annotators who agree with each other only moderately.
"""

from __future__ import annotations

import argparse
import copy
import math
import json
import os
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader

from data import (
    ImageContext,
    Sample,
    build_contexts,
    load_contexts,
    load_samples,
    make_folds,
    save_contexts,
    spine_alignment,
)
from dataset_torch import FilamentTiles
from infer import AMP_CHOICES, amp_dtype_for, disk_mask_for, predict_full
from losses import FilamentLoss
from metrics import evaluate, evaluate_image, format_report
from model import FilamentNet
from postprocess import PostprocessConfig, extract_instances
from preprocess import Disk


class EMA:
    """Exponential moving average of weights, with warm-up bias correction.

    The shadow starts as a copy of the *randomly initialised* model, so a fixed
    decay of 0.999 leaves 0.999^n of that noise behind after n steps - still 40%
    after a 900-step run, which is enough to pin the output at a constant and
    make early validation read zero.  Ramping the decay in as
    ``min(decay, (1 + n) / (10 + n))`` makes the first updates track the live
    weights almost exactly and converges to ``decay`` once training is underway.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.steps = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.steps += 1
        decay = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for shadow, live in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if shadow.dtype.is_floating_point:
                shadow.mul_(decay).add_(live.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(live)


def validate(
    model: torch.nn.Module,
    samples: list[Sample],
    image_dir: str,
    contexts: dict[str, ImageContext],
    config: PostprocessConfig,
    device: str,
    tta: int = 1,
    max_files: int | None = None,
    amp_dtype: str = "auto",
) -> dict:
    """Full-disk PQ over held-out observations, averaged across annotators."""
    by_file: dict[str, list[Sample]] = {}
    for sample in samples:
        by_file.setdefault(sample.file_name, []).append(sample)

    # File names begin with a timestamp, so sorting them orders by date and
    # taking a prefix would validate exclusively on the earliest observations
    # (all of 2011).  Stride through instead, so the subset spans the full
    # 2011-2022 range and every GONG site.
    file_names = sorted(by_file)
    if max_files is not None and 0 < max_files < len(file_names):
        step = len(file_names) / max_files
        file_names = [file_names[int(i * step)] for i in range(max_files)]

    results = []
    # Probability statistics are logged alongside PQ because a score of zero is
    # otherwise ambiguous: a collapsed model that emits a constant and a model
    # that simply finds nothing above threshold look identical in the metric,
    # but demand completely different fixes.
    inside, outside, peaks, counts = [], [], [], []

    for file_name in file_names:
        image = cv2.imread(os.path.join(image_dir, file_name), cv2.IMREAD_GRAYSCALE)
        context = contexts[file_name]
        probability = predict_full(
            model, image, context, tta=tta, device=device, amp_dtype=amp_dtype
        )
        on_disk = disk_mask_for(context)
        instances = extract_instances(probability, config, on_disk)
        counts.append(len(instances))
        peaks.append(float(probability[on_disk].max()))

        for sample in by_file[file_name]:
            results.append(
                evaluate_image(sample.image_id, sample.instances, instances)
            )
            if sample.instances:
                truth = mask_utils.decode(sample.semantic_rle).astype(bool)
                inside.append(float(probability[truth].mean()))
                outside.append(float(probability[on_disk & ~truth].mean()))

    report = evaluate(results)
    report["diag_prob_inside_gt"] = float(np.mean(inside)) if inside else 0.0
    report["diag_prob_outside_gt"] = float(np.mean(outside)) if outside else 0.0
    report["diag_prob_max"] = float(np.mean(peaks)) if peaks else 0.0
    report["diag_instances_per_image"] = float(np.mean(counts)) if counts else 0.0
    return report


def spine_preflight(dataset, n: int = 40, min_alignment: float = 0.80) -> None:
    """Assert the spine annotation has actually been understood.

    Cheap - it touches no image, only the annotation geometry - and decisive.
    Spine pixels are supposed to lie on their own filament (95.4% do, measured at
    thickness 3), so alignment is a single number that fails loudly for every
    silent way of getting this wrong: a spine wrapped one list deep that
    rasterises to nothing at all, or ``(row, column)`` coordinates handed to
    ``cv2.polylines``, which reads ``(x, y)`` and would draw every spine
    reflected about the disk diagonal - landing it on quiet Sun while still
    producing a perfectly plausible-looking target.
    """
    samples = [s for s in dataset.samples if s.instances]
    if not samples:
        raise RuntimeError("preflight: no annotated samples to check the spine against")
    step = max(len(samples) // n, 1)
    chosen = samples[::step][:n]

    scores = [spine_alignment(s) for s in chosen]
    empty = sum(1 for inside, _ in scores if inside == 0.0)
    inside = float(np.mean([s[0] for s in scores]))
    covered = float(np.mean([s[1] for s in scores]))
    print(
        f"  spine target: {inside:.1%} of spine pixels inside their filament, "
        f"covering {covered:.1%} of its area ({empty}/{len(chosen)} readings empty)",
        flush=True,
    )
    if inside < min_alignment:
        raise RuntimeError(
            f"preflight: only {inside:.1%} of spine pixels fall inside their own "
            f"filament (expected ~95%).  The spine annotation is being misread - "
            f"check the coordinate order and the nesting of the 'spine' field "
            f"before spending GPU hours training against a wrong target."
        )


def preflight(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    val_samples: list[Sample],
    image_dir: str,
    contexts: dict[str, ImageContext],
    device: str,
    epochs: int,
    amp_dtype: str = "auto",
) -> None:
    """Exercise every path that can fail silently, before committing hours to a run.

    Unit tests run on synthetic fixtures and cannot see the failures that
    actually cost us runs: EfficientNet-B4 overflowing fp16 under autocast, a
    frozen dataloader epoch, a corrupted probability map.  Those need the real
    model, real weights, real half precision and real data - which is exactly
    what this does, once, in about a minute.

    It also times one step and one full-disk inference so the run's total cost is
    known up front rather than discovered three hours later.
    """
    print("--- preflight ---", flush=True)
    dtype = amp_dtype_for(amp_dtype, device)
    print(f"  autocast: {dtype if dtype is not None else 'disabled (fp32)'}", flush=True)

    # 0. The auxiliary spine target, if it is switched on.  This path had never
    # run against real annotations, and every way of misreading the field fails
    # *silently* to an all-zero channel: the head then learns to predict nothing,
    # the loss curve looks perfectly healthy, and hours of GPU time report that
    # the spine "does not help" without ever having tested it.
    if getattr(loader.dataset, "with_spine", False):
        spine_preflight(loader.dataset)

    # 1. One real training step, in the same precision the run will use.
    t0 = time.time()
    features, target, weight = next(iter(loader))
    load_time = time.time() - t0

    if getattr(loader.dataset, "with_spine", False):
        if target.shape[1] != 2:
            raise RuntimeError(
                f"preflight: spine training wants a 2-channel target, got {target.shape[1]}"
            )
        if float(target[:, 1].sum()) <= 0.0:
            raise RuntimeError(
                "preflight: the spine channel is empty across a whole batch of "
                "filament-centred tiles - the target is not reaching the loss"
            )
    features, target, weight = (
        features.to(device), target.to(device), weight.to(device)
    )
    if not torch.isfinite(features).all():
        raise RuntimeError("preflight: non-finite features from the dataloader")

    model.train()
    # Time three steps and keep the fastest.  The first pass on CUDA pays for
    # kernel autotuning and allocator warm-up - it measured 14 s against a true
    # 0.26 s, which turned a 2 h estimate into a nonsensical 106 h.
    step_time = float("inf")
    for _ in range(3):
        t0 = time.time()
        with torch.autocast(
            device_type=device.split(":")[0],
            dtype=dtype or torch.float32,
            enabled=dtype is not None,
        ):
            loss = criterion(model(features), target, weight)
        loss.backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_time = min(step_time, time.time() - t0)
    if not math.isfinite(loss.item()):
        raise RuntimeError(f"preflight: non-finite loss {loss.item()}")

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError("preflight: no gradients reached the parameters")
    if not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError(
            "preflight: non-finite gradients in half precision - this encoder "
            "overflows fp16 and would train with silently skipped steps"
        )
    model.zero_grad(set_to_none=True)

    # 2. One real full-disk inference, which is where the B4 NaN surfaced.
    file_name = sorted({s.file_name for s in val_samples})[0]
    image = cv2.imread(os.path.join(image_dir, file_name), cv2.IMREAD_GRAYSCALE)
    t0 = time.time()
    probability = predict_full(
        model, image, contexts[file_name], tta=1, device=device, amp_dtype=amp_dtype
    )
    infer_time = time.time() - t0
    instances = extract_instances(
        probability, PostprocessConfig(), disk_mask_for(contexts[file_name])
    )

    steps = len(loader)
    epoch_estimate = steps * step_time
    print(
        f"  batch load {load_time:.2f}s | train step {step_time:.3f}s | "
        f"full-disk infer {infer_time:.1f}s",
        flush=True,
    )
    print(
        f"  probability in [{probability.min():.3f}, {probability.max():.3f}], "
        f"{len(instances)} instances on {file_name}",
        flush=True,
    )
    print(
        f"  {steps} steps/epoch -> ~{epoch_estimate / 60:.1f} min/epoch, "
        f"~{epochs * epoch_estimate / 3600:.1f} h for {epochs} epochs",
        flush=True,
    )
    if len(instances) > 200:
        raise RuntimeError(
            f"preflight: {len(instances)} instances from an untrained model suggests "
            "a corrupted probability map"
        )
    print("  preflight OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--disk-cache", default="artifacts/disk_cache.json")
    parser.add_argument("--context-cache", default="artifacts/contexts.npz")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--encoder", default="tf_efficientnet_b4")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tiles-per-sample", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="cap the positive target at 1-eps; counters memorising noisy labels")
    parser.add_argument("--spine-weight", type=float, default=0.0,
                        help="auxiliary spine head weight; 0 disables the head entirely")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--val-files", type=int, default=40)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=0, help="debug: cap steps/epoch")
    parser.add_argument("--amp-dtype", choices=AMP_CHOICES, default="auto",
                        help="autocast precision; 'auto' picks bf16 on Ampere and "
                             "later, fp16 on older cards such as the Kaggle T4")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1234 + args.fold)
    np.random.seed(1234 + args.fold)

    train_dir = os.path.join(args.data_root, "train", "train_images")
    annotations = os.path.join(
        args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )

    samples = load_samples(annotations)
    folds = make_folds(samples, n_folds=args.n_folds)
    train_samples = [s for s in samples if folds[s.file_name] != args.fold]
    val_samples = [s for s in samples if folds[s.file_name] == args.fold]
    print(
        f"fold {args.fold}: {len(train_samples)} train / {len(val_samples)} val samples "
        f"({len({s.file_name for s in train_samples})}/{len({s.file_name for s in val_samples})} files)",
        flush=True,
    )

    if os.path.exists(args.context_cache):
        contexts = load_contexts(args.context_cache)
        print(f"loaded {len(contexts)} contexts from cache", flush=True)
    else:
        with open(args.disk_cache) as fh:
            disk_cache = json.load(fh)
        names = sorted({s.file_name for s in samples})
        t0 = time.time()
        contexts = build_contexts(train_dir, names, disk_cache)
        save_contexts(contexts, args.context_cache)
        print(f"built {len(contexts)} contexts in {time.time() - t0:.0f}s", flush=True)

    dataset = FilamentTiles(
        train_samples,
        train_dir,
        contexts,
        tile_size=args.tile_size,
        tiles_per_sample=args.tiles_per_sample,
        augment=True,
        with_spine=args.spine_weight > 0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        # persistent_workers MUST stay off.  Worker processes fork a copy of the
        # dataset when they start, so with persistence they never observe
        # set_epoch() and every epoch replays byte-identical crops with identical
        # augmentation - which silently turns "30 epochs of fresh samples" into
        # "one epoch repeated 30 times" and drives severe overfitting.  Respawning
        # workers each epoch costs a couple of seconds against a ~230 s epoch.
        persistent_workers=False,
    )

    model = FilamentNet(
        encoder_name=args.encoder,
        pretrained=not args.no_pretrained,
        out_channels=2 if args.spine_weight > 0 else 1,
    ).to(args.device)
    ema = EMA(model)
    criterion = FilamentLoss(
        pos_weight=args.pos_weight,
        dice_weight=args.dice_weight,
        spine_weight=args.spine_weight,
        smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    steps_per_epoch = args.max_steps or len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1,
    )
    # Loss scaling exists to keep fp16 gradients off the bottom of its range.
    # bf16 has fp32's exponent range and needs none of it, and leaving the scaler
    # on there would make the AMP-SKIPPED diagnostic below meaningless - it can
    # only ever report zero, which reads as "no overflow" rather than "not
    # applicable".
    amp_dtype = amp_dtype_for(args.amp_dtype, args.device)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    preflight(
        model, criterion, loader, val_samples, train_dir, contexts, args.device,
        args.epochs, args.amp_dtype,
    )

    best_pq = -1.0
    history = []
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running, seen, skipped, t0 = 0.0, 0, 0, time.time()

        for step, (features, target, weight) in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break
            features = features.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            weight = weight.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=args.device.split(":")[0],
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                loss = criterion(model(features), target, weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # A skipped step is invisible unless you look for it: GradScaler
            # silently drops the update when gradients are non-finite and lowers
            # the loss scale.  EfficientNet-B4 overflows fp16 on a T4, so a run
            # can be badly undertrained while its loss curve looks perfectly
            # healthy - which is exactly what the B4 run may have suffered.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
            scheduler.step()
            ema.update(model)

            # A non-finite loss is not survivable and must not be averaged away
            # into a plausible-looking epoch mean.  Silent corruption is the
            # failure mode that cost us the B4 run.
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch + 1} step {step}: {value}"
                )
            running += value * features.size(0)
            seen += features.size(0)

        message = (
            f"epoch {epoch + 1:3d}/{args.epochs}  loss {running / max(seen, 1):.4f}  "
            f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s"
        )
        if skipped:
            message += f"  AMP-SKIPPED {skipped} steps"

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            report = validate(
                ema.shadow,
                val_samples,
                train_dir,
                contexts,
                PostprocessConfig(),
                args.device,
                tta=1,
                max_files=args.val_files,
                amp_dtype=args.amp_dtype,
            )
            message += (
                f"  |  val PQ {report['pq_micro']:.4f} (SQ {report['sq']:.3f} "
                f"RQ {report['rq']:.3f})  p_in {report['diag_prob_inside_gt']:.3f} "
                f"p_out {report['diag_prob_outside_gt']:.3f} "
                f"p_max {report['diag_prob_max']:.3f} "
                f"inst {report['diag_instances_per_image']:.1f}"
            )
            history.append({"epoch": epoch + 1, **{k: v for k, v in report.items() if not isinstance(v, list)}})
            if report["pq_micro"] > best_pq:
                best_pq = report["pq_micro"]
                torch.save(
                    {
                        "model": ema.shadow.state_dict(),
                        "args": vars(args),
                        "epoch": epoch + 1,
                        "pq": best_pq,
                    },
                    os.path.join(args.out_dir, f"fold{args.fold}_best.pt"),
                )
                message += "  *saved*"

        print(message, flush=True)

    torch.save(
        {"model": ema.shadow.state_dict(), "args": vars(args), "epoch": args.epochs},
        os.path.join(args.out_dir, f"fold{args.fold}_last.pt"),
    )
    with open(os.path.join(args.out_dir, f"fold{args.fold}_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"best val PQ {best_pq:.4f}", flush=True)


if __name__ == "__main__":
    main()

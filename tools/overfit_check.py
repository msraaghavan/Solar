"""Sanity check: can the network memorise a handful of tiles?

This separates two failure modes that look identical from a validation score of
zero - a genuine bug in the model, loss or targets, versus a model that is
simply undertrained.  If the ratio of mean predicted probability inside the
target to outside it climbs well above 1, the machinery is sound and the fix is
more training rather than more debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data import build_contexts, load_samples  # noqa: E402
from dataset_torch import FilamentTiles  # noqa: E402
from losses import FilamentLoss  # noqa: E402
from model import FilamentNet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--disk-cache", default="artifacts/disk_cache.json")
    parser.add_argument("--encoder", default="tf_efficientnet_b0")
    parser.add_argument("--tile", type=int, default=192)
    parser.add_argument("--n-tiles", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train", "train_images")
    with open(args.disk_cache) as fh:
        disk_cache = json.load(fh)

    samples = [
        s
        for s in load_samples(
            os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
        )
        if s.instances
    ][: args.n_tiles]
    contexts = build_contexts(train_dir, sorted({s.file_name for s in samples}), disk_cache)

    dataset = FilamentTiles(
        samples, train_dir, contexts,
        tile_size=args.tile, tiles_per_sample=1, augment=False,
    )
    batch = [dataset[i] for i in range(len(samples))]
    x = torch.stack([b[0] for b in batch]).to(args.device)
    y = torch.stack([b[1] for b in batch]).to(args.device)
    w = torch.stack([b[2] for b in batch]).to(args.device)
    print(f"batch {tuple(x.shape)}  positive fraction {float(y.mean()):.4f}", flush=True)

    torch.manual_seed(0)
    model = FilamentNet(encoder_name=args.encoder, pretrained=False).to(args.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = FilamentLoss(pos_weight=args.pos_weight, dice_weight=args.dice_weight)

    t0 = time.time()
    model.train()
    for step in range(args.steps + 1):
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(x), y, w)
        loss.backward()
        optimiser.step()

        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(x))
            inside = float(p[y > 0.5].mean())
            outside = float(p[(y < 0.5) & (w > 0)].mean())
            print(
                f"step {step:4d}  loss {loss.detach().item():.4f}  "
                f"p_in {inside:.3f}  p_out {outside:.3f}  "
                f"ratio {inside / max(outside, 1e-9):6.1f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            model.train()

    print("\nA ratio far above 1 means architecture, loss and targets are sound.", flush=True)


if __name__ == "__main__":
    main()

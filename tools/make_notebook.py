"""Generate ``notebooks/pipeline.ipynb``, the end-to-end walkthrough.

The organisers require "a jupyter notebook [that] illustrates the entire
pipeline".  It is generated from this script rather than hand-edited so it
cannot drift from ``src/`` and carries no stale execution output into review.
"""

from __future__ import annotations

import itertools
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_COUNTER = itertools.count()


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"md{next(_COUNTER):02d}",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"code{next(_COUNTER):02d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# Solar Filament Segmentation — end-to-end pipeline

This notebook walks the full path from a raw GONG H-alpha observation to a
submission row, and reproduces the measurements that motivated each design
choice. Every step calls the same modules used for the competition runs
(`src/`), so nothing here is a re-implementation.

The competition metric is **Panoptic Quality**, an instance-level score:

$$\\mathrm{PQ} = \\frac{\\sum_{(y,\\hat{y}) \\in TP} \\mathrm{IoU}(y,\\hat{y})}{|TP| + 0.5|FP| + 0.5|FN|}$$

A predicted segment matches a ground-truth segment when their IoU exceeds 0.5.
Splitting one filament into two therefore costs two false positives *and* a
false negative — pixel-overlap intuitions do not transfer.
"""
    ),
    code(
        """
import os, sys, json
sys.path.insert(0, os.path.join('..', 'src'))
import numpy as np, cv2
import matplotlib.pyplot as plt

DATA = '../data/MAGFiLO_1.0_Kaggle_2026'
TRAIN_IMAGES = f'{DATA}/train/train_images'
ANNOTATIONS = f'{DATA}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
"""
    ),
    md(
        """
## 1. The corpus, and why the labels are the hard part

707 observations carry **1154 independent readings**: a third of the images were
annotated by two or three people working separately.
"""
    ),
    code(
        """
from data import load_samples, make_folds
import collections

samples = load_samples(ANNOTATIONS)
by_file = collections.Counter(s.file_name for s in samples)
print(f'{len(samples)} readings of {len(by_file)} observations')
print('annotators per observation:', dict(sorted(collections.Counter(by_file.values()).items())))
print('filaments:', sum(len(s.instances) for s in samples))
"""
    ),
    md(
        """
### Measurement 1 — how well do humans agree with each other?

This is the single most important number in the project: it is the effective
ceiling, and it says what a good score even looks like.
"""
    ),
    code(
        """
import itertools
from metrics import evaluate, evaluate_image, format_report

readings = collections.defaultdict(list)
for s in samples:
    readings[s.file_name].append(s)

pairs = []
for group in readings.values():
    if len(group) < 2:
        continue
    for a, b in itertools.combinations(group, 2):
        pairs.append(evaluate_image(f'{a.image_id}|{b.image_id}', a.instances, b.instances))

print(format_report(evaluate(pairs)))
"""
    ),
    md(
        """
Human annotators agree at **PQ ≈ 0.34**, with a hit rate near 0.52 — one
annotator finds only about half of what another marks. Segmentation quality
(mean IoU of matched pairs) is ≈ 0.63.

Two consequences drive the whole design:

1. A model scoring in the 0.3s is at human level. The public leaderboard's
   scores of 1.00 are an artefact of the test images also appearing, with
   ground truth, in the public MAGFiLO release.
2. The per-pixel target is genuinely stochastic, so the network should be
   trained to predict *P(a randomly drawn annotator marks this pixel)*. Binary
   cross-entropy is a proper scoring rule, so its minimiser is exactly that.
"""
    ),
    md(
        """
### Measurement 2 — is a learned instance head necessary?

Running connected components on the *ground-truth* union recovers the
ground-truth instances almost perfectly, which means annotated filaments
essentially never touch.
"""
    ),
    code(
        """
from pycocotools import mask as mask_utils

results = []
for s in samples[:150]:
    if not s.instances:
        continue
    union = np.zeros((2048, 2048), np.uint8)
    for rle in s.instances:
        union |= mask_utils.decode(rle)
    n, labels = cv2.connectedComponents(union, 8)
    preds = [mask_utils.encode(np.asfortranarray((labels == k).astype(np.uint8))) for k in range(1, n)]
    results.append(evaluate_image(s.image_id, s.instances, preds))

print(f"connected-components oracle PQ = {evaluate(results)['pq_micro']:.4f}")
"""
    ),
    md(
        """
At ≈ 0.997 there is nothing left for a Mask R-CNN or a learned grouping head to
win. **The task reduces to producing one good binary mask**, which is why the
model below is a plain segmentation network.
"""
    ),
    md(
        """
## 2. Preprocessing: disk geometry and limb darkening

Filaments are *dark* features, and the solar limb is also dark — so a naive
dark-region detector fires on the limb annulus. Dividing each pixel by the
median intensity of its radial shell flattens the disk. The median matters:
filaments are dark outliers and a mean profile would partly normalise them away.
"""
    ),
    code(
        """
from preprocess import detect_disk, flat_field

name = sorted(by_file)[0]
image = cv2.imread(f'{TRAIN_IMAGES}/{name}', cv2.IMREAD_GRAYSCALE)
disk = detect_disk(image)
flat = flat_field(image, disk)
print(f'fitted disk: centre ({disk.cx:.1f}, {disk.cy:.1f}), radius {disk.r:.1f}')

truth = np.zeros((2048, 2048), bool)
for s in readings[name]:
    for rle in s.instances:
        truth |= mask_utils.decode(rle).astype(bool)

fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
ax[0].imshow(image, cmap='gray'); ax[0].set_title('raw H-alpha')
ax[1].imshow(flat, cmap='gray', vmin=-0.4, vmax=0.4); ax[1].set_title('limb-corrected contrast')
ax[2].imshow(flat, cmap='gray', vmin=-0.4, vmax=0.4)
ax[2].contour(truth, levels=[0.5], colors='red', linewidths=0.6)
ax[2].set_title('annotated filaments')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
"""
    ),
    md(
        """
## 3. Model input

Three planes, each standardised to roughly unit variance with fixed corpus
constants: limb-corrected contrast, raw intensity (absolute brightness separates
sunspots from filaments), and normalised radius r/R (tells the network where the
limb is). The constants are fixed rather than per-tile — per-tile normalisation
would erase the brightness cue and behave differently at test time.
"""
    ),
    code(
        """
from data import ImageContext

context = ImageContext.build(image, disk)
features = context.tile_features(image, 700, 700, 512)
print('feature planes:', features.shape)
for i, plane in enumerate(['contrast', 'raw intensity', 'radius']):
    print(f'  {plane:14s} mean {features[i].mean():+.3f}  std {features[i].std():.3f}')
"""
    ),
    md(
        """
## 4. Model

A `timm` encoder with a U-Net decoder that runs all the way back to stride 1 —
filament barbs are a few pixels wide, and the usual "decode to stride 2, then
upsample the logits" shortcut erases them. Deep supervision at strides 2 and 4
stabilises the early epochs against a ~0.36% positive rate.
"""
    ),
    code(
        """
import torch
from model import FilamentNet

model = FilamentNet(encoder_name='tf_efficientnet_b0', pretrained=False)
with torch.no_grad():
    model.eval()
    out = model(torch.from_numpy(features)[None])
print('logits:', tuple(out.shape), '| parameters: %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6))
"""
    ),
    md(
        """
Training is one command per fold; see `src/train.py`. Folds are grouped by
observation (so the several readings of one image never straddle a split) and
stratified by GONG site.

```bash
python src/train.py --fold 0 --encoder tf_efficientnet_b4 --epochs 30
```
"""
    ),
    md(
        """
## 5. Inference and instance extraction

Full-disk prediction stitches overlapping tiles under a raised-cosine window —
a seam that splits a filament is expensive under PQ. Then hysteresis
thresholding separates two decisions a single threshold has to conflate: *which*
components to emit (they need a confident core) and *how far* each extends.
"""
    ),
    code(
        """
from postprocess import PostprocessConfig, extract_instances, marginal_threshold

# The operating point is derived, not guessed. Emitting one more instance always
# adds 0.5 to the PQ denominator, and its IoU to the numerator only if it hits,
# so it pays whenever  p * SQ > 0.5 * PQ.
print('break-even hit probability at (PQ=0.34, SQ=0.64): p >',
      round(marginal_threshold(0.34, 0.64), 3))
print(PostprocessConfig())
"""
    ),
    md(
        """
## 6. Submission

One row per predicted filament, carrying the COCO RLE *counts* at a fixed
2048x2048 size. The encoding only emits bytes 48–111, so the payload can never
contain a comma or quote — asserted rather than assumed, since a stray delimiter
would corrupt every following row.
"""
    ),
    code(
        """
from submit import rle_to_counts, write_submission, validate_submission

demo = np.zeros((2048, 2048), np.uint8); demo[900:960, 1000:1120] = 1
counts = rle_to_counts(mask_utils.encode(np.asfortranarray(demo)))
print('RLE counts (truncated):', counts[:60], '...')
restored = mask_utils.decode({'size': [2048, 2048], 'counts': counts.encode()})
print('round-trips exactly:', bool((restored == demo).all()))
"""
    ),
    md(
        """
## 7. Reproducing everything

```bash
pip install -r requirements.txt
python tools/analysis.py       # regenerates every measurement quoted above
python tests/test_pipeline.py  # 21 correctness checks
python src/train.py --fold 0
python src/evaluate_fold.py --checkpoint artifacts/fold0_best.pt --fold 0
python src/predict_test.py --checkpoints artifacts/fold*_best.pt --config artifacts/fold0_tuned.json
```
"""
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(REPO, "notebooks", "pipeline.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(notebook, fh, indent=1)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

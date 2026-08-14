"""Correctness tests for the filament pipeline.

Run with ``python tests/test_pipeline.py`` (no pytest required) or ``pytest``.

These cover the failures that would otherwise be silent - a metric that scores a
broken prediction highly, a mask that drifts off its image, a submission whose
RLE cannot be decoded.  Several of them caught real bugs during development: the
loss mask was once derived by inverting a feature plane, which broke the moment
that plane was rescaled.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pycocotools import mask as mask_utils  # noqa: E402

import metrics  # noqa: E402
from data import FEATURE_MEAN, FEATURE_STD, ImageContext, make_folds, Sample, stride_split  # noqa: E402
import postprocess  # noqa: E402
from postprocess import PostprocessConfig, extract_instances, marginal_threshold  # noqa: E402
from preprocess import Disk, detect_disk, flat_field, limb_profile  # noqa: E402
from submit import rle_to_counts, validate_submission, write_submission  # noqa: E402

PASSED: list[str] = []


def check(name: str):
    def decorator(fn):
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL  {name}: {exc}")
            raise
        PASSED.append(name)
        print(f"ok    {name}")
        return fn

    return decorator


def box(size, y0, y1, x0, x1):
    m = np.zeros((size, size), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def rle(mask):
    return mask_utils.encode(np.asfortranarray(mask))


# --------------------------------------------------------------------------- #
# Panoptic Quality
# --------------------------------------------------------------------------- #


@check("PQ is 1.0 for an exact prediction")
def _():
    gt = [box(64, 0, 10, 0, 10), box(64, 20, 30, 20, 30)]
    assert abs(metrics.evaluate_image("a", gt, gt).pq - 1.0) < 1e-9


@check("PQ matches a hand-computed value")
def _():
    # One match at IoU 0.8, one false positive, one false negative.
    # PQ = 0.8 / (1 + 0.5 + 0.5) = 0.4
    gt = [box(64, 0, 10, 0, 10), box(64, 40, 50, 40, 50)]
    pred = [box(64, 0, 8, 0, 10), box(64, 20, 30, 20, 30)]
    r = metrics.evaluate_image("b", gt, pred)
    assert (r.tp, r.fp, r.fn) == (1, 1, 1), (r.tp, r.fp, r.fn)
    assert abs(r.iou_sum - 0.8) < 1e-9
    assert abs(r.pq - 0.4) < 1e-9


@check("an empty prediction scores 0, not 1")
def _():
    # The original competition metric rewarded empty masks; PQ must not.
    assert metrics.evaluate_image("c", [box(64, 0, 10, 0, 10)], []).pq == 0.0


@check("fragmentation costs two false positives and a false negative")
def _():
    gt = [box(64, 0, 10, 0, 20)]
    pred = [box(64, 0, 10, 0, 9), box(64, 0, 10, 11, 20)]
    r = metrics.evaluate_image("d", gt, pred)
    assert (r.tp, r.fp, r.fn) == (0, 2, 1)
    assert r.one_to_many == 1


@check("over-merging is detected as many-to-one")
def _():
    gt = [box(64, 0, 10, 0, 9), box(64, 0, 10, 11, 20)]
    pred = [box(64, 0, 10, 0, 20)]
    assert metrics.evaluate_image("e", gt, pred).many_to_one == 1


@check("matching is one-to-one above the IoU threshold")
def _():
    # Two predictions over one ground-truth segment: only the better one matches.
    gt = [box(64, 0, 20, 0, 20)]
    pred = [box(64, 0, 19, 0, 20), box(64, 0, 18, 0, 20)]
    r = metrics.evaluate_image("f", gt, pred)
    assert (r.tp, r.fp) == (1, 1)
    assert abs(r.matched_ious[0] - 19 / 20) < 1e-9


@check("PQ decomposes as SQ * RQ")
def _():
    gt = [box(64, 0, 10, 0, 10), box(64, 40, 50, 40, 50)]
    pred = [box(64, 0, 8, 0, 10), box(64, 20, 30, 20, 30)]
    report = metrics.evaluate([metrics.evaluate_image("g", gt, pred)])
    assert abs(report["sq"] * report["rq"] - report["pq_micro"]) < 1e-9


@check("the marginal-emission rule matches its definition")
def _():
    assert abs(marginal_threshold(0.34, 0.64) - 0.5 * 0.34 / 0.64) < 1e-12
    assert marginal_threshold(0.0, 0.64) == 0.0


# --------------------------------------------------------------------------- #
# Solar geometry and photometry
# --------------------------------------------------------------------------- #


def synthetic_disk(size=512, cx=250.0, cy=260.0, r=200.0, limb=True):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    image = np.zeros((size, size), dtype=np.float32)
    inside = rho <= 1.0
    # Classic limb darkening: brightness falls towards the edge.
    image[inside] = 200.0 * (1.0 - 0.6 * rho[inside] ** 2)
    if not limb:
        image[inside] = 200.0
    return image.astype(np.uint8), rho


@check("disk fitting recovers the centre exactly and errs small on radius")
def _():
    image, _ = synthetic_disk()
    disk = detect_disk(image)
    assert abs(disk.cx - 250.0) < 2.0, disk
    assert abs(disk.cy - 260.0) < 2.0, disk

    # Otsu cuts the dimmest limb annulus, so the radius comes out ~1.6% small.
    # That bias is deliberate to leave in place: it is the *safe* direction, and
    # on the real corpus no annotated filament pixel exceeds r/R = 0.991 under
    # the fitted radius (measured over 3.77M ground-truth pixels), while the
    # on-disk mask cuts at 0.995.  An over-estimate, by contrast, would pull the
    # noisy off-disk ring into the loss.
    assert 0.97 * 200.0 <= disk.r <= 200.0 + 1.0, disk


@check("the flat field removes limb darkening")
def _():
    image, rho = synthetic_disk()
    disk = Disk(250.0, 260.0, 200.0)
    flat = flat_field(image, disk)
    # After correction the disk should be uniform: spread must collapse.
    inner = (rho > 0.1) & (rho < 0.9)
    assert flat[inner].std() < 0.02, flat[inner].std()
    assert abs(float(flat[inner].mean())) < 0.02


@check("the flat field keeps a dark feature dark")
def _():
    image, rho = synthetic_disk()
    image = image.copy()
    image[250:260, 300:340] = (image[250:260, 300:340] * 0.7).astype(np.uint8)
    flat = flat_field(image, Disk(250.0, 260.0, 200.0))
    feature = np.zeros_like(image, dtype=bool)
    feature[250:260, 300:340] = True
    assert flat[feature].mean() < -0.2, flat[feature].mean()


@check("limb profile is monotonically decreasing on a limb-darkened disk")
def _():
    image, _ = synthetic_disk()
    profile = limb_profile(image, Disk(250.0, 260.0, 200.0), 64)
    # Ignore the outermost bins where the disk edge is partially sampled.
    core = profile[:55]
    assert (np.diff(core) <= 1e-3).mean() > 0.9, "profile should fall towards the limb"


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


@check("tile features are standardised and the disk mask comes from geometry")
def _():
    image, _ = synthetic_disk(size=512)
    disk = Disk(250.0, 260.0, 200.0)
    context = ImageContext.build(image, disk)

    features = context.tile_features(image, 0, 0, 256)
    assert features.shape == (3, 256, 256)
    assert np.isfinite(features).all()

    mask = context.tile_disk_mask(0, 0, 256)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    # The mask must not be recoverable-by-luck from the scaled radius plane:
    # recompute it independently and require an exact match.
    radius = context.tile_radius(0, 0, 256)
    assert np.array_equal(mask, (radius <= 1.0).astype(np.float32))


@check("feature standardisation constants are actually applied")
def _():
    image, _ = synthetic_disk(size=512)
    context = ImageContext.build(image, Disk(250.0, 260.0, 200.0))
    features = context.tile_features(image, 100, 100, 128)
    radius = context.tile_radius(100, 100, 128)
    expected = (np.clip(radius, 0.0, 1.5) - 0.75 - FEATURE_MEAN[2]) / FEATURE_STD[2]
    assert np.abs(features[2] - expected).max() < 1e-5


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #


@check("folds are deterministic, complete and site-stratified")
def _():
    samples = [
        Sample(image_id=f"{b:06d}-2011010100000{i % 10}{site}", file_name=f"2011010100000{i % 10}{site}.jpeg", instances=[])
        for site in ("Bh", "Ch", "Lh")
        for i, b in enumerate(range(10101, 10131))
    ]
    a = make_folds(samples, n_folds=5, seed=42)
    b = make_folds(samples, n_folds=5, seed=42)
    assert a == b, "fold assignment must be reproducible"
    assert set(a) == {s.file_name for s in samples}
    counts = {}
    for name, fold in a.items():
        counts.setdefault(name[14:16], {}).setdefault(fold, 0)
        counts[name[14:16]][fold] += 1
    for site, per_fold in counts.items():
        spread = max(per_fold.values()) - min(per_fold.values())
        assert spread <= 1, f"site {site} unevenly split: {per_fold}"


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


def hysteresis_fixture():
    p = np.zeros((200, 200), dtype=np.float32)
    p[20:40, 20:80] = 0.8            # a confident filament
    p[60:70, 20:40] = 0.32           # a blob with no confident core
    p[100:110, 100:160] = 0.8        # filament, part A
    p[100:110, 160:164] = 0.33       # a faint bridge
    p[100:110, 164:200] = 0.8        # filament, part B
    return p


@check("hysteresis beats every single threshold on the same fixture")
def _():
    p = hysteresis_fixture()
    # Pin morphology off: this test is about threshold semantics, and the
    # fitted default close_radius=2 would bridge the gap by itself.
    base = dict(min_area=10, min_seed_area=1, dilate_radius=0, close_radius=0)
    areas = lambda inst: sorted(int(mask_utils.area(x)) for x in inst)  # noqa: E731

    high = extract_instances(p, PostprocessConfig(seed_threshold=0.45, mask_threshold=0.45, **base))
    low = extract_instances(p, PostprocessConfig(seed_threshold=0.30, mask_threshold=0.30, **base))
    hyst = extract_instances(p, PostprocessConfig(seed_threshold=0.45, mask_threshold=0.30, **base))

    assert areas(high) == [360, 600, 1200], "a high threshold splits the bridged filament"
    assert areas(low) == [200, 1000, 1200], "a low threshold admits the coreless blob"
    assert areas(hyst) == [1000, 1200], "hysteresis should do both correctly"


@check("dilation grows instances but stays inside the disk")
def _():
    p = hysteresis_fixture()
    disk = np.zeros((200, 200), dtype=np.uint8)
    disk[:, :90] = 1
    plain = extract_instances(
        p, PostprocessConfig(seed_threshold=0.45, mask_threshold=0.45, min_area=10, min_seed_area=1, close_radius=0), disk
    )
    grown = extract_instances(
        p,
        PostprocessConfig(seed_threshold=0.45, mask_threshold=0.45, min_area=10, min_seed_area=1, dilate_radius=3, close_radius=0),
        disk,
    )
    assert mask_utils.area(grown[0]) > mask_utils.area(plain[0])
    decoded = mask_utils.decode(grown[0])
    assert decoded[:, 90:].sum() == 0, "dilation must not leak off-disk"


@check("min_area and min_seed_area both reject")
def _():
    p = hysteresis_fixture()
    assert extract_instances(p, PostprocessConfig(min_area=10**6)) == []
    assert extract_instances(p, PostprocessConfig(min_seed_area=10**6)) == []


@check("the tuning split separates observations, not annotator readings")
def _():
    # An observation read by three annotators produces three readings.  Splitting
    # the reading list puts them on both sides, so the tuner reports its own
    # optimism on images it already fitted to.  Splitting observations cannot.
    names = [f"2016092023{i:04d}Lh.jpeg" for i in range(20)]
    readings = [n for n in names for _ in range(3)]  # every image read 3 times

    tune, report = stride_split(readings)
    assert not (tune & report), "an observation landed in both halves"
    assert tune | report == set(names), "the split lost or invented observations"
    assert abs(len(tune) - len(report)) <= 1, "the halves should be balanced"

    naive = {r for i, r in enumerate(readings) if i % 2 == 0}
    assert naive & {r for i, r in enumerate(readings) if i % 2 == 1}, (
        "the reading-level split this replaced is supposed to leak; if it no "
        "longer does, this test is not measuring what it claims"
    )


@check("uint8 and float probability maps extract identical instances")
def _():
    # Out-of-fold tuning holds all 707 training maps at once, which only fits as
    # uint8.  Quantisation must not move a single instance boundary.
    p = hysteresis_fixture()
    q = np.round(p * 255).astype(np.uint8)
    for seed, mask_t in [(0.45, 0.30), (0.8, 0.35), (0.32, 0.32), (0.95, 0.05)]:
        config = PostprocessConfig(
            seed_threshold=seed, mask_threshold=mask_t, min_area=10, min_seed_area=1
        )
        a = [rle_to_counts(x) for x in extract_instances(p, config)]
        b = [rle_to_counts(x) for x in extract_instances(q, config)]
        assert a == b, f"uint8 path diverged at seed={seed} mask={mask_t}"


@check("min_seed_fraction rejects on confidence ratio, not absolute size")
def _():
    # A large vague blob and a small crisp one, with the same absolute seed area.
    # min_seed_area cannot tell them apart; the fraction can.
    p = np.zeros((200, 200), dtype=np.float32)
    p[10:70, 10:70] = 0.40          # 3600 px, vague
    p[10:30, 10:30] = 0.80          #  400 px of it confident -> fraction 0.11
    p[120:140, 120:140] = 0.80      #  400 px, entirely confident -> fraction 1.0

    base = dict(seed_threshold=0.7, mask_threshold=0.35, min_area=100, close_radius=0)
    both = extract_instances(p, PostprocessConfig(min_seed_area=400, **base))
    assert len(both) == 2, "both components clear an absolute seed area of 400"

    crisp = extract_instances(
        p, PostprocessConfig(min_seed_area=400, min_seed_fraction=0.5, **base)
    )
    assert len(crisp) == 1, "the ratio should drop the vague blob and keep the crisp one"
    assert int(mask_utils.area(crisp[0])) == 400, "the surviving instance is the crisp one"

    assert extract_instances(p, PostprocessConfig(min_seed_fraction=0.0, **base)) == both, (
        "0.0 must be an exact no-op, so the fitted value decides whether it is used"
    )


@check("every configuration ever fitted sits strictly inside its grid")
def _():
    # The defaults test below only guards the *default* config.  Fitted configs
    # live in artefacts, and a value pinned at a grid ceiling there is the same
    # boundary artefact - it is how seed_threshold=0.70 went unnoticed across
    # five folds, and how min_seed_fraction=0.4 was then missed in the pilot.
    import glob

    checked = 0
    for path in glob.glob("kernels/_runs/out_*/*_tuned.json"):
        with open(path) as fh:
            config = json.load(fh)["config"]
        for name, value in config.items():
            grid = postprocess.TUNING_GRIDS.get(name)
            if grid is None or not isinstance(value, (int, float)):
                continue
            assert value <= max(grid), f"{path}: {name}={value} exceeds its grid"
            if value == max(grid):
                raise AssertionError(
                    f"{path}: {name}={value} was fitted at the ceiling of its grid; "
                    f"widen the axis before trusting that configuration"
                )
            checked += 1
    print(f"      ({checked} fitted values checked across artefacts)", end="")


@check("the tuning grid brackets every fitted value on both sides")
def _():
    # Folds 0-2 all selected seed_threshold=0.70 when 0.70 was the largest value
    # offered, so the search was censored and the "optimum" was a grid edge.  A
    # value that a fold can select must never again be the ceiling of its axis.
    fitted = PostprocessConfig()
    for name, grid in postprocess.TUNING_GRIDS.items():
        value = getattr(fitted, name)
        assert value in grid, f"{name}={value} is not reachable on its own grid"
        assert value < max(grid), (
            f"{name}={value} sits at the grid ceiling {max(grid)}; widen the axis "
            f"or the fitted value is a boundary artefact, not an optimum"
        )


@check("the ensemble threshold transfer names the top of a distribution, not its floor")
def _():
    from calibrate_ensemble import quantile_of, threshold_for

    # All the mass at zero, a spike at 210/255.  Every threshold from 1/255 to
    # 210/255 admits exactly that 1% spike, so "first bin under the target" - the
    # obvious implementation - answers 1/255, which is not where the top 1%
    # begins.  This is the bug the transfer had.
    counts = np.zeros(256, dtype=np.int64)
    counts[0], counts[210] = 9900, 100
    assert abs(quantile_of(counts, 10000, 200 / 255) - 0.01) < 1e-9
    assert abs(threshold_for(counts, 10000, 0.01) - 210 / 255) < 1e-9

    rng = np.random.default_rng(0)
    single = np.clip(rng.beta(2, 8, 200_000), 0, 1)
    hist = np.bincount(np.round(single * 255).astype(np.uint8), minlength=256)
    for level in (0.2, 0.3, 0.4, 0.5, 0.6):
        fraction = quantile_of(hist, single.size, level)
        assert abs(threshold_for(hist, single.size, fraction) - level) <= 2 / 255

    # Averaging two independent draws is what ensembling does to a probability
    # map: it pulls disputed pixels toward the middle and shaves the peaks.  The
    # transferred threshold must therefore come *down*, which is the whole point.
    averaged = (single + np.clip(rng.beta(2, 8, 200_000), 0, 1)) / 2
    hist_avg = np.bincount(np.round(averaged * 255).astype(np.uint8), minlength=256)
    moved = threshold_for(hist_avg, averaged.size, quantile_of(hist, single.size, 0.6))
    assert moved < 0.6, "averaging shaves peaks; the threshold must fall"


@check("connected components recover disjoint instances exactly")
def _():
    p = np.zeros((128, 128), dtype=np.float32)
    p[10:30, 10:30] = 0.9
    p[60:90, 60:90] = 0.9
    truth = [rle(box(128, 10, 30, 10, 30)), rle(box(128, 60, 90, 60, 90))]
    got = extract_instances(p, PostprocessConfig(seed_threshold=0.5, mask_threshold=0.5, min_area=10, min_seed_area=1, close_radius=0))
    assert abs(metrics.evaluate_image("cc", truth, got).pq - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #


@check("RLE payloads are CSV-safe and round-trip exactly")
def _(tmp="artifacts/_test_submission.csv"):
    source = box(2048, 100, 160, 200, 340)
    encoded = rle(source)
    counts = rle_to_counts(encoded)
    assert "," not in counts and '"' not in counts
    decoded = mask_utils.decode({"size": [2048, 2048], "counts": counts.encode("ascii")})
    assert np.array_equal(decoded, source)

    os.makedirs("artifacts", exist_ok=True)
    rows = write_submission([("20150125172714Mh", [encoded, rle(box(2048, 300, 340, 400, 460))])], tmp)
    assert rows == 2
    report = validate_submission(tmp, ["20150125172714Mh"])
    assert report["rows"] == 2
    assert report["unknown_ids"] == []
    assert report["empty_masks"] == 0
    assert report["images_without_predictions"] == []
    os.remove(tmp)


@check("training never uses persistent dataloader workers")
def _():
    # Workers fork a copy of the dataset, so with persistent_workers=True a call
    # to set_epoch() in the parent never reaches them and every epoch replays
    # byte-identical crops.  Verified empirically: identical batches across three
    # epochs with persistence, different without.  Guard the call site.
    import re

    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "train.py")).read()
    match = re.search(r"persistent_workers\s*=\s*([^,\n]+)", source)
    assert match, "DataLoader must state persistent_workers explicitly"
    assert match.group(1).strip() == "False", f"got persistent_workers={match.group(1)!r}"


@check("validation subsets stride the fold rather than take a date-ordered prefix")
def _():
    # File names start with a timestamp, so sorted()[:n] would validate only on
    # the earliest observations.
    names = [f"20{y:02d}0101000000Bh.jpeg" for y in range(11, 23)]
    max_files = 4
    step = len(names) / max_files
    chosen = [names[int(i * step)] for i in range(max_files)]
    years = [n[:4] for n in chosen]
    assert years == ["2011", "2014", "2017", "2020"], years
    assert years[-1] != names[max_files - 1][:4], "must not collapse to a prefix"


@check("inference reads only channel 0 from a two-channel model")
def _():
    import torch

    from infer import predict_full
    from preprocess import Disk

    class TwoChannel(torch.nn.Module):
        """Channel 0 constant 0 (p=0.5); channel 1 saturated, and must be ignored."""

        def forward(self, x):
            out = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])
            out[:, 1] = 20.0
            return out

    image = np.full((2048, 2048), 128, dtype=np.uint8)
    disk = Disk(1020.0, 1027.0, 900.0)
    context = ImageContext(disk, np.full(256, 128.0, dtype=np.float32))
    probability = predict_full(
        TwoChannel(), image, context, tile_size=512, stride=384, tta=1,
        device="cpu", amp=False,
    )
    on_disk = disk.mask(2048)
    assert abs(float(probability[on_disk].max()) - 0.5) < 1e-5, (
        "spine channel leaked into the filament probability"
    )


@check("post-processing defaults match the configuration fitted against PQ")
def _():
    # Training validates with these defaults, so a stale value would select the
    # best checkpoint at the wrong operating point.  Fitted on fold 0 -> PQ 0.4062.
    c = PostprocessConfig()
    assert (c.seed_threshold, c.mask_threshold) == (0.70, 0.35), c
    assert (c.min_area, c.min_seed_area, c.close_radius) == (400, 20, 2), c
    assert c.mask_threshold < c.seed_threshold, "hysteresis needs a looser extent than seed"


@check("submission validation catches unknown ids")
def _(tmp="artifacts/_test_bad.csv"):
    os.makedirs("artifacts", exist_ok=True)
    write_submission([("not_a_test_image", [rle(box(2048, 0, 10, 0, 10))])], tmp)
    report = validate_submission(tmp, ["20150125172714Mh"])
    assert report["unknown_ids"], "an id outside the manifest must be reported"
    assert report["images_without_predictions"] == ["20150125172714Mh"]
    os.remove(tmp)


if __name__ == "__main__":
    print(f"\n{len(PASSED)} checks passed")

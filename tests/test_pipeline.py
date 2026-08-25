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
from data import (  # noqa: E402
    FEATURE_MEAN,
    FEATURE_STD,
    ImageContext,
    Sample,
    make_folds,
    rasterise_spines,
    stride_split,
)
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


@check("the empty-prediction fallback emits one candidate, and only when empty")
def _():
    # 10 of 707 observations emit nothing, and no training reading contains zero
    # filaments, so an empty prediction set is certainly wrong.  The fallback
    # must rescue exactly those cases without touching anything else - and must
    # not flood a disk the model genuinely left blank.
    disk = np.zeros((256, 256), dtype=bool)
    yy, xx = np.mgrid[:256, :256]
    disk[(yy - 128) ** 2 + (xx - 128) ** 2 <= 120**2] = True
    off = PostprocessConfig()
    on = PostprocessConfig(fallback_min_area=100)

    # Above mask_threshold (0.35) but below seed_threshold (0.70): rejected.
    faint = np.zeros((256, 256), dtype=np.float32)
    faint[100:112, 90:150] = 0.45
    assert extract_instances(faint, off, disk) == []
    rescued = extract_instances(faint, on, disk)
    assert len(rescued) == 1, f"expected one rescued candidate, got {len(rescued)}"
    assert mask_utils.area(rescued[0]) == 12 * 60, mask_utils.area(rescued[0])

    # A blank map must stay blank.  A fixed relaxation would admit the whole
    # disk here; seeding from the map's own peak cannot.
    assert extract_instances(np.zeros((256, 256), dtype=np.float32), on, disk) == []

    # A peak below mask_threshold means no pixels reach the extent level either.
    below = np.zeros((256, 256), dtype=np.float32)
    below[100:112, 90:150] = 0.20
    assert extract_instances(below, on, disk) == []

    # Two weak blobs: only the more confident one is emitted, because each extra
    # candidate must clear the marginal bar on its own and the second is weaker.
    two = np.zeros((256, 256), dtype=np.float32)
    two[100:112, 90:150] = 0.45
    two[160:172, 90:150] = 0.40
    assert len(extract_instances(two, on, disk)) == 1

    # A configuration that already found something is untouched.
    good = np.zeros((256, 256), dtype=np.float32)
    good[100:112, 90:150] = 0.9
    good[160:172, 90:150] = 0.9
    assert [mask_utils.area(m) for m in extract_instances(good, off, disk)] == \
           [mask_utils.area(m) for m in extract_instances(good, on, disk)]

    # 0 must be an exact no-op so the tuner decides whether the axis earns a place.
    assert PostprocessConfig().fallback_min_area == 0
    assert 0 in postprocess.TUNING_GRIDS["fallback_min_area"]


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


@check("the threshold transfer cancels abundance only when both halves share a population")
def _():
    from calibrate_ensemble import quantile_of, threshold_for

    # The transfer is meant to correct one thing: the shift from single-model
    # maps to averaged ones.  Drawing its two histograms from different image
    # populations silently folds in a second thing - how many filaments each
    # population actually contains - and "corrects" a real difference in the sky.
    def population(rng, n, abundance, members=5):
        filament = rng.random(n) < abundance
        a = np.where(filament, 8.0, 1.0)
        b = np.where(filament, 2.0, 30.0)
        draws = np.stack([rng.beta(a, b) for _ in range(members)])
        hist = lambda x: np.bincount(np.round(x * 255).astype(np.uint8), minlength=256)
        return hist(draws[0]), hist(draws.mean(0)), n

    rng = np.random.default_rng(0)
    n = 1_500_000
    single_a, ensemble_a, total_a = population(rng, n, 0.010)  # train-like
    single_b, ensemble_b, total_b = population(rng, n, 0.030)  # test-like, 3x denser

    for level in (0.95, 0.40):
        fraction_a = quantile_of(single_a, total_a, level)
        within_a = threshold_for(ensemble_a, total_a, fraction_a)
        within_b = threshold_for(ensemble_b, total_b, quantile_of(single_b, total_b, level))
        crossed = threshold_for(ensemble_b, total_b, fraction_a)

        # Same family shift, so measuring inside either population agrees.
        assert abs(within_a - within_b) < 0.02, (level, within_a, within_b)
        # Mixing populations does not, and errs towards admitting too little.
        assert crossed > within_b, (level, crossed, within_b)
    assert crossed - within_b > 0.05, (
        "the fixture must actually separate the two estimators"
    )


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


# --------------------------------------------------------------------------- #
# The auxiliary spine head
#
# The host sanctioned spine metadata as training supervision, and this path is
# wired end to end but had never run against a real annotation.  Every one of
# its failure modes is silent - a target that rasterises to nothing, or one
# drawn transposed, still produces a healthy-looking loss curve - so it is
# tested here rather than discovered after a GPU run reports "no effect".
# --------------------------------------------------------------------------- #


@check("a spine parses identically however COCO nested it")
def _():
    from data import spine_points

    flat = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    wrapped = [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]      # like `segmentation`
    paired = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]   # already (x, y) pairs
    want = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)

    for name, form in (("flat", flat), ("wrapped", wrapped), ("paired", paired)):
        got = spine_points(form)
        assert np.array_equal(got, want), f"{name} form parsed as {got.tolist()}"

    assert len(spine_points([])) == 0
    assert len(spine_points(None)) == 0
    # A wrapped spine used to fall through a `len(spine) < 4` guard and rasterise
    # to nothing, which is the whole reason this test exists.
    assert len(rasterise_spines([wrapped[0]], size=128).nonzero()[0]) > 0

    try:
        spine_points([1.0, 2.0, 3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("an odd coordinate count must not be silently truncated")


@check("a spine crossing a tile boundary is drawn continuously")
def _():
    # The tile target is cut from a full-frame rasterisation, which is what makes
    # a spine entering from off-tile continuous at the seam.  Drawing it per tile
    # in a shifted frame would be cheaper, but cv2.polylines clips to integer
    # canvas bounds and lands the line up to a pixel off (IoU 0.89 against this,
    # worst case 0.69) - for 0.2% of a data pipeline dominated by the JPEG
    # decode.  Guard the property, not the micro-optimisation.
    crossing = [100.0, 640.0, 900.0, 660.0]         # enters the tile from the left
    y0, x0, size = 512, 512, 512
    tile = rasterise_spines([crossing])[y0 : y0 + size, x0 : x0 + size]

    columns = np.flatnonzero(tile.any(axis=0))
    assert columns[0] == 0, "a spine from off-tile must reach the tile edge"
    assert np.array_equal(columns, np.arange(columns[0], columns[-1] + 1)), (
        "the drawn spine has a gap; it is not one connected polyline"
    )
    # A spine nowhere near the tile contributes nothing to it.
    assert rasterise_spines([[50.0, 50.0, 120.0, 90.0]])[y0 : y0 + size, x0 : x0 + size].sum() == 0


@check("spine alignment detects a transposed coordinate convention")
def _():
    from data import Sample, spine_alignment

    # A horizontal bar filament with its spine running along the axis.
    mask = box(256, 100, 116, 40, 200)
    axis = [45.0, 108.0, 195.0, 108.0]          # (x, y): along the bar
    transposed = [108.0, 45.0, 108.0, 195.0]    # (y, x) fed to a (x, y) drawer

    good = Sample("i", "f.jpeg", [rle(mask)], spines=[axis])
    bad = Sample("i", "f.jpeg", [rle(mask)], spines=[transposed])

    inside_good, covered_good = spine_alignment(good)
    inside_bad, _ = spine_alignment(bad)
    assert inside_good > 0.95, inside_good
    assert 0.0 < covered_good < 0.6, covered_good  # a core, not the whole filament
    assert inside_bad < 0.2, (
        f"a transposed spine scored {inside_bad:.2f}; preflight would not catch it"
    )
    # An empty spine reads as zero rather than raising, so preflight reports it.
    assert spine_alignment(Sample("i", "f.jpeg", [rle(mask)], spines=[[]])) == (0.0, 0.0)


@check("the spine channel reaches the loss only when it is switched on")
def _():
    import torch

    from losses import FilamentLoss

    torch.manual_seed(0)
    logits = torch.zeros(2, 2, 32, 32, requires_grad=True)
    target = torch.zeros(2, 2, 32, 32)
    target[:, 0, 8:24, 8:24] = 1.0   # filament
    target[:, 1, 14:18, 8:24] = 1.0  # its spine
    weight = torch.ones(2, 1, 32, 32)

    def spine_grad(spine_weight):
        logits.grad = None
        FilamentLoss(spine_weight=spine_weight)(logits, target, weight).backward()
        return float(logits.grad[:, 1].abs().sum())

    assert spine_grad(0.0) == 0.0, "the spine channel must be inert when disabled"
    assert spine_grad(0.3) > 0.0, (
        "spine_weight is set but no gradient reaches channel 1 - the auxiliary "
        "head is being trained on nothing"
    )
    # A 1-channel target must not silently drop the spine term into a shape error.
    FilamentLoss(spine_weight=0.3)(logits[:, :1], target[:, :1], weight).backward()


@check("training pairs the spine head with the spine target, or neither")
def _():
    # Two independent switches - the model's out_channels and the dataset's
    # with_spine - must move together.  Either one alone fails silently: a second
    # head with no target gets no gradient, and a second target with no head is
    # dropped by the loss's shape guard.
    import re

    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "train.py")).read()
    for pattern, what in (
        (r"with_spine\s*=\s*args\.spine_weight\s*>\s*0", "dataset target"),
        (r"out_channels\s*=\s*2\s+if\s+args\.spine_weight\s*>\s*0\s+else\s+1", "model head"),
        (r"spine_weight\s*=\s*args\.spine_weight", "loss term"),
    ):
        assert re.search(pattern, source), f"{what} is not gated on --spine-weight"


@check("a spine-enabled dataset emits a two-channel target")
def _():
    import torch

    from dataset_torch import FilamentTiles
    from data import Sample

    disk = Disk(1024.0, 1024.0, 900.0)
    context = ImageContext(disk, np.full(256, 128.0, dtype=np.float32))
    mask = box(2048, 1000, 1040, 900, 1200)
    sample = Sample(
        "010401-x", "x.jpeg", [rle(mask)], spines=[[905.0, 1020.0, 1195.0, 1020.0]]
    )

    class OneImage(dict):
        def __getitem__(self, key):
            return context

    image_dir = "artifacts/_spine_fixture"
    os.makedirs(image_dir, exist_ok=True)
    cv2.imwrite(os.path.join(image_dir, "x.jpeg"), np.full((2048, 2048), 128, np.uint8))

    for with_spine, channels in ((False, 1), (True, 2)):
        dataset = FilamentTiles(
            [sample], image_dir, OneImage(), tiles_per_sample=4,
            augment=False, with_spine=with_spine,
        )
        _, target, _ = dataset[0]
        assert target.shape[0] == channels, (with_spine, target.shape)
        if with_spine:
            assert float(target[1].sum()) > 0.0, "spine channel is empty on a hit tile"
            assert float(target[1].sum()) < float(target[0].sum()), (
                "the spine must be a core inside the filament, not larger than it"
            )
    os.remove(os.path.join(image_dir, "x.jpeg"))
    os.rmdir(image_dir)


@check("autocast picks bf16 on Ampere and leaves the T4 exactly as it was")
def _():
    import torch

    from infer import choose_amp_dtype

    # Every autocast site used to take torch's CUDA default, which is fp16.  On
    # the T4 that is right.  On a rented 3090 it silently reproduces the fp16
    # overflow that made EfficientNet-B4 train with skipped steps - the caveat
    # standing against the one capacity measurement this project has.
    assert choose_amp_dtype("auto", "cuda", (7, 5)) is torch.float16, "T4 must not change"
    assert choose_amp_dtype("auto", "cuda", (8, 6)) is torch.bfloat16, "Ampere"
    assert choose_amp_dtype("auto", "cuda", (8, 9)) is torch.bfloat16, "Ada"
    assert choose_amp_dtype("auto", "cuda", (9, 0)) is torch.bfloat16, "Hopper"
    assert choose_amp_dtype("auto", "cpu", None) is None
    assert choose_amp_dtype("fp32", "cuda", (8, 6)) is None
    assert choose_amp_dtype("fp16", "cuda", (8, 6)) is torch.float16, "override honoured"

    # Selection must come from compute capability, not
    # torch.cuda.is_bf16_supported(), which reports True for *emulated* bf16 on
    # pre-Ampere cards and would put the T4 on a slow path.
    import inspect

    source = inspect.getsource(choose_amp_dtype)
    assert "is_bf16_supported" not in source.split('"""')[2], (
        "capability, not is_bf16_supported, decides"
    )

    try:
        choose_amp_dtype("float16", "cuda", (8, 6))
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown precision name must be rejected, not guessed")

    # bf16 is the point: fp32's range, and a mantissa still finer than the
    # uint8 quantisation the tuning path already rounds probabilities to.
    assert torch.finfo(torch.bfloat16).max > 1e38
    assert torch.finfo(torch.float16).max < 1e5
    assert 2 ** -8 <= 1.0 / 255


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

# Solar Filament Segmentation Challenge 2026 — handover

Paste this whole file as the first message of a new Claude Code session, working
directory `C:\Users\msraa\Downloads\Solar`.

## The task

Kaggle / IEEE BigData Cup, hosted by the Earth-Space AI Research Lab with the NSF
National Solar Observatory. $3,000. **Deadline 15 Nov 2026.** Segment individual
solar filaments in 2048x2048 GONG H-alpha full-disk images.

Repo: https://github.com/msraaghavan/Solar (public, MIT, currently up to date).
Data: `data/MAGFiLO_1.0_Kaggle_2026` — 707 train observations (1154 annotator
readings), 180 test images.

**Judging is 70% quantitative + 30% qualitative, not leaderboard rank.** The host
has said twice that the LB is "only a preliminary way of filtering some
submissions out" and that "submissions without a decent strategy will NOT be
rewarded". Deliverables: a 4-page PDF report (Overleaf template is at
`kaggle_comp_report_fi_segm_2026.pdf`, gitignored) plus a public repo with
`requirements.txt` and a pipeline notebook, submitted via a Google Form that
needs the user's own Google login.

The quantitative 70% is PQ **plus** the Dice distribution, the IoU distribution,
and the one-to-many / many-to-one counts. Fragmentation is graded directly.

## Current standing (26 Aug 2026)

**Rank 40 of 421, public LB PQ 0.37.** Two submissions ever, both 0.37.

The leaderboard has a shape that matters more than our position in it:

| rank | score |
|---|---|
| 1-2 | 0.55 |
| 3 | 0.54 |
| 4 | 0.53 |
| **5-7** | **0.40** |
| 8-10 | 0.39 |
| **40 (us)** | **0.37** |

There is a **0.13 discontinuity between rank 4 and rank 5** and almost nothing
between 0.40 and 0.37. A gap that large in the top four, with a dense continuum
below it, is the signature of a different *method*, not better tuning. A 12th
place competitor said publicly: *"I guess training with test labels from original
dataset."* 139-180 of the 180 test images have public ground truth in the MAGFiLO
1.0 release. The host, unprompted: *"I would not compare very high scores with
mine. Any score higher than 0.35 is of great value for us and we will examine
them carefully."*

**We do not use the leak.** So the honest target is the top of the lower cohort,
not the top of the board:

- **+0.02 reaches rank 10.** +0.03 reaches rank 5.
- **+0.05 (LB ~0.42) would be first in the honest cohort** and 5th overall.
- Rank 3 needs **+0.17**, which would mean CV ~0.59 against a measured ceiling of
  0.7687. Not impossible, but nothing identified is worth that on its own.

Since judging is 70% quantitative + 30% qualitative and *not* leaderboard rank,
being demonstrably the best leak-free submission is the winning position, and the
paired unconfounded experiments are what make that case.

## Calibration: a clean null result (26 Aug 2026)

The calibrated operating point (seed 0.950 -> 0.929, mask 0.400 -> 0.392) scored
**0.37 - identical to the uncalibrated submission** at the two decimals Kaggle
reports. It moved instances/image from 6.083 to 6.433, so it did what it was
designed to do; it just was not worth anything.

Taken with the measurement itself - both the train-side and test-side estimators
agree the threshold only moves ~0.02 - **the ensemble operating point is settled
and closed.** It was never the explanation for the -0.047 CV-to-LB gap, and no
further effort belongs there. What remains is generalisation and mask quality.

## The metric — verified, not assumed## The metric — verified, not assumed

The host published a self-evaluation notebook and confirmed on 13 Aug that it
"has the exact implementation of PQ". `tests/test_official_metric.py`
re-implements their rules independently and asserts agreement (7 checks).

Rules that matter:

- `PQ = sum(IoU of TP) / (TP + 0.5*FP + 0.5*FN)`, match at **IoU > 0.5 strictly**,
  micro-pooled over every annotator-image entry, divided once at the end.
- GT is keyed `<annotator>-<image>_<n>`; predictions are keyed `<image>_<n>`. The
  scorer iterates GT entries and looks predictions up **by image alone**, so one
  prediction set is scored once per annotator. **Predicting once per observation
  is the only thing the format can express** — there is no duplication strategy.
  A top-10 team asked this publicly 19 days ago and never got an answer; it is
  settled from their code.
- Fragmentation uses `IoU > 0`, any overlap. Ours used `> 0.1` and was fixed.
- Predictions for an image with no GT entry are never scored.

## Honest results, all like-for-like under current code

**Five-fold CV, per-fold configs:** 0.4065 / 0.3904 / 0.4029 / 0.3953 / 0.3777
= **0.3946 +/- 0.0113**.

**Pooled out-of-fold fit (the headline): PQ 0.4167**, selection optimism only
+0.0024, fitted on 573 readings / 354 observations, reported on 353 disjoint
observations. Every per-fold config scored on that same half lands 0.3843-0.3965,
so pooling is worth **+0.020 to +0.032**.

OOF report: SQ 0.6748, RQ 0.6175, TP 2501 / FP 1458 / FN 1640, precision 0.632,
hit rate 0.604, one-to-many 144, many-to-one 70, mean matched Dice 0.8027, mean
semantic Dice 0.6517.

**Fold-0 ablations (identical code, identical data, only the model differs):**

| variant | PQ | SQ | RQ |
|---|---|---|---|
| B0 baseline | **0.4387** | 0.6724 | 0.6524 |
| B0 + label smoothing 0.05 | 0.4281 | 0.6724 | 0.6368 |
| EfficientNet-B4 | 0.4177 | 0.6711 | 0.6223 |

**Label smoothing (-0.011) is rejected.** Beware: against fold 0's *old-code*
0.4065, it looked like a +0.022 gain. It is a loss. **Never compare across code
versions.**

**B4 (-0.021) is NOT settled, and this is the most important open question.**
That run triggered fp32 recomputation for non-finite fp16 logits, and GradScaler
drops a step silently whenever gradients overflow — so B4 may simply have been
undertrained rather than genuinely worse. The T4 is Turing and has only fp16, so
on Kaggle the confound is unavoidable. **Any Ampere card (3090/4090) has bf16,
which carries fp32's exponent range and cannot overflow that way.**

As of commit 7471e57 the code no longer hardcodes torch's fp16 default:
`--amp-dtype auto` resolves to bf16 on compute capability >= 8.0 and stays fp16
on the T4, so existing results remain reproducible. **Re-testing capacity on
Ampere with bf16 is the single highest-value experiment available**, because it
is the only lever anyone has identified that could be worth more than +0.05, and
the one measurement against it is confounded.

## The single most important correction to earlier reasoning

Earlier work concluded the task was "label-noise-limited, not capacity-limited",
by comparing our PQ to the 0.3397 inter-annotator agreement. **That comparison is
invalid** and it suppressed the right experiments for weeks.

0.3397 is annotator-A-vs-annotator-B, excluding the self-match. The real scorer
evaluates one prediction against *every* reading including the one it matches.
Measured achievable ceiling, if a method emitted human-quality masks:

| | observations | ceiling |
|---|---|---|
| single-annotator | 411 (58%) | **1.0000** |
| multi-annotator | 296 (42%) | **0.6307** |
| overall | 707 | **0.7687** |

We are at 0.4167 against ~0.77. And OOF single-annotator PQ is **0.4163** vs
multi-annotator **0.4169** — identical. If disagreement bound us we would score
far better where there is no disagreement. **We are capacity-limited. There is
enormous headroom.** Recompute by scoring each annotator's own masks against all
readings of that image, pooled micro.

Other live diagnostics from the OOF run: 321 near-misses (unmatched GT at IoU
0.4-0.5) worth **+0.0359** if converted; `median_area_ratio` **1.082**, i.e. we
over-segment by 8%, so growing instances is the wrong direction and
`dilate_radius=0` in all five folds is correct; 10 of 707 observations (1.4%)
emit nothing, a guaranteed loss since **no training reading has zero filaments**
(minimum 1, mean 7.1).

## Mask quality is the largest lever, measured (26 Aug 2026)

`tools/iou_headroom.py` reads the stored matched-IoU distribution out of an OOF
artefact and asks what a uniform change in mask quality would be worth. On
`out_ooffull` (2501 matched pairs, PQ 0.4167):

| uniform IoU shift | PQ | delta |
|---|---|---|
| -0.05 | 0.3597 | -0.057 |
| +0.02 | 0.4391 | **+0.022** |
| +0.05 | 0.4700 | **+0.053** |
| +0.10 | 0.5225 | **+0.106** |

The response is very nearly linear at about **1.06 PQ per unit of mean IoU**, in
both directions. Three things follow, and they reorder the priority list:

1. **This dominates the near-miss lever.** Converting *every one* of the 321 near
   misses at IoU 0.6 gives 0.4643 (+0.048) - less than a uniform +0.05 shift,
   because that shift also lifts all 2501 pairs that already match. Chasing the
   cliff alone is the smaller half of the opportunity.
2. **The distribution has no upper tail.** Matched IoUs peak at 0.65-0.75 and
   exactly **one** pair out of 2501 exceeds 0.90. The model never draws a nearly
   correct boundary, on any instance. That is not what a label-noise ceiling
   looks like; it is what limited capacity or a mismatched objective looks like.
3. **8.9% of matches (223) sit below IoU 0.55** and are one small regression away
   from becoming a false positive *and* a false negative. The -0.05 row is the
   same statement from the other side: mask quality is the dominant sensitivity
   of this metric, in both directions.

**The objective does not optimise what the metric measures.** `FilamentLoss` is
`BCE + 0.5 * soft_dice`, and that Dice is computed over the whole tile with every
filament pooled - it is a *semantic* overlap term. The metric scores *per
instance* IoU. The gap is visible in the run's own diagnostics: mean semantic
Dice 0.6517 against mean matched Dice 0.8027.

Candidates, in order of how well they fit the evidence:

- a boundary-weighted term (filaments are thin, so boundary pixels are a large
  fraction of each mask and errors there dominate IoU);
- a Lovász-hinge term alongside the existing pair, which is a direct surrogate
  for IoU;
- more capacity, which the B0-vs-B4 pod is testing right now - `evaluate_fold`
  reports SQ, which *is* mean matched IoU, so that run measures this lever
  directly.

**Do not simply replace BCE.** The docstring in `losses.py` argues, correctly,
that BCE is a proper scoring rule whose minimiser is P(a random annotator marks
this pixel) - which is exactly what the operating-point tuning in
`postprocess.py` assumes it is thresholding. Add a term; do not remove that one.

## SQ saturates in six epochs and never moves again (26 Aug 2026)

Read out of every stored `fold*_history.json`. The validation curves say two
things, and the second one reframes the problem.

**1. Thirty epochs is too many; every run peaks around 9-15 and then declines.**

| run | epochs | best PQ | at epoch | PQ at the end |
|---|---|---|---|---|
| out_fix (B0) | 30 | 0.3884 | 15 | 0.363 |
| out_b0 | 30 | 0.3364 | 10 | 0.310 |
| out_f3 | 20 | 0.4050 | 15 | 0.395 |
| out_f1 | 20 | 0.3846 | 9 | 0.368 |

`*_best.pt` is selected on validation PQ, so results are not wrong - but roughly
half of every 30-epoch run is spent getting worse. **Use ~20 epochs.** The 20-
epoch runs peak in the same place as the 30-epoch ones, so this is overfitting
rather than an artefact of the one-cycle schedule being stretched. (That caveat
is real and worth keeping in mind: with one-cycle, epoch 15 of a 30-epoch
schedule is at a different learning rate than epoch 15 of a 20-epoch one, so
"peaks at 15" does not by itself license "train for 15".)

**2. SQ is flat. All of the movement in PQ is RQ.**

    out_fix SQ by epoch:  0.627 0.666 0.665 0.671 0.668 0.669 0.666 0.671 0.670 0.668
    out_fix PQ by epoch:  0.258 0.337 0.355 0.377 0.388 0.387 0.360 0.357 0.363 0.363

Mean matched IoU reaches ~0.67 by epoch 6 and then does not move for the
remaining 24 epochs, while PQ swings by 0.13. Every run in the table sits in
0.63-0.68, including both B4 runs (0.6595, 0.6647) which are **no better than
B0's 0.6680**.

So the model learns *where* filaments are - that is RQ, and it keeps improving -
but the quality of its boundaries hits a wall almost immediately and stays there,
regardless of training time, encoder, or fold.

**This is the single most useful diagnostic in the project.** Combined with the
headroom measurement (~1.06 PQ per unit of mean IoU) and the shape of the IoU
distribution (one matched pair in 2501 above 0.90), it says the boundary ceiling
is structural: it is not a matter of training longer, and the one capacity
comparison available - confounded though it is - shows no SQ benefit either.

It is *not* annotator disagreement. Out-of-fold PQ on single-annotator
observations (0.4163), where an exact match is achievable by construction, equals
the multi-annotator figure (0.4169).

That leaves the objective. `FilamentLoss` is BCE plus a *semantic* soft-Dice
pooled over the whole tile; neither term rewards putting an edge in exactly the
right place, and predictions are only 8% too large by area, so the error is edge
*position*, not mask size. **Two direct tests of this are running now.**

`--boundary-weight` puts extra BCE weight on a rim either side of each edge.

`--stem-skip` fixes something the model docstring *claimed* and the code did not
do. A timm `features_only` encoder emits nothing above stride 2, and the last
entry of the decoder's skip list was a literal `0` - so the final stride-2 ->
stride-1 block received **no skip at all**. It saw a nearest-neighbour upsample
of the /2 features through 16 channels and nothing else, and so could not place
an edge more precisely than the /2 grid allows. That is exactly the shape of a
hard ceiling on mean matched IoU that no amount of training moves. The flag adds
one full-resolution convolution over the input (+3K parameters, 0.05%) and hands
it to that block. If it moves SQ off 0.67 it is worth far more than anything else
on the list; if it does not, the next candidates are output resolution and tile
size, not more epochs and not a bigger encoder.

## The CV-to-LB gap, and the unfinished fix

CV 0.4167 -> LB 0.37, a gap of **-0.047**. Leading hypothesis: the operating
point is fitted on single-model out-of-fold maps but applied to a five-model
average. Averaging shaves probability peaks, so `seed_threshold=0.95` admits
less. Symptom: 6.15 instances/image on test vs 6.8 per observation out-of-fold.

`src/calibrate_ensemble.py` fixes this without labels — it holds the *admitted
pixel mass* fixed rather than the threshold.

**DONE, 25 Aug 2026.** Measured on the 180 **test** images (`--on test`), five
models, 0.48 G on-disk pixels, 18.8 min:

| threshold | fitted | admits | ensemble | admits |
|---|---|---|---|---|
| `seed_threshold` | 0.950 | 3.607e-03 | **0.929** | 3.605e-03 |
| `mask_threshold` | 0.400 | 6.331e-03 | **0.392** | 6.341e-03 |

Stored at `kernels/_runs/out_calib_test/ensemble_config.json`, tagged
`"measured_on": "test"` — check that field before trusting any such file, since a
train-measured one looks identical otherwise.

**The prediction made here was wrong, and in the interesting direction.** The
train-measured shift (0.95 -> 0.9255) was written up as a *lower bound*, on the
argument that four of five models have seen each training image so the ensemble
histogram there is artificially sharp. The honest test-side measurement gives
**0.929**, a *smaller* shift (-0.021 against -0.025), not a larger one. So that
correction was already at full size, and the reasoning behind calling it a lower
bound does not survive contact with the measurement — do not repeat it.

What that settles: **calibration is not the explanation for the -0.047 CV-to-LB
gap.** Both estimators agree the operating point only moves by ~0.02 on seed and
0.008 on mask. The gap has to come from somewhere else — test-set annotator
composition (unknown, unknowable without labels) and ordinary generalisation are
the remaining candidates, and the second is the one worth attacking.

Measuring both halves on the *same* images is the point, not an incidental
detail: on a synthetic population 3x denser in filaments, transferring within
either population agrees to 0.004, while mixing them moves `mask_threshold` by
0.19 — "correcting" a real difference in the sky as though it were an artefact
of averaging. Asserted in `tests/test_pipeline.py`.

The predict kernel now selects its operating point by explicit name
(`CONFIG_PREFERENCE` in `src/postprocess.py`, preferring `ensemble_config.json`
over `oof_tuned.json`). Before that fix it *could not* have used a calibration
even when one was attached: `ensemble_config.json` does not end in
`_tuned.json`, so the glob that looked for configurations never saw it, and the
run would have submitted the uncalibrated point looking entirely normal.

## Two newly-sanctioned levers (host replies, 4 days and 13 hours ago)

1. **Spine / bbox / area / category_id metadata are ALLOWED as training
   supervision.** Host: *"You can use other meta data available in the dataset
   (i.e., spine, bbox, area, and category_id)."* The spine auxiliary head is
   **already built** (`--spine-weight`, `FilamentNet(out_channels=2)`, inference
   already slices channel 0) and wired through the kernel. The hosts explicitly
   name "structural continuity / fragmented segmentations" as a core challenge,
   and fragmentation is graded. This is the single most promising untested idea.

   **The path is now covered by six tests (commit c828ece), but it has still
   never touched a real annotation** — no session has had the data and a GPU at
   the same time. Two silent failure modes were found and closed by inspection:
   a spine wrapped one list deep (the way `segmentation` is) tripped the
   `len(spine) < 4` guard and rasterised to an **all-zero target**, which would
   have trained the head on a blank image behind a healthy-looking loss curve;
   and `cv2.polylines` reads `(x, y)`, so `(row, column)` data would draw every
   spine reflected about the diagonal onto quiet Sun. `spine_points` now accepts
   flat, wrapped and paired forms; `spine_alignment` measures the quantity that
   catches the rest.

   **`train.py` preflight now refuses to start** unless ≥80% of spine pixels
   fall inside their own filament (published figure: 95.4%). It runs on
   annotation geometry alone, costs no GPU time, and prints the alignment. **If
   that line does not appear, or the run dies on it, the spine field is being
   misread — fix the parse, do not lower the bar.**
2. **Self-supervised pretraining on external unlabeled GONG H-alpha images is
   ALLOWED.** Host answered "Yes." A legitimate way to add data.

## Infrastructure — hard-won, do not relearn

- **Kaggle P100 is unusable**: `cuda.is_available()` is True and every kernel
  launch fails. Always set `"machine_shape": "NvidiaTeslaT4"` and verify with
  `kaggle kernels pull -m` — invalid values silently revert.
- **Kaggle limits: 2 concurrent GPU sessions, 30 h/week.** ~25.7 h logged so far.
  Logs are only readable after a run completes.
- **The local machine cannot run inference.** CPU-only torch; one full-disk
  2048px `predict_full` takes ~11 minutes. Never plan a local inference smoke
  test — use a small `--max-files` GPU kernel instead (~15 min). Local CPU is
  fine for annotation statistics, metric tests, submission validation, fold
  logic, and disk detection.
- `tools/push_train.py --name X --fold N [--label-smoothing F] [--spine-weight F]`
  launches a run; it re-publishes `src/` and **waits for publication** first,
  because a stale dataset silently trains old code.
- `tools/push_oof.py --max-files 0` runs the pooled OOF fit (~2.5 h).
- `tools/bootstrap_pod.sh` sets up a rented GPU pod.
- Runtime confirmed matching `requirements.txt`: torch 2.10.0+cu128, timm 1.0.26,
  numpy 2.0.2, cv2 4.13.0, scipy 1.16.3, Tesla T4.
- **A cloud Claude session cannot do any of the above.** It gets a fresh
  container: no `data/`, no `artifacts/`, no `kernels/_runs/`, no Kaggle or
  RunPod credentials, no GPU, and outbound HTTPS restricted enough that
  kaggle.com, arxiv.org and the competition site are all blocked. It can install
  deps from PyPI, run both test suites, and read and write code — nothing that
  touches the data or the leaderboard. Anything needing data or a GPU has to run
  from the user's own machine.

## Rented GPU — read before spending

`tools/run_pod_experiment.sh` trains, evaluates, pushes results and **terminates
the pod from a trap**, so it fires on success, on a failed job, on a crash and
on Ctrl-C. `bootstrap_pod.sh` only prepares a pod and stops at a prompt; a
finished pod bills exactly like a training one, so never leave that unattended.

**What five smoke pods cost, and what they bought: $0.09 and five real bugs.**
Every one of them would have destroyed a multi-hour experiment, and four were
*silent* - the pod reported success:

1. `timm` depends on torch, so pip satisfied it by pulling a PyPI build compiled
   against a newer CUDA than the host driver. torch imported, reported its CUDA
   version cheerfully, and saw no device.
2. **CUDA error 804**, "forward compatibility was attempted on non supported HW":
   the host driver was older than the image's CUDA. Pin `allowedCudaVersions`.
3. The runpod images are bare - torch and torchaudio, no numpy, no torchvision -
   and timm imports torchvision during `create_model`.
4. The runner grepped `"pq_micro":` from a log that prints a *formatted report*,
   so every result row read `PQ=unknown`.
5. `kaggle datasets create` prints "title is already in use" and **exits 0**, so
   `create || version` never falls through. A completed pod uploaded nothing
   while reporting success, because its own bootstrap heartbeat had already
   created the dataset.

The general lesson, which is worth more than any of the five: **on a pod, check
the message, not the exit status, and verify an extraction against a log you
already know the answer to.** The PQ grep was validated by pointing it at the
stored B0 fold-0 evaluation and confirming it returned 0.4387 / 0.6724 / 0.6524.

Two rules the runner enforces, both of which cost money to learn otherwise:

1. **Always `--smoke` first.** One epoch, five steps, two validation files.
   Costs cents, exercises spine preflight, a real step, full-disk inference,
   instance extraction, the checkpoint round-trip and the result push.
2. **Every spine run is paired with its own `spine-weight 0` baseline on the
   same pod.** Fold 0's 0.4387 was measured under older code, on a T4, in fp16.
   A pod is none of those three. Compare against the paired row, never 0.4387.

**Checkpoints do not survive a pod** — they are ~50 MB and gitignored. This is
now solved without a network volume: the runner publishes everything, including
the `.pt` files, to a **Kaggle dataset** `raaghavanms/filament-pod-<tag>`. That
is strictly better than a volume, because a Kaggle dataset can be attached to
the prediction kernel through `dataset_sources` — so a fold trained on rented
hardware is immediately usable on Kaggle, with no upload step in between.

Three things learned by spending money, all of them now handled in code:

- **There is no GitHub token on this account.** The runner's original
  `git push` path was dead code, so every result would have evaporated with the
  pod. Kaggle is the transport; git remains as an optional extra if a token ever
  appears in `GITHUB_TOKEN`.
- **RunPod's REST API cannot read a running pod's console, and a terminated pod
  takes its logs with it.** The first pod died four minutes in and reported
  *nothing*. `tools/launch_pod.py` now redirects the whole pod session to a file
  and uploads it to `raaghavanms/pod-<tag>-log` from the exit trap, before
  destroying the pod. Read that dataset first when a pod fails.
- **`/workspace` is the default *volume* mount point, and these pods are created
  with no volume**, so it is not guaranteed to exist; `mkdir -p` it before use.

`tools/launch_pod.py launch|list|kill` drives the whole thing. Secrets come from
the environment (`RUNPOD_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`) and are never
printed, never passed as arguments, and never committed. `list` shows what is
billing; `kill all` stops everything.

Pricing checked 25 Aug 2026 on Community: **RTX 4090 $0.34/h, High stock**;
3090 $0.22/h Medium; A40 $0.35/h High. All three are Ampere-or-later and have
native bf16, which is the entire reason to rent rather than use the Kaggle T4.
$20 is roughly **58 GPU-hours** at 4090 prices.

**Spend it only on what Kaggle physically cannot do.** Kaggle's free quota is 30
h/week, and 15 Nov is ~12 weeks out: that is ~360 T4-hours, which dwarfs the
rented budget. The one thing a T4 cannot do is bf16 — it is Turing, fp16 only —
so the rented cards exist for the *capacity* question and nothing else. Five-fold
retrains, pooled OOF fits, calibration and prediction all belong on Kaggle, free.
Note the weekly quota resets: the 25.7 h recorded earlier was from the week of
14 Aug and does not still apply.

## Tests — 51 checks, all passing

`python tests/test_pipeline.py` (44) and `python tests/test_official_metric.py`
(7). No pytest needed. Runs in ~4 s on CPU. Several encode real bugs:

- the tuning grid must bracket every fitted value **and** every fitted value in
  every artefact (checks 73 values; `seed_threshold` was pinned at its ceiling in
  5/5 folds, then again at 0.95 after widening — check this every time);
- tune/report split must separate **observations**, not readings (59.8% of
  "held-out" observations were also in the tuning half);
- uint8 and float probability maps must extract identical instances;
- fragmentation must use the host's any-overlap rule;
- the ensemble threshold transfer must name the top of a distribution, not its
  floor (a step-function tail means whole ranges of thresholds admit the same
  pixels), and both its histograms must come from one image population;
- the spine target must parse whatever nesting COCO used, must sit on its own
  filament, and must reach the loss only when `--spine-weight` is set;
- the operating point a submission uses must be chosen by an explicit name, never
  by sort order, and the list of names must include `ensemble_config.json` —
  which does *not* end in `_tuned.json` and so was invisible to the glob that
  looked for one.

Note that "every fitted value sits inside its grid" checks **0 values** unless
`kernels/_runs/` is present — it walks the run artefacts, which are gitignored.
It reports the count it checked; on the user's machine that is 73. A fresh clone
will pass it vacuously.

## Bugs already found and fixed — do not reintroduce

Earlier sessions: `persistent_workers=True` froze the dataset epoch (+0.052);
fp16 NaN on B4 was silent because NaN comparisons are False; validation used
`sorted()[:n]` so it only validated on 2011; `predict_full` assumed 1 output
channel; censored tuning grid; reading-level tune/report leak; fragmentation
threshold mismatch; predict kernel choosing its config by alphabetical sort
order.

25 Aug: `requirements.txt` pinned `opencv-python-headless==4.13.0`, which does
not exist on PyPI — `cv2.__version__` reports `4.13.0` but the wheel is
`4.13.0.90`, so the whole file failed to install. It is one of the two required
repository deliverables, so this was the first thing a judge reproducing the
pipeline would have hit. Also: the spine parse and coordinate-order traps above.

## Suggested priority order

0. **`--smoke` on the first pod, before anything else.** Nothing in this repo
   has ever run on a non-Kaggle GPU. Costs cents; proves the pipeline.
1. **Re-test capacity on Ampere with bf16** (B4, and larger if it holds up).
   The only identified lever that could be worth more than +0.05, and the one
   measurement against it is confounded by fp16 overflow on a T4. Pair it with
   a matched B0 baseline on the same pod.
2. **Submit a calibrated config.** Recompute it first with
   `src/calibrate_ensemble.py --on test` (the stored 0.9255 is a lower bound,
   see above), then one predict run. Both are GPU-only.
3. **Test the spine auxiliary head** — sanctioned, built, covered by tests, and
   verified end to end on synthetic data (spine alignment 93.4%, gradient
   reaches head channel 1, checkpoint round-trips, inference reads channel 0)
   but still never run against real annotations. `--spine-weight 0.3` on fold 0,
   against its paired baseline. Watch the preflight alignment line.
4. **Attack the near-miss cliff** (+0.0359 available). We over-segment by 8%, so
   the lever is boundary *shape*, not size. Consider a boundary-aware loss.
5. **Re-run the pooled OOF fit** once a better model exists; worth +0.02-0.03
   over per-fold configs, and the grid may still be censored. This now also
   fits `fallback_min_area` (the empty-observation rescue, commit 4e801b9),
   which is on the grid at 0 and has never been fitted.
6. **Write the 4-page report.** Required figures: a pipeline diagram and example
   segmentations. Required content: train/val/test split strategy, error bars (we
   have per-fold sd and measured selection optimism), and
   strengths/weaknesses/improvements — the near-miss and ceiling analyses are
   ready-made for that section.
7. Larger tiles (768px) for context; 5-fold retrain of whatever wins.

## Working agreements with the user

- **Never submit to Kaggle without explicit approval.** Prepare the CSV, show the
  validation report, wait. This was reversed once and then reinstated — it holds.
- Otherwise work autonomously: train, tune, fix, read the competition site, commit.
- Full git permissions granted; commit and push directly to `main`.
- The user has ~$30 for GPU rental, not yet spent. Recommendation on file:
  **RunPod Community, RTX 3090/4090, ephemeral pods, self-terminating** — buy
  *width* (many cheap parallel instances), not speed, because training is
  input-bound (233 s/epoch on a T4) and a faster card would idle. **A100 and
  Instant Clusters are both wrong**: A100 is 4x the cost for memory we never use,
  and clusters cost *more* ($1.79/h vs $1.49/h) to provide interconnect that our
  embarrassingly-parallel single-GPU jobs never touch and that `train.py` has no
  DDP support to use. Storage bills on stopped pods on every provider — hence
  ephemeral. If a RunPod API key is provided it will be in the `RUNPOD_API_KEY`
  environment variable; never print it.
- The user cares enormously about this competition. Be straight about what the
  evidence supports and what it does not.

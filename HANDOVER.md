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

## Current standing (25 Aug 2026)

- **One submission ever: public LB PQ 0.37**, rank ~39 of ~300.
- Top of board is now **0.55 / 0.55 / 0.54 / 0.53**, then a gap to 0.40.
- 13h ago the host said of those top scores: *"I would not compare very high
  scores with mine. Any score higher than 0.35 is of great value for us and we
  will examine them carefully."* A 12th-place competitor: *"I guess training with
  test labels from original dataset."* 139-180 of the 180 test images have public
  ground truth in the MAGFiLO 1.0 release. **We do not use the leak** — it is the
  user's stated principle and the host says it will not be rewarded.
- Our 0.37 already clears the host's stated 0.35 bar.

## The metric — verified, not assumed

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

**Label smoothing (-0.011) and B4 (-0.021) are both rejected.** B4 still triggers
fp32 recomputation for non-finite fp16 logits and costs 378 s/epoch against B0's
233 s. Beware: against fold 0's *old-code* 0.4065, label smoothing looked like a
+0.022 gain. It is a loss. **Never compare across code versions.**

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

## The CV-to-LB gap, and the unfinished fix

CV 0.4167 -> LB 0.37, a gap of **-0.047**. Leading hypothesis: the operating
point is fitted on single-model out-of-fold maps but applied to a five-model
average. Averaging shaves probability peaks, so `seed_threshold=0.95` admits
less. Symptom: 6.15 instances/image on test vs 6.8 per observation out-of-fold.

`src/calibrate_ensemble.py` fixes this without labels — it holds the *admitted
pixel mass* fixed rather than the threshold. **It has run and produced a result
that has never been used:**

`kernels/_runs/out_calib/ensemble_config.json` —
`seed_threshold 0.95 -> 0.9255`, `mask_threshold 0.40 -> 0.392`, everything else
unchanged (areas are pixel counts; the ratio moves with the transfer).

**IMMEDIATE NEXT ACTION: re-run `filament-predict` with a calibrated config and
submit (after asking the user).** Point the predict kernel at `filament-calib`,
or pass `--config` explicitly. The predict kernel prefers `oof_tuned.json` by
name, so this needs a deliberate change — do not let it pick by sort order; that
was a real bug, see commit 9e5f4c0.

**Recompute the config with `--on test` first (commit a9e1d82).** The stored
0.9255 was measured on *training* images, where 4 of 5 models have seen each
image, so the ensemble maps there are sharper than they will ever be on test and
the correction is a **lower bound**, not an estimate. `--on test` measures both
histograms on the 180 test images, where no model has seen anything, and pools
all five models for the single-model family instead of picking one. It reads
test *pixels*, never test annotations, so it stays label-free.

Measuring both halves on the *same* images is the point, not an incidental
detail: on a synthetic population 3x denser in filaments, transferring within
either population agrees to 0.004, while mixing them moves `mask_threshold` by
0.19 — "correcting" a real difference in the sky as though it were an artefact
of averaging. Asserted in `tests/test_pipeline.py`.

The shift is smaller than expected (-0.025 on seed), so calibration probably does
not explain the whole -0.047. Other candidates: test-set annotator composition
(unknown), and ordinary generalisation gap.

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

## Tests — 45 checks, all passing

`python tests/test_pipeline.py` (38) and `python tests/test_official_metric.py`
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
  filament, and must reach the loss only when `--spine-weight` is set.

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

1. **Submit a calibrated config.** Recompute it first with
   `src/calibrate_ensemble.py --on test` (the stored 0.9255 is a lower bound,
   see above), then one predict run. Both are GPU-only.
2. **Test the spine auxiliary head** — sanctioned, built, now covered by tests
   but still never run against real annotations. `--spine-weight 0.3` on fold 0,
   compare against **0.4387** and nothing else; never across code versions.
   Watch the preflight alignment line before the first epoch.
3. **Attack the near-miss cliff** (+0.0359 available). We over-segment by 8%, so
   the lever is boundary *shape*, not size. Consider a boundary-aware loss.
4. **Re-run the pooled OOF fit** once a better model exists; worth +0.02-0.03
   over per-fold configs, and the grid may still be censored.
5. **Fix the 10 empty observations** with a relax-until-non-empty fallback.
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

#!/usr/bin/env bash
# Run an experiment queue on a rented pod, then STOP BILLING - whatever happens.
#
# Usage, from the pod's shell (or as the pod's docker command):
#
#     export RUNPOD_API_KEY=...            # only needed if runpodctl is absent
#     export GITHUB_TOKEN=...              # only needed to push results back
#     bash tools/run_pod_experiment.sh --spine 0.3
#
# `bootstrap_pod.sh` prepares a pod and stops at a shell prompt, which is the
# right shape when a human is watching and the wrong one when nobody is.  A pod
# that finishes its job and keeps running bills at the same rate as one that is
# training, so the single most expensive failure mode here is not a bad
# hyperparameter - it is a successful run nobody noticed had finished.
#
# Three things therefore hold unconditionally:
#
#   1. Termination runs from a trap, so it fires on success, on a crash, on a
#      failed job, and on Ctrl-C.  It is not the last line of the happy path.
#   2. A wall-clock watchdog kills the pod even if the training process wedges
#      somewhere a trap cannot see.
#   3. Results are pushed *before* termination, never after.  A pod that has
#      billed for two hours and produced nothing recoverable is the whole loss.
#
# What does NOT survive a pod: anything outside a network volume.  Checkpoints
# are ~50 MB each and gitignored, so if the five-fold ensemble is the goal,
# mount a network volume and set ARTIFACT_DIR to a path on it - otherwise the
# folds train, report their PQ, and evaporate.

set -uo pipefail   # deliberately not -e: a failed job must still reach cleanup

REPO=${REPO:-https://github.com/msraaghavan/Solar.git}
WORK=${WORK:-/workspace}
ARTIFACT_DIR=${ARTIFACT_DIR:-$WORK/artifacts}
RESULT_BRANCH=${RESULT_BRANCH:-pod-results}
# Names the Kaggle dataset this pod publishes into.  Distinct per pod, because
# two pods running in parallel would otherwise version the same dataset and race.
RUN_TAG=${RUN_TAG:-$(date -u +%m%d%H%M)}
KAGGLE_USER=${KAGGLE_USER:-raaghavanms}
MAX_HOURS=${MAX_HOURS:-6}
KEEP_ALIVE=${KEEP_ALIVE:-0}       # 1 = do not terminate (debugging only)

SPINE_WEIGHTS=()
ENCODERS=()
FOLD=${FOLD:-0}
EPOCHS=${EPOCHS:-30}
BASELINE=${BASELINE:-1}    # also run spine-weight 0 on this pod; see below
SMOKE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --spine)  SPINE_WEIGHTS+=("$2"); shift 2 ;;
        --fold)   FOLD="$2";   shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --encoder) ENCODERS+=("$2"); shift 2 ;;
        --no-baseline) BASELINE=0; shift ;;
        --smoke)  SMOKE=1; shift ;;
        --keep-alive) KEEP_ALIVE=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ ${#SPINE_WEIGHTS[@]} -eq 0 ] && SPINE_WEIGHTS=(0.3)
[ ${#ENCODERS[@]} -eq 0 ] && ENCODERS=(tf_efficientnet_b0)

# --smoke: one epoch, five steps, two validation files.  Costs a few cents and
# exercises every path that can fail after an hour of billing - the spine
# preflight, a real training step, a full-disk inference, instance extraction,
# the checkpoint round-trip and the result push.  Run this first, always.
if [ "$SMOKE" = "1" ]; then
    EPOCHS=1
    SMOKE_ARGS=(--max-steps 5 --val-every 1 --val-files 2)
    # evaluate_fold scores the whole fold and tunes over the grid by default -
    # ~141 full-disk inferences at TTA 4, which is far and away the longest part
    # of a smoke run and tests nothing that four images do not.  Cap it, or the
    # cheap preflight costs most of an hour.
    EVAL_ARGS=(--max-files 4 --tta 1)
    MAX_HOURS=1
    echo "=== SMOKE RUN: 1 epoch, 5 steps. Validates the pipeline, not the model. ==="
else
    SMOKE_ARGS=()
    EVAL_ARGS=()
fi

# A spine run on its own cannot answer the question it is asked.  Fold 0's
# baseline of 0.4387 was measured with different code, on a T4, in fp16; this
# pod is different code, on a different card, in bf16.  Comparing across that is
# how label smoothing once looked like a +0.022 gain when it was a -0.011 loss.
# So the baseline is re-run here, in the same pod and the same precision, and
# the comparison is against *that*.  It doubles the cost of the experiment and
# is the only thing that makes the result mean anything.
if [ "$BASELINE" = "1" ]; then
    SPINE_WEIGHTS=(0 "${SPINE_WEIGHTS[@]}")
fi

# The queue is every encoder crossed with every spine weight, which is what lets
# the *capacity* question be asked the same way:
#
#   --encoder tf_efficientnet_b0 --encoder tf_efficientnet_b4 --spine 0 --no-baseline
#
# runs B0 and B4 back to back on one pod, one card, one precision, one code
# version - so the difference between their PQs is the encoder and nothing else.
# The single existing B4 number cannot be used for this: it was measured on a T4
# in fp16, where B4 overflowed and GradScaler dropped steps silently.
JOBS=()
for enc in "${ENCODERS[@]}"; do
    for w in "${SPINE_WEIGHTS[@]}"; do
        JOBS+=("$enc|$w")
    done
done
echo "queue (${#JOBS[@]} jobs): ${JOBS[*]}"

STARTED=$(date +%s)
SUMMARY="$ARTIFACT_DIR/pod_summary.txt"

# --------------------------------------------------------------------------- #
# Termination.  Everything below exists to make this run exactly once, always.
# --------------------------------------------------------------------------- #

terminate_pod() {
    if [ "$KEEP_ALIVE" = "1" ]; then
        echo "!!! KEEP_ALIVE=1: pod left running and BILLING. Remove it yourself."
        return
    fi
    local id="${RUNPOD_POD_ID:-}"
    if [ -z "$id" ]; then
        echo "!!! RUNPOD_POD_ID is unset - cannot self-terminate. REMOVE THIS POD MANUALLY."
        return
    fi

    echo "=== terminating pod $id ==="
    if command -v runpodctl >/dev/null 2>&1; then
        runpodctl remove pod "$id" && return
        echo "runpodctl failed; falling back to the REST API"
    fi

    if [ -n "${RUNPOD_API_KEY:-}" ]; then
        # The key goes in via a header file rather than on the command line:
        # arguments are world-readable in /proc on a shared host.
        local hdr; hdr=$(mktemp); chmod 600 "$hdr"
        printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" > "$hdr"
        curl -s -X DELETE "https://rest.runpod.io/v1/pods/$id" -H @"$hdr" >/dev/null
        rm -f "$hdr"
    else
        echo "!!! no runpodctl and no RUNPOD_API_KEY. REMOVE THIS POD MANUALLY."
    fi
}

# Kaggle is the primary way results leave this pod, for two reasons.  There is
# no GitHub token on this account, so the git path below is usually dead; and
# checkpoints do not survive a pod without a network volume, while a Kaggle
# dataset both stores them and can be attached directly to the prediction kernel
# by `dataset_sources` - so a fold trained here is immediately usable there.
publish_to_kaggle() {
    command -v kaggle >/dev/null 2>&1 || python -m kaggle --version >/dev/null 2>&1 || {
        echo "!!! no kaggle CLI; results cannot leave this pod"; return 1; }
    local slug="$KAGGLE_USER/filament-pod-$RUN_TAG"
    local stage="$WORK/publish"
    rm -rf "$stage"; mkdir -p "$stage"

    cp -f "$SUMMARY" "$stage/" 2>/dev/null
    # $1 = "light" for a progress heartbeat: text only.  Checkpoints are ~50 MB
    # each and would make a heartbeat cost minutes of upload, which defeats the
    # purpose of having one.
    [ "${1:-full}" = "light" ] || cp -f "$ARTIFACT_DIR"/*.pt "$stage/" 2>/dev/null
    cp -f "$ARTIFACT_DIR"/*_history.json "$stage/" 2>/dev/null
    cp -f "$ARTIFACT_DIR"/*_tuned.json "$stage/" 2>/dev/null
    # Tails, not whole logs: enough to read why a job failed or whether it was
    # still improving when it stopped, without uploading megabytes.
    for log in "$ARTIFACT_DIR"/train_*.log "$ARTIFACT_DIR"/eval_*.log; do
        [ -f "$log" ] && tail -n 200 "$log" > "$stage/$(basename "$log")"
    done

    cat > "$stage/dataset-metadata.json" <<META
{"title": "filament-pod-$RUN_TAG", "id": "$slug", "licenses": [{"name": "CC0-1.0"}]}
META

    echo "=== publishing $(ls -1 "$stage" | wc -l) files (${1:-full}) to kaggle $slug ==="
    # create the first time, version every time after.  Either can be the one
    # that works, so try both and let the summary below report what landed.
    python -m kaggle datasets create -p "$stage" -r zip -q 2>&1 | tail -3 ||
    python -m kaggle datasets version -p "$stage" -r zip -m "pod $RUN_TAG" -q 2>&1 | tail -3
    echo "=== kaggle dataset: $slug ==="
}

push_results() {
    [ -f "$SUMMARY" ] || return 0
    publish_to_kaggle || echo "!!! Kaggle publish failed; see the summary below"
    [ -n "${GITHUB_TOKEN:-}" ] || { cat "$SUMMARY"; return 0; }
    echo "=== pushing results to $RESULT_BRANCH ==="
    cd "$WORK/Solar" || return 0
    mkdir -p results
    cp -f "$SUMMARY" results/ 2>/dev/null
    cp -f "$ARTIFACT_DIR"/fold*_history.json results/ 2>/dev/null
    cp -f "$ARTIFACT_DIR"/*_tuned.json       results/ 2>/dev/null
    # Tails, not whole logs: enough to read why a job failed or whether it
    # was still improving when it stopped, without committing megabytes.
    for log in "$ARTIFACT_DIR"/train_*.log "$ARTIFACT_DIR"/eval_*.log; do
        [ -f "$log" ] && tail -n 120 "$log" > "results/$(basename "$log")"
    done

    git config user.email "pod@localhost"
    git config user.name  "runpod"
    git checkout -B "$RESULT_BRANCH" >/dev/null 2>&1
    git add -f results
    git commit -q -m "Pod results: fold $FOLD, jobs ${JOBS[*]}" 2>/dev/null
    git push -q -f "https://x-access-token:${GITHUB_TOKEN}@github.com/msraaghavan/Solar.git" \
        "$RESULT_BRANCH" 2>&1 | sed 's/x-access-token:[^@]*@/x-access-token:***@/g'
}

# Ctrl-C fires the INT trap *and then* the EXIT trap, so an unguarded handler
# pushes twice and issues two terminate calls (verified, not assumed).  Run once.
CLEANED=0
cleanup() {
    local code=$?
    [ "$CLEANED" = "1" ] && return
    CLEANED=1
    # Kill the sleep first, then its subshell.  A bare `sleep` may still be
    # orphaned here; that is harmless, because the `&&` below means a sleep that
    # did not run to completion never reaches the terminate call, and the pod is
    # destroyed moments later anyway.
    if [ -n "${WATCHDOG:-}" ]; then
        pkill -P "$WATCHDOG" 2>/dev/null
        kill "$WATCHDOG" 2>/dev/null
    fi
    echo ""
    echo "=== cleanup (exit $code, $(( ($(date +%s) - STARTED) / 60 )) min elapsed) ==="
    push_results
    terminate_pod
}

# The watchdog covers what the trap cannot: a wedged CUDA call, a hung mount, a
# dataloader deadlock.  None of those return control to this shell, and all of
# them bill.  A background subshell does not run the parent's EXIT trap (checked),
# so this terminates directly rather than going through cleanup.
#
# `&&`, not `;`.  With `;` a sleep killed during cleanup falls through to the
# next command, so tearing the watchdog down would *fire* it - the exact inverse
# of the intent.  Short-circuiting on the sleep's exit status is what makes
# cancellation mean cancellation.
( sleep $(( MAX_HOURS * 3600 )) && {
      echo "!!! watchdog: ${MAX_HOURS}h wall clock exceeded, terminating"
      terminate_pod
  } ) &
WATCHDOG=$!

trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

mkdir -p "$ARTIFACT_DIR"

# bootstrap_pod.sh needs ~/.kaggle/kaggle.json to fetch the competition data and
# runs under `set -e`, so a missing file kills it before anything trains.  The
# credentials arrive as environment variables rather than a file, because a pod's
# shell history is not private and a token pasted into a command persists there.
if [ ! -f ~/.kaggle/kaggle.json ] && [ -n "${KAGGLE_KEY:-}" ]; then
    mkdir -p ~/.kaggle
    ( umask 077; printf '{"username":"%s","key":"%s"}'         "${KAGGLE_USERNAME:-$KAGGLE_USER}" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json )
    chmod 600 ~/.kaggle/kaggle.json
    echo "wrote ~/.kaggle/kaggle.json from the environment"
fi

cd "$WORK"
[ -d Solar ] || git clone -q "$REPO"
cd Solar
git pull -q 2>/dev/null

bash tools/bootstrap_pod.sh || { echo "bootstrap failed"; exit 1; }

WORKERS=$(( $(nproc) > 16 ? 16 : $(nproc) ))
{
    echo "run started $(date -u +%FT%TZ)"
    echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
    echo "vcpus: $(nproc)  workers: $WORKERS"
    echo "queue: ${JOBS[*]}"
    echo "bootstrap: ok (data fetched, both test suites passed)"
} > "$SUMMARY"

# Publish once here, before any training.  There is no way to read a running
# pod's logs from the API, so without this a pod that dies during the first
# epoch is indistinguishable from one that never got its data - and the two
# call for completely different fixes.
publish_to_kaggle light || true

# --------------------------------------------------------------------------- #
# The queue.  Each job appends one line to the summary, so a pod that dies
# halfway still reports what it managed.
# --------------------------------------------------------------------------- #

for job in "${JOBS[@]}"; do
    ENCODER="${job%%|*}"
    weight="${job##*|}"
    # Tags every artefact for this job.  It has to carry the encoder as well as
    # the weight, or a two-encoder queue silently overwrites its own results.
    tag="${ENCODER#tf_efficientnet_}_spine${weight}"
    echo ""
    echo "=== fold $FOLD, encoder $ENCODER, spine-weight $weight ==="
    default_val=(--val-every 3 --val-files 40)
    [ "$SMOKE" = "1" ] && default_val=()
    python src/train.py \
        --fold "$FOLD" --encoder "$ENCODER" \
        --tile-size 512 --batch-size 8 --tiles-per-sample 8 \
        --epochs "$EPOCHS" "${default_val[@]}" "${SMOKE_ARGS[@]}" \
        --workers "$WORKERS" --spine-weight "$weight" \
        --out-dir "$ARTIFACT_DIR" 2>&1 | tee "$ARTIFACT_DIR/train_${tag}.log"

    checkpoint="$ARTIFACT_DIR/fold${FOLD}_best.pt"
    if [ ! -f "$checkpoint" ]; then
        echo "$ENCODER spine=$weight  FAILED (no checkpoint)" >> "$SUMMARY"
        continue
    fi
    mv "$checkpoint" "$ARTIFACT_DIR/fold${FOLD}_${tag}.pt"
    # train.py names its history by fold alone, so the second job in the
    # queue would overwrite the first one's curve.  Tag it by weight while
    # it still exists; the summary line survives either way, but the loss
    # curve is what tells an undertrained run apart from a worse one.
    mv -f "$ARTIFACT_DIR/fold${FOLD}_history.json" \
          "$ARTIFACT_DIR/fold${FOLD}_${tag}_history.json" 2>/dev/null

    # Score it the same way fold 0's 0.4387 baseline was scored.  Comparing to
    # anything measured under a different code version is how label smoothing
    # once looked like a +0.022 gain when it was a loss.
    #
    # --out must be per weight and inside ARTIFACT_DIR.  Its default is the
    # fixed path artifacts/fold0_tuned.json, so both jobs would write one
    # file *outside* the directory push_results collects: the baseline's
    # fitted operating point would be overwritten by the spine run's, and
    # neither would ever leave the pod.
    python src/evaluate_fold.py \
        --checkpoint "$ARTIFACT_DIR/fold${FOLD}_${tag}.pt" \
        --fold "$FOLD" \
        --out "$ARTIFACT_DIR/fold${FOLD}_${tag}_tuned.json" "${EVAL_ARGS[@]}" \
        2>&1 | tee "$ARTIFACT_DIR/eval_${tag}.log"

    pq=$(grep -oE '"pq_micro":[[:space:]]*[0-9.]+' "$ARTIFACT_DIR/eval_${tag}.log" \
         | tail -1 | grep -oE '[0-9.]+$')
    label="$ENCODER  spine=$weight"
    [ "$weight" = "0" ] && label="$ENCODER  baseline (spine off)"
    echo "$label  PQ=${pq:-unknown}" >> "$SUMMARY"
    publish_to_kaggle light || true
done

if [ "$BASELINE" = "1" ]; then
    cat >> "$SUMMARY" <<'NOTE'

Compare every spine row against the baseline row ABOVE, not against 0.4387.
That figure was measured on a T4 in fp16 under older code; this pod is none of
those three things, and comparing across them is how a -0.011 loss once read as
a +0.022 gain.
NOTE
fi

echo ""
echo "=== summary ==="
cat "$SUMMARY"

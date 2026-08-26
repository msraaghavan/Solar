"""Rent a RunPod GPU, run one experiment queue on it, and let it kill itself.

    python tools/launch_pod.py launch --smoke
    python tools/launch_pod.py launch -- --encoder tf_efficientnet_b0 \
                                         --encoder tf_efficientnet_b4 \
                                         --spine 0 --no-baseline
    python tools/launch_pod.py list
    python tools/launch_pod.py kill <podId>|all

Secrets come from the environment (``RUNPOD_API_KEY``, ``KAGGLE_USERNAME``,
``KAGGLE_KEY``) and are never printed, never passed as command-line arguments,
and never written to the repository.  They are placed in the pod's environment,
which is the only channel the pod can read them from without them also landing
in its shell history.

Two independent mechanisms stop this from billing forever, because the expensive
failure here is not a bad hyperparameter - it is a pod nobody noticed had
finished:

  * ``run_pod_experiment.sh`` terminates from a trap and carries its own
    wall-clock watchdog;
  * the start command below wraps that in a *second* trap and a *second*
    watchdog, covering the window the inner ones cannot - a failed git clone, a
    missing interpreter, an image that does not boot the way we expect.  If the
    repository never arrives the inner script never runs, and nothing inside it
    can help.

Results leave the pod through Kaggle rather than git: there is no GitHub token
on this account, and a Kaggle dataset also carries the checkpoints, which do not
survive an ephemeral pod at all.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
REPO = "https://github.com/msraaghavan/Solar.git"

# Anything Ampere (8.6) or later has native bf16; Turing and Volta do not, which
# rules out the T4 and the V100 and is the entire point of renting: the one B4
# measurement on record was taken in fp16 on a T4, where B4 overflowed and
# GradScaler dropped the offending steps silently, so it cannot distinguish
# "B4 is worse" from "B4 was undertrained".
#
# A wide list is what keeps a pod schedulable - "no instances currently
# available" is a real and frequent answer to a narrow request on Community, and
# training is input-bound anyway, so the card is rarely the bottleneck.
DEFAULT_GPUS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA RTX A4500",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX A4000",
    "NVIDIA L40S",
]
# cu1281, not cu1290.  A pod on a host whose driver is older than the image's
# CUDA falls back to the container's forward-compatibility libraries, which fail
# with "CUDA error 804: forward compatibility was attempted on non supported HW"
# - torch imports, reports its CUDA version happily, and sees no device.  That
# killed a smoke pod that had a perfectly healthy RTX 4090 in it.  A CUDA 12.8
# build runs on any 12.8+ driver, so pairing this image with ALLOWED_CUDA below
# makes the host pool as wide as it can safely be.
DEFAULT_IMAGE = "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204"

# Hosts whose driver is at least as new as the image's CUDA.  Without this,
# RunPod is free to place the pod on an 12.6 host and the run dies at bootstrap.
ALLOWED_CUDA = ["12.8", "12.9", "13.0"]


def request(method: str, path: str, body: dict | None = None) -> object:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as fh:
            raw = fh.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> {exc.code}: {exc.read().decode()[:600]}")


def start_command(args_line: str, hours: int) -> str:
    """The pod's whole life, as one shell command.

    Deliberately small.  Everything that can be version-controlled lives in the
    repository; this exists to fetch the repository, to guarantee termination in
    the cases where the repository never arrives, and to make those cases
    *legible*.

    That last part is not optional.  RunPod's REST API has no endpoint for a
    running pod's console, and a terminated pod's logs are gone with it - so a
    pod that dies during setup reports precisely nothing, and the first attempt
    here did exactly that.  Everything below therefore runs with stdout and
    stderr redirected to a file that is uploaded to Kaggle from the exit trap,
    before the pod is destroyed.  It is the only evidence that survives a failure
    happening *before* the repository, and hence the inner script, exists.
    """
    outer_seconds = (hours + 1) * 3600
    return "\n".join(
        [
            "set -x",
            "export DEBIAN_FRONTEND=noninteractive",
            "",
            "# /workspace is the default volume mount point, and this pod is",
            "# deliberately created with no volume - so it is not guaranteed to",
            "# exist.  `cd` into a missing directory would end the run here, with",
            "# no repository, no inner script and, before the log upload below,",
            "# no way to find out that was the reason.",
            "mkdir -p /workspace",
            "LOG=/workspace/pod_console.log",
            "exec >\"$LOG\" 2>&1",
            "",
            "upload_log() {",
            "  # Best effort, and separate from the results dataset: the inner",
            "  # script publishes results by replacing that dataset's contents,",
            "  # so writing the console into the same slug would delete them.",
            "  d=/workspace/_log; rm -rf $d; mkdir -p $d",
            '  tail -c 2000000 "$LOG" > $d/pod_console.log 2>/dev/null',
            "  printf '{\"title\":\"pod-%s-log\",\"id\":\"%s/pod-%s-log\","
            '"licenses":[{"name":"CC0-1.0"}]}\' '
            '"$RUN_TAG" "$KAGGLE_USERNAME" "$RUN_TAG" > $d/dataset-metadata.json',
            "  python -m pip install -q kaggle >/dev/null 2>&1",
            "  # `create` prints its refusal and exits 0, so `create || version`",
            "  # never falls through - the same trap that cost a finished pod its",
            "  # results.  Decide on the message.",
            "  o=$(python -m kaggle datasets create -p $d -q 2>&1)",
            "  case \"$o\" in *'already in use'*|*'already exists'*)",
            "    python -m kaggle datasets version -p $d -m console -q ;; esac",
            "}",
            "",
            "terminate() {",
            '  curl -s -X DELETE "' + API + '/pods/$RUNPOD_POD_ID"'
            ' -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null 2>&1',
            "}",
            "",
            "CLEANED=0",
            "cleanup() {",
            '  [ "$CLEANED" = "1" ] && return',
            "  CLEANED=1",
            "  upload_log",
            "  terminate",
            "}",
            "# Outer watchdog: covers a hang before the inner script is running.",
            "( sleep " + str(outer_seconds) + " && cleanup ) &",
            "trap cleanup EXIT INT TERM",
            "",
            "nvidia-smi || echo 'NO nvidia-smi'",
            "python -V || echo 'NO python'",
            "df -h /workspace",
            "command -v git >/dev/null ||"
            " (apt-get update -qq && apt-get install -y -qq git curl unzip)",
            "cd /workspace || exit 1",
            "rm -rf Solar",
            "git clone --depth 1 " + REPO + " || exit 1",
            "cd Solar || exit 1",
            "bash tools/run_pod_experiment.sh " + args_line,
            'echo "inner script exited $?"',
        ]
    )


def launch(argv: argparse.Namespace, passthrough: list[str]) -> None:
    env = {}
    for name in ("RUNPOD_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        value = os.environ.get(name)
        if not value:
            raise SystemExit(f"{name} is not set; the pod needs it")
        env[name] = value
    env["RUN_TAG"] = argv.tag
    env["MAX_HOURS"] = str(argv.hours)
    # Two pods running the same configuration at different seeds is the only
    # measurement of run-to-run spread, and that number is what says whether a
    # difference between two variants is a result or noise.
    env["SEED"] = str(argv.seed)

    args_line = " ".join(passthrough)
    body = {
        "name": f"filament-{argv.tag}",
        "imageName": argv.image,
        "gpuTypeIds": argv.gpu,
        "gpuCount": 1,
        "cloudType": "COMMUNITY",
        "containerDiskInGb": argv.disk,
        # No network volume and no persistent volume: storage bills on stopped
        # pods on every provider, and this pod is not meant to outlive its queue.
        "volumeInGb": 0,
        # Training was measured input-bound, not compute-bound, so the loader
        # decides throughput.  A fast card behind four workers idles.
        "minVCPUPerGPU": argv.vcpus,
        "allowedCudaVersions": argv.cuda,
        "env": env,
        "dockerStartCmd": ["bash", "-lc", start_command(args_line, argv.hours)],
        "ports": ["22/tcp"],
    }

    printable = {k: v for k, v in body.items() if k not in ("env", "dockerStartCmd")}
    print(json.dumps(printable, indent=2))
    print(f"env keys passed: {sorted(env)}  (values not shown)")
    print(f"queue args     : {args_line or '(defaults)'}")
    if argv.dry_run:
        print("\n--dry-run: nothing was created and nothing is billing.")
        print("\n--- start command ---")
        print(start_command(args_line, argv.hours))
        return

    pod = request("POST", "/pods", body)
    pod_id = pod.get("id") if isinstance(pod, dict) else None
    print(f"\npod {pod_id} created; billing stops when it terminates itself")
    print(f"results   -> kaggle dataset raaghavanms/filament-pod-{argv.tag}")
    print("watch     -> python tools/launch_pod.py list")
    print(f"kill now  -> python tools/launch_pod.py kill {pod_id}")


def pods_list() -> None:
    pods = request("GET", "/pods")
    if not pods:
        print("no pods; nothing is billing.")
        return
    for p in pods:
        gpu = (p.get("machine") or {}).get("gpuDisplayName") or p.get("machineId")
        print(
            f"{p.get('id')}  {str(p.get('name')):<24} {str(gpu):<18} "
            f"{p.get('desiredStatus')}  ${p.get('costPerHr')}/hr"
        )


def kill(argv: argparse.Namespace) -> None:
    targets = [p["id"] for p in request("GET", "/pods")] if argv.pod == "all" else [argv.pod]
    for pod_id in targets:
        request("DELETE", f"/pods/{pod_id}")
        print(f"terminated {pod_id}")
    if not targets:
        print("nothing to terminate")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("launch")
    p.add_argument("--tag", default=None, help="names the pod and its Kaggle dataset")
    p.add_argument("--gpu", action="append", default=None,
                   help="gpu type id, repeatable; tried in order")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--hours", type=int, default=6, help="wall-clock kill switch")
    p.add_argument("--disk", type=int, default=60)
    p.add_argument("--vcpus", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cuda", action="append", default=None,
                   help="acceptable host CUDA versions; repeatable")
    p.add_argument("--dry-run", action="store_true")

    sub.add_parser("list")
    k = sub.add_parser("kill")
    k.add_argument("pod")

    argv, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if argv.command == "launch":
        if not argv.tag:
            argv.tag = time.strftime("%m%d%H%M", time.gmtime())
        argv.gpu = argv.gpu or DEFAULT_GPUS
        argv.cuda = argv.cuda or ALLOWED_CUDA
        launch(argv, passthrough)
    elif argv.command == "list":
        pods_list()
    else:
        kill(argv)


if __name__ == "__main__":
    main()

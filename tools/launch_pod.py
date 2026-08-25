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

# Ada (8.9) and Ampere (8.6) both have native bf16, which is the entire point of
# renting rather than using the Kaggle T4: the one B4 measurement on record was
# taken in fp16, where B4 overflowed and GradScaler dropped the offending steps
# silently, so it cannot distinguish "B4 is worse" from "B4 was undertrained".
DEFAULT_GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA A40"]
DEFAULT_IMAGE = "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2204"


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

    Deliberately tiny.  Everything that can be version-controlled lives in the
    repository; this exists only to fetch the repository and to guarantee
    termination in the cases where the repository never arrives.
    """
    outer_seconds = (hours + 1) * 3600
    return "\n".join(
        [
            "set -x",
            "export DEBIAN_FRONTEND=noninteractive",
            "terminate() {",
            '  curl -s -X DELETE "' + API + '/pods/$RUNPOD_POD_ID"'
            ' -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null 2>&1',
            "}",
            "# Outer watchdog: covers a hang before the inner script is running.",
            "( sleep " + str(outer_seconds) + " && terminate ) &",
            "trap terminate EXIT INT TERM",
            "command -v git >/dev/null ||"
            " (apt-get update -qq && apt-get install -y -qq git curl unzip)",
            "cd /workspace || exit 1",
            "rm -rf Solar",
            "git clone --depth 1 " + REPO + " || exit 1",
            "cd Solar || exit 1",
            "bash tools/run_pod_experiment.sh " + args_line,
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
    p.add_argument("--vcpus", type=int, default=12)
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
        launch(argv, passthrough)
    elif argv.command == "list":
        pods_list()
    else:
        kill(argv)


if __name__ == "__main__":
    main()

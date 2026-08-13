"""Launch the out-of-fold evaluation kernel.

Shares push_train's source-publication gate: Kaggle processes dataset versions
asynchronously, and a kernel launched against a half-written version silently
runs the previous revision of the code.

    python tools/push_oof.py --max-files 40      # pilot, ~15 min
    python tools/push_oof.py --max-files 0       # full run over all 707
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from push_train import USER, dataset_updated_at, run, wait_for_src  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(REPO, "kernels", "oof")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="oof", help="kernel slug suffix")
    parser.add_argument("--max-files", type=int, default=40, help="0 runs all 707")
    parser.add_argument("--tta", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--diagnose-ensemble", action="store_true")
    parser.add_argument("--no-sync", action="store_true", help="skip re-publishing src/")
    args = parser.parse_args()

    if not args.no_sync:
        previous = dataset_updated_at()
        print(f"publishing src/ (previous version {previous or 'none'}) ...")
        subprocess.run([sys.executable, os.path.join(REPO, "tools", "sync_src.py")], check=True)
        wait_for_src(previous)

    slug = f"filament-{args.name}"
    stage = os.path.join(REPO, "kernels", "_runs", slug)
    os.makedirs(stage, exist_ok=True)

    config = {
        "MAX_FILES": args.max_files,
        "TTA": args.tta,
        "ROUNDS": args.rounds,
        "DIAGNOSE_ENSEMBLE": int(args.diagnose_ensemble),
    }

    source = open(os.path.join(TEMPLATE_DIR, "oof_kernel.py")).read()
    patched, n = re.subn(
        r"CONFIG = \{.*?\n\}", "CONFIG = " + json.dumps(config, indent=4), source, count=1, flags=re.S
    )
    if n != 1:
        raise SystemExit("could not substitute CONFIG block in oof_kernel.py")
    with open(os.path.join(stage, "oof_kernel.py"), "w") as fh:
        fh.write(patched)

    metadata = json.load(open(os.path.join(TEMPLATE_DIR, "kernel-metadata.json")))
    metadata["id"] = f"{USER}/{slug}"
    metadata["title"] = slug
    with open(os.path.join(stage, "kernel-metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"pushing {slug}: {json.dumps(config)}")
    print(run("kernels", "push", "-p", stage).strip())
    print(f"\n  status: python -m kaggle kernels status {USER}/{slug}")
    print(f"  output: python -m kaggle kernels output {USER}/{slug} -p out")


if __name__ == "__main__":
    main()

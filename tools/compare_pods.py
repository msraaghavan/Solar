"""Collect pod results and compare only the things that are comparable.

    python tools/compare_pods.py cap1234c spine5678b bnd1234 stem1234

Each pod publishes a ``pod_summary.txt`` to ``raaghavanms/filament-pod-<tag>``
holding one row per job, plus the GPU and seed the whole pod ran under.  This
downloads them and does two separate things with them, because they answer
different questions and mixing them is how a loss once read as a gain:

**Within a pod** the two arms differ in exactly one variable - same card, same
precision, same code version, same seed, same data order.  That difference is a
result, and it is the only kind of difference this project treats as one.

**Across pods** the baselines are identical configurations, so whatever they
differ by is *noise*: seed, hardware, and run-to-run variation together.  That
spread is the yardstick every within-pod delta has to clear before it means
anything.  Reporting it is the whole point - a +0.02 result is worthless if
identical configurations also differ by 0.02.

Nothing here compares a treatment in one pod against a treatment in another, and
nothing compares against the historical 0.4387: that was a T4, in fp16, under
older code, and none of those things is true of a pod.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

# An axis absent from a label means that axis did not exist when the pod ran,
# which is not a difference - it means the default.  Requiring the literal text
# "stem=0" made every pod launched before the stem axis existed unrecognisable
# as a baseline, so the whole comparison silently reported nothing.  It failed
# safe, but it failed.
DEFAULT_AXES = {"spine": 0.0, "bnd": 0.0, "stem": 0.0, "tile": 512.0}


def fetch(tag: str, into: str) -> str | None:
    """Download one pod's results; return its summary text, or None."""
    target = os.path.join(into, tag)
    os.makedirs(target, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", f"raaghavanms/filament-pod-{tag}", "-p", target, "--unzip", "-q"],
        capture_output=True, text=True,
    )
    path = os.path.join(target, "pod_summary.txt")
    if not os.path.exists(path):
        print(f"  {tag}: no results yet ({result.stderr.strip()[:60]})")
        return None
    return open(path, encoding="utf-8").read()


def parse(text: str) -> tuple[dict, list[dict]]:
    """Split a summary into its pod-level header and its per-job rows."""
    header, rows = {}, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("gpu:"):
            header["gpu"] = line.split(":", 1)[1].strip()
        elif line.startswith("seed:"):
            header["seed"] = line.split(":", 1)[1].split()[0]
        elif "PQ=" in line:
            row = {"label": line.split("  PQ=")[0].strip()}
            for key in ("PQ", "SQ", "RQ"):
                m = re.search(rf"{key}=([0-9.]+)", line)
                row[key] = float(m.group(1)) if m else None
            rows.append(row)
    return header, rows


def axes(label: str) -> dict:
    """The configuration a job ran, read out of its summary label."""
    written = dict(re.findall(r"([a-z_]+)=([-\d.]+)", label))
    values = {k: float(written.get(k, default)) for k, default in DEFAULT_AXES.items()}
    # The encoder is the first token, and it is part of the identity: a B4
    # baseline is not the same configuration as a B0 baseline.
    values["encoder"] = label.split()[0] if label.split() else "?"
    return values


def is_baseline(label: str) -> bool:
    values = axes(label)
    return all(values[k] == default for k, default in DEFAULT_AXES.items())


def treatment_of(label: str, base: str) -> str:
    """What this job changed relative to its pod's baseline."""
    a, b = axes(label), axes(base)
    changed = [f"{k}: {b[k]} -> {a[k]}" for k in a if a[k] != b[k]]
    return ", ".join(changed) or "(identical)"


def main() -> None:
    tags = sys.argv[1:]
    if not tags:
        raise SystemExit(__doc__)

    workdir = tempfile.mkdtemp(prefix="pods-")
    pods = {}
    try:
        for tag in tags:
            text = fetch(tag, workdir)
            if text:
                pods[tag] = parse(text)
    finally:
        pass

    if not pods:
        print("\nno completed pods yet.")
        shutil.rmtree(workdir, ignore_errors=True)
        return

    print(f"\n{'pod':<13}{'gpu':<26}{'seed':>5}  {'job':<44}{'PQ':>8}{'SQ':>8}{'RQ':>8}")
    for tag, (header, rows) in pods.items():
        for row in rows:
            print(f"{tag:<13}{header.get('gpu', '?')[:25]:<26}{header.get('seed', '?'):>5}  "
                  f"{row['label'][:43]:<44}"
                  f"{row['PQ'] if row['PQ'] is not None else float('nan'):>8.4f}"
                  f"{row['SQ'] if row['SQ'] is not None else float('nan'):>8.4f}"
                  f"{row['RQ'] if row['RQ'] is not None else float('nan'):>8.4f}")

    # --- the noise floor, first: every later number is judged against it ------
    baselines = [
        (tag, row) for tag, (_, rows) in pods.items()
        for row in rows if is_baseline(row["label"]) and row["PQ"] is not None
    ]
    floor = None
    print("\n--- noise floor: identical configurations across pods ---")
    if len(baselines) < 2:
        print("  fewer than two baselines finished; no floor yet, so treat every")
        print("  within-pod delta below as UNVERIFIED.")
    else:
        for tag, row in baselines:
            print(f"  {tag:<13}PQ {row['PQ']:.4f}   SQ {row['SQ']:.4f}   {row['label'][:40]}")
        pq = [row["PQ"] for _, row in baselines]
        sq = [row["SQ"] for _, row in baselines if row["SQ"] is not None]
        encoders = {axes(row["label"])["encoder"] for _, row in baselines}
        if len(encoders) > 1:
            print(f"  WARNING: baselines span {encoders}; that is not one "
                  f"configuration and the spread below is not a noise floor.")
        floor = max(pq) - min(pq)
        print(f"  spread: PQ {floor:.4f}   SQ {max(sq) - min(sq):.4f}"
              f"   over {len(baselines)} runs of the same configuration")
        print("  A within-pod delta smaller than this is not distinguishable from noise.")

    # --- within-pod deltas, the only real comparisons -------------------------
    print("\n--- within-pod deltas (same card, seed, code: the only valid comparison) ---")
    for tag, (_, rows) in pods.items():
        base = next((r for r in rows if is_baseline(r["label"]) and r["PQ"] is not None), None)
        if base is None:
            print(f"  {tag}: no baseline arm finished; nothing here is interpretable")
            continue
        for row in rows:
            if row is base or row["PQ"] is None:
                continue
            d_pq = row["PQ"] - base["PQ"]
            d_sq = (row["SQ"] - base["SQ"]) if None not in (row["SQ"], base["SQ"]) else float("nan")
            if floor is None:
                verdict = "unverified (no noise floor)"
            elif abs(d_pq) <= floor:
                verdict = f"within noise ({floor:.4f})"
            else:
                verdict = "EXCEEDS NOISE" + (" - improvement" if d_pq > 0 else " - regression")
            treatment = treatment_of(row["label"], base["label"])
            print(f"  {tag:<13}{treatment[:40]:<42}"
                  f"dPQ {d_pq:+.4f}  dSQ {d_sq:+.4f}   {verdict}")

    print("\n  SQ is mean matched IoU.  It has sat at 0.63-0.68 in every run ever")
    print("  measured; a treatment that moves it is worth more than one that moves")
    print("  PQ through RQ alone, because ~1.06 PQ is available per unit of SQ.")
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

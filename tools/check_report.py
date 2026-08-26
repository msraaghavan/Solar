"""Assert that every figure quoted in report/report.tex matches its artefact.

A report is a claim about a run.  Numbers get transcribed by hand, runs get
repeated, and the two drift silently - at which point the most carefully
written section in the document is also the least trustworthy.  This re-reads
the out-of-fold artefact and checks the text against it.

    python tools/check_report.py
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTEFACT = os.path.join(REPO, "kernels", "_runs", "out_ooffull", "oof_tuned.json")
REPORT = os.path.join(REPO, "report", "report.tex")


def main() -> None:
    if not os.path.exists(ARTEFACT):
        print(f"no artefact at {ARTEFACT}; nothing to check against")
        return
    data = json.load(open(ARTEFACT))
    report = data["report"]
    text = open(REPORT, encoding="utf-8").read()

    # (as written in the report, value in the artefact, tolerance)
    checks = [
        ("0.4167", report["pq_micro"], 5e-4),
        ("0.6748", report["sq"], 5e-4),
        ("0.6175", report["rq"], 5e-4),
        ("2501", report["tp"], 0.5),
        ("1458", report["fp"], 0.5),
        ("1640", report["fn"], 0.5),
        ("0.632", report["precision"], 5e-4),
        ("0.604", report["hit_rate"], 5e-4),
        ("144", report["one_to_many"], 0.5),
        ("70", report["many_to_one"], 0.5),
        ("0.8027", report["mean_matched_dice"], 5e-4),
        ("0.6517", report["mean_semantic_dice"], 5e-4),
        ("321", report["near_miss_count"], 0.5),
    ]

    failures = []
    for written, actual, tolerance in checks:
        if written not in text:
            failures.append(f"{written} is no longer quoted in the report")
        elif abs(float(written) - float(actual)) > tolerance:
            failures.append(f"report says {written}, artefact says {actual}")

    above = sum(1 for v in data["iou_distribution"] if v > 0.90)
    if above != 1 and "one matched pair in 2501" in text:
        failures.append(f"report claims one pair above IoU 0.90; artefact has {above}")

    for message in failures:
        print("  FAIL", message)
    if failures:
        sys.exit(1)
    print(f"report/report.tex agrees with the artefact on {len(checks)} figures")


if __name__ == "__main__":
    main()

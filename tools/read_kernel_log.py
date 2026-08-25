"""Print a Kaggle kernel log as the plain text the kernel actually emitted.

``kaggle kernels output`` writes the log as a JSON array of stream records, one
per flushed line, each with the text under ``data``.  Reading that with grep
means reading escape sequences, so every log inspection ends up re-deriving the
same unescaping.  This does it once.

    python tools/read_kernel_log.py kernels/_runs/out_calib_test/filament-calib.log [pattern]
"""

from __future__ import annotations

import json
import re
import sys


def text(path: str) -> str:
    raw = open(path, encoding="utf-8", errors="ignore").read()
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        # A running kernel's log can be truncated mid-array; salvage what parses.
        records = []
        for chunk in raw.splitlines():
            chunk = chunk.strip().lstrip(",")
            if chunk.startswith("{"):
                try:
                    records.append(json.loads(chunk))
                except json.JSONDecodeError:
                    pass
    return "".join(r.get("data", "") for r in records if isinstance(r, dict))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    body = text(sys.argv[1])
    if len(sys.argv) > 2:
        pattern = re.compile(sys.argv[2])
        body = "\n".join(line for line in body.splitlines() if pattern.search(line))
    print(body)


if __name__ == "__main__":
    main()

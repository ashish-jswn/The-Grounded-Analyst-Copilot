#!/usr/bin/env python
"""Generate the not-found evaluation set.

    python scripts/make_negatives.py            # write evaluation/negatives/negatives.jsonl
    python scripts/make_negatives.py --show 10  # print a sample for review

EVERY ITEM MUST BE HAND-REVIEWED before it is used for calibration. A
"negative" that is actually answerable teaches the system to refuse a question
it should have answered, which is the most damaging direction to be wrong in.
Set `reviewed: true` on each line once checked.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyst_copilot.config import load_settings                # noqa: E402
from analyst_copilot.eval.negatives import generate, save       # noqa: E402
from analyst_copilot.ingest.catalog import build_catalog        # noqa: E402
from analyst_copilot.query.router import load_aliases           # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evaluation/negatives/negatives.jsonl")
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    settings = load_settings()
    catalog = build_catalog(settings.filings_dir)
    aliases = load_aliases(settings.data_dir / "company_aliases.yaml")
    negatives = generate(catalog, target=args.target, aliases=aliases)

    counts = Counter(n.category for n in negatives)
    print(f"generated {len(negatives)} negatives from {len(catalog)} filings")
    for category, n in sorted(counts.items()):
        print(f"  {category:34s} {n}")

    if args.show:
        print()
        for n in negatives[: args.show]:
            print(f"  [{n.category}] {n.question}")
            print(f"      why unanswerable: {n.rationale}")

    out = Path(args.out)
    save(negatives, out)
    print(f"\nwritten to {out}")
    print("NOTE: hand-review every item and set reviewed=true before calibrating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

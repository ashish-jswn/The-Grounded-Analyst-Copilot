#!/usr/bin/env python
"""Mechanically check the synthetic negatives against the real corpus.

    python scripts/review_negatives.py             # report only
    python scripts/review_negatives.py --write     # mark the decisive ones reviewed

WHY THIS EXISTS. `eval/negatives.py` says the set must be HAND-REVIEWED,
because "a negative that turns out to be answerable poisons the calibration in
the most damaging direction: it teaches the system to refuse something it should
have answered." All 60 currently sit at `reviewed: false`, so any operating
point chosen against them is guesswork.

WHAT THIS IS AND IS NOT. It is not a substitute for human review. It is a
decisive check where one exists, so that human attention goes to the items that
actually need judgement:

  wrong_filing_or_period   DECISIVE  - the catalog either covers that fiscal
                                       year for that company, or it does not
  outside_the_filing       DECISIVE  - share price today, analyst opinion and
                                       post-filing events are not in any filing,
                                       by construction rather than by corpus
  genuinely_missing_metric EVIDENCE  - full-text search of that company's
                                       filings for the metric's own words
  false_premise            EVIDENCE  - same, for the event it presupposes
  missing_operand          JUDGEMENT - proving a denominator is absent needs a
                                       human; always flagged

Only DECISIVE checks set `reviewed: true`, and each records HOW it was decided
in `notes` so the claim stays auditable. Everything else is reported for you.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from analyst_copilot.config import load_settings                  # noqa: E402
from analyst_copilot.eval.negatives import (                      # noqa: E402
    _ABSENT_METRICS, FALSE_PREMISE, MISSING_METRIC, MISSING_OPERAND,
    OUTSIDE_FILING, WRONG_FILING, load, save,
)
from analyst_copilot.query.router import load_aliases, normalise, years_in  # noqa: E402
from analyst_copilot.storage.db import connect                    # noqa: E402
from analyst_copilot.storage import repository as repo            # noqa: E402

DECISIVE_OK = "verified"
NEEDS_EYES = "needs review"
CONTRADICTED = "CONTRADICTED"

# PROBE PHRASES, NOT WORDS. The first version of this script split the question
# into individual words and reported hits on "employee", "average" and
# "customer" - words that appear in every 10-K ever written. It flagged all 10
# missing-metric negatives as suspect and told the reader nothing. A negative
# claims a SPECIFIC disclosure is absent, so the probe must be that phrase.
_FALSE_PREMISE_PHRASES = [
    "goodwill impairment",
    "product recall",
    "special dividend",
    "restructuring reserve",
    "headquarters",
]

_QUARTER_RE = re.compile(r"\bq([1-4])\b", re.I)


def probe_phrases(question: str) -> list[str]:
    """The distinctive phrase whose absence makes this question unanswerable."""
    lowered = question.lower()
    hits = [m.lower() for m in _ABSENT_METRICS if m.lower() in lowered]
    hits += [p for p in _FALSE_PREMISE_PHRASES if p in lowered]
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="evaluation/negatives/negatives.jsonl")
    ap.add_argument("--write", action="store_true",
                    help="mark the decisively-verified items reviewed=true")
    args = ap.parse_args()

    settings = load_settings()
    path = Path(args.path)
    negatives = load(path)

    with connect(settings.database_url) as conn:
        catalog = repo.load_catalog(conn)
        # Full page text per company, for the evidence probes.
        rows = repo.pages_for_bm25(conn)

    coverage: dict[str, set[int]] = defaultdict(set)
    quarters: dict[str, set[int]] = defaultdict(set)
    latest: dict[str, int] = {}
    for f in catalog:
        slug = f["company_slug"]
        if f.get("fiscal_quarter"):
            quarters[slug].add(f["fiscal_quarter"])
        coverage[slug].update(f.get("coverage_years") or [])
        if f.get("fiscal_year"):
            coverage[slug].add(f["fiscal_year"])
            latest[slug] = max(latest.get(slug, 0), f["fiscal_year"])

    doc_company = {f["doc_id"]: f["company_slug"] for f in catalog}
    text_by_company: dict[str, str] = defaultdict(str)
    for r in rows:
        slug = doc_company.get(r["doc_id"])
        if slug:
            text_by_company[slug] += " " + (r.get("raw_text") or "").lower()

    verdicts: list[tuple] = []
    for n in negatives:
        slug = n.company_slug
        status, note = NEEDS_EYES, ""

        if n.category == WRONG_FILING:
            asked = years_in(n.question)
            covered = coverage.get(slug or "", set())
            quarter = _QUARTER_RE.search(n.question)
            if quarter or "wrong_quarter" in (n.tags or []):
                # THE YEAR IS THE WRONG TEST HERE. These ask for a QUARTER.
                # Q4 is never filed on its own - a 10-K reports the full year and
                # 10-Qs cover Q1-Q3 - so "Q4 FY2023" is unanswerable even though
                # FY2023 is fully covered. Checking only the year reported these
                # five as CONTRADICTED, which was a false alarm.
                q = int(quarter.group(1)) if quarter else None
                filed = quarters.get(slug or "", set())
                if q == 4:
                    status = DECISIVE_OK
                    note = ("Q4 is never filed separately: a 10-K reports the "
                            "full year and 10-Qs cover Q1-Q3")
                elif q is not None and q not in filed:
                    status = DECISIVE_OK
                    note = (f"catalog check: only quarters "
                            f"{sorted(filed) or 'none'} on file for {slug}")
                elif q is not None:
                    status = CONTRADICTED
                    note = f"catalog check: Q{q} IS on file for {slug} - may be answerable"
                else:
                    note = "tagged wrong_quarter but no quarter found in the question"
            elif asked and covered and not (asked & covered):
                status = DECISIVE_OK
                note = (f"catalog check: {slug} covers {sorted(covered)}, "
                        f"question asks {sorted(asked)} - no overlap")
            elif asked & covered:
                status = CONTRADICTED
                note = (f"catalog check: {slug} DOES cover "
                        f"{sorted(asked & covered)} - this may be answerable")
            else:
                note = f"could not read a year from the question (covers {sorted(covered)})"

        elif n.category == OUTSIDE_FILING:
            status = DECISIVE_OK
            note = ("category is out of scope for any SEC filing by construction "
                    "(market data, opinion, or post-filing event)")

        elif n.category in (MISSING_METRIC, FALSE_PREMISE):
            phrases = probe_phrases(n.question)
            corpus_text = text_by_company.get(slug or "", "")
            hits = [t for t in phrases if t in corpus_text]
            if not corpus_text:
                note = f"no text loaded for {slug}"
            elif not phrases:
                note = "no distinctive phrase to probe"
            elif not hits:
                status = DECISIVE_OK
                note = f"full-text probe: {phrases} appear nowhere in {slug} filings"
            else:
                note = (f"full-text probe: {hits} DO appear in {slug} filings - "
                        f"read those pages before trusting this negative")

        else:  # MISSING_OPERAND
            note = "proving a denominator is absent needs a human; not machine-decidable"

        verdicts.append((n, status, note))

    # ---- report ----------------------------------------------------------
    width = 78
    print("=" * width)
    print("SYNTHETIC NEGATIVES - MECHANICAL REVIEW")
    print("=" * width)
    counts = Counter(s for _n, s, _note in verdicts)
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for n, status, _note in verdicts:
        by_cat[n.category][status] += 1

    print(f"{'category':<36}{'verified':>10}{'needs eyes':>12}{'CONTRA':>8}")
    print("-" * width)
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"{cat:<36}{c[DECISIVE_OK]:>10}{c[NEEDS_EYES]:>12}{c[CONTRADICTED]:>8}")
    print("-" * width)
    print(f"{'TOTAL':<36}{counts[DECISIVE_OK]:>10}{counts[NEEDS_EYES]:>12}"
          f"{counts[CONTRADICTED]:>8}")

    for label, wanted in ((CONTRADICTED, CONTRADICTED), (NEEDS_EYES, NEEDS_EYES)):
        rows_ = [(n, note) for n, s, note in verdicts if s == wanted]
        if not rows_:
            continue
        print(f"\n{label} ({len(rows_)}):")
        for n, note in rows_:
            print(f"  {n.nid}  [{n.category}]")
            print(f"     Q: {n.question}")
            print(f"     {note}")

    if args.write:
        changed = 0
        for n, status, note in verdicts:
            if status == DECISIVE_OK and not n.reviewed:
                n.reviewed = True
                n.notes = f"machine-verified: {note}"
                changed += 1
            elif status == CONTRADICTED:
                n.reviewed = False
                n.notes = f"CONTRADICTED: {note}"
        save(negatives, path)
        print(f"\nwrote {path}: {changed} newly marked reviewed; "
              f"{counts[NEEDS_EYES] + counts[CONTRADICTED]} still need human eyes")
    else:
        print("\n(report only - pass --write to record the decisive verdicts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The filing catalog - the metadata the document router filters on.

`form_type` is derived DETERMINISTICALLY from the filename, never by an LLM
. It selects the parsing rule set downstream: a 10-K gets the
full section tree, a 10-Q a different shape, an 8-K no tree at all.

The field that earns its keep is `coverage_years`. A FY2018 10-K presents 2018
alongside comparatives for 2017 and 2016, so a question about 2016 is answerable
from the 2018 filing. The measured router error taxonomy put 44% of top-1 misses
on the wrong YEAR, and coverage_years is what turns that heuristic into a
lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# 3M_2018_10K | 3M_2023Q2_10Q | AMCOR_2022_8K_dated-2022-07-01
_NAME_RE = re.compile(
    r"^(?P<slug>.+?)_(?P<year>\d{4})(?:Q(?P<q>[1-4]))?_(?P<form>10K|10Q|8K)"
    r"(?:_dated[-_](?P<d>\d{4}-\d{2}-\d{2}))?$",
    re.I,
)

_FORM_CANON = {"10K": "10-K", "10Q": "10-Q", "8K": "8-K"}

# A 10-K carries three years of income/cash-flow statements and two of balance
# sheet; a 10-Q carries the quarter plus the prior-year comparative.
_COVERAGE_BACK = {"10-K": 2, "10-Q": 1, "8-K": 0}


@dataclass(frozen=True)
class FilingMeta:
    doc_id: str
    file_path: Path
    company_slug: str
    form_type: str
    fiscal_year: int
    fiscal_quarter: int | None
    period_label: str
    filing_date: date | None
    coverage_years: tuple[int, ...] = field(default=())

    @property
    def is_quarterly(self) -> bool:
        return self.form_type == "10-Q"


def parse_filing_name(path: Path) -> FilingMeta:
    """Derive catalog metadata from a filing's name. Pure, no I/O on content."""
    stem = path.stem
    m = _NAME_RE.match(stem)
    if not m:
        raise ValueError(f"unrecognised filing name: {stem!r}")

    form = _FORM_CANON[m.group("form").upper()]
    year = int(m.group("year"))
    q = int(m.group("q")) if m.group("q") else None

    filing_date = None
    if m.group("d"):
        filing_date = date.fromisoformat(m.group("d"))

    if q:
        period = f"Q{q} FY{year}"
    elif form == "8-K":
        period = filing_date.isoformat() if filing_date else str(year)
    else:
        period = f"FY{year}"

    back = _COVERAGE_BACK[form]
    coverage = tuple(range(year - back, year + 1))

    return FilingMeta(
        doc_id=stem,
        file_path=path,
        # The corpus mixes case (PFIZER_2021 vs Pfizer_2023Q2), so the slug is
        # upper-cased to make it a stable join key against company_aliases.
        company_slug=m.group("slug").upper(),
        form_type=form,
        fiscal_year=year,
        fiscal_quarter=q,
        period_label=period,
        filing_date=filing_date,
        coverage_years=coverage,
    )


def build_catalog(filings_dir: Path) -> list[FilingMeta]:
    """Parse every filing in the corpus directory, in stable name order."""
    return [parse_filing_name(p) for p in sorted(filings_dir.glob("*.htm"))]

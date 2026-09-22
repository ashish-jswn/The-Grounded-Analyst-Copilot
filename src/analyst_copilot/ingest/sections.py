"""The section tree - four signals in priority order.

SEC FILINGS HAVE NO SEMANTIC HEADING TAGS. MEASURED: only 6 of 78 filings
contain ANY <h1>-<h6>, median 1 tag. Heading detection that expects <h2> finds
nothing. The four signals, in the order they are trusted:

  1. ANCHORS      <a href="#x"> -> id="x". Present in 72/78 filings; the 6
                  without are all 8-Ks, which get no tree at all by design.
  2. SEC PATTERNS universal textual fallback (Item 1., Part II, Note 3, and the
                  statement titles) - works on any filing, including uploads.
  3. STYLING      bold / all-caps short lines. NOT <h*>.
  4. FRONT TOC    enrichment ONLY. Its printed page numbers must NEVER be
                  used for location - they index the filing's own pagination,
                  which drifts from our derived index by -2 to +16.

EXTRACT ANCHORS WITH lxml, NEVER WITH A REGEX. A regex probe reported
Microsoft as having zero anchors; it has 33. The <a> wraps more than 200
characters of nested <span>, so a naive pattern never matches.

The tree is built from structure alone. The optional per-filing LLM pass only
CLEANS titles that HTML split mid-word ("ITEM 1. B USINESS"); it never invents a
section, and it is behind `ingest.llm_title_cleanup`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

# Measured 73% gold-page recall from these patterns alone, no LLM.
STMT_PATTERNS: dict[str, re.Pattern[str]] = {
    "income": re.compile(
        r"(consolidated|condensed).{0,40}statements? of (income|operations|earnings)", re.I
    ),
    "balance": re.compile(
        r"(consolidated|condensed).{0,40}(balance sheets?|statements? of financial position)", re.I
    ),
    "cashflow": re.compile(
        r"(consolidated|condensed).{0,40}statements? of cash flows?", re.I
    ),
    "equity": re.compile(
        r"statements? of (stockholders|shareholders).{0,20}equity", re.I
    ),
    "compinc": re.compile(r"statements? of comprehensive (income|loss)", re.I),
    "segment": re.compile(
        r"(segment|business segment).{0,30}(information|reporting|results)", re.I
    ),
    "mdna": re.compile(r"management.s discussion and analysis", re.I),
}

_ITEM = re.compile(r"^item\s+(\d+[a-z]?)\s*[.\-:]?\s*(.*)$", re.I)
_PART = re.compile(r"^part\s+([ivx]+)\s*[.\-:]?\s*(.*)$", re.I)
_NOTE = re.compile(r"^note\s+(\d+)\s*[.\-:—–]?\s*(.*)$", re.I)

PART, ITEM, NOTE, STATEMENT, MDNA, EXHIBIT, OTHER = (
    "part", "item", "note", "statement", "mdna", "exhibit", "other"
)

# A TOC entry repeats the same titles on one early page. Nodes found on a page
# that holds many candidate headings are table-of-contents links, not real
# section starts, and must never become the citation target.
_TOC_DENSITY = 6


@dataclass
class Section:
    ordinal: int
    raw_title: str
    kind: str
    source: str                      # anchor | sec_pattern | styling | toc
    page_start: int
    level: int = 1
    stmt_type: str | None = None
    parent_ordinal: int | None = None
    page_end: int | None = None
    clean_title: str | None = None
    summary: str | None = None
    anchor_id: str | None = field(default=None)


def extract_anchor_targets(root: etree._Element) -> dict[str, str]:
    """Map an element id to the link text that points at it.

    lxml only - see the module docstring. This is the strongest tree signal.
    """
    targets: dict[str, str] = {}
    for a in root.iter("a"):
        href = a.get("href") or ""
        if not href.startswith("#"):
            continue
        text = re.sub(r"\s+", " ", "".join(a.itertext())).strip()
        if text:
            targets.setdefault(href[1:], text)
    return targets


def classify_title(title: str) -> tuple[str, str | None]:
    """Return (kind, stmt_type) for a candidate heading."""
    text = title.strip()
    for stmt, pattern in STMT_PATTERNS.items():
        if pattern.search(text):
            return (MDNA if stmt == "mdna" else STATEMENT), (
                None if stmt == "mdna" else stmt
            )
    if _PART.match(text):
        return PART, None
    if _ITEM.match(text):
        return ITEM, None
    if _NOTE.match(text):
        return NOTE, None
    if re.match(r"^exhibit\s", text, re.I):
        return EXHIBIT, None
    return OTHER, None


# A heading is a label, not a sentence. Measured failure this guards against:
# the paragraph "The following tables contain sales and operating income results
# by business segment for the fourth quarters of 2018 and 2017, ..." matched the
# `segment` statement pattern and was stored as a section title, polluting both
# `context_header` and `lexical_text` for every page in its range.
_MAX_HEADING_CHARS = 120
_MAX_HEADING_WORDS = 14


def _is_candidate(text: str) -> bool:
    """True when a line looks like a section heading rather than prose.

    THE STRUCTURAL TEST RUNS FIRST, AND THE ORDER WAS A REAL BUG.
    A full-stop rule used to sit above it: "prose ends in a full stop and runs
    on; 'Item 1.' and 'Note 3.' are fine because the stop follows the
    enumerator". That reasoning only holds when the TITLE does not also end in a
    period - and SEC filings routinely write:

        "Item 7. Management's Discussion and Analysis of Financial Condition
         and Results of Operations."

    13 words and a trailing stop, so it was discarded as prose. MEASURED: 86
    such headings across 20 of the 78 filings, including Item 7 on every one of
    them. 3M_2022_10K ended up with NO MD&A section at all - pages 18-47 were
    absorbed into "PART II", and one Note appeared to span 48 pages because the
    headings between it and the next survivor had been dropped too.

    AND THE RULE COULD NEVER HAVE HELPED. This function only ever returns
    True through a structural match, so a prose line returns False anyway - the
    full-stop test could only ever DISCARD genuine enumerated headings, never
    reject a false positive. The prose case its comment cited ("The following
    tables contain sales and operating income results by business segment for
    the fourth quarters of...") is already stopped by the length and word-count
    guards below, which are kept for exactly that purpose.
    """
    t = text.strip()
    if not (3 < len(t) <= _MAX_HEADING_CHARS):
        return False
    if len(t.split()) > _MAX_HEADING_WORDS:
        return False

    # An enumerated heading is a heading whatever punctuation trails it.
    if _PART.match(t) or _ITEM.match(t) or _NOTE.match(t):
        return True
    # The statement title must OPEN the line - buried in a sentence it is a
    # cross-reference ("...as shown in the consolidated statements of cash
    # flows"), not the start of that statement.
    for pattern in STMT_PATTERNS.values():
        m = pattern.search(t)
        if m and m.start() <= 4:
            return True
    return False


def build_tree(
    pages: list[tuple[int, str]],
    anchor_targets: dict[str, str] | None = None,
    *,
    form_type: str = "10-K",
) -> list[Section]:
    """Build the section tree for one filing.

    An 8-K gets NO tree: it has no Items in the 10-K sense,
    and 6 of 78 filings carry no anchors at all - all of them 8-Ks. Returning an
    empty tree is correct, not a failure.
    """
    if form_type == "8-K":
        return []

    # Signals 2 and 3: scan the head of each page for candidate headings. The
    # head is where a section starts; scanning the whole page would match every
    # cross-reference in the prose.
    per_page: dict[int, list[str]] = {}
    for seq, text in pages:
        lines = [ln.strip() for ln in text[:1200].split("\n") if ln.strip()]
        per_page[seq] = [ln for ln in lines[:40] if _is_candidate(ln)]

    # Signal 4 handling: drop pages that are clearly the front table of
    # contents. Their printed page numbers must never drive location.
    toc_pages = {seq for seq, cands in per_page.items() if len(cands) >= _TOC_DENSITY}

    sections: list[Section] = []
    seen: set[str] = set()
    for seq, candidates in sorted(per_page.items()):
        if seq in toc_pages:
            continue
        for title in candidates:
            key = re.sub(r"\W+", "", title.lower())[:60]
            if key in seen:
                continue
            seen.add(key)
            kind, stmt = classify_title(title)
            source = "anchor" if anchor_targets and title in anchor_targets.values() else "sec_pattern"
            sections.append(
                Section(
                    ordinal=len(sections),
                    raw_title=title,
                    kind=kind,
                    stmt_type=stmt,
                    source=source,
                    page_start=seq,
                    level=1 if kind == PART else 2,
                )
            )

    # Parent linkage: an Item belongs to the Part above it; a Note to the Item.
    last_part: int | None = None
    last_item: int | None = None
    for s in sections:
        if s.kind == PART:
            last_part, last_item = s.ordinal, None
        elif s.kind == ITEM:
            s.parent_ordinal = last_part
            last_item = s.ordinal
        else:
            s.parent_ordinal = last_item if last_item is not None else last_part
            s.level = 3 if last_item is not None else 2

    # A section runs until the next one starts.
    for i, s in enumerate(sections):
        s.page_end = (
            sections[i + 1].page_start - 1 if i + 1 < len(sections) else pages[-1][0]
        )
        if s.page_end < s.page_start:
            s.page_end = s.page_start
    return sections


def statement_pages(pages: list[tuple[int, str]]) -> dict[str, list[int]]:
    """Structure anchors: pages holding a primary financial statement.

    MEASURED 73.0% gold-page recall from these regexes alone, with no LLM and
    no embedding. Roughly 75% of gold evidence sits in the three primary
    statements, which is why this cheap signal is so strong.

    The following page is included too: a statement routinely spans a seam.
    """
    hits: dict[str, list[int]] = {k: [] for k in STMT_PATTERNS}
    by_seq = dict(pages)
    for seq, text in pages:
        head = text[:600]
        for stmt, pattern in STMT_PATTERNS.items():
            if pattern.search(head):
                hits[stmt].append(seq)
                if seq + 1 in by_seq:
                    hits[stmt].append(seq + 1)
    return {k: sorted(set(v)) for k, v in hits.items() if v}

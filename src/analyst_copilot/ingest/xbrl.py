"""Inline XBRL fact extraction - ingest step S8.

XBRL IS A FIRST-CLASS *FACT* PATH, NOT A FIRST-CLASS *ANSWER* PATH.
MEASURED: a tagged fact carries the gold answer for only 38% of questions
(the widely quoted 77% is *filing* coverage, which is a different claim). Its
larger value is supplying TYPED OPERANDS to the deterministic calculator where
the answer itself is never tagged - a ratio like capex/revenue is not a fact,
but both of its operands are.

WHY IT MATTERS HERE: the HTML table alignment gate passes on only 28% of data
tables corpus-wide, so 72% of tables yield markdown and no typed cells. XBRL
bypasses table parsing entirely for tagged facts, converting a would-be
abstention into an answer - on exactly the half of the benchmark that needs
computed operands.

THREE NON-NEGOTIABLE RULES, all enforced below:

  1. QUERY BY CONCEPT + PERIOD, NEVER BY VALUE. A value-first search was tried
     and produced garbage: an Activision question matched an unrelated
     share-based-compensation concept on a coincidental number, and a
     `CommonStockValue` search returned 468 coincidental matches. This module
     therefore never indexes on value; `facts_lookup` is (doc_id, qname,
     period_end).
  2. DIMENSIONAL CONTEXTS ARE FLAGGED. A fact carrying `explicitMember` is a
     segment or geography breakdown. Including them by default makes "revenue"
     return the consolidated figure PLUS every segment.
  3. THE ROW LABEL IS THE SAFETY NET. Walking from the fact element up
     to its enclosing <tr> recovers the label the filing actually displays, so a
     wrong-concept mapping is caught deterministically with no LLM - and the
     verbatim citation quote comes free.

Pure lxml. No Arelle, no model call, negligible ingest cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

from lxml import etree

# Inline XBRL puts facts in the ix: namespace and contexts in xbrli:, but
# prefixes vary between filers and some filings declare a default namespace.
# Matching on LOCAL NAME is the only robust option.
_FACT_TAGS = {"nonfraction", "nonnumeric"}

_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_NUMERIC_LABEL = re.compile(r"^[\s\d.,()$%+\-–—]*$")


def _local(tag) -> str:
    """Local name of an element tag, lowercased. Non-elements yield ''.

    TWO SPELLINGS, AND THE HTML ONE IS THE ONE WE ACTUALLY GET.
    An XML parser yields a namespaced tag `{http://...}nonFraction`. But we parse
    filings with lxml's HTML parser (SEC markup is far too malformed for the XML
    one), and that keeps the prefix IN the tag name, lowercased: `ix:nonfraction`.
    MEASURED on MICROSOFT_2023_10K - 1,563 `<ix:nonFraction>` in the raw bytes
    all arrive as tag `ix:nonfraction`, so stripping only on `}` matched nothing
    and the extractor silently found ZERO facts.

    Strip both separators. Attribute names are lowercased by the same parser,
    which is why `contextRef` is read as `contextref` below.
    """
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()


def _text(el: etree._Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).replace("\xa0", " ").strip()


def _parse_date(text: str) -> date | None:
    m = _DATE.search(text or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


@dataclass
class Context:
    """One xbrli:context - the period and dimensions a fact is reported for."""

    context_id: str
    period_start: date | None = None
    period_end: date | None = None
    is_instant: bool = False
    dimensions: dict[str, str] = field(default_factory=dict)

    @property
    def has_dimensions(self) -> bool:
        return bool(self.dimensions)


@dataclass
class XbrlFact:
    qname: str
    value: Decimal | None
    unit: str | None
    scale: int | None
    sign: int
    context: Context
    row_label: str | None = None
    text: str = ""

    @property
    def has_dimensions(self) -> bool:
        return self.context.has_dimensions


# ---------------------------------------------------------------------------
# Contexts and units
# ---------------------------------------------------------------------------
def build_context_map(root: etree._Element) -> dict[str, Context]:
    """Every `<xbrli:context>` in the document, keyed by id.

    Contexts live in a hidden `<ix:header>` block, so they must be read from the
    whole document rather than from any one page.
    """
    contexts: dict[str, Context] = {}
    for el in root.iter():
        if _local(el.tag) != "context":
            continue
        cid = el.get("id")
        if not cid:
            continue
        ctx = Context(context_id=cid)
        for child in el.iter():
            name = _local(child.tag)
            if name == "startdate":
                ctx.period_start = _parse_date(child.text or "")
            elif name == "enddate":
                ctx.period_end = _parse_date(child.text or "")
            elif name == "instant":
                ctx.is_instant = True
                instant = _parse_date(child.text or "")
                ctx.period_start = ctx.period_end = instant
            elif name in ("explicitmember", "typedmember"):
                dimension = child.get("dimension") or "dimension"
                ctx.dimensions[dimension] = (child.text or "").strip()
        contexts[cid] = ctx
    return contexts


def build_unit_map(root: etree._Element) -> dict[str, str]:
    """`<xbrli:unit>` id -> a readable unit ('USD', 'shares', 'USD/shares')."""
    units: dict[str, str] = {}
    for el in root.iter():
        if _local(el.tag) != "unit":
            continue
        uid = el.get("id")
        if not uid:
            continue
        measures = [
            (m.text or "").rsplit(":", 1)[-1].strip()
            for m in el.iter()
            if _local(m.tag) == "measure"
        ]
        measures = [m for m in measures if m]
        if not measures:
            continue
        # A divide unit renders as numerator/denominator, e.g. USD/shares.
        has_divide = any(_local(m.tag) == "divide" for m in el.iter())
        units[uid] = "/".join(measures[:2]) if has_divide and len(measures) > 1 else measures[0]
    return units


# ---------------------------------------------------------------------------
# The row-label safety net
# ---------------------------------------------------------------------------
def row_label_for(el: etree._Element) -> str | None:
    """The label the FILING displays for this fact's row.

    Walk up to the enclosing <tr> and take the first cell that is not itself a
    figure. MEASURED to recover the displayed label every time:
      PaymentsToAcquirePropertyPlantAndEquipment -> "Additions to property and
      equipment" (Microsoft), "Purchases of property, plant and equipment" (3M).

    Two payoffs beyond the check itself: the verbatim citation quote comes free,
    and the same us-gaap concept normalises across companies' different wording -
    which attacks the analyst-vs-filing vocabulary gap that held oracle-document
    BM25 to 18.3% R@10, with no hand-maintained synonym list.
    """
    row = None
    node = el.getparent()
    while node is not None:
        if _local(node.tag) == "tr":
            row = node
            break
        node = node.getparent()
    if row is None:
        return None

    for cell in row.iter():
        if _local(cell.tag) not in ("td", "th"):
            continue
        text = _text(cell)
        if text and not _NUMERIC_LABEL.match(text):
            return text
    return None


def label_matches_metric(row_label: str | None, metric_terms: list[str]) -> bool:
    """Deterministic wrong-concept detector - no LLM.

    Ask for capex and get a row labelled "Share-based compensation" and the fact
    is discarded before it can reach an answer. A missing label cannot refute
    anything, so it passes: this gate exists to REJECT on positive evidence of a
    mismatch, never to reject on absence.
    """
    if not row_label or not metric_terms:
        return True
    label = row_label.lower()
    return any(term.lower() in label for term in metric_terms if term)


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
def parse_fact_value(el: etree._Element) -> Decimal | None:
    """Apply the inline-XBRL `scale` and `sign` attributes to the shown text.

    BOTH ATTRIBUTES CHANGE THE NUMBER. A filing renders "1,577" with
    scale="6" and sign="-" and means -1,577,000,000. Ignoring `scale` under-reports
    by a factor of a million; ignoring `sign` flips a cash outflow into an
    inflow, which reads as entirely plausible and is exactly the silent error
    the rubric prices at -1.
    """
    raw = _text(el)
    if not raw:
        return None
    cleaned = raw.replace(",", "").replace("$", "").replace("%", "").strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").strip()
    if cleaned.startswith("-"):
        negative, cleaned = True, cleaned[1:].strip()
    if not cleaned or cleaned in {"—", "–", "-"}:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    scale = el.get("scale")
    if scale:
        try:
            value = value.scaleb(int(scale))
        except (ValueError, InvalidOperation):
            pass
    if (el.get("sign") or "").strip() == "-":
        negative = True
    return -value if negative else value


def fact_from_element(
    el: etree._Element,
    contexts: dict[str, Context],
    units: dict[str, str],
) -> XbrlFact | None:
    """Build one fact, or None when it carries no resolvable concept/context."""
    qname = (el.get("name") or "").strip()
    if not qname:
        return None
    context = contexts.get((el.get("contextRef") or el.get("contextref") or "").strip())
    if context is None:
        # A fact whose context we cannot resolve has no period, and a figure
        # without a period cannot be checked against the question. Drop it.
        return None

    is_numeric = _local(el.tag) == "nonfraction"
    value = parse_fact_value(el) if is_numeric else None
    unit_ref = (el.get("unitRef") or el.get("unitref") or "").strip()
    scale = el.get("scale")

    return XbrlFact(
        qname=qname,
        value=value,
        unit=units.get(unit_ref, unit_ref or None),
        scale=int(scale) if (scale or "").lstrip("-").isdigit() else None,
        sign=-1 if (value is not None and value < 0) else 1,
        context=context,
        row_label=row_label_for(el),
        text=_text(el),
    )


def iter_fact_elements(elements) -> list[etree._Element]:
    """Every inline-XBRL fact element among `elements`, in document order."""
    return [el for el in elements if _local(getattr(el, "tag", None)) in _FACT_TAGS]


def extract_facts(
    root: etree._Element,
    elements=None,
) -> list[XbrlFact]:
    """Extract facts from a parsed filing.

    `root` supplies contexts and units (they live in a hidden header, not on any
    page). `elements` restricts the scan to one page's elements so a fact can be
    attributed to the page it is rendered on - which is what makes an XBRL fact
    carry its own citation. Omit it to scan the whole document.
    """
    contexts = build_context_map(root)
    units = build_unit_map(root)
    scan = elements if elements is not None else list(root.iter())

    facts: list[XbrlFact] = []
    for el in iter_fact_elements(scan):
        fact = fact_from_element(el, contexts, units)
        if fact is not None:
            facts.append(fact)
    return facts


class FactExtractor:
    """Contexts and units resolved ONCE per filing, then applied page by page.

    Contexts live in a hidden `<ix:header>`, so they are document-scoped while
    facts are page-scoped — and a fact must be attributed to the page it renders
    on, because that page is its citation. Rebuilding the context map for each
    of ~110 pages would re-walk the whole DOM every time; a filing like
    MICROSOFT_2023_10K carries 469 contexts and 1,839 facts.
    """

    def __init__(self, roots: list[etree._Element]) -> None:
        self.contexts: dict[str, Context] = {}
        self.units: dict[str, str] = {}
        for root in roots:
            self.contexts.update(build_context_map(root))
            self.units.update(build_unit_map(root))

    @property
    def is_empty(self) -> bool:
        """True for the 20 pre-2019 filings that carry no inline XBRL at all.
        That is the documented HTML-only path, not a failure."""
        return not self.contexts

    def facts_for(self, elements) -> list[XbrlFact]:
        facts: list[XbrlFact] = []
        for el in iter_fact_elements(elements):
            fact = fact_from_element(el, self.contexts, self.units)
            if fact is not None:
                facts.append(fact)
        return facts


def fact_rows(
    facts: list[XbrlFact], doc_id: str, page_id: str | None
) -> list[dict]:
    """Rows ready for the `facts` table.

    `fact_id` is DERIVED from position, never random, so a re-ingest overwrites
    rather than duplicating - the same property every other id in this system
    has.
    """
    rows: list[dict] = []
    for i, f in enumerate(facts):
        rows.append(
            {
                "fact_id": f"{page_id or doc_id}.f{i}",
                "doc_id": doc_id,
                "qname": f.qname,
                "value": f.value,
                "unit": f.unit,
                "scale": f.scale,
                "sign": f.sign,
                "period_start": f.context.period_start,
                "period_end": f.context.period_end,
                "is_instant": f.context.is_instant,
                "dimensions": f.context.dimensions or None,
                "has_dimensions": f.has_dimensions,
                "row_label": f.row_label,
                "page_id": page_id,
                "block_id": None,
            }
        )
    return rows

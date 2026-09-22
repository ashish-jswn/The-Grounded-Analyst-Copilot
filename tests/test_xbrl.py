"""Inline XBRL fact extraction.

Every fixture here is parsed with lxml's HTML parser, which is what the ingest
pipeline uses - SEC markup is far too malformed for the XML parser. That choice
has a consequence the first version of this module got wrong, so it is pinned by
`test_local_name_strips_the_html_prefix` below.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from lxml import html as lxml_html

from analyst_copilot.ingest.xbrl import (
    _local, build_context_map, build_unit_map, extract_facts, fact_rows,
    label_matches_metric, parse_fact_value, row_label_for,
)

# A cash-flow row as EDGAR actually writes it: the figure is tagged, scaled to
# millions, and marked negative by `sign` while being DISPLAYED in parentheses.
FILING = """
<html><body>
<div style="display:none">
  <ix:header><ix:resources>
    <xbrli:context id="FY2023">
      <xbrli:period>
        <xbrli:startDate>2022-07-01</xbrli:startDate>
        <xbrli:endDate>2023-06-30</xbrli:endDate>
      </xbrli:period>
    </xbrli:context>
    <xbrli:context id="AT2023">
      <xbrli:period><xbrli:instant>2023-06-30</xbrli:instant></xbrli:period>
    </xbrli:context>
    <xbrli:context id="SEG2023">
      <xbrli:period>
        <xbrli:startDate>2022-07-01</xbrli:startDate>
        <xbrli:endDate>2023-06-30</xbrli:endDate>
      </xbrli:period>
      <xbrli:entity><xbrli:segment>
        <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">msft:ProductiveSegment</xbrldi:explicitMember>
      </xbrli:segment></xbrli:entity>
    </xbrli:context>
    <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
    <xbrli:unit id="shares"><xbrli:measure>xbrli:shares</xbrli:measure></xbrli:unit>
  </ix:resources></ix:header>
</div>
<table>
  <tr>
    <td>Additions to property and equipment</td>
    <td>$</td>
    <td><ix:nonFraction name="us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
         contextRef="FY2023" unitRef="usd" scale="6" sign="-"
         decimals="-6">28,107</ix:nonFraction></td>
  </tr>
  <tr>
    <td>Total current assets</td>
    <td><ix:nonFraction name="us-gaap:AssetsCurrent" contextRef="AT2023"
         unitRef="usd" scale="6" decimals="-6">25,202</ix:nonFraction></td>
  </tr>
  <tr>
    <td>Segment revenue</td>
    <td><ix:nonFraction name="us-gaap:Revenues" contextRef="SEG2023"
         unitRef="usd" scale="6" decimals="-6">9,000</ix:nonFraction></td>
  </tr>
</table>
</body></html>
"""


def parse(markup: str):
    parser = lxml_html.HTMLParser(encoding="utf-8", recover=True, huge_tree=True)
    return lxml_html.document_fromstring(markup.encode("utf-8"), parser=parser)


def facts_by_concept(root):
    return {f.qname: f for f in extract_facts(root)}


def fact_elements(root):
    """Find fact elements the way the module does.

    NOT with XPath `local-name()`. After HTML parsing the colon is part of the
    tag name rather than a namespace separator, so `local-name()` returns
    'ix:nonfraction' and an XPath match on 'nonfraction' finds nothing - the same
    trap the module's `_local` exists to avoid, one level up. Two of these tests
    failed on exactly that before this helper existed.
    """
    return [el for el in root.iter() if _local(getattr(el, "tag", None)) == "nonfraction"]


# ---------------------------------------------------------------------------
# The bug that made this module silently return nothing
# ---------------------------------------------------------------------------
def test_local_name_strips_the_html_prefix():
    """lxml's HTML parser keeps the prefix IN the tag name, lowercased.

    An XML parser gives `{http://...}nonFraction`; the HTML parser gives
    `ix:nonfraction`. Stripping only on '}' matched neither, and the extractor
    found ZERO facts on a filing with 1,563 of them - with no error.
    """
    assert _local("ix:nonfraction") == "nonfraction"
    assert _local("{http://www.xbrl.org/2013/inlineXBRL}nonFraction") == "nonfraction"
    assert _local("xbrli:context") == "context"
    assert _local("td") == "td"
    assert _local(None) == ""


def test_facts_are_found_at_all():
    root = parse(FILING)
    assert len(extract_facts(root)) == 3


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------
def test_duration_and_instant_contexts():
    contexts = build_context_map(parse(FILING))
    duration = contexts["FY2023"]
    assert duration.period_start == date(2022, 7, 1)
    assert duration.period_end == date(2023, 6, 30)
    assert duration.is_instant is False

    instant = contexts["AT2023"]
    assert instant.is_instant is True
    assert instant.period_start == instant.period_end == date(2023, 6, 30)


def test_dimensional_contexts_are_flagged():
    """A fact with explicitMember is a segment breakdown, not the consolidated
    figure. Without this flag, "revenue" returns the total PLUS every segment."""
    contexts = build_context_map(parse(FILING))
    assert contexts["FY2023"].has_dimensions is False
    assert contexts["SEG2023"].has_dimensions is True
    assert "us-gaap:StatementBusinessSegmentsAxis" in contexts["SEG2023"].dimensions


def test_units_resolve_to_readable_names():
    units = build_unit_map(parse(FILING))
    assert units["usd"] == "USD"
    assert units["shares"] == "shares"


# ---------------------------------------------------------------------------
# Value: scale and sign both change the number
# ---------------------------------------------------------------------------
def test_scale_and_sign_are_both_applied():
    """Ignoring `scale` under-reports by 1e6; ignoring `sign` turns a cash
    outflow into an inflow - a plausible-looking figure and a silent -1."""
    fact = facts_by_concept(parse(FILING))["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"]
    assert fact.value == Decimal("-28107000000")
    assert fact.sign == -1
    assert fact.scale == 6
    assert fact.unit == "USD"


def test_unsigned_fact_stays_positive():
    fact = facts_by_concept(parse(FILING))["us-gaap:AssetsCurrent"]
    assert fact.value == Decimal("25202000000")
    assert fact.sign == 1


def test_parse_fact_value_handles_accounting_parentheses_and_dashes():
    def value(markup):
        el = fact_elements(parse(f"<html><body><table><tr><td>{markup}</td></tr></table></body></html>"))
        return parse_fact_value(el[0]) if el else None

    assert value('<ix:nonFraction name="a">(1,577)</ix:nonFraction>') == Decimal("-1577")
    assert value('<ix:nonFraction name="a" scale="3">1.5</ix:nonFraction>') == Decimal("1500")
    assert value('<ix:nonFraction name="a">—</ix:nonFraction>') is None
    assert value('<ix:nonFraction name="a"></ix:nonFraction>') is None


# ---------------------------------------------------------------------------
# The row-label safety net
# ---------------------------------------------------------------------------
def test_row_label_is_recovered_from_the_enclosing_row():
    """This is what makes a wrong-concept mapping detectable with no LLM, and it
    yields the verbatim citation quote for free."""
    facts = facts_by_concept(parse(FILING))
    assert facts["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"].row_label == (
        "Additions to property and equipment"
    )
    assert facts["us-gaap:AssetsCurrent"].row_label == "Total current assets"


def test_row_label_skips_the_currency_symbol_cell():
    """The '$' occupies its own cell in SEC statements; it is not a label."""
    fact = facts_by_concept(parse(FILING))["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"]
    assert fact.row_label != "$"


def test_row_label_is_none_outside_a_table():
    root = parse(
        '<html><body><p><ix:nonFraction name="us-gaap:Revenues" '
        'contextRef="c">5</ix:nonFraction></p></body></html>'
    )
    assert row_label_for(fact_elements(root)[0]) is None


def test_label_mismatch_detector_rejects_on_evidence_never_on_absence():
    assert label_matches_metric("Additions to property and equipment", ["property"])
    assert not label_matches_metric("Share-based compensation expense", ["property", "capital expenditure"])
    # A missing label cannot refute anything - it must not cause a rejection.
    assert label_matches_metric(None, ["property"])
    assert label_matches_metric("anything", [])


# ---------------------------------------------------------------------------
# Rows for the database
# ---------------------------------------------------------------------------
def test_fact_ids_are_derived_so_reingest_overwrites():
    root = parse(FILING)
    rows = fact_rows(extract_facts(root), "MSFT_2023_10K", "MSFT_2023_10K#p61")
    again = fact_rows(extract_facts(root), "MSFT_2023_10K", "MSFT_2023_10K#p61")
    assert [r["fact_id"] for r in rows] == [r["fact_id"] for r in again]
    assert rows[0]["fact_id"].startswith("MSFT_2023_10K#p61.f")
    assert rows[0]["page_id"] == "MSFT_2023_10K#p61"
    assert rows[0]["doc_id"] == "MSFT_2023_10K"


def test_rows_carry_period_dimension_and_label():
    rows = fact_rows(extract_facts(parse(FILING)), "D", "D#p1")
    by_concept = {r["qname"]: r for r in rows}
    capex = by_concept["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"]
    assert capex["period_end"] == date(2023, 6, 30)
    assert capex["has_dimensions"] is False
    assert capex["row_label"] == "Additions to property and equipment"
    assert by_concept["us-gaap:Revenues"]["has_dimensions"] is True


def test_a_fact_with_no_resolvable_context_is_dropped():
    """A figure without a period cannot be checked against the question."""
    root = parse(
        '<html><body><ix:nonFraction name="us-gaap:Revenues" '
        'contextRef="MISSING">5</ix:nonFraction></body></html>'
    )
    assert extract_facts(root) == []


def test_a_filing_with_no_inline_xbrl_yields_nothing_quietly():
    """20 of the 78 filings are pre-2019 and carry no inline XBRL at all.
    That is the documented HTML-only path, not an error."""
    assert extract_facts(parse("<html><body><p>No XBRL here.</p></body></html>")) == []

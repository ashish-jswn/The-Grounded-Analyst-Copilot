"""MD&A anchors to its SECTION SPAN, not to its title page.

THE MEASURED FAILURE. `3M_2022_10K` has 131 pages. The MD&A anchor reached
5 of them (9, 10, 11, 19, 20), because the anchor rule is "the page whose head
carries the title, plus the next one" — a rule written for financial
statements, which run 2-3 pages. Item 7 runs ~30. The word "organic", which
the gold answer to `financebench_id_01865` turns on ("the consumer segment
shrunk by 0.9% organically"), appears on 12 pages; only 2 were reachable by
any anchor at all.
"""

from __future__ import annotations

from analyst_copilot.retrieval.anchors import (
    AnchorRetriever,
    build_from_rows,
    statement_hints,
)

MDNA_TITLE = "Item 7. Management's Discussion and Analysis of Financial Condition"


def _pages(doc: str, n: int, *, mdna_on: int = 5) -> list[dict]:
    """A filing whose MD&A title sits on one page and whose prose runs on."""
    rows = []
    for seq in range(1, n + 1):
        if seq == mdna_on:
            text = MDNA_TITLE + "\nOverview of results."
        elif seq == 2:
            text = "Consolidated Statements of Income\nRevenue ... 1,000"
        else:
            text = f"Body text for page {seq}. Organic growth was 0.9%."
        rows.append(
            {"doc_id": doc, "page_seq": seq, "page_id": f"{doc}:{seq}", "raw_text": text}
        )
    return rows


def _span(doc: str, start: int, end: int, kind: str = "mdna") -> dict:
    return {
        "doc_id": doc,
        "kind": kind,
        "stmt_type": None,
        "raw_title": MDNA_TITLE,
        "page_start": start,
        "page_end": end,
    }


def _mdna_pages(r: AnchorRetriever, doc: str) -> list[int]:
    """Every page the MD&A anchor can reach: head-matched plus span body.

    They live in two structures on purpose - the span body is held apart in
    `_spans` and folded in by `search` only when the question hints `mdna`, so
    26 body pages do not compete for the k=40 budget on a balance-sheet
    question. `test_span_pages_only_enter_search_for_a_narrative_question`
    covers that seam.
    """
    head = r._index.get(doc, {}).get("mdna", [])
    return sorted({seq for seq, _pid in head} | {seq for seq, _pid in r._spans.get(doc, [])})


def test_without_spans_mdna_is_only_the_title_page_and_its_successor():
    """The old behaviour, pinned so the regression is visible if it returns."""
    r = build_from_rows(_pages("D", 40, mdna_on=5))
    assert _mdna_pages(r, "D") == [5, 6]


def test_span_anchoring_covers_every_page_of_item_7():
    r = build_from_rows(_pages("D", 40, mdna_on=5), [_span("D", 5, 34)])
    assert _mdna_pages(r, "D") == list(range(5, 35))


def test_span_anchoring_keeps_the_title_page_rule_too():
    """Spanning ADDS pages; it never drops the ones the head match found."""
    r = build_from_rows(_pages("D", 40, mdna_on=5), [_span("D", 12, 20)])
    got = _mdna_pages(r, "D")
    assert {5, 6}.issubset(got)          # from the head match
    assert set(range(12, 21)).issubset(got)  # from the span


def test_pages_outside_the_filing_are_not_invented():
    """A span whose end runs past the last page must not produce phantom pages."""
    r = build_from_rows(_pages("D", 10, mdna_on=3), [_span("D", 3, 99)])
    assert max(_mdna_pages(r, "D")) == 10


def test_a_runaway_span_is_capped():
    """A mis-detected heading must not turn the whole filing into one anchor.

    Guard, not a nicety: before the `_is_candidate` fix a single Note appeared
    to span 48 pages. If that recurs for MD&A, an uncapped span would drown
    every other anchor in the ranking.
    """
    r = build_from_rows(_pages("D", 200, mdna_on=1), [_span("D", 1, 200)])
    assert len(_mdna_pages(r, "D")) <= 62


def test_a_statement_span_is_not_expanded():
    """Statements are 2-3 pages; the title+next rule already covers them."""
    r = build_from_rows(
        _pages("D", 40, mdna_on=5), [_span("D", 2, 30, kind="statement")]
    )
    assert _mdna_pages(r, "D") == [5, 6]


def test_kind_item_counts_when_the_title_says_discussion_and_analysis():
    """Item 7 is classified `item` in some filings and `mdna` in others.

    The tree classifies by the FIRST matching signal, so "Item 7. Management's
    Discussion..." can land under either. Anchoring must not depend on which.
    """
    r = build_from_rows(_pages("D", 40, mdna_on=5), [_span("D", 8, 25, kind="item")])
    assert set(range(8, 26)).issubset(_mdna_pages(r, "D"))


def test_an_item_span_without_mdna_in_its_title_is_ignored():
    span = _span("D", 8, 25, kind="item")
    span["raw_title"] = "Item 1A. Risk Factors"
    r = build_from_rows(_pages("D", 40, mdna_on=5), [span])
    assert _mdna_pages(r, "D") == [5, 6]


def test_spans_for_another_document_do_not_leak():
    rows = _pages("A", 20, mdna_on=5) + _pages("B", 20, mdna_on=5)
    r = build_from_rows(rows, [_span("A", 5, 18)])
    assert len(_mdna_pages(r, "A")) == 14
    assert _mdna_pages(r, "B") == [5, 6]


def test_a_span_for_an_unknown_document_is_harmless():
    r = build_from_rows(_pages("D", 20, mdna_on=5), [_span("MISSING", 1, 10)])
    assert _mdna_pages(r, "D") == [5, 6]


def test_missing_page_end_falls_back_to_the_start_page():
    span = _span("D", 7, 7)
    span["page_end"] = None
    r = build_from_rows(_pages("D", 20, mdna_on=5), [span])
    assert 7 in _mdna_pages(r, "D")


# ---------------------------------------------------------------- hints

def test_the_organic_growth_question_now_hints_mdna():
    """financebench_id_01865, verbatim in shape.

    THE MEASURED FAILURE: this hinted `segment` alone, so retrieval returned
    the segment operating-income TABLE — which reports dollars, not organic
    growth — and the composer answered from it.
    """
    q = ("Which region dragged down 3M's overall growth in 2022, excluding "
         "the impact of M&A?")
    hints = statement_hints(q)
    assert "mdna" in hints


def test_organic_alone_is_mdna_vocabulary():
    assert "mdna" in statement_hints("What was organic sales growth in FY2022?")


def test_excluding_acquisitions_is_mdna_vocabulary():
    assert "mdna" in statement_hints("Revenue growth excluding acquisitions?")


def test_constant_currency_is_mdna_vocabulary():
    assert "mdna" in statement_hints("Sales on a constant currency basis?")


def test_hints_still_rank_rather_than_filter():
    """A question can hint both; anchors ORDER by hint and drop nothing."""
    q = "Which segment dragged down organic growth?"
    hints = statement_hints(q)
    assert "segment" in hints and "mdna" in hints


def test_a_plain_balance_sheet_question_does_not_become_mdna():
    """The widened vocabulary must not swallow ordinary lookups."""
    assert statement_hints("What were total assets at year end?") == ["balance"]


# ------------------------------------------------- the conditional seam

def _searched_pages(r: AnchorRetriever, query: str, doc: str) -> set[int]:
    return {h.page_seq for h in r.search(query, [doc], k=200)}


def test_span_pages_only_enter_search_for_a_narrative_question():
    """The reason the body is held apart from `_index`.

    MEASURED: folding 26 MD&A body pages into the ranked buckets for EVERY
    question cost 1.6 points of router-top-4 gold-page recall - `search` keeps
    k pages across the whole scope, and on a balance-sheet question those body
    pages displaced balance-sheet pages from the document the router ranked
    first.
    """
    r = build_from_rows(_pages("D", 60, mdna_on=5), [_span("D", 5, 40)])
    narrative = _searched_pages(r, "What drove organic growth?", "D")
    lookup = _searched_pages(r, "What were total assets at year end?", "D")
    assert 30 in narrative
    assert 30 not in lookup


def test_span_pages_are_ordered_by_question_overlap_not_page_number():
    """A 26-page bucket ordered by page number is ordered by nothing.

    Every page in a bucket carries the same anchor score, so ties fall through
    to `page_seq`. Harmless for a 2-page statement; for the MD&A body it put
    the pages that answer the question behind every page that preceded them.
    """
    rows = _pages("D", 40, mdna_on=2)
    for r_ in rows:
        if r_["page_seq"] == 30:
            r_["raw_text"] = "Restructuring charges reduced segment margin sharply."
    r = build_from_rows(rows, [_span("D", 2, 35)])
    ranked = [seq for seq, _pid in r._rank_span("D", "restructuring charges margin")]
    assert ranked[0] == 30, ranked[:5]


def test_span_ranking_survives_a_question_of_pure_stopwords():
    r = build_from_rows(_pages("D", 20, mdna_on=2), [_span("D", 2, 15)])
    assert r._rank_span("D", "what is the") == r._spans["D"]


# ------------------------------------------------- ranking across documents

def test_ties_break_by_router_rank_not_alphabetically():
    """`k` is a budget over the WHOLE scope, and four documents compete.

    MEASURED: breaking ties on `doc_id` sorted candidates by company name
    and discarded the router's confidence, costing 0.8 points of gold-page
    recall at router top-4. `scope` arrives in router-confidence order, so the
    document the router ranked first must win a tie.
    """
    rows = _pages("ZEBRA", 30, mdna_on=5) + _pages("ACME", 30, mdna_on=5)
    r = build_from_rows(rows)
    # ZEBRA first: the router's top choice, but last alphabetically.
    hits = r.search("What were total assets?", ["ZEBRA", "ACME"], k=4)
    assert hits[0].doc_id == "ZEBRA", [h.doc_id for h in hits]


def test_a_document_outside_scope_never_ranks():
    rows = _pages("A", 20, mdna_on=5) + _pages("B", 20, mdna_on=5)
    r = build_from_rows(rows)
    hits = r.search("What were total assets?", ["A"], k=40)
    assert {h.doc_id for h in hits} == {"A"}

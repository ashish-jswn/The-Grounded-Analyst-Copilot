"""Regression tests for the 7-step table pipeline.

The gate in step 6 is what keeps a misaligned figure out of the database. These
tests pin both halves of it: the cases that MUST align, and the guarantee that a
failure produces markdown with no typed facts rather than a plausible wrong one.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from lxml import html as lxml_html

from analyst_copilot.config import load_settings
from analyst_copilot.ingest.pages import parse_document
from analyst_copilot.ingest.tables import (
    align,
    classify_columns,
    detect_header_row,
    drop_empty_columns,
    expand_grid,
    header_tokens_of,
    normalise_number,
    parse_table,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _tables_of(settings, doc_id):
    doc = parse_document(
        (settings.filings_dir / f"{doc_id}.htm").read_bytes(),
        settings.ingest.min_page_chars,
    )
    for page in doc.pages:
        for el in doc.elements_for(page):
            if isinstance(el.tag, str) and el.tag == "table":
                yield page, parse_table(
                    el,
                    min_rows=settings.ingest.table_min_rows,
                    min_numeric_cells=settings.ingest.table_min_numeric_cells,
                    context_text=page.raw_text[:400],
                )


# ---------------------------------------------------------------------------
# Numeric normalisation - a sign flip here is invisible downstream
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("(1,577)", Decimal("-1577")),
        ("1,577", Decimal("1577")),
        ("$ 184,257", Decimal("184257")),
        ("(0.5)", Decimal("-0.5")),
        ("12.4%", Decimal("12.4")),
        ("-42", Decimal("-42")),
    ],
)
def test_accounting_parentheses_mean_negative(text, expected):
    value, _sign = normalise_number(text)
    assert value == expected


@pytest.mark.parametrize("text", ["", "  ", "—", "$", "n/a"])
def test_non_numeric_cells_yield_no_value(text):
    assert normalise_number(text)[0] is None


# ---------------------------------------------------------------------------
# Steps 1-5 on the canonical layout
# ---------------------------------------------------------------------------
def test_colspan_expands_into_a_dense_grid():
    html = """<table>
      <tr><td colspan="2">Year Ended</td><td>2018</td></tr>
      <tr><td>a</td><td>b</td><td>1,000</td></tr>
    </table>"""
    grid = expand_grid(lxml_html.fromstring(html))
    assert grid[0] == ["Year Ended", "", "2018"]
    assert grid[1] == ["a", "b", "1,000"]


def test_ordinal_alignment_survives_the_currency_symbol_column():
    """The header row does not share the data rows' layout. Index alignment
    yields `2018 -> "$"`; ordinal alignment over value columns is correct."""
    grid = [
        ["(Millions)", "2018", "", "2017", "", "2016", ""],
        ["Purchases", "$", "(1,577)", "$", "(1,373)", "$", "(1,420)"],
    ]
    grid = drop_empty_columns(grid)
    header_row = detect_header_row(grid)
    assert header_row == 0
    kinds = classify_columns(grid, header_row)
    tokens = header_tokens_of(grid, header_row, label_col=0)
    assert tokens == ["2018", "2017", "2016"]
    ok, mapping = align(tokens, kinds)
    assert ok
    # The k-th header token lands on the k-th VALUE column, not the k-th column.
    assert [grid[1][c] for c in sorted(mapping)] == ["(1,577)", "(1,373)", "(1,420)"]


def test_header_row_is_detected_before_columns_are_classified():
    """A year like 2018 matches any numeric regex. Classifying over the whole
    grid mislabels the header's own cells as values - a real measured bug."""
    grid = [["(Millions)", "2018", "2017"], ["Revenue", "1,000", "900"]]
    kinds = classify_columns(grid, detect_header_row(grid))
    assert kinds[0] == "label"
    assert kinds[1] == kinds[2] == "value"


def test_spanning_period_descriptor_is_not_a_column_label():
    """'Year Ended June 30,' names the period for the whole table. Counting it
    gave 4 tokens against 3 value columns on 30 MICROSOFT_2023 tables."""
    grid = [["Year Ended June 30,", "2023", "2022", "2021"], ["Revenue", "1", "2", "3"]]
    assert header_tokens_of(grid, 0, label_col=None) == ["2023", "2022", "2021"]


def test_non_period_column_label_is_kept():
    """'Percentage Change' IS a real column but carries no period token.
    Dropping it gave 2 tokens against 3 value columns."""
    grid = [["(In millions)", "2023", "2022", "Percentage Change"], ["R", "1", "2", "3"]]
    assert header_tokens_of(grid, 0, label_col=0) == ["2023", "2022", "Percentage Change"]


# ---------------------------------------------------------------------------
# Step 0 and step 6 - the two gates
# ---------------------------------------------------------------------------
def test_layout_artifacts_are_rejected(settings):
    """46% of <table> elements are <=2-row spacers. They must never be indexed
    as tables, and must never carry typed cells."""
    rejected = [t for _p, t in _tables_of(settings, "JPMORGAN_2022_10K") if not t.is_data_table]
    assert rejected, "expected layout tables in JPMorgan"
    assert all(not t.cells for t in rejected)
    assert all(t.markdown is not None for t in rejected)


def test_failing_alignment_stores_markdown_but_no_typed_cells(settings):
    """FAIL CLOSED. This is the whole point of step 6: where provenance is not
    provable there is no typed fact, only markdown for the model to read."""
    seen = False
    for _page, table in _tables_of(settings, "JOHNSON_JOHNSON_2022_10K"):
        if table.is_data_table and not table.alignment_ok:
            seen = True
            assert table.cells == []
            assert table.markdown
    assert seen, "expected at least one alignment failure in J&J"


# ---------------------------------------------------------------------------
# The two cases that MUST align
# ---------------------------------------------------------------------------
def test_3m_capex_cell_aligns_to_the_right_year(settings):
    """The gold answer for financebench_id_03029 is $1,577M of FY2018 capex,
    and the gold justification names the line item verbatim."""
    hits = [
        cell
        for _page, table in _tables_of(settings, "3M_2018_10K")
        for cell in table.cells
        if "property, plant and equipment" in cell.row_header_path.lower()
        and cell.col_header_path == "2018"
        and cell.numeric_value == Decimal("-1577")
    ]
    assert hits, "3M FY2018 capex did not align"


def test_microsoft_2023_cell_aligns_with_scale(settings):
    hits = [
        (table, cell)
        for _page, table in _tables_of(settings, "MICROSOFT_2023_10K")
        for cell in table.cells
        if cell.numeric_value == Decimal("184257") and cell.col_header_path == "2023"
    ]
    assert hits, "Microsoft FY2023 figure did not align"
    table, cell = hits[0]
    assert cell.row_header_path == "Total current assets"
    # Scale must survive: gate G5 cannot reconcile operands without it.
    assert cell.scale == "millions"


def test_bare_millions_units_note_is_recognised(settings):
    """3M writes '(Millions)', not '(In millions)'. Missing the bare form leaves
    scale NULL and makes G5 abstain for a fixable reason."""
    scales = {
        cell.scale
        for _page, table in _tables_of(settings, "3M_2018_10K")
        for cell in table.cells
    }
    assert "millions" in scales

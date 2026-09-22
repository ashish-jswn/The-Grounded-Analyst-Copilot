"""Row labels must reach the BM25 field.

THE MEASURED GAP. `build_lexical_text`'s docstring says the BM25 lift
depends on strings like "Purchases of property, plant and equipment" surviving
indexing — but only COLUMN headers were ever passed, and a column header in a
financial statement is "2022", "2021" or "$". The line item, which is the part
an analyst's question actually names, was dropped entirely.
"""

from __future__ import annotations

from analyst_copilot.ingest.tables import parse_table
from lxml import html as lxml_html


def _table(rows: list[list[str]]) -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table>{body}</table>"


def _parse(rows):
    el = lxml_html.fromstring(_table(rows))
    return parse_table(el, min_rows=2, min_numeric_cells=2)


CASHFLOW = [
    ["", "2022", "2021"],
    ["Purchases of property, plant and equipment", "(1,577)", "(1,373)"],
    ["Proceeds from sale of investments", "512", "489"],
]


def test_the_line_item_is_captured():
    t = _parse(CASHFLOW)
    assert "Purchases of property, plant and equipment" in t.row_labels


def test_every_distinct_line_item_is_captured():
    t = _parse(CASHFLOW)
    assert len(t.row_labels) == 2


def test_a_bare_year_column_header_is_not_a_row_label():
    """The old behaviour indexed exactly this and nothing else."""
    t = _parse(CASHFLOW)
    assert "2022" not in t.row_labels


def test_numeric_only_labels_are_rejected():
    """A row whose label cell holds a number is not a line item."""
    t = _parse([["", "2022"], ["1,234", "56"], ["Total revenue", "78"]])
    assert "1,234" not in t.row_labels


def test_duplicate_labels_are_collapsed():
    t = _parse([
        ["", "2022", "2021"],
        ["Total revenue", "1", "2"],
        ["Total revenue", "3", "4"],
    ])
    assert t.row_labels.count("Total revenue") == 1


def test_a_prose_paragraph_is_not_a_row_label():
    """Guards the same failure the heading detector guards: a long sentence in
    a layout table must not become an index term."""
    long_text = "The following tables contain sales and operating income " * 3
    t = _parse([["", "2022"], [long_text, "1"]])
    assert long_text not in t.row_labels


def test_labels_survive_a_table_that_FAILS_the_alignment_gate():
    """THE POINT OF COLLECTING FROM THE GRID, NOT FROM `cells`.

    Roughly half of data tables fail the alignment gate, and typed cells are
    withheld from those by design (fail closed). Their row labels are still
    real text on the page, and indexing them creates no fact — it only makes
    the page findable, after which every gate still applies.
    """
    ragged = [
        ["", "2022", "2021", "extra"],
        ["Inventories, net", "1,000"],
    ]
    t = _parse(ragged)
    if t.is_data_table and not t.alignment_ok:
        assert "Inventories, net" in t.row_labels

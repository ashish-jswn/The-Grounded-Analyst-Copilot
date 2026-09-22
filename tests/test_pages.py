"""Regression tests for the page splitter and the EDGAR envelope handling.

The page is both the retrieval unit and the citation unit, so a regression here
silently costs the location half of the rubric on every question. Each test
below pins a failure that was measured on the real corpus.

Pure functions: no network, no database, no LLM.
"""

from __future__ import annotations

import pytest

from analyst_copilot.config import load_settings
from analyst_copilot.ingest.edgar import is_full_submission, iter_documents
from analyst_copilot.ingest.pages import count_break_markers, split_pages


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _read(settings, doc_id: str) -> bytes:
    return (settings.filings_dir / f"{doc_id}.htm").read_bytes()


def _pages(settings, doc_id: str):
    return split_pages(_read(settings, doc_id), settings.ingest.min_page_chars)


# ---------------------------------------------------------------------------
# The <hr>-only splitter these tests replace produced ZERO pages for these two.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc_id", ["GENERALMILLS_2022_10K", "ULTABEAUTY_2023_10K"])
def test_css_only_filings_still_paginate(settings, doc_id):
    """These filings declare every seam in CSS and contain no <hr> at all."""
    markers = count_break_markers(_read(settings, doc_id))
    assert markers["hr"] == 0
    assert markers["css"] > 50
    assert len(_pages(settings, doc_id)) > 50


def test_hr_inside_a_table_is_not_a_page_seam(settings):
    """MICROSOFT_2016_10K has 1,836 <hr>, of which 1,728 are cell rules inside
    a <td>. Counting those as seams split it into 343 pages instead of ~107,
    shredding every financial statement across several 'pages'."""
    pages = _pages(settings, "MICROSOFT_2016_10K")
    assert 90 <= len(pages) <= 130, f"got {len(pages)} pages"


def test_double_marked_seams_are_not_double_counted(settings):
    """CVS marks each seam with an <hr> that ALSO carries page-break-after.
    Splitting on both, without the short-fragment merge, doubles the count -
    the real cause of the old '+64 page offset' scare."""
    markers = count_break_markers(_read(settings, "CVSHEALTH_2018_10K"))
    pages = _pages(settings, "CVSHEALTH_2018_10K")
    assert markers["hr"] == markers["css"] > 0
    # ~one page per seam, not two.
    assert len(pages) <= markers["hr"] + 2


# ---------------------------------------------------------------------------
# EDGAR full submissions - measured 2026-08-31
# ---------------------------------------------------------------------------
FULL_SUBMISSIONS = [
    "AMCOR_2022_8K_dated-2022-07-01",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
    "FOOTLOCKER_2022_8K_dated_2022-08-19",
]


@pytest.mark.parametrize("doc_id", FULL_SUBMISSIONS)
def test_full_submission_content_is_not_lost(settings, doc_id):
    """These three files wrap several concatenated <html> documents. lxml parses
    only the first, so we were extracting ~3.5k chars of EDGAR navigation chrome
    and none of the filing."""
    raw = _read(settings, doc_id).decode("utf-8", "replace")
    assert is_full_submission(raw)
    assert len(iter_documents(raw)) >= 1

    text = "".join(p.raw_text for p in _pages(settings, doc_id))
    assert len(text) > 8_000, f"only {len(text)} chars recovered"


def test_amcor_8k_body_is_recovered(settings):
    """The AMCOR 8-K body sits at byte 82,776 of 142,400 and was never reaching
    the index at all."""
    text = "".join(p.raw_text for p in _pages(settings, "AMCOR_2022_8K_dated-2022-07-01"))
    # Body of the 8-K itself...
    assert "Amcor Finance" in text
    assert "Item 8.01" in text
    # ...and the exhibits it incorporates (the title is set in caps there).
    assert "second supplemental indenture" in text.lower()


def test_footlocker_news_release_exhibit_is_kept(settings):
    """Gold evidence for several 8-K questions lives in an EX-99.1 news release
    rather than in the 8-K body, so exhibits must not be discarded."""
    raw = _read(settings, "FOOTLOCKER_2022_8K_dated_2022-08-19").decode("utf-8", "replace")
    types = {t for t, _ in iter_documents(raw)}
    assert any(t.upper().startswith("EX-99") for t in types), types
    # The XBRL viewer scaffolding must NOT be kept: it would flood the lexical
    # index with context ids and namespace tokens.
    assert "XML" not in {t.upper() for t in types}


def test_single_document_filings_are_passed_through(settings):
    """75 of 78 filings are ordinary single documents; the envelope handling
    must be a no-op for them."""
    raw = _read(settings, "3M_2018_10K").decode("utf-8", "replace")
    assert not is_full_submission(raw)
    assert len(iter_documents(raw)) == 1


# ---------------------------------------------------------------------------
# Corpus-level invariants
# ---------------------------------------------------------------------------
def test_every_filing_yields_at_least_one_page(settings):
    """An 8-K still has to be citable. A filing must never ingest to zero pages
    - that is the failure mode that silently removes a document from the corpus."""
    for path in sorted(settings.filings_dir.glob("*.htm")):
        pages = split_pages(path.read_bytes(), settings.ingest.min_page_chars)
        assert pages, f"{path.stem} produced no pages"
        assert all(p.raw_text.strip() for p in pages), path.stem


def test_page_seq_is_dense_and_one_based(settings):
    pages = _pages(settings, "3M_2018_10K")
    assert [p.page_seq for p in pages] == list(range(1, len(pages) + 1))

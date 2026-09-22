"""Regression tests for the deterministic router and the filing catalog.

The measured numbers are assertions: if a refactor drops router top-4 below
98.5%, the test fails.

These run with no network, no database and no LLM.
"""

from __future__ import annotations

import json

import pytest

from analyst_copilot.config import load_settings
from analyst_copilot.ingest.catalog import build_catalog, parse_filing_name
from analyst_copilot.query.router import DocumentRouter, load_aliases, years_in

from pathlib import Path


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def router(settings):
    catalog = build_catalog(settings.filings_dir)
    aliases = load_aliases(settings.data_dir / "company_aliases.yaml")
    return DocumentRouter(
        catalog,
        aliases,
        top_k=settings.routing.top_k_filings,
        prefer_coverage_years=settings.routing.prefer_coverage_years,
    )


@pytest.fixture(scope="module")
def questions(settings):
    with open(settings.practice_questions, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ---------------------------------------------------------------------------
# The trap that cost a measurement once already.
# ---------------------------------------------------------------------------
def test_fy_year_pattern_matches_fy2022():
    """`\\b20\\d{2}\\b` does NOT match "FY2022" - there is no word boundary
    between "Y" and "2". That bug reported router top-1 as 48.5%."""
    assert years_in("What was revenue in FY2022?") == {2022}
    assert years_in("revenue in fy22") == {2022}
    assert years_in("from FY2020 to FY2022") == {2020, 2022}
    # A four-digit figure that is not a year must not be picked up.
    assert 20180 not in years_in("a charge of 20180 million")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def test_catalog_covers_the_whole_corpus(settings):
    catalog = build_catalog(settings.filings_dir)
    assert len(catalog) == 78
    assert {f.form_type for f in catalog} == {"10-K", "10-Q", "8-K"}


def test_coverage_years_include_comparatives():
    """A FY2018 10-K presents 2018 with comparatives for 2017 and 2016."""
    meta = parse_filing_name(Path("3M_2018_10K.htm"))
    assert meta.form_type == "10-K"
    assert meta.fiscal_year == 2018
    assert meta.coverage_years == (2016, 2017, 2018)


def test_quarterly_and_8k_names_parse():
    q = parse_filing_name(Path("3M_2023Q2_10Q.htm"))
    assert (q.form_type, q.fiscal_quarter, q.period_label) == ("10-Q", 2, "Q2 FY2023")

    k = parse_filing_name(Path("AMCOR_2022_8K_dated-2022-07-01.htm"))
    assert k.form_type == "8-K"
    assert k.filing_date is not None and k.filing_date.isoformat() == "2022-07-01"


def test_every_company_has_an_alias_entry(settings):
    """The alias table closes the 14/136 questions that name a company only by
    an alias (AMEX, JnJ, JPM). A slug with no entry is silently unroutable."""
    catalog = build_catalog(settings.filings_dir)
    aliases = load_aliases(settings.data_dir / "company_aliases.yaml")
    missing = sorted({f.company_slug for f in catalog if f.company_slug not in aliases})
    assert missing == []


# ---------------------------------------------------------------------------
# The measured numbers
# ---------------------------------------------------------------------------
def test_router_reproduces_measured_accuracy(router, questions):
    top1 = top2 = top4 = 0
    for q in questions:
        ids = [c.doc_id for c in router.route(q["question"]).candidates]
        gold = q["doc_name"]
        top1 += ids[:1] == [gold]
        top2 += gold in ids[:2]
        top4 += gold in ids[:4]

    n = len(questions)
    # Measured 2026-08-31: 95.6% / 97.8% / 98.5%. The plan's recorded figures
    # are 94.1% / 97.1% / 98.5%; these are floors, not equalities, so an
    # improvement passes and a regression fails.
    assert top1 / n >= 0.941, f"top-1 regressed to {top1}/{n}"
    assert top2 / n >= 0.971, f"top-2 regressed to {top2}/{n}"
    assert top4 / n >= 0.985, f"top-4 regressed to {top4}/{n}"


def test_the_only_top4_misses_name_no_company(router, questions):
    """An unroutable question must take the clarify path (score 0), never
    guess a document (score -1)."""
    for q in questions:
        result = router.route(q["question"])
        ids = [c.doc_id for c in result.candidates]
        if q["doc_name"] not in ids[:4]:
            assert not result.company_named
            assert result.needs_clarification


def test_latest_year_wins_when_several_are_named(router):
    """Rule A, measured at +7.3 points: a 10-K carries comparatives, so a
    question spanning FY2020-FY2022 is answered from the FY2022 filing."""
    ids = [
        c.doc_id
        for c in router.route(
            "What was 3M's revenue CAGR from FY2018 to FY2022?"
        ).candidates
    ]
    assert ids[0] == "3M_2022_10K"


def test_quarterly_question_prefers_the_10q(router):
    ids = [
        c.doc_id
        for c in router.route("What was 3M's Q2 2023 revenue?").candidates
    ]
    assert ids[0] == "3M_2023Q2_10Q"


def test_alias_only_question_still_routes(router):
    """JPM, AMEX and JnJ never appear as the literal company_slug."""
    for question, expected in [
        ("What was JPM's net income in 2022?", "JPMORGAN"),
        ("What was AMEX's total revenue in 2022?", "AMERICANEXPRESS"),
        ("What was JnJ's revenue in 2022?", "JOHNSON_JOHNSON"),
    ]:
        candidates = router.route(question).candidates
        assert candidates, f"no candidate for {question!r}"
        assert candidates[0].company_slug == expected


def test_same_year_8k_pair_separated_by_event_date(router):
    """PepsiCo filed two 8-Ks in 2023; only the event date separates them."""
    ids = [
        c.doc_id
        for c in router.route(
            "At the PepsiCo AGM held on May 3, 2023, what was the outcome of "
            "the shareholder vote?"
        ).candidates
    ]
    assert ids[0].startswith("PEPSICO_2023_8K")

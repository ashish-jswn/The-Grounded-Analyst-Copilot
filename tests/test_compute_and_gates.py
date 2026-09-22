"""Tests for the calculator, the formula book and gates G1-G7.

All deterministic: no network, no database, no LLM. These are the
components that prevent a -1, so they are tested for what they REJECT at least
as hard as for what they accept.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from analyst_copilot.query.compute import (
    FormulaError,
    Operand,
    compute,
    evaluate,
    to_base_units,
)
from analyst_copilot.query.formula_book import FormulaBook, FormulaSource
from analyst_copilot.query.gates import (
    AnswerCandidate,
    Citation,
    normalize,
    run_gates,
)


# ---------------------------------------------------------------------------
# The calculator must never execute anything but arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "formula",
    [
        "__import__('os').system('echo hi')",
        "().__class__.__mro__",
        "open('/etc/passwd').read()",
        "[x for x in range(10)]",
        "lambda: 1",
        "revenue if revenue else 0",
        "exec('x=1')",
    ],
)
def test_dangerous_formulas_are_rejected(formula):
    """A level-3 formula is written by an LLM. It may compute a number and
    nothing else."""
    with pytest.raises(FormulaError):
        evaluate(formula, {"revenue": Decimal(1)})


def test_arithmetic_is_exact_decimal():
    """Decimal, never float: 0.1 + 0.2 must be 0.3."""
    assert evaluate("a + b", {"a": Decimal("0.1"), "b": Decimal("0.2")}) == Decimal("0.3")


def test_division_by_zero_is_an_error_not_an_answer():
    with pytest.raises(FormulaError):
        evaluate("a / b", {"a": Decimal(1), "b": Decimal(0)})


def test_missing_operand_raises_rather_than_defaulting():
    """Defaulting a missing operand to zero would produce a plausible number
    from incomplete evidence - the -1 case."""
    with pytest.raises(FormulaError):
        evaluate("a / b", {"a": Decimal(1)})


def test_avg_needs_the_prior_period():
    with pytest.raises(FormulaError):
        evaluate("avg(inventory, prev, current)", {"inventory": Decimal(10)})


def test_period_helpers():
    values = {"inventory": Decimal(120), "inventory__prev": Decimal(80)}
    assert evaluate("avg(inventory, prev, current)", values) == Decimal(100)
    assert evaluate("delta(inventory, prev, current)", values) == Decimal(40)
    assert evaluate("prev(inventory)", values) == Decimal(80)


# ---------------------------------------------------------------------------
# Scale - a 1000x error still looks like a number
# ---------------------------------------------------------------------------
def test_scale_normalisation():
    assert to_base_units(Decimal("1577"), "millions") == Decimal("1577000000")
    assert to_base_units(Decimal("5"), "thousands") == Decimal("5000")
    assert to_base_units(Decimal("5"), None) == Decimal("5")


def test_mixed_scales_are_reconciled_before_arithmetic():
    """One operand in millions and one in thousands must not silently divide."""
    result = compute(
        "a / b",
        [
            Operand("a", Decimal("2000"), unit="currency", scale="thousands"),
            Operand("b", Decimal("1"), unit="currency", scale="millions"),
        ],
    )
    assert result.result == Decimal(2)


def test_percent_rendering_happens_once_in_code():
    result = compute(
        "gross_profit / revenue",
        [
            Operand("gross_profit", Decimal("50"), unit="currency"),
            Operand("revenue", Decimal("200"), unit="currency"),
        ],
        render="percent",
        dp=1,
    )
    assert result.rendered() == "25.0%"


# ---------------------------------------------------------------------------
# The formula book and its precedence ladder
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def book():
    return FormulaBook()


def test_book_loads_all_metrics(book):
    assert len(book.metrics) == 25
    assert len(book.operands) == 23


def test_book_is_also_the_concept_map(book):
    """`operands.concepts` feeds the XBRL path directly."""
    assert "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment" in book.concepts_for("capex")
    assert "us-gaap:Assets" in book.concepts_for("total_assets")


def test_alias_matching_prefers_the_longest_match(book):
    """'operating cash flow ratio' must not be shadowed by 'cash flow'."""
    assert book.find("what is the operating cash flow ratio").id == "operating_cash_flow_ratio"
    assert book.find("compute the quick ratio for 3M").id == "quick_ratio"


def test_question_supplied_definition_overrides_the_book(book):
    """Precedence level 1 - measured on 14/136 questions."""
    choice = book.choose(
        "What is the inventory turnover ratio, defined as COGS divided by the "
        "average of beginning and ending inventory balances?"
    )
    assert choice is not None
    assert choice.source is FormulaSource.QUESTION
    # The question's averaging convention wins over the book's.
    assert choice.period_rule == "average_of(prev, current)"


def test_book_is_used_when_the_question_supplies_no_definition(book):
    choice = book.choose("What was the gross margin in FY2022?")
    assert choice is not None
    assert choice.source is FormulaSource.BOOK
    assert choice.metric_id == "gross_margin"
    assert choice.definition


def test_unknown_metric_returns_none_so_the_caller_can_abstain(book):
    assert book.choose("What colour is the CEO's car?") is None


def test_every_book_formula_is_evaluable(book):
    """A formula that cannot parse would fail only at answer time."""
    for metric in book.metrics:
        values = {name: Decimal(2) for name in metric.operand_names()}
        values |= {f"{name}__prev": Decimal(1) for name in metric.operand_names()}
        values["n"] = Decimal(3)
        try:
            evaluate(metric.formula, values)
        except FormulaError as exc:
            if "not available" in str(exc):
                continue          # a named period operand, resolved at runtime
            pytest.fail(f"{metric.id}: {exc}")


# ---------------------------------------------------------------------------
# G1 - the -1 preventer
# ---------------------------------------------------------------------------
PAGE = "Purchases of property, plant and equipment (PP&E) $ (1,577) $ (1,373)"


def _candidate(**kw) -> AnswerCandidate:
    base = dict(
        question="FY2018 capex for 3M?",
        answer_text="$1,577 million",
        citations=[Citation(doc_id="D", page_seq=61, quote="Purchases of property, plant and equipment")],
        page_texts={"D": {61: PAGE}},
        candidate_doc_ids=["D"],
    )
    base.update(kw)
    return AnswerCandidate(**base)


def test_g1_accepts_a_verbatim_quote():
    assert run_gates(_candidate()).passed


def test_g1_normalisation_ignores_currency_punctuation():
    """`$ , ( ) %` and whitespace differences must not fail a real quote."""
    assert normalize("$ (1,577)") == normalize("1577")


def test_g1_rejects_an_invented_figure():
    """THE CASE THE WHOLE SYSTEM EXISTS TO PREVENT."""
    report = run_gates(
        _candidate(
            citations=[Citation(doc_id="D", page_seq=61, quote="Purchases of property, plant and equipment (PP&E) $ (9,999)")]
        )
    )
    assert not report.passed
    assert report.abstain_reason == "G1"


def test_g1_rejects_a_paraphrase():
    report = run_gates(
        _candidate(citations=[Citation(doc_id="D", page_seq=61, quote="Capital expenditures were 1577")])
    )
    assert not report.passed and report.abstain_reason == "G1"


def test_g1_rejects_a_citation_to_a_page_that_was_never_retrieved():
    report = run_gates(_candidate(citations=[Citation(doc_id="D", page_seq=999, quote="Purchases of property, plant")]))
    assert not report.passed and report.abstain_reason == "G1"


def test_g1_rejects_an_answer_with_no_citation_at_all():
    report = run_gates(_candidate(citations=[]))
    assert not report.passed and report.abstain_reason == "G1"


# ---------------------------------------------------------------------------
# G2-G7
# ---------------------------------------------------------------------------
def test_g2_rejects_a_document_outside_the_routed_scope():
    report = run_gates(_candidate(candidate_doc_ids=["OTHER"]))
    assert not report.passed and report.abstain_reason == "G2"


def test_g3_allows_a_comparative_year_from_coverage_years():
    """A FY2018 10-K answers a 2016 question from its comparative columns."""
    report = run_gates(
        _candidate(coverage_years={"D": [2016, 2017, 2018]}, evidence_years=[2016])
    )
    assert report.passed


def test_g3_rejects_a_year_the_filing_does_not_cover():
    report = run_gates(
        _candidate(coverage_years={"D": [2016, 2017, 2018]}, evidence_years=[2022])
    )
    assert not report.passed and report.abstain_reason == "G3"


def test_g3_allows_a_forward_looking_period_for_forecast_intent():
    """'What is Boeing forecasting for FY2023?' is answered from an earlier
    filing, so a later question period is correct here and wrong otherwise."""
    report = run_gates(
        _candidate(
            coverage_years={"D": [2020, 2021, 2022]},
            evidence_years=[2023],
            question_years=[2023],
            intent="forecast",
        )
    )
    assert report.passed


def test_g4_rejects_when_a_slot_is_missing():
    report = run_gates(_candidate(missing_slots=["total current liabilities"]))
    assert not report.passed and report.abstain_reason == "G4"


def test_g5_rejects_incompatible_units():
    report = run_gates(
        _candidate(
            operands=[
                Operand("a", Decimal(1), unit="currency"),
                Operand("b", Decimal(2), unit="shares"),
            ]
        )
    )
    assert not report.passed and report.abstain_reason == "G5"


def test_g6_rejects_prose_that_disagrees_with_the_arithmetic():
    computation = compute(
        "a / b",
        [Operand("a", Decimal(50)), Operand("b", Decimal(200))],
        render="percent",
        dp=1,
    )
    report = run_gates(_candidate(answer_text="The margin was 42.0%", computation=computation))
    assert not report.passed and report.abstain_reason == "G6"

    ok = run_gates(_candidate(answer_text="The margin was 25.0%", computation=computation))
    assert ok.passed


def test_a_disabled_gate_is_skipped_for_ablation():
    """The ablation runner switches one gate off and measures it."""
    bad = _candidate(citations=[Citation(doc_id="D", page_seq=61, quote="totally invented text here")])
    assert not run_gates(bad).passed
    assert run_gates(bad, enabled=("G2", "G3", "G4", "G5", "G6")).passed

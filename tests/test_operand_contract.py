"""The extractor -> calculator operand-name contract.

THE DEFECT THIS PINS. `evaluate` resolves operands by EXACT name
(`values[o.name]`), but nothing ever told the extractor what those names were.
It invented its own (`fy2019_revenue`, `total_revenue`), every derived answer
died with "operand 'revenue' is not available", gate G4 fired, and the question
abstained AS THOUGH THE FILING LACKED THE NUMBERS.

Measured live: 3 of the first 11 questions in a stratified run failed exactly
this way - `revenue`, `payables__prev`, `current_assets` - which is the whole
domain-relevant category and most ratio questions failing silently.

Two layers are tested here:
  1. `required_operand_names` - what the extractor is now TOLD to return
  2. `align_operands`         - the safety net when it returns something else
"""

from __future__ import annotations

from decimal import Decimal

from analyst_copilot.query.compute import (
    FormulaError, Operand, align_operands, compute, required_operand_names,
)
from analyst_copilot.query.formula_book import FormulaBook


def op(name, value, period=None, scale=None):
    return Operand(name=name, value=Decimal(str(value)), period=period, scale=scale)


# ---------------------------------------------------------------------------
# What the calculator actually demands
# ---------------------------------------------------------------------------
def test_simple_formula_names_its_operands():
    assert required_operand_names("current_assets / current_liabilities") == [
        "current_assets", "current_liabilities"
    ]


def test_an_averaged_operand_also_demands_its_prior_period():
    """`avg(ppe_net, prev, current)` needs BOTH `ppe_net` and `ppe_net__prev`,
    because `_Helpers._lookup` keys a non-current period as `name__period`.
    Missing that suffix is what killed fixed-asset turnover."""
    names = required_operand_names("revenue / avg(ppe_net, prev, current)")
    assert "revenue" in names
    assert "ppe_net" in names
    assert "ppe_net__prev" in names


def test_period_words_are_not_mistaken_for_operands():
    """`prev` and `current` name periods, not line items."""
    names = required_operand_names("revenue / avg(ppe_net, prev, current)")
    assert "prev" not in names and "current" not in names


def test_delta_and_prev_both_demand_the_prior_period():
    names = required_operand_names("delta(revenue, prev, current) / prev(revenue)")
    assert set(names) == {"revenue", "revenue__prev"}


def test_the_dpo_formula_resolves_completely():
    """The formula that failed live with "operand 'payables__prev' is not
    available"."""
    names = required_operand_names(
        "365 * avg(payables, prev, current) / (cogs + delta(inventory, prev, current))"
    )
    assert set(names) == {
        "payables", "payables__prev", "cogs", "inventory", "inventory__prev"
    }


def test_a_malformed_formula_yields_nothing_rather_than_raising():
    assert required_operand_names("revenue /") == []


def test_every_metric_in_the_book_resolves_its_own_operands():
    """A book entry whose operands cannot be named is unanswerable by
    construction - the extractor could never be told what to return."""
    book = FormulaBook()
    for metric in book.metrics:
        names = required_operand_names(metric.formula)
        assert names, f"{metric.id}: formula {metric.formula!r} named no operands"


# ---------------------------------------------------------------------------
# The safety net
# ---------------------------------------------------------------------------
def test_period_decorated_names_are_aligned():
    """`fy2019_revenue` is `revenue`. The prompt now asks for the bare name, but
    a model that decorates it must not cost the answer."""
    aligned = align_operands(
        [op("fy2019_revenue", 6489, period="FY2019")], ["revenue"]
    )
    assert aligned[0].name == "revenue"
    assert aligned[0].value == Decimal("6489")


def test_two_periods_of_one_item_split_into_current_and_prev():
    """The LATER period is the current one. Getting this backwards computes a
    plausible wrong average, which no gate can catch."""
    aligned = align_operands(
        [op("ppe_net_fy2018", 250, period="FY2018"),
         op("ppe_net_fy2019", 285, period="FY2019")],
        ["ppe_net", "ppe_net__prev"],
    )
    by_name = {o.name: o.value for o in aligned}
    assert by_name["ppe_net"] == Decimal("285")        # FY2019, the later
    assert by_name["ppe_net__prev"] == Decimal("250")  # FY2018


def test_an_already_correct_name_is_left_alone():
    aligned = align_operands([op("revenue", 100)], ["revenue"])
    assert aligned[0].name == "revenue" and aligned[0].value == Decimal("100")


def test_an_unmatched_operand_is_passed_through_not_guessed():
    """A mis-assigned operand computes a plausible WRONG number - the -1 this
    system exists to prevent. Leaving it unresolved fails G4 loudly instead."""
    aligned = align_operands([op("goodwill", 5)], ["revenue"])
    assert [o.name for o in aligned] == ["goodwill"]
    assert not any(o.name == "revenue" for o in aligned)


def test_alignment_does_not_reuse_one_slot_for_two_operands():
    aligned = align_operands([op("revenue", 100, period="FY2019")],
                             ["revenue", "revenue__prev"])
    names = [o.name for o in aligned]
    assert names.count("revenue") == 1
    assert "revenue__prev" not in names       # only one period was supplied


# ---------------------------------------------------------------------------
# End to end: the live failure, now computing
# ---------------------------------------------------------------------------
def test_fixed_asset_turnover_computes_after_alignment():
    """The exact question that abstained live: "FY2019 revenue / (average PP&E
    between FY2018 and FY2019)"."""
    formula = "revenue / avg(ppe_net, prev, current)"
    extracted = [
        op("fy2019_revenue", 6489, period="FY2019"),
        op("ppe_net_fy2019", 253, period="FY2019"),
        op("ppe_net_fy2018", 282, period="FY2018"),
    ]
    aligned = align_operands(extracted, required_operand_names(formula))
    result = compute(formula, aligned, dp=2)
    assert result.result == Decimal("6489") / ((Decimal("253") + Decimal("282")) / 2)


def test_a_genuinely_missing_operand_still_fails_closed():
    """The fix must not paper over absent evidence - that would trade an honest
    0 for a possible -1."""
    formula = "revenue / avg(ppe_net, prev, current)"
    aligned = align_operands([op("fy2019_revenue", 6489, period="FY2019")],
                             required_operand_names(formula))
    try:
        compute(formula, aligned)
    except FormulaError as exc:
        assert "ppe_net" in str(exc)
    else:                                     # pragma: no cover
        raise AssertionError("a missing operand must raise, not compute")


# ---------------------------------------------------------------------------
# G6 - the gate that rejected the answers it had just computed correctly
# ---------------------------------------------------------------------------
def test_g6_accepts_the_answer_at_its_declared_precision():
    """MEASURED: fixed asset turnover computed 24.2579..., rendered "24.26" at
    dp=2, and G6 rejected it at 8.6e-5 against the raw quotient. The question
    abstained on an answer that matched gold exactly. Every rounded metric in
    the book failed this way."""
    from analyst_copilot.query.compute import Computation, recomputes

    c = Computation(formula="revenue / ppe", result=Decimal("24.257943"),
                    operands={}, dp=2)
    assert c.rendered() == "24.26"
    assert recomputes("24.26", c, Decimal("0.000001"))


def test_g6_reads_every_number_not_just_the_first():
    """A prose answer naming its period first was checked against the YEAR - the
    same defect already fixed in the eval scorer."""
    from analyst_copilot.query.compute import Computation, recomputes

    c = Computation(formula="a/b", result=Decimal("24.26"), operands={}, dp=2)
    assert recomputes("The FY2019 fixed asset turnover ratio is 24.26.", c,
                      Decimal("0.000001"))


def test_g6_still_rejects_a_number_the_model_invented():
    """The gate must keep doing its real job: computed 24.26, wrote 27."""
    from analyst_copilot.query.compute import Computation, recomputes

    c = Computation(formula="a/b", result=Decimal("24.257943"), operands={}, dp=2)
    assert not recomputes("The ratio is 27.", c, Decimal("0.000001"))
    assert not recomputes("no figures here", c, Decimal("0.000001"))


def test_g6_handles_a_percent_rendering():
    from analyst_copilot.query.compute import Computation, recomputes

    c = Computation(formula="a/b", result=Decimal("0.0512"), operands={},
                    render="percent", dp=1)
    assert c.rendered() == "5.1%"
    assert recomputes("5.1%", c, Decimal("0.000001"))
    assert not recomputes("8.4%", c, Decimal("0.000001"))

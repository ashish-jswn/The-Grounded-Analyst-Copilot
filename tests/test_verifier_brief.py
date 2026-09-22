"""For a COMPUTED answer, the verifier must judge evidence — not redo the sum.

MEASURED, AND IT COST A CORRECT ANSWER. On the Activision fixed-asset-turnover
question the pipeline computed **24.26 — the gold answer exactly** — and both
verifiers rejected it by re-deriving the ratio from a revenue figure that was not
FY2019's:

    "The quoted figures imply FY2019 revenue = 7,017 ... yielding
     7,017/267.5 = 26.23, not 24.26, so the proposed answer is incorrect."

That is the model doing arithmetic (which the design forbids), duplicating a check gate
G6 already performed deterministically in Decimal, and overriding it wrongly.
"""

from __future__ import annotations

from decimal import Decimal

from analyst_copilot.query.compute import Computation
from analyst_copilot.query.pipeline import QueryPipeline


def turnover() -> Computation:
    return Computation(
        formula="revenue / avg(ppe_net, prev, current)",
        result=Decimal("24.257943"),
        operands={
            "revenue": Decimal("6489"),
            "ppe_net": Decimal("253"),
            "ppe_net__prev": Decimal("282"),
        },
        dp=2,
    )


def test_no_brief_when_nothing_was_computed():
    """A narrative answer has no arithmetic to protect; the verifier's normal
    entailment check is exactly right there."""
    assert QueryPipeline._computation_brief(None) == ""


def test_the_brief_states_the_arithmetic_is_already_proven():
    brief = QueryPipeline._computation_brief(turnover())
    assert "ALREADY PROVEN" in brief
    assert "G6" in brief
    assert "DO NOT re-derive" in brief


def test_the_brief_tells_the_verifier_its_own_sum_is_not_the_authority():
    """The measured failure was a verifier trusting its own mental arithmetic
    over a deterministic Decimal evaluation."""
    brief = QueryPipeline._computation_brief(turnover())
    assert "your arithmetic is wrong" in brief


def test_the_brief_carries_every_operand_so_the_quotes_can_be_checked():
    """Narrowing the job is only useful if the verifier is given the thing it
    SHOULD check: that each operand really appears in the quotes."""
    brief = QueryPipeline._computation_brief(turnover())
    for name, value in (("revenue", "6489"), ("ppe_net", "253"), ("ppe_net__prev", "282")):
        assert name in brief and value in brief


def test_the_brief_reports_the_rendered_result_not_the_raw_quotient():
    """The answer states 24.26; showing the verifier 24.257943 would invite it
    to flag a mismatch that G6 has already resolved."""
    brief = QueryPipeline._computation_brief(turnover())
    assert "24.26" in brief
    assert "24.257943" not in brief


def test_the_brief_redirects_to_evidence():
    brief = QueryPipeline._computation_brief(turnover())
    assert "Judge the EVIDENCE, not the sum." in brief

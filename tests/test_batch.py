"""Batch runner: planning, the stop policy, explanations, and metering.

Entirely offline - no database, no model call. The batch runner decides when to
STOP SPENDING, so its logic has to be testable without spending anything.
"""

from __future__ import annotations

import threading
from decimal import Decimal

import pytest

from analyst_copilot.eval.batch import (
    QuestionRun, StopOn, Tally, explain_answer, explain_location, plan_batches,
    render_question, stop_reason,
)
from analyst_copilot.eval.gold import GoldQuestion
from analyst_copilot.eval.metering import CostLedger, MeteredProvider, Usage
from analyst_copilot.eval.scorer import (
    Citation, ScoredResult, SystemAnswer, numbers_match, parse_numbers,
)
from analyst_copilot.eval.shapes import AnswerShape

TOL = Decimal("0.005")


def gold(**kw) -> GoldQuestion:
    base = dict(
        qid="q1",
        question="What is the FY2018 capital expenditure amount for 3M?",
        answer="$1577.00",
        justification="",
        doc_name="3M_2018_10K",
        company="3M",
        question_type="metrics-generated",
        shape=AnswerShape.NUMERIC,
        evidence_texts=["Purchases of property, plant and equipment (PP&E) (1,577)"],
        evidence_full_pages=[],
        gold_page_seqs=[61],
    )
    base.update(kw)
    return GoldQuestion(**base)


def scored(score: int, *, answer=True, location=True, abstained=False) -> ScoredResult:
    return ScoredResult(
        qid="q1", shape=AnswerShape.NUMERIC, score=score,
        answer_correct=answer, location_correct=location, abstained=abstained,
    )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def test_batches_escalate_then_repeat_the_last_size():
    """The ramp widens, then holds - a long run must keep its checkpoints."""
    batches = plan_batches(100, [5, 10, 20])
    assert [b.size for b in batches] == [5, 10, 20, 20, 20, 20, 5]
    assert batches[0].start == 0 and batches[0].end == 5
    assert batches[1].start == 5 and batches[1].end == 15
    assert batches[-1].end == 100


def test_batches_cover_every_question_exactly_once():
    batches = plan_batches(37, [5, 10])
    covered = [i for b in batches for i in range(b.start, b.end)]
    assert covered == list(range(37))


def test_batches_handle_edges():
    assert plan_batches(0, [5]) == []
    assert [b.size for b in plan_batches(3, [5, 10])] == [3]
    # A malformed ramp must not wedge the loop.
    assert [b.size for b in plan_batches(4, [0, -1])] == [4]


# ---------------------------------------------------------------------------
# Stop policy - the part that decides how much money a bad run costs
# ---------------------------------------------------------------------------
def test_default_policy_stops_only_on_a_wrong_answer():
    """An abstention is the DESIGNED-CORRECT outcome, not a failure.

    This system over-abstains today, so a default that halted on every non-+1
    would stop on the first question of every run and measure nothing.
    """
    assert stop_reason(StopOn.WRONG, scored(-1, answer=False, location=False))
    assert stop_reason(StopOn.WRONG, scored(0, answer=False, location=False, abstained=True)) is None
    assert stop_reason(StopOn.WRONG, scored(0, answer=True, location=False)) is None
    assert stop_reason(StopOn.WRONG, scored(1)) is None


def test_wrong_or_location_also_stops_on_a_misplaced_citation():
    policy = StopOn.WRONG_OR_LOCATION
    assert stop_reason(policy, scored(0, answer=True, location=False))
    assert stop_reason(policy, scored(0, answer=False, abstained=True)) is None


def test_any_stops_on_an_abstention_too():
    assert stop_reason(StopOn.ANY, scored(0, answer=False, abstained=True))
    assert stop_reason(StopOn.ANY, scored(1)) is None


def test_none_never_stops():
    assert stop_reason(StopOn.NONE, scored(-1, answer=False, location=False)) is None


def test_policy_parsing_rejects_a_typo_loudly():
    assert StopOn.parse("WRONG") is StopOn.WRONG
    assert StopOn.parse("wrong+location") is StopOn.WRONG_OR_LOCATION
    with pytest.raises(ValueError, match="unknown --stop-on"):
        StopOn.parse("wrng")


# ---------------------------------------------------------------------------
# Explanations - a run has to be readable instead of re-run
# ---------------------------------------------------------------------------
def test_numeric_explanation_reports_the_gap():
    q = gold()
    ok = explain_answer(q, SystemAnswer(text="$1,577 million in FY2018"), TOL)
    assert "the answer states 1577" in ok

    bad = explain_answer(q, SystemAnswer(text="$1,373 million"), TOL)
    assert "appears in none of" in bad and "1373" in bad.replace(",", "")


def test_yes_no_explanation_names_both_verdicts():
    q = gold(answer="No, the company is managing capex efficiently.", shape=AnswerShape.YES_NO)
    text = explain_answer(q, SystemAnswer(text="Yes, it is capital intensive."), TOL)
    assert "got yes" in text and "gold no" in text


def test_abstention_is_explained_as_a_decline_not_a_mismatch():
    assert "declined" in explain_answer(gold(), SystemAnswer(abstained=True), TOL)


def test_location_explanation_flags_the_wrong_document():
    q = gold()
    answer = SystemAnswer(
        text="$1,577", citations=[Citation("NIKE_2023_10K", 12, "something")]
    )
    text = explain_location(q, answer, {}, containment=0.6, page_slack=1)
    assert "WRONG DOCUMENT" in text


def test_location_explanation_accepts_an_adjacent_page():
    q = gold()
    answer = SystemAnswer(text="x", citations=[Citation("3M_2018_10K", 62, "q")])
    text = explain_location(q, answer, {}, containment=0.6, page_slack=1)
    assert "within" in text


def test_location_explanation_reports_overlap_when_the_page_is_wrong():
    q = gold()
    answer = SystemAnswer(text="x", citations=[Citation("3M_2018_10K", 5, "q")])
    pages = {5: "totally unrelated narrative about risk factors"}
    text = explain_location(q, answer, pages, containment=0.6, page_slack=1)
    assert "overlap" in text and "needs 60%" in text


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_render_shows_gold_answer_and_the_verbatim_proof():
    run = QuestionRun(
        question=gold(),
        answer=SystemAnswer(
            text="$1,577 million in FY2018",
            citations=[Citation(
                "3M_2018_10K", 61,
                "Purchases of property, plant and equipment (PP&E) (1,577)",
            )],
        ),
        scored=scored(1),
        seconds=38.0,
    )
    out = render_question(
        run, "[1/5]", {}, tolerance=TOL, containment=0.6, page_slack=1
    )
    assert "PASS  +1" in out
    assert "$1577.00" in out                       # the gold answer, for cross-check
    assert "$1,577 million in FY2018" in out       # what the system said
    assert "Purchases of property, plant and equipment" in out   # the proof
    assert "3M_2018_10K p.61" in out


def test_render_marks_an_answer_with_no_citation():
    run = QuestionRun(
        question=gold(),
        answer=SystemAnswer(text="1577"),
        scored=scored(0, answer=True, location=False),
    )
    out = render_question(run, "[1/5]", {}, tolerance=TOL, containment=0.6, page_slack=1)
    assert "can never score +1" in out


def test_render_surfaces_the_failing_gate_for_an_abstention():
    run = QuestionRun(
        question=gold(),
        answer=SystemAnswer(abstained=True),
        scored=scored(0, answer=False, location=False, abstained=True),
        abstain_reason="verifier:b",
        gate_detail="tier1 G1: quote not found on 3M_2018_10K#p62",
        verifiers={"a": True, "b": False},
    )
    out = render_question(run, "[2/5]", {}, tolerance=TOL, containment=0.6, page_slack=1)
    assert "DECLINED 0" in out
    assert "verifier:b" in out
    assert "quote not found" in out
    assert "b=REJECTED" in out


def test_tally_reports_the_false_answer_rate_beside_the_score():
    """A change that raises the score AND the -1 rate is a regression."""
    t = Tally()
    t.add(scored(1))
    t.add(scored(0, answer=False, abstained=True))
    t.add(scored(-1, answer=False, location=False))
    assert t.total == 0 and t.n == 3
    assert t.false_answer_rate == pytest.approx(1 / 3)
    assert "-1=1" in t.line("run")


# ---------------------------------------------------------------------------
# Metering
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, i=100, o=20, r=5):
        self.input_tokens, self.output_tokens, self.reasoning_tokens = i, o, r
        self.data, self.raw = {}, ""


class _FakeProvider:
    def __init__(self, boom=False):
        self.boom = boom

    def complete(self, *, system, user, schema, stage):
        if self.boom:
            raise RuntimeError("429")
        return _FakeResponse()


def test_metering_records_tokens_without_changing_the_response():
    ledger = CostLedger()
    provider = MeteredProvider(_FakeProvider(), ledger, "extractor")
    ledger.begin()
    response = provider.complete(system="s", user="u", schema={}, stage="extractor")
    usage = ledger.take()

    assert response.input_tokens == 100          # untouched
    assert usage.input_tokens == 100 and usage.output_tokens == 20
    assert usage.calls == 1
    assert ledger.total.input_tokens == 100


def test_metering_records_a_failed_call_rather_than_hiding_it():
    """A rate-limit storm must look expensive, not free."""
    ledger = CostLedger()
    provider = MeteredProvider(_FakeProvider(boom=True), ledger, "extractor")
    ledger.begin()
    with pytest.raises(RuntimeError):
        provider.complete(system="s", user="u", schema={}, stage="extractor")
    assert ledger.take().errors == 1
    assert ledger.total.errors == 1


def test_per_question_usage_does_not_leak_across_threads():
    """The harness runs questions concurrently; attribution must stay separate."""
    ledger = CostLedger()
    provider = MeteredProvider(_FakeProvider(), ledger, "extractor")
    seen: dict[int, int] = {}

    def worker(n: int) -> None:
        ledger.begin()
        for _ in range(n):
            provider.complete(system="s", user="u", schema={}, stage="extractor")
        seen[n] = ledger.take().calls

    threads = [threading.Thread(target=worker, args=(n,)) for n in (1, 2, 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {1: 1, 2: 2, 3: 3}
    assert ledger.total.calls == 6


def test_cost_is_none_until_a_rate_is_supplied():
    """Dollars are derived, never guessed - an unset rate reports no number."""
    usage = Usage()
    usage.record("extractor", input_tokens=1_000_000, output_tokens=500_000)
    assert usage.cost(None, None) is None
    assert usage.cost(0.25, None) is None
    assert usage.cost(0.25, 2.0) == pytest.approx(0.25 + 1.0)


# ---------------------------------------------------------------------------
# The scoring defect that reported a CORRECT answer as confidently wrong
# ---------------------------------------------------------------------------
def test_a_year_in_the_prose_does_not_become_the_answer():
    """MEASURED. The system replied "The FY2018 capital expenditure ... was
    $1,577 million." - exactly right. Taking the FIRST number parsed 2018 out of
    "FY2018", compared it to gold 1577 and scored the question -1.

    A harness that turns a correct answer into a confident wrong answer corrupts
    the false-answer rate, which is the number calibration is chosen on.
    """
    got = "The FY2018 capital expenditure (Purchases of property, plant and equipment) was $1,577 million."
    assert numbers_match("$1577.00", got, TOL)


def test_the_same_fact_scores_the_same_whatever_the_word_order():
    """The value-first phrasing scored +1 and the period-first phrasing scored
    -1 for the SAME fact. Phrasing is not correctness."""
    for got in (
        "$1,577 million in FY2018 for purchases of property, plant and equipment",
        "The FY2018 capital expenditure was $1,577 million.",
        "In 2018, 3M spent 1,577 million on PP&E.",
    ):
        assert numbers_match("$1577.00", got, TOL), got


def test_a_genuinely_wrong_figure_still_fails():
    """The fix must not turn the scorer into a rubber stamp."""
    assert not numbers_match("$1577.00", "The FY2018 figure was $1,373 million.", TOL)
    assert not numbers_match("$1577.00", "In 2018 the company reported nothing relevant.", TOL)


def test_years_rank_last_but_are_still_available():
    """A gold answer can legitimately BE a year, so a year must not be dropped -
    only out-ranked by a figure standing next to it."""
    assert parse_numbers("In 2018 the total was $1,577 million") == [
        Decimal("1577000000"), Decimal("2018")
    ]
    assert numbers_match("2018", "The fiscal year referenced is 2018.", TOL)


def test_scale_words_still_apply_and_negatives_survive():
    values = parse_numbers("cash outflow of (1,577) against $8.70 billion of assets")
    assert Decimal("-1577") in values
    assert Decimal("8700000000") in values


def test_explanation_names_every_figure_it_considered():
    q = gold()
    text = explain_answer(q, SystemAnswer(text="The FY2018 amount was $1,373 million."), TOL)
    assert "1373" in text.replace(",", "") and "nearest" in text


# ---------------------------------------------------------------------------
# Stratified sampling - an A/B is worthless if the arms see different questions
# ---------------------------------------------------------------------------
def _population():
    """40 questions ordered by company, as the real practice set is."""
    shapes = ([AnswerShape.NUMERIC] * 20 + [AnswerShape.YES_NO] * 10
              + [AnswerShape.PHRASE] * 5 + [AnswerShape.MULTI_SENTENCE] * 5)
    return [
        gold(qid=f"q{i:02d}", company=f"CO{i // 5}", shape=shape)
        for i, shape in enumerate(shapes)
    ]


def test_sample_preserves_the_answer_shape_mix():
    """The harness reports by shape, so the mix must survive sampling."""
    from collections import Counter
    from analyst_copilot.eval.batch import stratified_sample

    sample = stratified_sample(_population(), 20)
    mix = Counter(q.shape for q in sample)
    assert mix[AnswerShape.NUMERIC] > mix[AnswerShape.YES_NO] > mix[AnswerShape.PHRASE]
    assert 18 <= len(sample) <= 22


def test_sample_spreads_companies_where_limit_would_not():
    """`--limit N` takes the first N, and the practice set is ordered by company:
    the first five questions are all one filing. A score from that measures the
    company, not the system."""
    from analyst_copilot.eval.batch import stratified_sample

    population = _population()
    assert len({q.company for q in population[:10]}) == 2      # what --limit gives
    assert len({q.company for q in stratified_sample(population, 10)}) > 2


def test_sample_is_deterministic_so_both_arms_see_the_same_questions():
    from analyst_copilot.eval.batch import stratified_sample

    population = _population()
    first = [q.qid for q in stratified_sample(population, 15)]
    assert first == [q.qid for q in stratified_sample(population, 15)]


def test_sample_returns_everything_when_n_covers_the_set():
    from analyst_copilot.eval.batch import stratified_sample

    population = _population()
    assert stratified_sample(population, 999) == population
    assert stratified_sample(population, 0) == population

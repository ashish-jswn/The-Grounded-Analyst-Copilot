"""Escalating batch evaluation with an early stop and a full evidence trail.

WHY BATCHES. A full 136-question run is 4-6 model calls per question against a
rate-limited deployment, and measured latency is 60-130 s per question. Spending
that on a configuration that was already broken at question 3 wastes both credit
and an afternoon. So the run starts small (5), widens (10, 20, 40, ...), and
reports a scoreboard at every boundary — a bad change shows up in the first
batch, for the price of five questions.

WHAT COUNTS AS A FAILURE IS NOT OBVIOUS, AND THE DEFAULT MATTERS.
Under this rubric an abstention scores 0 and is the DESIGNED-CORRECT outcome
when the evidence does not verify. The system currently over-abstains,
so a policy of "stop unless +1" would halt on the first or
second question of every run and measure nothing.

The default is therefore `wrong`: stop on a -1, the confidently wrong answer
that actually costs two points against a refusal. `any` is available for the
stricter reading, and says so in the header so a run is never ambiguous about
which rule produced it.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from .gold import GoldQuestion, tokens
from .metering import Usage
from .scorer import (
    Citation, ScoredResult, SystemAnswer, numbers_match, parse_number, parse_numbers,
)
from .shapes import AnswerShape

_YES_NO = re.compile(r"^\s*(yes|no)\b", re.I)
_RULE = "-" * 78


# ---------------------------------------------------------------------------
# Batch planning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Batch:
    index: int
    start: int
    end: int          # exclusive

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def label(self) -> str:
        return f"batch {self.index} — questions {self.start + 1}-{self.end}"


def plan_batches(total: int, sizes: list[int]) -> list[Batch]:
    """Split `total` questions into escalating batches.

    `sizes` gives the ramp, e.g. [5, 10, 20, 40]. Once the ramp is exhausted the
    LAST size repeats, so a long run does not degenerate into one enormous batch
    with no checkpoint in the middle — the point of batching is the checkpoint.
    """
    if total <= 0:
        return []
    sizes = [s for s in sizes if s > 0] or [total]
    batches: list[Batch] = []
    start = 0
    index = 0
    while start < total:
        size = sizes[min(index, len(sizes) - 1)]
        end = min(start + size, total)
        batches.append(Batch(index=index + 1, start=start, end=end))
        start = end
        index += 1
    return batches


def stratified_sample(questions: list[GoldQuestion], n: int) -> list[GoldQuestion]:
    """A deterministic sample that preserves the answer-shape mix and spreads
    companies.

    `--limit N` IS NOT A SAMPLE. The practice set is ordered by company, so
    the first five questions are all 3M and the first twenty-five are four
    companies. A score from that measures those companies, not the system, and
    the shape mix - 52 numeric / 35 yes-no / 26 phrase / 23 multi-sentence - is
    the axis the harness reports on, so it must survive sampling.

    Proportional by shape; within a shape, questions are ordered by company and
    taken with an even stride so no single filing dominates. No RNG: the same
    sample every run, because an A/B whose two arms saw different questions
    measures nothing.
    """
    if n <= 0 or n >= len(questions):
        return questions

    by_shape: dict[AnswerShape, list[GoldQuestion]] = {}
    for q in questions:
        by_shape.setdefault(q.shape, []).append(q)

    picked: list[GoldQuestion] = []
    for shape, group in sorted(by_shape.items(), key=lambda kv: kv[0].value):
        want = max(1, round(n * len(group) / len(questions)))
        ordered = sorted(group, key=lambda q: (q.company, q.qid))
        stride = max(1, len(ordered) // want)
        picked.extend(ordered[::stride][:want])

    # Keep the original corpus order so a run reads in a stable sequence.
    order = {q.qid: i for i, q in enumerate(questions)}
    return sorted(picked, key=lambda q: order[q.qid])


# ---------------------------------------------------------------------------
# Stop policy
# ---------------------------------------------------------------------------
class StopOn(str, Enum):
    NONE = "none"                       # run everything
    WRONG = "wrong"                     # stop on -1  (the default)
    WRONG_OR_LOCATION = "wrong+location"  # also stop on right answer / wrong location
    ANY = "any"                         # stop on anything that is not +1

    @classmethod
    def parse(cls, text: str) -> "StopOn":
        try:
            return cls(text.strip().lower())
        except ValueError:
            raise ValueError(
                f"unknown --stop-on {text!r}; choose from "
                f"{', '.join(m.value for m in cls)}"
            ) from None

    @property
    def explanation(self) -> str:
        return {
            StopOn.NONE: "never stop early — run every batch",
            StopOn.WRONG: "stop on a confidently WRONG answer (-1)",
            StopOn.WRONG_OR_LOCATION: "stop on -1, or a right answer cited to the wrong place",
            StopOn.ANY: "stop on anything that is not +1, INCLUDING an honest abstention",
        }[self]


def stop_reason(policy: StopOn, scored: ScoredResult) -> str | None:
    """Why this result should halt the run, or None to continue."""
    if policy is StopOn.NONE:
        return None
    if scored.score == -1:
        return "answered and the answer is wrong (-1)"
    if policy is StopOn.WRONG:
        return None
    if scored.answer_correct and not scored.location_correct:
        return "right answer, wrong location (0)"
    if policy is StopOn.WRONG_OR_LOCATION:
        return None
    if scored.score != 1:
        return "abstained" if scored.abstained else "did not score +1"
    return None


# ---------------------------------------------------------------------------
# Why a question scored what it scored
# ---------------------------------------------------------------------------
def explain_answer(
    question: GoldQuestion, answer: SystemAnswer, tolerance: Decimal
) -> str:
    """A human-readable reason for the answer verdict, per shape predicate."""
    if answer.abstained:
        return "declined — no answer to compare"
    if answer.clarified:
        return "asked a clarifying question — no answer to compare"

    gold, got = question.answer, answer.text
    if question.shape is AnswerShape.NUMERIC:
        g = parse_number(gold)
        found = parse_numbers(got)
        if g is None:
            return "gold answer has no parseable figure"
        if not found:
            return f"no figure found in the answer (gold {g})"
        if numbers_match(gold, got, tolerance):
            return f"the answer states {g} (within {tolerance:%}); figures seen: {found[:4]}"
        # Report EVERY figure considered, not just the first. The scorer accepts
        # any of them, so naming only one would misexplain its own verdict.
        nearest = min(found, key=lambda c: abs(c - g) if g else abs(c))
        delta = abs(g - nearest) / abs(g) if g else Decimal(0)
        return (f"gold {g} appears in none of {found[:5]}; "
                f"nearest {nearest} is {delta:.2%} off")

    if question.shape is AnswerShape.YES_NO:
        g, c = _YES_NO.match(gold), _YES_NO.match(got)
        if not g:
            return "gold answer does not open with yes/no"
        if not c:
            return f"answer does not open with yes/no (gold says {g.group(1).lower()})"
        if g.group(1).lower() != c.group(1).lower():
            return f"verdict differs: got {c.group(1).lower()}, gold {g.group(1).lower()}"
        return f"verdict matches ({g.group(1).lower()}); justification graded by the judge"

    return "free-text answer, graded by the LLM judge against gold + justification"


def explain_location(
    question: GoldQuestion,
    answer: SystemAnswer,
    pages_by_seq: dict[int, str],
    *,
    containment: float,
    page_slack: int,
) -> str:
    """Why the location verdict came out as it did.

    Location is scored by EVIDENCE-TEXT OVERLAP against our own derived pages,
    never by FinanceBench's `evidence_page_num` (it indexes a third-party PDF).
    """
    if not answer.citations:
        return "no citations"
    cited = {c.doc_id for c in answer.citations}
    if question.doc_name not in cited:
        return (
            f"cited {sorted(cited)} but the gold evidence is in "
            f"{question.doc_name} — WRONG DOCUMENT"
        )
    if not question.gold_page_seqs:
        return (
            "gold page could not be mapped onto our derived pages; "
            "location scored by text overlap alone"
        )

    pages = sorted({c.page_seq for c in answer.citations if c.doc_id == question.doc_name})
    near = [
        p for p in pages
        if any(abs(p - g) <= page_slack for g in question.gold_page_seqs)
    ]
    if near:
        return f"cited p.{near} against gold p.{question.gold_page_seqs} (within ±{page_slack})"

    best = 0.0
    for citation in answer.citations:
        if citation.doc_id != question.doc_name:
            continue
        page = tokens(pages_by_seq.get(citation.page_seq, ""))
        for gold_text in question.evidence_texts:
            gold = tokens(gold_text)
            if gold and page:
                best = max(best, len(gold & page) / len(gold))
    return (
        f"cited p.{pages}, gold p.{question.gold_page_seqs}; "
        f"best evidence overlap {best:.0%} (needs {containment:.0%})"
    )


# ---------------------------------------------------------------------------
# One evaluated question
# ---------------------------------------------------------------------------
@dataclass
class QuestionRun:
    """Everything worth knowing about one question, for display and for JSONL."""

    question: GoldQuestion
    answer: SystemAnswer
    scored: ScoredResult
    abstain_reason: str | None = None
    gate_detail: str = ""
    verifiers: dict[str, bool] = field(default_factory=dict)
    verifier_reasons: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0
    trace: dict = field(default_factory=dict)

    def as_row(self, price_in: float | None, price_out: float | None) -> dict:
        return {
            "qid": self.question.qid,
            "doc_name": self.question.doc_name,
            "company": self.question.company,
            "shape": self.question.shape.value,
            "question": self.question.question,
            "gold": self.question.answer,
            "got": self.answer.text,
            "score": self.scored.score,
            "answer_correct": self.scored.answer_correct,
            "location_correct": self.scored.location_correct,
            "abstained": self.scored.abstained,
            "abstain_reason": self.abstain_reason,
            "gate_detail": self.gate_detail,
            "verifiers": self.verifiers,
            "verifier_reasons": self.verifier_reasons,
            "citations": [
                {"doc_id": c.doc_id, "page_seq": c.page_seq, "quote": c.quote}
                for c in self.answer.citations
            ],
            "gold_page_seqs": self.question.gold_page_seqs,
            "errors": self.errors,
            "seconds": round(self.seconds, 1),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "llm_calls": self.usage.calls,
            "cost_usd": self.usage.cost(price_in, price_out),
        }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _wrap(text: str, label: str, width: int = 78) -> str:
    # 8, not 7: the longest label ("STOPPED", "VERIFY") is 7 characters, and a
    # 7-wide field left no gap - the output read "STOPPEDverifier:a".
    body = " ".join((text or "").split())
    indent = " " * 8
    lines = textwrap.wrap(body, width=width - 8) or [""]
    out = [f"{label:<8}{lines[0]}"]
    out.extend(indent + line for line in lines[1:])
    return "\n".join(out)


def _verdict_badge(scored: ScoredResult) -> str:
    if scored.score == 1:
        return "PASS  +1"
    if scored.score == -1:
        return "WRONG -1"
    if scored.abstained:
        return "DECLINED 0"
    if scored.answer_correct:
        return "LOCATION 0"
    return "MISS   0"


def render_question(
    run: QuestionRun,
    position: str,
    pages_by_seq: dict[int, str],
    *,
    tolerance: Decimal,
    containment: float,
    page_slack: int,
    quote_chars: int = 240,
    price_in: float | None = None,
    price_out: float | None = None,
) -> str:
    """The per-question report: question, gold, got, why, and the proof."""
    q, a, s = run.question, run.answer, run.scored
    lines = [
        _RULE,
        f"{position} {q.qid}  [{q.shape.value}]  {q.doc_name}   {_verdict_badge(s)}",
        _wrap(q.question, "Q"),
        _wrap(q.answer, "GOLD"),
        _wrap(a.text or "(nothing)", "GOT"),
    ]

    lines.append(
        _wrap(
            ("answer OK — " if s.answer_correct else "answer NO — ")
            + explain_answer(q, a, tolerance),
            "WHY",
        )
    )
    if s.answer_correct or a.citations:
        lines.append(
            _wrap(
                ("location OK — " if s.location_correct else "location NO — ")
                + explain_location(
                    q, a, pages_by_seq,
                    containment=containment, page_slack=page_slack,
                ),
                "",
            )
        )

    if a.citations:
        lines.append("PROOF")
        for citation in a.citations:
            quote = " ".join((citation.quote or "").split())
            if len(quote) > quote_chars:
                quote = quote[:quote_chars] + " ..."
            mark = "  " if citation.doc_id == q.doc_name else " !"
            lines.append(f"     {mark}{citation.doc_id} p.{citation.page_seq}")
            lines.append(f"        \"{quote}\"")
    elif not s.abstained:
        lines.append("PROOF  (none — an answer with no citation can never score +1)")

    if run.abstain_reason:
        lines.append(_wrap(run.abstain_reason, "STOPPED"))
    if run.gate_detail:
        lines.append(_wrap(run.gate_detail, "GATE"))
    if run.verifiers:
        verdicts = ", ".join(
            f"{k}={'VALID' if v else 'REJECTED'}" for k, v in sorted(run.verifiers.items())
        )
        lines.append(_wrap(verdicts, "VERIFY"))
        # THE REJECTION REASON IS THE WHOLE POINT WHEN CALIBRATING.
        # Verifier rejection is the largest source of abstention here, and
        # "verifier:a" alone cannot tell you whether the verifier was RIGHT.
        # Only the rejections are shown - an approval's reasoning is noise.
        for who, verdict in sorted(run.verifiers.items()):
            reason = (run.verifier_reasons or {}).get(who)
            if reason and not verdict:
                lines.append(_wrap(f"{who}: {reason}", ""))
    if run.errors:
        lines.append(_wrap("; ".join(run.errors), "ERRORS"))

    cost = run.usage.cost(price_in, price_out)
    money = f"  ${cost:.4f}" if cost is not None else ""
    lines.append(
        f"COST   {run.usage.input_tokens:,} in / {run.usage.output_tokens:,} out"
        f"  ·  {run.usage.calls} calls  ·  {run.seconds:.0f}s{money}"
    )
    return "\n".join(lines)


@dataclass
class Tally:
    """A running scoreboard. Deliberately reports the -1 rate next to the score:
    a change that raises both is a regression, not an improvement."""

    n: int = 0
    total: int = 0
    plus: int = 0
    zero: int = 0
    minus: int = 0
    abstained: int = 0
    wrong_location: int = 0

    def add(self, scored: ScoredResult) -> None:
        self.n += 1
        self.total += scored.score
        if scored.score == 1:
            self.plus += 1
        elif scored.score == -1:
            self.minus += 1
        else:
            self.zero += 1
        if scored.abstained:
            self.abstained += 1
        if scored.answer_correct and not scored.location_correct:
            self.wrong_location += 1

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def false_answer_rate(self) -> float:
        return self.minus / self.n if self.n else 0.0

    def line(self, label: str) -> str:
        return (
            f"{label:<22}n={self.n:<4} score={self.total:<+5} mean={self.mean:<+7.3f} "
            f"+1={self.plus:<4} 0={self.zero:<4} -1={self.minus:<4} "
            f"declined={self.abstained:<4} RA/WL={self.wrong_location}"
        )


def render_batch_summary(
    batch: Batch, batch_tally: Tally, run_tally: Tally, usage: Usage,
    *, price_in: float | None, price_out: float | None,
) -> str:
    cost = usage.cost(price_in, price_out)
    money = f"   spend so far ${cost:.2f}" if cost is not None else ""
    return "\n".join([
        "",
        "=" * 78,
        f"{batch.label}",
        "=" * 78,
        batch_tally.line("this batch"),
        run_tally.line("run so far"),
        f"false-answer rate {100 * run_tally.false_answer_rate:.1f}% (target < 5%)"
        f"   ·  {usage.total_tokens:,} tokens over {usage.calls} calls{money}",
    ])

"""Eval reporting - per answer shape, always.

An aggregate score hides the failure that matters. 62% of gold answers are
non-numeric, so a system strong on the 52 numeric answers and weak on the other
84 reads as respectable in aggregate and then collapses in the live session.
Every report here breaks the score out by shape.

The dev/blind split is BY COMPANY, never by question: filings from one company
share boilerplate, so splitting by question leaks the answer's neighbourhood
across the split and inflates the blind score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gold import GoldQuestion
from .scorer import ScoredResult
from .shapes import AnswerShape

_SHAPE_ORDER = [
    AnswerShape.NUMERIC,
    AnswerShape.YES_NO,
    AnswerShape.PHRASE,
    AnswerShape.MULTI_SENTENCE,
]


@dataclass
class Bucket:
    n: int = 0
    total: int = 0
    plus_one: int = 0
    zero: int = 0
    minus_one: int = 0
    abstained: int = 0
    right_answer_wrong_location: int = 0

    def add(self, r: ScoredResult) -> None:
        self.n += 1
        self.total += r.score
        if r.score == 1:
            self.plus_one += 1
        elif r.score == -1:
            self.minus_one += 1
        else:
            self.zero += 1
        if r.abstained:
            self.abstained += 1
        if r.answer_correct and not r.location_correct:
            self.right_answer_wrong_location += 1

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def false_answer_rate(self) -> float:
        """The -1 rate. This is the number the abstention gate exists to hold down."""
        return self.minus_one / self.n if self.n else 0.0


@dataclass
class EvalReport:
    overall: Bucket = field(default_factory=Bucket)
    by_shape: dict[AnswerShape, Bucket] = field(default_factory=dict)
    unanswerable: list[tuple[str, str]] = field(default_factory=list)
    unmapped: list[tuple[str, str, float]] = field(default_factory=list)

    def add(self, question: GoldQuestion, result: ScoredResult) -> None:
        self.overall.add(result)
        self.by_shape.setdefault(question.shape, Bucket()).add(result)

    def render(self) -> str:
        lines: list[str] = []
        w = 78
        lines.append("=" * w)
        lines.append("RUBRIC SCORE   +1 answer&location | 0 refusal or wrong location | -1 wrong")
        lines.append("=" * w)
        head = f"{'shape':<16}{'n':>5}{'score':>8}{'mean':>8}{'+1':>6}{'0':>6}{'-1':>6}{'abst':>6}{'RA/WL':>7}"
        lines.append(head)
        lines.append("-" * w)
        for shape in _SHAPE_ORDER:
            b = self.by_shape.get(shape)
            if not b:
                continue
            lines.append(
                f"{shape.value:<16}{b.n:>5}{b.total:>8}{b.mean:>8.3f}"
                f"{b.plus_one:>6}{b.zero:>6}{b.minus_one:>6}{b.abstained:>6}"
                f"{b.right_answer_wrong_location:>7}"
            )
        b = self.overall
        lines.append("-" * w)
        lines.append(
            f"{'OVERALL':<16}{b.n:>5}{b.total:>8}{b.mean:>8.3f}"
            f"{b.plus_one:>6}{b.zero:>6}{b.minus_one:>6}{b.abstained:>6}"
            f"{b.right_answer_wrong_location:>7}"
        )
        lines.append("")
        lines.append(f"false-answer rate (-1): {100 * b.false_answer_rate:.1f}%   target < 5%")

        if self.unanswerable:
            lines.append("")
            lines.append(
                f"EXCLUDED as corpus defects ({len(self.unanswerable)}) - evidence is "
                "not in the supplied filing; abstaining is the best possible outcome:"
            )
            for qid, reason in self.unanswerable:
                lines.append(f"  {qid}  {reason}")

        if self.unmapped:
            lines.append("")
            lines.append(
                f"GOLD PAGE NOT MAPPED ({len(self.unmapped)}) - reported, never dropped; "
                "location cannot be scored for these:"
            )
            for qid, doc, score in self.unmapped:
                lines.append(f"  {qid}  {doc}  best jaccard={score:.2f}")
        return "\n".join(lines)


def split_by_company(
    questions: list[GoldQuestion], blind_fraction: float = 0.3
) -> tuple[list[GoldQuestion], list[GoldQuestion]]:
    """Deterministic dev/blind split BY COMPANY.

    Splitting by question would put two questions about the same filing on
    opposite sides, and the shared boilerplate would leak.
    """
    companies = sorted({q.company for q in questions})
    # Stable, seed-free assignment: every k-th company in sorted order is blind.
    stride = max(1, round(1 / blind_fraction))
    blind = {c for i, c in enumerate(companies) if i % stride == 0}
    dev = [q for q in questions if q.company not in blind]
    held = [q for q in questions if q.company in blind]
    return dev, held

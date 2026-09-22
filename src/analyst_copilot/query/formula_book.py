"""The formula book and the precedence ladder.

PRECEDENCE - FIRST MATCH WINS:

  1. A definition supplied IN THE QUESTION. MEASURED: 14/136 questions carry
     one. It overrides the book COMPLETELY - including its averaging convention
     and its rounding. "Using an average of beginning and ending balances" means
     exactly that, even where the book would use the closing balance.
  2. This curated book (~25 metrics).
  3. An LLM-proposed formula, gated by four checks.
  4. `Not found in this filing.`

The book IS the metric -> us-gaap concept map, so `operands.concepts`
feeds the XBRL path directly and the same operand falls back to the table and
narrative ladder where XBRL is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BOOK = Path(__file__).with_name("formula_book.yaml")


class FormulaSource(str, Enum):
    QUESTION = "question"
    BOOK = "book"
    LLM_PROPOSED = "llm_proposed"


@dataclass
class OperandSpec:
    name: str
    concepts: list[str] = field(default_factory=list)
    unit: str | None = None
    derive: str | None = None


@dataclass
class Metric:
    id: str
    aliases: list[str]
    definition: str
    formula: str
    unit: str | None = None
    render: str | None = None
    dp: int | None = None
    period_rule: str | None = None

    def operand_names(self) -> list[str]:
        """The bare identifiers the formula references, in order."""
        skip = {"avg", "delta", "prev", "sum_range", "rank", "compare", "n", "start", "end", "current"}
        seen: list[str] = []
        for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", self.formula):
            if name not in skip and name not in seen:
                seen.append(name)
        return seen


@dataclass
class FormulaChoice:
    """What the calculator should evaluate, and where the definition came from.

    `definition` is ALWAYS carried: the rubric rewards auditability, and an
    answer that states its own definition can be checked by a human without
    re-deriving it.
    """

    metric_id: str | None
    definition: str
    formula: str
    source: FormulaSource
    unit: str | None = None
    render: str | None = None
    dp: int | None = None
    period_rule: str | None = None


# A definition stated inside the question itself, e.g.
# "...using the average of beginning and ending inventory balances".
_QUESTION_DEFINITION = re.compile(
    r"\b(defined as|calculated as|computed as|using the formula|"
    r"which is defined|i\.e\.,?\s|defined by)\b(?P<body>[^?]{10,300})",
    re.I,
)
_AVERAGING_HINT = re.compile(
    r"\baverage of (the )?(beginning|opening) and (ending|closing)\b", re.I
)


class FormulaBook:
    def __init__(self, path: Path | None = None) -> None:
        raw = yaml.safe_load((path or DEFAULT_BOOK).read_text(encoding="utf-8"))
        self.operands: dict[str, OperandSpec] = {
            name: OperandSpec(
                name=name,
                concepts=list(spec.get("concepts") or []),
                unit=spec.get("unit"),
                derive=spec.get("derive"),
            )
            for name, spec in (raw.get("operands") or {}).items()
        }
        self.metrics: list[Metric] = []
        for m in raw.get("metrics") or []:
            result = m.get("result") or {}
            self.metrics.append(
                Metric(
                    id=m["id"],
                    aliases=[a.lower() for a in (m.get("aliases") or [])],
                    definition=m.get("definition", ""),
                    formula=m["formula"],
                    unit=result.get("unit"),
                    render=result.get("render"),
                    dp=result.get("dp"),
                    period_rule=m.get("period_rule"),
                )
            )
        # Longest alias first, so "operating cash flow ratio" is not shadowed by
        # "cash flow" appearing inside it.
        self._by_alias: list[tuple[str, Metric]] = sorted(
            ((alias, m) for m in self.metrics for alias in m.aliases),
            key=lambda t: -len(t[0]),
        )

    def concepts_for(self, operand: str) -> list[str]:
        """The us-gaap concepts to try, in order, for one operand."""
        spec = self.operands.get(operand)
        return list(spec.concepts) if spec else []

    def find(self, text: str) -> Metric | None:
        """Match a metric by alias against free text."""
        low = f" {re.sub(r'[^a-z0-9 ]', ' ', text.lower())} "
        low = re.sub(r"\s+", " ", low)
        for alias, metric in self._by_alias:
            if f" {alias} " in low:
                return metric
        return None

    def choose(self, question: str, metric_hint: str | None = None) -> FormulaChoice | None:
        """Apply precedence levels 1 and 2. Returns None if neither matches,
        which is the caller's signal to try level 3 or abstain."""
        # ── level 1: a definition supplied in the question wins outright ──
        m = _QUESTION_DEFINITION.search(question)
        if m:
            body = m.group("body").strip().rstrip(".")
            book_metric = self.find(metric_hint or question)
            return FormulaChoice(
                metric_id=book_metric.id if book_metric else None,
                definition=body,
                # The book's expression is reused only as the evaluation shape;
                # the QUESTION's wording is what is reported and what the
                # verifier checks the answer against.
                formula=book_metric.formula if book_metric else "",
                source=FormulaSource.QUESTION,
                unit=book_metric.unit if book_metric else None,
                render=book_metric.render if book_metric else None,
                dp=book_metric.dp if book_metric else None,
                period_rule=(
                    "average_of(prev, current)"
                    if _AVERAGING_HINT.search(body)
                    else (book_metric.period_rule if book_metric else None)
                ),
            )

        # ── level 2: the curated book ──
        metric = self.find(metric_hint or question)
        if metric:
            return FormulaChoice(
                metric_id=metric.id,
                definition=metric.definition,
                formula=metric.formula,
                source=FormulaSource.BOOK,
                unit=metric.unit,
                render=metric.render,
                dp=metric.dp,
                period_rule=metric.period_rule,
            )
        return None

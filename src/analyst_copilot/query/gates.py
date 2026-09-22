"""Deterministic gates G1-G7 with exact predicates.

Run in order, cheapest first. ANY failure => `Not found in this filing.` with
`abstain_reason` set to the gate id.

G1 IS THE SINGLE MOST IMPORTANT PIECE OF CODE IN THE SYSTEM. It makes an
invented figure or a fabricated citation STRUCTURALLY IMPOSSIBLE: if the quoted
text does not appear verbatim on the cited page, the answer is discarded. That
is exactly the -1 case the rubric punishes, and G1 costs one string search.

These gates are DETERMINISTIC and FAMILY-INDEPENDENT, which is why they carry
the system even though verifier independence is currently degraded (both
verifiers are gpt-5-mini). The LLM verifiers are the LAST net, not the first.

Gates are objects in a list, not an if-chain, so `eval/ablate.py` can disable
exactly one and measure its contribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Protocol

from .compute import Computation, Operand, recomputes, units_compatible


# ---------------------------------------------------------------------------
# The candidate under test
# ---------------------------------------------------------------------------
@dataclass
class Citation:
    doc_id: str
    page_seq: int
    quote: str
    page_printed: int | None = None
    section_path: str | None = None


@dataclass
class AnswerCandidate:
    """Everything the gates need. No gate reaches out to the database."""

    question: str
    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    operands: list[Operand] = field(default_factory=list)
    computation: Computation | None = None
    # doc_id -> page_seq -> raw_text. RAW TEXT ONLY - never `summary` or
    # `lexical_text`, which are paraphrases (see G1).
    page_texts: dict[str, dict[int, str]] = field(default_factory=dict)
    candidate_doc_ids: list[str] = field(default_factory=list)
    coverage_years: dict[str, list[int]] = field(default_factory=dict)
    evidence_years: list[int] = field(default_factory=list)
    intent: str = "historical"          # historical | forecast
    question_years: list[int] = field(default_factory=list)
    # scalar | ratio | comparison | ranking | qualitative. G4 reads this: a
    # "missing slot" means something different for a computed answer than for a
    # narrative one.
    answer_type: str = "scalar"


@dataclass
class GateResult:
    gate_id: str
    passed: bool
    detail: str = ""


class Gate(Protocol):
    id: str

    def check(self, candidate: AnswerCandidate) -> GateResult: ...


# ---------------------------------------------------------------------------
# G1 - the quote must appear verbatim on the cited page
# ---------------------------------------------------------------------------
_STRIP = str.maketrans("", "", "$,()%")


def normalize(text: str) -> str:
    """Casefold, collapse whitespace, strip `$ , ( ) %`.

    Deliberately narrow. Normalising more - stemming, removing letters,
    stripping digits - would let a paraphrase through, and the whole value of
    G1 is that it cannot be argued with.
    """
    text = (text or "").replace("\xa0", " ").translate(_STRIP)
    return re.sub(r"\s+", " ", text).strip().casefold()


class G1QuoteOnPage:
    """The quote must be found VERBATIM in the cited page's `raw_text`."""

    id = "G1"

    def __init__(self, min_quote_chars: int = 12) -> None:
        self._min = min_quote_chars

    def check(self, c: AnswerCandidate) -> GateResult:
        if not c.citations:
            return GateResult(self.id, False, "no citations")
        for cite in c.citations:
            quote = (cite.quote or "").strip()
            if len(quote) < self._min:
                return GateResult(
                    self.id, False, f"quote too short to verify: {quote!r}"
                )
            page_text = c.page_texts.get(cite.doc_id, {}).get(cite.page_seq)
            if page_text is None:
                return GateResult(
                    self.id, False,
                    f"cited page {cite.doc_id}#p{cite.page_seq} was not retrieved",
                )
            if normalize(quote) not in normalize(page_text):
                return GateResult(
                    self.id, False,
                    f"quote not found on {cite.doc_id}#p{cite.page_seq}: {quote[:80]!r}",
                )
        return GateResult(self.id, True)


# ---------------------------------------------------------------------------
# G2 - every cited document must be in the router's candidate set
# ---------------------------------------------------------------------------
class G1bValueInQuote:
    """A reported figure must appear IN the quote that supports it.

    G1 proves the quote is on the page. This proves the quote actually carries
    the number, which is a different claim and just as necessary: a quote of the
    line-item LABEL alone is on the page, passes G1, and supports nothing.

    MEASURED: the extractor initially quoted "Purchases of property, plant and
    equipment (PP&E)" with no figures. Every gate passed, and both LLM verifiers
    then rejected the answer - correctly - because the number was not in the
    evidence. That cost a full escalation round on a question we had right.
    Checking it deterministically is free and catches it a tier earlier.

    Digits only, after G1's normalisation, so "(1,577)" in the quote supports a
    stated value of "1577" or "-1577".
    """

    id = "G1b"

    def check(self, c: AnswerCandidate) -> GateResult:
        numeric = [o for o in c.operands if o.citation]
        if not numeric:
            return GateResult(self.id, True, "no numeric operands to check")
        for operand in numeric:
            # Check against THIS operand's own quote. Keying quotes by page
            # was a bug: two slots frequently cite the same page (a statement
            # supplies several line items), and the page-keyed lookup then
            # checked operand A's digits against operand B's quote, rejecting
            # correct answers.
            quote = (operand.citation or {}).get("quote", "")
            quote_digits = re.sub(r"\D", "", quote or "")
            if not quote_digits:
                return GateResult(
                    self.id, False,
                    f"quote for {operand.name!r} contains no figures",
                )
            # Compare the integer part only: the quote shows "(1,577)" while the
            # operand may be -1577, and a scale note may add decimals.
            digits = re.sub(r"\D", "", str(abs(operand.value)).split(".")[0])
            if digits and digits not in quote_digits:
                return GateResult(
                    self.id, False,
                    f"value {operand.value} for {operand.name!r} does not appear "
                    f"in its quote",
                )
        return GateResult(self.id, True)


class G2DocInScope:
    id = "G2"

    def check(self, c: AnswerCandidate) -> GateResult:
        if not c.candidate_doc_ids:
            return GateResult(self.id, True, "no router scope recorded")
        scope = set(c.candidate_doc_ids)
        for cite in c.citations:
            if cite.doc_id not in scope:
                return GateResult(
                    self.id, False, f"{cite.doc_id} is outside the routed candidates"
                )
        return GateResult(self.id, True)


# ---------------------------------------------------------------------------
# G3 - the evidence period must match what was asked
# ---------------------------------------------------------------------------
class G3PeriodMatch:
    """Evidence year must fall in the filing's `coverage_years`.

    A FY2018 10-K legitimately answers a 2016 question from its comparative
    columns, which is why coverage_years exists rather than a bare fiscal_year.

    The forecast exception matters: "What is Boeing forecasting for FY2023?"
    is answered from an EARLIER filing, so a question period AFTER the filing
    period is correct for forecast intent and wrong for historical intent.
    """

    id = "G3"

    def check(self, c: AnswerCandidate) -> GateResult:
        if not c.evidence_years or not c.coverage_years:
            return GateResult(self.id, True, "no period information to check")
        for cite in c.citations:
            covered = c.coverage_years.get(cite.doc_id)
            if not covered:
                continue
            if any(y in covered for y in c.evidence_years):
                continue
            if c.intent == "forecast" and c.question_years and any(
                y > max(covered) for y in c.question_years
            ):
                continue
            return GateResult(
                self.id, False,
                f"evidence years {c.evidence_years} outside {cite.doc_id} "
                f"coverage {covered}",
            )
        return GateResult(self.id, True)


# ---------------------------------------------------------------------------
# G4 - no missing slots
# ---------------------------------------------------------------------------
class G4SlotsComplete:
    """The extractor lists what it could not find; we abstain rather than infer
    it - an explicit, cheap way for the model to stop.

    SCOPED TO ANSWERS BUILT FROM OPERANDS. A missing operand makes a computed
    answer unsound, full stop. But for a narrative question ("which segment
    dragged down growth?", "why did margin fall?") the extractor routinely lists
    figures it would have liked without needing them, and enforcing emptiness
    there abstained on questions the evidence fully supported - measured as the
    single largest source of abstention once the composer was added.

    For narrative answers the composer's own `answerable` flag and the two
    verifiers decide, which is what those stages are for.
    """

    id = "G4"

    _OPERAND_TYPES = {"scalar", "ratio"}

    def check(self, c: AnswerCandidate) -> GateResult:
        if not c.missing_slots:
            return GateResult(self.id, True)
        if c.computation is not None or c.answer_type in self._OPERAND_TYPES:
            return GateResult(self.id, False, f"missing: {', '.join(c.missing_slots)}")
        return GateResult(
            self.id, True,
            f"advisory only for a {c.answer_type} answer: {', '.join(c.missing_slots)}",
        )


# ---------------------------------------------------------------------------
# G5 - units and scale must be compatible
# ---------------------------------------------------------------------------
class G5UnitsCompatible:
    id = "G5"

    def check(self, c: AnswerCandidate) -> GateResult:
        if len(c.operands) < 2:
            return GateResult(self.id, True)
        if not units_compatible(c.operands):
            units = sorted({o.unit or "?" for o in c.operands})
            return GateResult(self.id, False, f"incompatible units: {units}")
        return GateResult(self.id, True)


# ---------------------------------------------------------------------------
# G6 - the formula must reproduce the stated answer
# ---------------------------------------------------------------------------
class G6FormulaRecomputes:
    """Catches the case where the arithmetic was right but the prose disagrees."""

    id = "G6"

    def __init__(self, tolerance: Decimal = Decimal("0.000001")) -> None:
        self._tol = tolerance

    def check(self, c: AnswerCandidate) -> GateResult:
        if c.computation is None:
            return GateResult(self.id, True, "no computation to check")
        if not recomputes(c.answer_text, c.computation, self._tol):
            return GateResult(
                self.id, False,
                f"stated answer does not match {c.computation.formula} "
                f"= {c.computation.rendered()}",
            )
        return GateResult(self.id, True)


# ---------------------------------------------------------------------------
# G7 - the cited location must exist
# ---------------------------------------------------------------------------
class G7LocationResolves:
    id = "G7"

    def check(self, c: AnswerCandidate) -> GateResult:
        for cite in c.citations:
            pages = c.page_texts.get(cite.doc_id, {})
            if cite.page_seq not in pages:
                return GateResult(
                    self.id, False,
                    f"{cite.doc_id}#p{cite.page_seq} does not exist",
                )
        return GateResult(self.id, True)


# ---------------------------------------------------------------------------
# The gate chain
# ---------------------------------------------------------------------------
DEFAULT_GATES: list[Gate] = [
    G1QuoteOnPage(),
    G1bValueInQuote(),
    G2DocInScope(),
    G3PeriodMatch(),
    G4SlotsComplete(),
    G5UnitsCompatible(),
    G6FormulaRecomputes(),
    G7LocationResolves(),
]


@dataclass
class GateReport:
    passed: bool
    results: list[GateResult] = field(default_factory=list)
    failed_gate: str | None = None

    @property
    def abstain_reason(self) -> str | None:
        return self.failed_gate


def run_gates(
    candidate: AnswerCandidate,
    gates: list[Gate] | None = None,
    enabled: tuple[str, ...] | None = None,
) -> GateReport:
    """Run the chain, stopping at the first failure.

    `enabled` comes from `verification.enabled_gates` in config.yaml so the
    ablation runner can switch one off and measure what it was worth.
    """
    chain = gates if gates is not None else DEFAULT_GATES
    results: list[GateResult] = []
    for gate in chain:
        if enabled is not None and gate.id not in enabled:
            continue
        result = gate.check(candidate)
        results.append(result)
        if not result.passed:
            return GateReport(passed=False, results=results, failed_gate=gate.id)
    return GateReport(passed=True, results=results)

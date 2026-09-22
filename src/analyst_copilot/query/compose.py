"""The ANSWER stage - composing the final answer from evidence.

THE MAJORITY CASE IS NOT NUMBER EXTRACTION. MEASURED: 84 of 136 gold
answers (62%) are non-numeric - judgements, phrases, explanations. A pipeline
that renders only a figure answers 38% of questions in the right shape and
returns a bare number for the rest.

MEASURED FAILURE that produced this module: asked "which segment has dragged
down 3M's overall growth in 2022?" (gold: "The consumer segment shrunk by 0.9%
organically") the pipeline returned `8,902` - the first numeric slot. Every
deterministic gate passed, because the figure was correctly quoted and correctly
located. It was simply not an answer to the question. That is a -1 the gates
cannot catch, because it is a RELEVANCE failure, not a grounding failure.

So the answer shape switches here:
  * a computed metric  -> the calculator's result, rendered (no model involved)
  * a bare scalar      -> the matching slot's value
  * anything else      -> composed from the quotes by the model, under the same
                          evidence-only constraint as extraction

The composer may only use the supplied quotes, and it may decline. G1 still
checks every citation afterwards, so composition cannot introduce a figure that
is not on the cited page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm import schemas
from ..llm.base import LLMError, LLMProvider
from ..llm.registry import load_prompt
from .compute import Computation
from .extract import Extraction

@dataclass
class ComposedAnswer:
    text: str
    answerable: bool = True
    used_model: bool = False
    # Which quotes the composer says its answer rests on. Advisory for now -
    # recorded so its predictive value can be measured before it gates anything.
    supporting_quote_indexes: list[int] = field(default_factory=list)
    # True when a refusal was caught in the prose rather than the flag. Worth
    # counting: it means the prompt's rule 5 is not landing.
    declined_in_prose: bool = False


# A REFUSAL WRITTEN AS PROSE IS STILL A REFUSAL, AND SHIPPING IT AS AN ANSWER
# COSTS TWO POINTS.
# MEASURED (financebench_id_00678): the composer replied
#     "I cannot determine whether Boeing has an improving gross margin profile
#      as of FY2022 because the only supplied data are total revenues ... and no
#      gross profit or cost of goods sold figures are provided."
# That is EXACTLY the behaviour this system is built to produce — it recognised
# its evidence was insufficient and said so. But it set `answerable: true`, so
# the text went out as `status="answered"` and the rubric scored it -1 instead
# of the 0 an honest refusal earns. The system was right and the plumbing
# punished it.
#
# The schema flag alone cannot be trusted: the model reasons its way to "I can't
# answer this" in the prose while leaving the boolean at its default. So the
# TEXT is checked too.
#
# Deliberately anchored to the OPENING of the answer. A real answer may well
# contain "not disclosed" partway through ("segment detail is not disclosed, but
# total revenue was $X"); one that OPENS by declining is declining.
_DECLINE_OPENING = re.compile(
    r"^\W*(?:"
    r"i\s+(?:cannot|can't|am\s+unable|do\s+not\s+have)"
    r"|(?:it\s+is\s+)?not\s+possible\s+to\s+determine"
    r"|unable\s+to\s+(?:determine|answer|verify)"
    r"|cannot\s+be\s+determined"
    r"|there\s+is\s+(?:no|insufficient)\s+"
    r"|(?:the\s+)?(?:provided|supplied|available|given)\s+"
    r"(?:pages?|passages?|excerpts?|evidence|data|information|quotes?)\s+"
    r"(?:do(?:es)?\s+not|don't|doesn't)"
    r"|no\s+(?:information|evidence|data)\s+(?:is\s+)?(?:provided|available|given)"
    r")",
    re.I,
)


def reads_as_decline(text: str) -> bool:
    """True when composed prose is itself a refusal.

    Routing it to the abstain path converts a -1 into a 0 — and emits the exact
    refusal string, which is what the rubric actually scores.
    """
    return bool(_DECLINE_OPENING.match((text or "").strip()))


def _pick_slot(extraction: Extraction, question_years: list[int]):
    """The slot whose period matches the question.

    A statement presents three years side by side, so the extractor routinely
    returns the same line item for several periods. Taking the first is a
    plausible figure from the wrong year - the -1 no gate can catch.
    """
    valued = [s for s in extraction.slots if s.value is not None] or extraction.slots
    if not valued:
        return None
    if question_years:
        target = str(max(question_years))
        matching = [s for s in valued if target in (s.period or "")]
        if matching:
            return matching[0]
    return valued[0]


def compose_answer(
    extraction: Extraction,
    computation: Computation | None,
    question: str,
    question_years: list[int],
    composer: LLMProvider | None = None,
) -> ComposedAnswer:
    """Render the final answer in the shape the question asked for."""
    # A computed metric is rendered by the calculator, never by the model.
    if computation is not None:
        return ComposedAnswer(computation.rendered())

    if not extraction.slots:
        return ComposedAnswer("", answerable=False)

    # EVEN A SCALAR GOES THROUGH THE COMPOSER when one is available.
    # MEASURED: returning the bare slot value gave the answer "1,577" for
    # "What is the FY2018 capital expenditure amount (in USD millions) for 3M?".
    # Every gate passed, and verifier A then rejected it - correctly - because a
    # naked figure states neither its unit nor its period, so it cannot be
    # checked for the scale and period consistency the verifier is asked to
    # confirm. "$1,577 million in FY2018" is the same fact, verifiable.
    if composer is None:
        slot = _pick_slot(extraction, question_years)
        return ComposedAnswer(
            str(slot.value) if slot and slot.value is not None else (slot.quote if slot else ""),
        )

    quotes = "\n".join(
        f"[{i}] ({s.doc_id} p.{s.page_seq}) {s.name}: {s.value} — \"{s.quote}\""
        for i, s in enumerate(extraction.slots)
    )
    prompt_name, schema = schemas.BY_STAGE["composer"]
    try:
        data = composer.complete(
            system=load_prompt(prompt_name),
            user=f"QUESTION:\n{question}\n\nEVIDENCE QUOTES:\n{quotes}",
            schema=schema,
            stage="composer",
        ).data
    except LLMError:
        # Composition failing is not evidence of absence, but we cannot answer.
        return ComposedAnswer("", answerable=False)

    if not data.get("answerable", True):
        return ComposedAnswer("", answerable=False)

    text = (data.get("answer") or "").strip()
    # The flag says answerable, but the prose may say otherwise. Trust the prose:
    # shipping a written refusal as an answer scores -1 where declining scores 0.
    # The prompt now forbids prose refusals outright; this stays as the backstop,
    # because the cost of one slipping through is two points.
    if reads_as_decline(text):
        return ComposedAnswer("", answerable=False, declined_in_prose=True)

    # GROUNDING SIGNAL THAT WAS BEING THROWN AWAY. `supporting_quote_indexes`
    # has always been in the schema and was never read. A composer that cannot
    # name a single quote its answer rests on is not answering from the
    # evidence - which is exactly the relevance failure that produced a -1 on
    # financebench_id_01935 (asked about supplemental indentures, answered about
    # an accounting-standard adoption).
    #
    # Treated as ADVISORY, not a gate: the field is model-populated, so making it
    # mandatory would convert a formatting lapse into a lost answer. It is
    # recorded so the next run can measure whether it predicts wrong answers
    # before anything is gated on it.
    supporting = data.get("supporting_quote_indexes")
    return ComposedAnswer(
        text,
        answerable=True,
        used_model=True,
        supporting_quote_indexes=list(supporting) if supporting else [],
    )

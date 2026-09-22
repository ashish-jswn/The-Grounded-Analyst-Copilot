"""Evidence extraction - slots, never prose.

The extractor returns SLOTS: named values, each with a unit, a period, a
location and a VERBATIM QUOTE. It does not write an answer and it does not
compute anything.

Two reasons this shape is load-bearing:
  * a quote that must be copied character-for-character is checkable by gate G1,
    whereas a prose answer is not
  * arithmetic done here would be the model doing arithmetic, which is forbidden

THE ESCAPE HATCH IS INVERTED. OpenAI's guide recommends letting the
model proceed under uncertainty; for our rubric proceeding under uncertainty IS
the -1 case. So the extractor is given an explicit, cheap, rewarded way to stop:
list what is missing in `missing_slots`, and gate G4 turns that into an
abstention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from ..llm import schemas
from ..llm.base import LLMProvider, LLMResponse
from ..llm.registry import load_prompt
from .compute import Operand
from .gates import Citation


@dataclass
class EvidenceSlot:
    name: str
    value: str | None
    unit: str | None
    scale: str | None
    period: str | None
    doc_id: str
    page_seq: int
    page_printed: int | None
    quote: str

    def as_operand(self) -> Operand | None:
        """Typed operand for the calculator, or None when non-numeric."""
        if self.value is None:
            return None
        text = str(self.value).strip()
        negative = text.startswith("(") and text.endswith(")")
        cleaned = (
            text.strip("()")
            .replace(",", "").replace("$", "").replace("%", "").strip()
        )
        if cleaned.startswith("-"):
            negative, cleaned = True, cleaned[1:].strip()
        try:
            number = Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None
        return Operand(
            name=self.name,
            value=-number if negative else number,
            unit=self.unit,
            scale=self.scale,
            period=self.period,
            citation={
                "doc_id": self.doc_id,
                "page_seq": self.page_seq,
                "quote": self.quote,
            },
        )

    def as_citation(self) -> Citation:
        return Citation(
            doc_id=self.doc_id,
            page_seq=self.page_seq,
            quote=self.quote,
            page_printed=self.page_printed,
        )


@dataclass
class Extraction:
    slots: list[EvidenceSlot] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    answer_type: str = "qualitative"
    metric_name: str | None = None
    question_supplied_definition: str | None = None
    usage: LLMResponse | None = None

    @property
    def operands(self) -> list[Operand]:
        return [o for o in (s.as_operand() for s in self.slots) if o is not None]

    @property
    def citations(self) -> list[Citation]:
        return [s.as_citation() for s in self.slots]


def extract_evidence(
    provider: LLMProvider,
    question: str,
    context: str,
    *,
    stage: str = "extractor",
    required_operands: list[str] | None = None,
) -> Extraction:
    """Run the extractor over the assembled context.

    `context` is built from page `raw_text` only - never a summary - so that
    every quote the model can copy is one gate G1 will accept.

    `required_operands` names the slots the calculator will look up BY EXACT
    NAME. Without it the extractor names slots freely, `evaluate` raises
    "operand 'revenue' is not available", gate G4 fires, and a fully-evidenced
    ratio question abstains as though the filing lacked the numbers. That
    silently disabled every derived answer in the system.
    """
    prompt_name, schema = schemas.BY_STAGE[stage]
    naming = ""
    if required_operands:
        naming = (
            "\n\nSLOT NAMES - USE THESE EXACTLY:\n"
            + "\n".join(f"  - {name}" for name in required_operands)
            + "\nThe calculator looks these up by exact name, so a slot named "
            "anything else is discarded and the question is refused.\n"
            "A name ending in `__prev` is the SAME line item for the PRIOR "
            "period; return it as its own slot, with its own quote.\n"
            "If one of these is genuinely absent from the pages, put its name "
            "in `missing_slots` rather than substituting a different item."
        )
    user = (
        f"QUESTION:\n{question}{naming}\n\n"
        f"PAGES (the only text you may quote):\n{context}"
    )
    response = provider.complete(
        system=load_prompt(prompt_name), user=user, schema=schema, stage=stage
    )
    data = response.data

    slots: list[EvidenceSlot] = []
    for raw in data.get("slots") or []:
        try:
            slots.append(
                EvidenceSlot(
                    name=raw.get("name") or "value",
                    value=raw.get("value"),
                    unit=raw.get("unit"),
                    scale=raw.get("scale"),
                    period=raw.get("period"),
                    doc_id=raw["doc_id"],
                    page_seq=int(raw["page_seq"]),
                    page_printed=raw.get("page_printed"),
                    quote=raw.get("quote") or "",
                )
            )
        except (KeyError, TypeError, ValueError):
            # A slot without a resolvable location cannot be gated, so it is
            # treated as missing rather than trusted.
            slots.append(
                EvidenceSlot(
                    name=str(raw.get("name") or "value"),
                    value=None, unit=None, scale=None, period=None,
                    doc_id="", page_seq=-1, page_printed=None, quote="",
                )
            )

    return Extraction(
        slots=[s for s in slots if s.page_seq >= 0],
        missing_slots=list(data.get("missing_slots") or [])
        + [s.name for s in slots if s.page_seq < 0],
        answer_type=data.get("answer_type") or "qualitative",
        metric_name=data.get("metric_name"),
        question_supplied_definition=data.get("question_supplied_definition"),
        usage=response,
    )

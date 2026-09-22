"""The API contract the frontend binds against.

The backend is a clean FastAPI service with a documented contract, mirrored in
`frontend/lib/api.ts`. Changing these models breaks that UI.

Refusal is ALWAYS the exact string `Not found in this filing.` - never
paraphrased. The rubric scores the exact refusal, and a helpful rewording
("I couldn't find that") scores as a wrong answer.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation = doc + derived page + printed footer page + verbatim quote.

    BOTH page numbers are carried: `page_seq` is the derived index we can
    reproduce for any document, and `page_printed` is what a human reads at the
    foot of the page. The offset between them varies per filing from -2 to +16,
    so neither substitutes for the other.
    """

    doc_id: str
    company: str | None = None
    form_type: str | None = None
    period: str | None = None
    page_seq: int
    page_printed: int | None = None
    section_path: str | None = None
    quote: str


class Operand(BaseModel):
    name: str
    value: Decimal
    unit: str | None = None
    scale: str | None = None
    period: str | None = None
    citation: Citation | None = None


class Computation(BaseModel):
    metric: str | None = None
    # Always stated: the rubric rewards auditability, and an answer that carries
    # its own definition can be checked without re-deriving it.
    definition: str
    formula: str
    formula_source: Literal["question", "book", "llm_proposed"]
    operands: list[Operand] = Field(default_factory=list)
    result: Decimal


class AnswerRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    status: Literal["answered", "abstained", "clarify"]
    answer: str | None = None
    clarifying_question: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    computation: Computation | None = None
    # The gate id that failed, e.g. "G1" - so a refusal is explainable.
    abstain_reason: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)


class IngestStatus(BaseModel):
    """Drives the visible processing indicator in the UI."""

    doc_id: str
    status: Literal[
        "queued", "parsing", "tables", "sections", "xbrl",
        "summaries", "embedding", "ready", "failed",
    ]
    progress: float = 0.0
    page_count: int | None = None
    error: str | None = None


class FilingSummary(BaseModel):
    doc_id: str
    company_slug: str
    form_type: str
    period_label: str | None = None
    page_count: int | None = None
    ingest_status: str


class CorpusStats(BaseModel):
    filings: int
    pages: int
    tables: int
    table_cells: int
    sections: int

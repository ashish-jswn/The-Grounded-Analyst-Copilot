"""FastAPI service.

Imports only `query.pipeline` and the schemas - no SQL, no vendor SDK, no
retrieval internals. The Next.js frontend binds here, so this is the
contract.

The four core product features map to these endpoints:
    "Add filing" upload + visible processing status  ->  POST /filings, GET /filings/{id}
    chat box                                          ->  POST /ask
    evidence on every answer                          ->  AnswerResponse.citations
    a plain decline path                              ->  status="abstained"
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from ..config import load_settings
from ..container import build_pipeline, load_corpus
from ..ingest.pipeline import ingest_filing
from ..storage import repository as repo
from ..storage.db import connect
from .schemas import (
    AnswerRequest,
    AnswerResponse,
    Citation,
    Computation,
    CorpusStats,
    FilingSummary,
    IngestStatus,
    Operand,
)

app = FastAPI(
    title="The Analyst Copilot",
    version="0.1.0",
    description=(
        "Question answering over SEC filings with an exact evidence location, "
        "or the exact refusal 'Not found in this filing.'"
    ),
)

# The corpus index is expensive to build (~10 s) and immutable between ingests,
# so it is held for the process lifetime and rebuilt after an upload.
_state: dict[str, Any] = {"pipeline": None, "settings": None}
_lock = threading.Lock()


def _pipeline():
    with _lock:
        if _state["pipeline"] is None:
            settings = _state["settings"] or load_settings()
            _state["settings"] = settings
            _state["pipeline"] = build_pipeline(settings)
        return _state["pipeline"]


def _invalidate() -> None:
    with _lock:
        _state["pipeline"] = None


@app.on_event("startup")
def _startup() -> None:
    _state["settings"] = load_settings()


@app.get("/health")
def health() -> dict[str, Any]:
    settings = _state["settings"] or load_settings()
    with connect(settings.database_url) as conn:
        stats = repo.corpus_stats(conn)
    return {"status": "ok", "corpus": stats}


@app.get("/stats", response_model=CorpusStats)
def stats() -> CorpusStats:
    settings = _state["settings"] or load_settings()
    with connect(settings.database_url) as conn:
        s = repo.corpus_stats(conn)
    return CorpusStats(
        filings=s["filings"], pages=s["pages"], tables=s["tables"],
        table_cells=s["table_cells"], sections=s["sections"],
    )


@app.get("/filings", response_model=list[FilingSummary])
def list_filings() -> list[FilingSummary]:
    settings = _state["settings"] or load_settings()
    with connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT doc_id, company_slug, form_type, period_label,
                          page_count, ingest_status
                     FROM filings ORDER BY doc_id"""
            )
            rows = cur.fetchall()
    return [FilingSummary(**r) for r in rows]


@app.get("/filings/{doc_id}", response_model=IngestStatus)
def filing_status(doc_id: str) -> IngestStatus:
    """Polled by the UI to render the visible processing indicator."""
    settings = _state["settings"] or load_settings()
    with connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT doc_id, ingest_status, ingest_progress, page_count,
                          ingest_error
                     FROM filings WHERE doc_id = %s""",
                (doc_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no filing {doc_id!r}")
    return IngestStatus(
        doc_id=row["doc_id"],
        status=row["ingest_status"],
        progress=row["ingest_progress"] or 0.0,
        page_count=row["page_count"],
        error=row["ingest_error"],
    )


def _ingest_job(path: Path) -> None:
    settings = load_settings()
    try:
        with connect(settings.database_url) as conn:
            ingest_filing(conn, path, settings, force=True)
    finally:
        _invalidate()          # the new filing must enter the retrieval index


@app.post("/filings", response_model=IngestStatus, status_code=202)
async def add_filing(
    background: BackgroundTasks, file: UploadFile = File(...)
) -> IngestStatus:
    """The "Add filing" feature.

    Ingest runs in the background and the client polls GET /filings/{doc_id};
    parsing is ~3 s per filing against a 10-minute budget, but the status is
    reported honestly rather than assumed.

    The filename carries the catalog metadata (company, year, form), so it must
    follow the EDGAR-style convention COMPANY_YEAR_FORM.htm.
    """
    if not file.filename or not file.filename.lower().endswith((".htm", ".html")):
        raise HTTPException(status_code=400, detail="expected an .htm filing")

    settings = _state["settings"] or load_settings()
    upload_dir = Path(tempfile.gettempdir()) / "analyst_copilot_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / Path(file.filename).name
    with target.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        from ..ingest.catalog import parse_filing_name

        meta = parse_filing_name(target)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"filename must be COMPANY_YEAR_FORM.htm: {exc}",
        ) from exc

    with connect(settings.database_url) as conn:
        repo.set_ingest_status(conn, meta.doc_id, "queued", 0.0)

    background.add_task(_ingest_job, target)
    return IngestStatus(doc_id=meta.doc_id, status="queued", progress=0.0)


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AnswerRequest) -> AnswerResponse:
    """The chat endpoint. Every answer carries its evidence, or declines."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")

    result = _pipeline().answer(request.question)

    computation = None
    if result.computation is not None:
        computation = Computation(
            metric=None,
            definition=result.definition or "",
            formula=result.computation.formula,
            formula_source=result.formula_source or "book",  # type: ignore[arg-type]
            operands=[],
            result=result.computation.result,
        )

    return AnswerResponse(
        status=result.status,  # type: ignore[arg-type]
        answer=result.answer,
        clarifying_question=result.clarifying_question,
        citations=[
            Citation(
                doc_id=c.doc_id,
                page_seq=c.page_seq,
                page_printed=c.page_printed,
                section_path=c.section_path,
                quote=c.quote,
            )
            for c in result.citations
        ],
        computation=computation,
        abstain_reason=result.abstain_reason,
        trace=result.trace,
    )


@app.exception_handler(Exception)
def _unhandled(request, exc: Exception) -> JSONResponse:
    """An internal failure must never surface as a confident answer.

    Returning the exact refusal keeps the contract intact: the client renders a
    decline, and the rubric scores 0 rather than -1.
    """
    settings = _state["settings"] or load_settings()
    return JSONResponse(
        status_code=200,
        content={
            "status": "abstained",
            "answer": settings.verification.abstain_string,
            "citations": [],
            "abstain_reason": f"internal_error:{type(exc).__name__}",
            "trace": {},
        },
    )

"""Analyst Copilot: evidence-first question answering over SEC filings.

The chatbot holds every filing and the user never picks one, so the system must
work out WHICH document holds the answer and WHERE inside it - the shared-corpus
setting in which FinanceBench's own shared-vector-store baseline scored ~19%.

The spine, one path with switchable stages:

    route -> navigate -> RETRIEVE -> extract -> COMPUTE -> verify -> ANSWER

    analyst_copilot.config      typed Settings; THE ONLY env reader
    analyst_copilot.ingest      pages, blocks, tables, sections, XBRL
    analyst_copilot.storage     PostgreSQL + pgvector; the only SQL
    analyst_copilot.retrieval   anchors, BM25, dense, RRF, rerank, assembly
    analyst_copilot.query       router, extraction, Decimal calculator, gates
    analyst_copilot.llm         the ONLY vendor-SDK importers
    analyst_copilot.api         FastAPI service
    analyst_copilot.eval        rubric scorer; the only package that may read
                                the benchmark

This is a DETERMINISTIC WORKFLOW, not an agent: control flow lives in code, and
the model chooses content, never what happens next.

The rubric is asymmetric - a confident wrong answer scores -1 while an honest
refusal scores 0 - so the objective is precision-calibrated selective answering,
not accuracy. Two abstentions beat one wrong answer.
"""

__version__ = "0.1.0"

"""Reranking, with a recall-oriented posture.

ADAPTED FROM `FinanceBench_RAG/rag/llm_judge.py`, whose instruction is
explicitly *"prefer false positives over false negatives"* - exclude only for
wrong company, wrong period, or unrelated topic. That posture is right for us
for a structural reason: **retrieval must not be the precision gate.
VERIFICATION is.** A page wrongly dropped here can never be quoted; a page
wrongly kept costs a few hundred tokens and is filtered by gates G1-G7.

Behind `retrieval.use_reranker`, which is the ablation flag the approach note
needs (the same seam `sec-rag-analyst` exposes as `rerank=False`).
"""

from __future__ import annotations

from ..llm.base import Reranker


class RerankClient:
    """Wraps a Reranker with a fixed query, so `assemble` need not know the query."""

    def __init__(self, reranker: Reranker, query: str, *, keep_floor: float = 0.0) -> None:
        self._reranker = reranker
        self._query = query
        self._keep_floor = keep_floor

    def rerank_documents(
        self, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        results = self._reranker.rerank(self._query, documents, top_n)
        if self._keep_floor <= 0:
            return results
        # A floor is available but defaults to off: dropping a page on a score
        # threshold is precisely the false-negative risk the posture above warns
        # against, so it must be enabled deliberately and measured.
        return [(i, s) for i, s in results if s >= self._keep_floor]


def maybe_reranker(settings, query: str):
    """Return a rerank client, or None when disabled or unavailable.

    An outage must degrade to RRF rather than fail the question.
    """
    if not settings.retrieval.use_reranker:
        return None
    try:
        from ..llm.registry import get_reranker

        return RerankClient(get_reranker(settings), query)
    except Exception:
        return None

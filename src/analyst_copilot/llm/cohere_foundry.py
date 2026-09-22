"""Cohere reranker on Azure AI Foundry (serverless endpoint, plain HTTP).

Called through `httpx` rather than a vendor SDK: the Foundry rerank endpoint is
a single POST, and adding a dependency to make one request would be worse than
the request. It still lives in `llm/` because that is where network calls to
model providers belong.

The reranker is behind `retrieval.use_reranker`, which is the ablation flag the
approach note needs - `sec-rag-analyst` shipped the same seam and it is the
cheapest way to state what the reranker was actually worth.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import ModelSettings, Settings
from .base import LLMError, Reranker


class CohereFoundryReranker(Reranker):
    def __init__(self, settings: Settings, model: ModelSettings, timeout: float = 60.0) -> None:
        self._endpoint = (model.endpoint or settings.cohere_rerank_endpoint or "").rstrip("/")
        self._api_key = settings.cohere_rerank_api_key
        self._model = model.deployment
        self._timeout = timeout
        if not self._endpoint:
            raise LLMError("COHERE_RERANK_ENDPOINT is not set")
        if not self._api_key:
            raise LLMError("COHERE_RERANK_API_KEY is not set")

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Return [(original_index, relevance_score)], best first."""
        if not documents:
            return []
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        try:
            response = httpx.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise LLMError(f"rerank: {type(exc).__name__}: {exc}") from exc

        results = body.get("results") or []
        return [
            (int(r["index"]), float(r.get("relevance_score", 0.0)))
            for r in results
        ]

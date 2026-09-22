"""Dense retrieval - the third RRF signal.

IN MEMORY, NOT A DATABASE QUERY PER QUESTION, and that is a deliberate
match to the rest of this stack. BM25, the structure anchors and the page text
are all already resident: the corpus is 8,389 pages and it is loaded once at
startup. A `<=>` query per question would add a Postgres round-trip to every
call for a similarity search that takes ~10 ms against a resident matrix, and
it would open a fresh connection inside the query path, which is exactly the
coupling `container.py` exists to prevent.

The whole matrix is 8,389 x 1024 float32 = ~34 MB. That is a rounding error
against the page text already held in the same process.

VECTORS ARE NORMALISED ONCE, AT LOAD. Cosine similarity between unit vectors
is a dot product, so the per-question cost is one matrix multiply. Normalising
per query instead would repeat 8,389 square roots on every question for a value
that never changes.

A PAGE WITH NO VECTOR MUST NOT RANK. `dense_search` in SQL filters
`WHERE embedding IS NOT NULL` for the same reason this class drops unembedded
pages at load: an all-zero row would score 0.0 against every query, which is
*better* than the negative similarities real pages can have - a missing vector
would outrank a genuinely dissimilar page. Partial coverage must degrade, never
corrupt.
"""

from __future__ import annotations

import numpy as np

from ..llm.base import Embedder
from .base import Hit


class DenseRetriever:
    """Cosine search over resident page vectors, always scoped to candidates."""

    def __init__(
        self,
        rows: list[dict],
        embedder: Embedder | None = None,
    ) -> None:
        """`rows` need page_id, doc_id, page_seq, raw_text and embedding."""
        self._embedder = embedder
        self._page_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._seqs: list[int] = []
        self._texts: list[str] = []
        vectors: list[np.ndarray] = []

        for r in rows:
            vec = _as_vector(r.get("embedding"))
            if vec is None:
                continue
            self._page_ids.append(r["page_id"])
            self._doc_ids.append(r["doc_id"])
            self._seqs.append(r["page_seq"])
            self._texts.append(r.get("raw_text") or "")
            vectors.append(vec)

        if vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            # A zero-norm row would divide by zero and poison the matrix with
            # NaN, which propagates silently through argsort.
            norms[norms == 0] = 1.0
            self._matrix = matrix / norms
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

        # doc_id -> row indices, so scoping is a slice rather than a scan.
        self._by_doc: dict[str, list[int]] = {}
        for i, doc_id in enumerate(self._doc_ids):
            self._by_doc.setdefault(doc_id, []).append(i)

    @property
    def coverage(self) -> int:
        """How many pages carry a vector. Reported, never assumed."""
        return len(self._page_ids)

    def search(self, query: str, scope: list[str], k: int = 20) -> list[Hit]:
        """Top-k pages by cosine similarity, restricted to `scope`.

        NEVER CORPUS-WIDE. Every retriever is scoped to the routed
        candidates: corpus-wide similarity search is the shared-store
        baseline this system exists to beat, and it measured 5.6%.
        """
        if self._matrix.size == 0 or self._embedder is None or not scope:
            return []

        rows = [i for doc_id in scope for i in self._by_doc.get(doc_id, [])]
        if not rows:
            return []

        try:
            raw = self._embedder.embed([query])
        except Exception:
            # A dead embedding endpoint must cost this ONE ranking, not the
            # question: anchors and BM25 still answer, RRF simply fuses two.
            return []
        if not raw:
            return []

        q = np.asarray(raw[0], dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        q /= norm
        if q.shape[0] != self._matrix.shape[1]:
            # A dimension mismatch means the query model and the stored vectors
            # disagree; the scores would be meaningless rather than merely bad.
            return []

        idx = np.asarray(rows, dtype=np.int64)
        scores = self._matrix[idx] @ q
        top = np.argsort(-scores)[:k]

        hits: list[Hit] = []
        for rank, j in enumerate(top, start=1):
            i = int(idx[int(j)])
            hits.append(
                Hit(
                    page_id=self._page_ids[i],
                    doc_id=self._doc_ids[i],
                    page_seq=self._seqs[i],
                    score=float(scores[int(j)]),
                    source="dense",
                    text=self._texts[i],
                    rank=rank,
                )
            )
        return hits


def _as_vector(value) -> np.ndarray | None:
    """pgvector arrives as a list, an ndarray, or its text form '[1,2,3]'."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value if value.size else None
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32) if len(value) else None
    if isinstance(value, str):
        body = value.strip().strip("[]")
        if not body:
            return None
        try:
            return np.asarray([float(x) for x in body.split(",")], dtype=np.float32)
        except ValueError:
            return None
    return None

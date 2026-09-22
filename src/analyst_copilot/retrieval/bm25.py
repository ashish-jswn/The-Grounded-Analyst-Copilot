"""In-memory BM25 over page `lexical_text`.

WHY IN-MEMORY: Azure PostgreSQL has NO BM25 extension. BM25 is only a scoring
function, so running it in the application costs nothing - MEASURED: the
index over all ~8,400 pages builds in 1.7 s and is rebuilt on ingest.

WHAT IT INDEXES: `lexical_text`, the COMPOSITE field - context header + verbatim
table and section headers + summary. NOT the summary alone. Query rewriting
to exact GAAP names moved R@10 from 16.7% to 37.3%, and that lift needs strings
like "Purchases of property, plant and equipment" to survive indexing; a
free-form summary paraphrases them away.

Scope matters: searching the whole corpus put the gold page in the top 10 only
5.6% of the time, while restricting to the router's candidate filings is what
makes lexical search useful at all. `scope` is therefore not an optimisation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .base import Hit

# Keep digits: fiscal years and figures are discriminative in filings.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


@dataclass
class _Doc:
    page_id: str
    doc_id: str
    page_seq: int
    text: str
    context_header: str


class BM25Index:
    """Corpus-wide index, queried with a document scope."""

    def __init__(self, rows: list[dict]) -> None:
        self._docs: list[_Doc] = [
            _Doc(
                page_id=r["page_id"],
                doc_id=r["doc_id"],
                page_seq=r["page_seq"],
                text=r.get("raw_text") or "",
                context_header=r.get("context_header") or "",
            )
            for r in rows
        ]
        corpus = [tokenize(r.get("lexical_text") or r.get("raw_text") or "") for r in rows]
        # BM25Okapi divides by the average document length, so an all-empty
        # corpus would raise. Guard it: an empty index must return no hits, not
        # crash the query pipeline.
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        self._by_doc: dict[str, list[int]] = {}
        for i, d in enumerate(self._docs):
            self._by_doc.setdefault(d.doc_id, []).append(i)

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def doc_ids(self) -> set[str]:
        return set(self._by_doc)

    def search(
        self,
        query: str,
        scope: list[str] | None,
        k: int,
        *,
        per_document: bool = True,
    ) -> list[Hit]:
        """Retrieve up to `k` pages.

        `per_document=True` allocates k to EACH candidate filing rather than
        across the scope as a whole, and it matters a lot once the router hands
        over 4 candidates.

        MEASURED at router top-4: a single global top-20 lets a strongly
        scoring wrong filing crowd the gold filing out, giving 74.8% recall.
        Allocating 20 per filing gives 80.3% - the plan's recorded 81.0%. The
        cost is more candidate pages, which the assembly stage then trims to the
        token budget; recall lost here cannot be recovered later.
        """
        if self._bm25 is None or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenize(query))

        if scope and per_document:
            hits: list[Hit] = []
            for doc_id in scope:
                hits.extend(self._top(scores, self._by_doc.get(doc_id, []), k))
            hits.sort(key=lambda h: (-h.score, h.page_id))
            for i, h in enumerate(hits, start=1):
                h.rank = i
            return hits

        candidates = (
            [i for d in scope for i in self._by_doc.get(d, [])]
            if scope
            else list(range(len(self._docs)))
        )
        return self._top(scores, candidates, k)

    def _top(self, scores, candidates, k: int) -> list[Hit]:
        ranked = sorted(candidates, key=lambda i: (-scores[i], self._docs[i].page_id))
        hits: list[Hit] = []
        for rank, i in enumerate(ranked[:k], start=1):
            if scores[i] <= 0:
                break
            d = self._docs[i]
            hits.append(
                Hit(
                    page_id=d.page_id,
                    doc_id=d.doc_id,
                    page_seq=d.page_seq,
                    score=float(scores[i]),
                    source="bm25",
                    text=d.text,
                    context_header=d.context_header,
                    rank=rank,
                )
            )
        return hits


def build_index(conn) -> BM25Index:
    """Build from the database. Import kept local so this module has no
    dependency on storage when used with pre-loaded rows in tests."""
    from ..storage import repository as repo

    return BM25Index(repo.pages_for_bm25(conn))

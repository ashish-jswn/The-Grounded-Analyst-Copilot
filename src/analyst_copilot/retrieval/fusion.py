"""Reciprocal Rank Fusion.

ADAPTED FROM `sec-rag-analyst/src/retrieve.py::_rrf` (references_details.md asks
for reuse over reimplementation; this is the ~8-line near-verbatim case).

RRF fuses rankings by POSITION, not by score, which is why it works across
retrievers whose scores are not comparable: a BM25 score and a cosine similarity
live on different scales, and normalising them would need a calibration we do
not have.

    score(d) = sum over retrievers of  1 / (k + rank(d))

`k` (default 60) damps the influence of the top rank so a single retriever
cannot dominate the fusion. It is `retrieval.rrf_k` in config.yaml.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .base import Hit


def rrf(
    rankings: Sequence[Sequence[Hit]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    """Fuse several ranked lists into one.

    Ties are broken deterministically by page_id so an eval run is reproducible;
    an unstable sort here would make two identical runs disagree.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: dict[str, float] = defaultdict(float)
    best: dict[str, Hit] = {}
    sources: dict[str, set[str]] = defaultdict(set)

    for ranking, weight in zip(rankings, weights):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.page_id] += weight / (k + rank)
            sources[hit.page_id].add(hit.source or "?")
            # Keep the richest copy of the row: a hit from one retriever may
            # carry text that another's did not.
            if hit.page_id not in best or (hit.text and not best[hit.page_id].text):
                best[hit.page_id] = hit

    fused: list[Hit] = []
    for page_id, score in scores.items():
        hit = best[page_id]
        fused.append(
            Hit(
                page_id=hit.page_id,
                doc_id=hit.doc_id,
                page_seq=hit.page_seq,
                score=score,
                source="+".join(sorted(sources[page_id])),
                text=hit.text,
                context_header=hit.context_header,
                extra=dict(hit.extra),
            )
        )
    fused.sort(key=lambda h: (-h.score, h.page_id))
    for i, hit in enumerate(fused, start=1):
        hit.rank = i
    return fused


def union_preserving_order(*groups: Iterable[Hit]) -> list[Hit]:
    """Concatenate hit groups, keeping the first occurrence of each page."""
    seen: set[str] = set()
    out: list[Hit] = []
    for group in groups:
        for hit in group:
            if hit.page_id in seen:
                continue
            seen.add(hit.page_id)
            out.append(hit)
    return out

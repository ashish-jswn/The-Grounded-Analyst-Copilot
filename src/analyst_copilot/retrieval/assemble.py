"""Candidate assembly - union, expand, rank, then fit the token budget.

The retrievers are RECALL stages and deliberately over-produce: at router top-4
the candidate pool is ~78 pages against an `assembly_token_budget` of 42k. This
module is what turns that pool into the context the extractor actually reads.

Order matters:
  1. UNION of anchor + BM25 + dense hits         (recall)
  2. NEIGHBOUR EXPANSION by +/-N pages           (recall - a statement spans a seam)
  3. RRF, then optional rerank                   (precision)
  4. TRIM to the token budget, best first        (cost)

Trimming last is deliberate. Recall lost in step 1 cannot be recovered later,
whereas a page dropped in step 4 was ranked worst by every signal we have.

THE EXTRACTOR ONLY EVER SEES `raw_text`. The summary is a retrieval artifact
and is never shown to the generator, because gate G1 checks quotes
against `raw_text` - a quote copied from a paraphrase would fail the gate, or
worse, pass a fabricated figure into an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Hit
from .fusion import rrf, union_preserving_order

# Matches the ingest estimate so the budget means the same thing on both sides.
_CHARS_PER_TOKEN = 4


@dataclass
class AssembledContext:
    hits: list[Hit]
    text: str
    token_estimate: int
    dropped_for_budget: int = 0
    pages_by_doc: dict[str, dict[int, str]] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.hits)


def expand_neighbours(
    hits: list[Hit],
    pages_by_doc: dict[str, dict[int, str]],
    distance: int,
    headers_by_page: dict[str, str] | None = None,
) -> list[Hit]:
    """Add the +/-`distance` pages around every hit.

    A financial statement routinely spans a page seam, and a footnote sits on
    the page after the table it annotates. Neighbours enter with a low score so
    they rank below genuine hits but remain available for quoting.
    """
    if distance <= 0:
        return hits
    headers_by_page = headers_by_page or {}
    seen = {(h.doc_id, h.page_seq) for h in hits}
    extra: list[Hit] = []
    for hit in hits:
        pages = pages_by_doc.get(hit.doc_id, {})
        for offset in range(-distance, distance + 1):
            if offset == 0:
                continue
            seq = hit.page_seq + offset
            if seq not in pages or (hit.doc_id, seq) in seen:
                continue
            seen.add((hit.doc_id, seq))
            page_id = f"{hit.doc_id}#p{seq}"
            extra.append(
                Hit(
                    page_id=page_id,
                    doc_id=hit.doc_id,
                    page_seq=seq,
                    score=hit.score * 0.25,
                    source="neighbour",
                    text=pages[seq],
                    context_header=headers_by_page.get(page_id, ""),
                )
            )
    return hits + extra


def assemble(
    rankings: list[list[Hit]],
    pages_by_doc: dict[str, dict[int, str]],
    *,
    token_budget: int,
    rrf_k: int = 60,
    neighbour_expand: int = 1,
    reranker=None,
    rerank_top_n: int = 15,
    headers_by_page: dict[str, str] | None = None,
) -> AssembledContext:
    """Build the extractor's context from several retriever rankings."""
    pool = union_preserving_order(*rankings)
    pool = expand_neighbours(pool, pages_by_doc, neighbour_expand, headers_by_page)

    # Neighbours were not in any ranking, so they must be fused as their own
    # list or RRF would drop them entirely.
    neighbours = [h for h in pool if h.source == "neighbour"]
    ranked = rrf([*rankings, neighbours] if neighbours else rankings, k=rrf_k)

    # RRF only knows pages that appeared in a ranking; anything else keeps its
    # place at the tail rather than vanishing.
    ranked_ids = {h.page_id for h in ranked}
    ranked += [h for h in pool if h.page_id not in ranked_ids]

    if reranker is not None and ranked:
        head = ranked[: max(rerank_top_n * 3, rerank_top_n)]
        documents = [
            (h.context_header + "\n" + h.text)[:4000] for h in head
        ]
        try:
            order = reranker.rerank_documents(documents, top_n=len(head))
        except Exception:
            # A reranker outage must degrade to RRF, never fail the question.
            order = None
        if order:
            reordered = [head[i] for i, _score in order]
            ranked = reordered + ranked[len(head):]

    kept: list[Hit] = []
    used = 0
    dropped = 0
    for hit in ranked:
        cost = len(hit.text) // _CHARS_PER_TOKEN + 32
        if used + cost > token_budget and kept:
            dropped += 1
            continue
        kept.append(hit)
        used += cost

    # Present in document order: a filing reads top-to-bottom, and adjacent
    # pages next to each other let the extractor follow a statement across a
    # seam instead of seeing it twice out of order.
    kept.sort(key=lambda h: (h.doc_id, h.page_seq))

    blocks = []
    texts: dict[str, dict[int, str]] = {}
    for hit in kept:
        # THE LOCATOR MUST BE MACHINE-READABLE AND EXACT.
        # MEASURED FAILURE: the human-facing header reads
        # "[3M | 10-K | FY2018 | ... | p.47 (printed 46)]", which never contains
        # the literal doc_id "3M_2018_10K". The extractor had to INVENT the
        # doc_id and page_seq it was required to return, and it duly cited
        # page 62 for a quote that lives on page 61 - gate G1 caught it and the
        # system abstained on an answerable question.
        # Emitting the exact strings the extractor must echo back removes the
        # guess entirely.
        locator = f"doc_id={hit.doc_id} page_seq={hit.page_seq}"
        header = hit.context_header or f"[{hit.doc_id} | p.{hit.page_seq}]"
        blocks.append(f"<<< {locator} >>>\n{header}\n{hit.text}")
        texts.setdefault(hit.doc_id, {})[hit.page_seq] = hit.text

    return AssembledContext(
        hits=kept,
        text="\n\n".join(blocks),
        token_estimate=used,
        dropped_for_budget=dropped,
        pages_by_doc=texts,
    )

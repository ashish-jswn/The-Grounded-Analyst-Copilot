"""Retrieval interfaces.

Protocols rather than base classes, so any stage is swappable and mockable
without inheritance - which is what lets `eval/ablate.py` disable one retriever
and measure its contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Hit:
    """One retrieved page. The page is the retrieval unit AND the citation unit."""

    page_id: str
    doc_id: str
    page_seq: int
    score: float
    source: str = ""              # bm25 | dense | anchor | rerank | neighbour
    text: str = ""
    context_header: str = ""
    rank: int = 0
    extra: dict[str, object] = field(default_factory=dict)


class Retriever(Protocol):
    def search(self, query: str, scope: list[str], k: int) -> list[Hit]: ...

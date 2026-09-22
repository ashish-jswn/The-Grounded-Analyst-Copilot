"""Derived identifiers - ONE definition, shared by ingest and storage.

Ids are derived from position, never random, which buys three things:

  * re-ingesting a filing produces the same ids, so a re-upload overwrites
    rather than duplicating (and a failed ingest is safe to retry)
  * a page id reads as a citation on its own: '3M_2018_10K#p59'
  * a stale row cannot survive a re-ingest under a different id

Kept out of both `ingest/` and `storage/` so neither has to import the other
just to name a row.
"""

from __future__ import annotations


def page_id(doc_id: str, page_seq: int) -> str:
    return f"{doc_id}#p{page_seq}"


def block_id(doc_id: str, page_seq: int, order_idx: int) -> str:
    return f"{doc_id}#p{page_seq}.b{order_idx}"


def table_id(doc_id: str, page_seq: int, order_idx: int) -> str:
    return f"{doc_id}#p{page_seq}.t{order_idx}"


def cell_id(table_id: str, row_idx: int, col_idx: int) -> str:
    return f"{table_id}.r{row_idx}c{col_idx}"


def section_id(doc_id: str, ordinal: int) -> str:
    return f"{doc_id}#s{ordinal}"


def fact_id(doc_id: str, index: int) -> str:
    return f"{doc_id}#f{index}"


def parse_page_id(page_id: str) -> tuple[str, int]:
    """Inverse of `page()`. Used by citation rendering and gate G7."""
    doc_id, _, tail = page_id.partition("#p")
    return doc_id, int(tail)

"""Block extraction - the children of a page.

Blocks are used for evidence extraction, verbatim quoting and neighbour
expansion. They are NEVER the primary ranking unit: the page is, because the
rubric grades location and the page is what a human verifies.

Blocks are linked prev/next in document order so the extractor can widen around
a hit without re-parsing, and each carries its character span within the page so
a quote can be traced back to its exact position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lxml import etree

HEADING, PARAGRAPH, LIST, FOOTNOTE, CAPTION, TABLE = (
    "heading", "paragraph", "list", "footnote", "caption", "table"
)

_BLOCK_LEVEL = {"p", "div", "li", "td", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

# A footnote reference marker: "(1)", "(a)", or a lone superscript digit at the
# start of a short line.
_FOOTNOTE = re.compile(r"^\(?[0-9a-z]\)\s+\S|^\*\s+\S", re.I)

# SEC filings have almost no <h1>-<h6>: only 6 of 78 filings contain ANY, median
# 1 tag. So a heading is recognised by shape and styling, never by tag name.
_HEADING_SHAPE = re.compile(
    r"^(item\s+\d+[a-z]?\.?|part\s+[ivx]+\.?|note\s+\d+|"
    r"(consolidated|condensed)\s+(statements?|balance))",
    re.I,
)
_BOLD_STYLE = re.compile(r"font-weight\s*:\s*(bold|[6-9]00)", re.I)


@dataclass
class Block:
    order_idx: int
    block_type: str
    text: str
    char_start: int
    char_end: int
    table_order_idx: int | None = None


def _text_of(el: etree._Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).replace("\xa0", " ").strip()


def _is_heading(el: etree._Element, text: str) -> bool:
    if len(text) > 200 or not text:
        return False
    tag = el.tag.lower() if isinstance(el.tag, str) else ""
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return True
    if _HEADING_SHAPE.match(text):
        return True
    # Visual styling is the fallback signal, since semantic tags are absent.
    style = el.get("style") or ""
    if _BOLD_STYLE.search(style) and len(text) < 120:
        return True
    if text.isupper() and 3 < len(text) < 120:
        return True
    return False


def extract_blocks(
    elements: list[etree._Element], page_text: str
) -> list[Block]:
    """Extract typed blocks from the elements composing one page.

    `elements` comes from `ParsedDocument.elements_for(page)`, so blocks and
    pages are produced by the same walk and cannot disagree about boundaries.
    """
    blocks: list[Block] = []
    seen: set[str] = set()
    cursor = 0
    table_ordinal = 0

    for el in elements:
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        if tag not in _BLOCK_LEVEL:
            continue

        if tag == "table":
            text = _text_of(el)
            if not text:
                continue
            start = page_text.find(text[:60], cursor) if len(text) > 60 else -1
            blocks.append(
                Block(
                    order_idx=len(blocks),
                    block_type=TABLE,
                    text=text[:4000],
                    char_start=max(start, 0),
                    char_end=max(start, 0) + len(text),
                    table_order_idx=table_ordinal,
                )
            )
            table_ordinal += 1
            continue

        # A <div> wrapping a <p> would otherwise emit the same text twice.
        if any(
            isinstance(child.tag, str) and child.tag.lower() in _BLOCK_LEVEL
            for child in el
        ):
            continue

        text = _text_of(el)
        if not text:
            continue
        key = f"{tag}:{text}"
        if key in seen:
            continue
        seen.add(key)

        if _is_heading(el, text):
            block_type = HEADING
        elif tag == "li":
            block_type = LIST
        elif _FOOTNOTE.match(text) and len(text) < 600:
            block_type = FOOTNOTE
        else:
            block_type = PARAGRAPH

        pos = page_text.find(text[:60], cursor)
        if pos >= 0:
            cursor = pos
        blocks.append(
            Block(
                order_idx=len(blocks),
                block_type=block_type,
                text=text,
                char_start=max(pos, 0),
                char_end=max(pos, 0) + len(text),
            )
        )
    return blocks


def nearest_heading_before(blocks: list[Block], order_idx: int) -> str | None:
    """The caption for a table is the nearest preceding heading (step 7)."""
    for b in reversed(blocks[:order_idx]):
        if b.block_type == HEADING:
            return b.text
    return None

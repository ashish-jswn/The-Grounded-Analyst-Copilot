"""EDGAR full-submission envelope handling.

WHY THIS EXISTS - a measured correctness bug, found on 2026-08-31.

Three of the 78 filings are not single HTML documents at all. They are EDGAR
*full submissions*: an SGML envelope wrapping several concatenated `<DOCUMENT>`
sections, each holding its own complete `<html>` document.

    AMCOR_2022_8K_dated-2022-07-01        5 embedded <html> blocks
    FOOTLOCKER_2022_8K_dated-2022-05-20   3 embedded <html> blocks
    FOOTLOCKER_2022_8K_dated_2022-08-19   6 embedded <html> blocks

`lxml.html.document_fromstring` parses only the FIRST `<html>` and discards the
rest, so for these three filings we were extracting the EDGAR submission-index
wrapper - about 3.5k characters of navigation chrome - and none of the actual
filing. AMCOR's 8-K body sits at byte 82,776 of 142,400 and never reached the
index at all.

This is also the real explanation for what looked like a separate fact: these
same three filings were the ONLY ones reported as having zero page-break
markers. They have no markers because we were never looking at their content.

The gold evidence for several 8-K questions lives in an EX-99.1 news release,
not in the 8-K body, so exhibits are kept rather than discarded.
"""

from __future__ import annotations

import re

# One complete embedded document. Non-greedy so each <html>...</html> is its own
# block; EDGAR never nests one HTML document inside another.
_HTML_BLOCK_RE = re.compile(r"<html[^>]*>.*?</html>", re.I | re.S)
_TYPE_RE = re.compile(r"<TYPE>([^\r\n<]+)", re.I)

# Document types that carry no readable filing text. EX-101.* is the XBRL
# instance/schema set, and TYPE XML is the inline-XBRL viewer scaffolding -
# both are machine artifacts that would pollute the lexical index with
# thousands of context ids and namespace tokens.
_SKIP_TYPES = re.compile(
    r"^(XML|EX-101(\.\w+)?|GRAPHIC|ZIP|EXCEL|JSON|EX-27(\.\w+)?)$", re.I
)


def is_full_submission(raw: str) -> bool:
    """True when the file is an EDGAR envelope rather than a single document."""
    if re.search(r"<SEC-DOCUMENT>", raw, re.I):
        return True
    return len(_HTML_BLOCK_RE.findall(raw)) > 1


def iter_documents(raw: str) -> list[tuple[str, str]]:
    """Split a filing into the (doc_type, html) documents worth reading.

    A single-document filing - 75 of 78 - is returned unchanged as one entry,
    so this is safe to run unconditionally on every filing.

    The first embedded block of a full submission is the EDGAR submission-index
    page: it precedes any `<TYPE>` declaration and is pure navigation chrome, so
    it is dropped. Every remaining block is attributed to the nearest preceding
    `<TYPE>`, and machine-only document types are skipped.
    """
    if not is_full_submission(raw):
        return [("primary", raw)]

    out: list[tuple[str, str]] = []
    for m in _HTML_BLOCK_RE.finditer(raw):
        types = _TYPE_RE.findall(raw[: m.start()])
        doc_type = types[-1].strip() if types else ""
        if not doc_type:
            # The submission-index wrapper, before any <TYPE>. Never content.
            continue
        if _SKIP_TYPES.match(doc_type):
            continue
        out.append((doc_type, m.group(0)))

    # Never return nothing: if every block was filtered, fall back to the raw
    # file so the filing is still ingestable rather than silently empty.
    return out or [("primary", raw)]

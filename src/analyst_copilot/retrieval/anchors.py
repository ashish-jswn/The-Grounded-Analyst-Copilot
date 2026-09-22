"""Structure anchors - the cheapest strong retriever in the system.

MEASURED: 73.0% gold-page recall from these regexes ALONE, with no LLM, no
embedding and no BM25, selecting ~27 pages. Combined with BM25@20 it reaches
85.7%. Roughly 75% of gold evidence sits in the three primary financial
statements, which is why a structural signal beats similarity search here by 3x.

THE HEADLINE: retrieval in filings is NAVIGATION, not similarity search.

The question->statement hint is a general mapping from the vocabulary of
financial analysis to the statement that reports it. It carries no knowledge of
any specific filing or question, so it applies unchanged to an unseen upload
.
"""

from __future__ import annotations

import re

from ..ingest.sections import STMT_PATTERNS
from .base import Hit

# Question vocabulary -> the statement that reports it. General finance
# knowledge: capex is a cash-flow line, inventory a balance-sheet line.
_QUESTION_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(capex|capital expenditure|cash flow|dividend|share repurchase|"
                r"buyback|financing activities|investing activities|depreciation|"
                r"amortization|free cash flow)\b", re.I), "cashflow"),
    (re.compile(r"\b(total assets|current assets|current liabilities|inventor(y|ies)|"
                r"receivable|payable|goodwill|debt|equity|working capital|"
                r"balance sheet|liquidity|quick ratio|current ratio)\b", re.I), "balance"),
    (re.compile(r"\b(revenue|sales|gross (profit|margin)|operating (income|margin)|"
                r"net income|earnings|eps|cost of (goods|revenue)|tax rate|"
                r"income statement|profit)\b", re.I), "income"),
    (re.compile(r"\bsegment\b|\bbusiness unit\b|\bwhich division\b", re.I), "segment"),
    # WIDENED FROM A MEASURED FAILURE. financebench_id_01865 asks "excluding
    # the impact of M&A, which segment dragged down growth?" — the gold answer
    # is "consumer shrunk 0.9% ORGANICALLY", which lives in MD&A prose. The old
    # vocabulary hinted only `segment`, so retrieval handed the composer the
    # segment operating-income TABLE and it answered from that. Growth,
    # organic/constant-currency and M&A-exclusion language are MD&A vocabulary,
    # not statement vocabulary.
    (re.compile(r"\b(why|drove|driver|dragged?|trend|outlook|guidance|forecast|"
                r"expects?|organic(ally)?|constant currency|like.for.like|"
                r"results of operations|management.s discussion)\b"
                r"|excluding (the )?(impact of )?(m&a|acquisitions?|divestitures?)",
                re.I), "mdna"),
    (re.compile(r"\bcomprehensive income\b", re.I), "compinc"),
    (re.compile(r"\b(stockholders|shareholders).{0,3} equity\b", re.I), "equity"),
]


# Pseudo-bucket for MD&A body pages. Deliberately not a STMT_PATTERNS key.
_MDNA_SPAN = "mdna_span"

_WORD = re.compile(r"[a-z][a-z0-9']+")
# Enough to stop the ranking being decided by "what" and "the". Not a stemmer
# and not an IDF model - the bucket is ~30 pages of one document, so the
# cheapest signal that separates them is sufficient.
_STOP = frozenset(
    "the a an and or of in on for to from by with as at is are was were be been "
    "what which how much many did does do this that these those it its their "
    "there has have had will would can could not no than then thus also".split()
)


def statement_hints(question: str) -> list[str]:
    """Which statements a question is likely answered from, most likely first."""
    return [stmt for pattern, stmt in _QUESTION_HINTS if pattern.search(question)]


class AnchorRetriever:
    """Selects pages whose head matches a financial-statement title.

    Built from the `sections` table when available, and otherwise directly from
    page text, so it works on a filing whose tree is empty (an 8-K).
    """

    def __init__(
        self,
        pages_by_doc: dict[str, list[tuple[int, str, str]]],
        narrative_spans: list[dict] | None = None,
    ) -> None:
        """pages_by_doc: doc_id -> [(page_seq, page_id, raw_text)]

        `narrative_spans` carries (doc_id, kind, raw_title, page_start,
        page_end) for MD&A-like sections, so those anchor to their WHOLE SPAN
        rather than to their title page. See `_add_narrative_spans`.
        """
        self._pages = pages_by_doc
        self._index: dict[str, dict[str, list[tuple[int, str]]]] = {}
        # Narrative body pages, held apart from `_index` - see _add_narrative_spans.
        self._spans: dict[str, list[tuple[int, str]]] = {}
        for doc_id, pages in pages_by_doc.items():
            by_stmt: dict[str, list[tuple[int, str]]] = {}
            seqs = {seq for seq, _pid, _t in pages}
            page_id_of = {seq: pid for seq, pid, _t in pages}
            for seq, pid, text in pages:
                head = text[:600]
                for stmt, pattern in STMT_PATTERNS.items():
                    if pattern.search(head):
                        by_stmt.setdefault(stmt, []).append((seq, pid))
                        # A statement routinely spans a page seam, so the
                        # FOLLOWING page is part of the same anchor.
                        if seq + 1 in seqs:
                            by_stmt[stmt].append((seq + 1, page_id_of[seq + 1]))
            self._index[doc_id] = {
                k: sorted(set(v)) for k, v in by_stmt.items()
            }
        self._add_narrative_spans(narrative_spans or [], pages_by_doc)

    # MD&A IS NOT A TWO-PAGE STATEMENT, AND THE ANCHOR RULE ASSUMED IT WAS.
    # "The page whose head carries the title, plus the next one" is right for a
    # financial statement, which runs 2-3 pages and repeats its title at the
    # top of each. Item 7 runs ~30 pages and names itself only once, so the
    # rule reached the overview and nothing else.
    #
    # MEASURED on a 131-page 10-K: the MD&A anchor covered 5 pages. The word
    # "organic" - the vocabulary a growth question is answered in - appeared on
    # 12 pages, of which only 2 were reachable by ANY anchor. Spanning the
    # section makes the other 10 retrievable.
    def _add_narrative_spans(self, spans, pages_by_doc) -> None:
        """Anchor a narrative section to its whole page span.

        Statements are deliberately NOT spanned - see the note above.
        """
        for span in spans:
            # A statement is never spanned whatever its title says: the
            # title+next rule already covers its 2-3 pages exactly.
            if span.get("kind") == "statement":
                continue
            # Item 7 lands under kind `item` in some filings and `mdna` in
            # others - the tree classifies by first matching signal - so the
            # title is checked too, and anchoring does not depend on which.
            title = (span.get("raw_title") or "").lower()
            if span.get("kind") != "mdna" and "discussion and analysis" not in title:
                continue
            doc_id = span["doc_id"]
            page_ids = {seq: pid for seq, pid, _t in pages_by_doc.get(doc_id, [])}
            start = span["page_start"]
            end = span.get("page_end") or start
            # A guard against a mis-detected heading swallowing the filing: an
            # "MD&A" that appears to run more than 60 pages is a tree defect,
            # and spanning it would drown every other anchor.
            if end - start > 60:
                end = start + 60
            # A SEPARATE BUCKET, NOT MERGED INTO `mdna`, AND THE
            # DIFFERENCE WAS MEASURED. Merging them cost 1.6 points of
            # router-top-4 recall: `search` keeps 40 pages, "hints rank, they
            # do not filter" puts EVERY bucket in the running, and 26 MD&A
            # body pages per document then displaced statement pages on
            # questions that had nothing to do with MD&A. Span pages earn
            # their slots only when the question is narrative, so they are
            # held aside and folded in by `search` only when `mdna` is hinted.
            bucket = self._spans.setdefault(doc_id, [])
            bucket.extend(
                (seq, page_ids[seq]) for seq in range(start, end + 1) if seq in page_ids
            )
            self._spans[doc_id] = sorted(set(bucket))

    # PAGE ORDER IS NOT RELEVANCE ORDER, AND FOR A 26-PAGE BUCKET THAT
    # DECIDES EVERYTHING. Every page in a bucket carries the same anchor score,
    # so ties fall through to `page_seq`. For a statement that is harmless -
    # the bucket is 2-3 pages and all of them get through. For an MD&A body it
    # meant the whole discussion was ordered by page number, so the pages that
    # actually answer the question sat behind every page that preceded them.
    #
    # MEASURED: with the body ordered by page number, span expansion raised
    # whole-hit-set recall but moved recall AT THE RANK CUT by 0.0 points -
    # the pages were retrieved and then never seen.
    def _rank_span(self, doc_id: str, query: str) -> list[tuple[int, str]]:
        """MD&A body pages, most lexically similar to the question first."""
        terms = {t for t in _WORD.findall(query.lower()) if t not in _STOP}
        if not terms:
            return self._spans.get(doc_id, [])
        text_of = {seq: t.lower() for seq, _p, t in self._pages.get(doc_id, [])}
        scored = []
        for seq, pid in self._spans.get(doc_id, []):
            body = text_of.get(seq, "")
            # Distinct terms present, not raw frequency: a page that mentions
            # every term of the question once is a better answer than one that
            # repeats a single common term forty times.
            scored.append((sum(t in body for t in terms), seq, pid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(seq, pid) for _score, seq, pid in scored]

    def search(self, query: str, scope: list[str], k: int = 40) -> list[Hit]:
        # HINTS RANK, THEY DO NOT FILTER. Anchors are a RECALL stage and the
        # precision gate is verification, so discarding a statement because the
        # question did not name its vocabulary is a pure loss.
        #
        # MEASURED: filtering to hinted statements gave 68.5% recall from
        # 15.5 pages; keeping every statement page and merely ORDERING by hint
        # gives 73.2% from 26.6 pages - which is the plan's recorded
        # "73.0% recall, ~27 pages". Eleven extra pages is a trivial slice of a
        # 42k-token budget; 4.7 points of recall is not recoverable later.
        hints = statement_hints(query)
        wanted = hints + [s for s in STMT_PATTERNS if s not in hints]
        # The MD&A body ranks directly behind the hinted buckets, and only for
        # a narrative question. `_MDNA_SPAN` is not a statement pattern, so it
        # never enters `wanted` by the ordinary path.
        if "mdna" in hints:
            wanted.insert(len(hints), _MDNA_SPAN)

        hits: list[Hit] = []
        rank = 0
        for stmt in wanted:
            for doc_id in scope:
                pages = (
                    self._rank_span(doc_id, query)
                    if stmt == _MDNA_SPAN
                    else self._index.get(doc_id, {}).get(stmt, [])
                )
                for seq, pid in pages:
                    rank += 1
                    text = next(
                        (t for s, p, t in self._pages.get(doc_id, []) if s == seq), ""
                    )
                    hits.append(
                        Hit(
                            page_id=pid,
                            doc_id=doc_id,
                            page_seq=seq,
                            # Earlier hints are stronger; decay with position.
                            score=1.0 / (1 + wanted.index(stmt)),
                            source="anchor",
                            text=text,
                            rank=rank,
                            extra={"stmt_type": stmt},
                        )
                    )
        # De-duplicate, keeping the strongest hint for each page.
        best: dict[str, Hit] = {}
        for h in hits:
            if h.page_id not in best or h.score > best[h.page_id].score:
                best[h.page_id] = h
        # TIES BREAK BY ROUTER RANK, NOT ALPHABETICALLY. `k` is a budget over
        # the WHOLE scope, and under router top-4 four documents compete for
        # it. Breaking ties on `doc_id` sorted the candidates by company name
        # and threw the router's confidence away, so pages from the document
        # the router ranked 4th displaced pages from the one it ranked 1st
        # whenever the name sorted earlier.
        order = {doc_id: i for i, doc_id in enumerate(scope)}
        out = sorted(
            best.values(),
            key=lambda h: (-h.score, order.get(h.doc_id, len(order)), h.page_seq),
        )
        for i, h in enumerate(out[:k], start=1):
            h.rank = i
        return out[:k]


def build_from_rows(
    rows: list[dict], narrative_spans: list[dict] | None = None
) -> AnchorRetriever:
    pages_by_doc: dict[str, list[tuple[int, str, str]]] = {}
    for r in rows:
        pages_by_doc.setdefault(r["doc_id"], []).append(
            (r["page_seq"], r["page_id"], r.get("raw_text") or "")
        )
    return AnchorRetriever(pages_by_doc, narrative_spans)

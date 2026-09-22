"""Gold loading and gold-page mapping.

DO NOT USE FinanceBench's `evidence_page_num`. It indexes a THIRD-PARTY PDF
rendering that we do not have and could never reproduce for a newly uploaded
filing. Location is scored by EVIDENCE-TEXT OVERLAP against our own derived
pages, which is reproducible for any document the product ingests.

MEASURED 2026-08-31: 127/136 questions map at Jaccard >= 0.35. The questions
that do NOT map are reported, never silently dropped - several of them are not
mapping failures at all but corpus defects, and scoring them as system errors
would misdirect the whole build. See `UNANSWERABLE`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .shapes import AnswerShape, classify_answer

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


# ---------------------------------------------------------------------------
# Corpus defects - measured, and NOT system failures
# ---------------------------------------------------------------------------
# The gold evidence for these questions is not present in the supplied filing at
# all, so the best achievable outcome is an honest abstention (score 0). They are
# excluded from the "answerable" denominator and reported separately.
#
# This CONTRADICTS the recorded fact that 0/136 practice questions are
# unanswerable - they are unanswerable for a different reason than the benchmark
# intended, and they are the only natural abstention signal the dev set contains.
UNANSWERABLE: dict[str, str] = {
    # The supplied file is the JANUARY earnings 8-K (XBRL cover date 2023-01-24),
    # not the August Kenvue-separation 8-K its filename claims, and the
    # Exhibit 99.1 the gold text comes from is not included in it.
    "financebench_id_01488": "J&J 8-K: wrong filing supplied; Exhibit 99.1 absent",
    "financebench_id_01490": "J&J 8-K: wrong filing supplied; Exhibit 99.1 absent",
    "financebench_id_01491": "J&J 8-K: wrong filing supplied; Exhibit 99.1 absent",
    # Cover date 2023-02-09; the question asks about the May 3 AGM.
    "financebench_id_01482": "PepsiCo 8-K: wrong filing supplied; exhibit absent",
    "financebench_id_00705": "PepsiCo 8-K: wrong filing supplied; exhibit absent",
    "financebench_id_00882": "PepsiCo 8-K: wrong filing supplied; exhibit absent",
    # The income-statement figures (183,910 / 194,579) do not appear anywhere in
    # the supplied HTML, in any formatting.
    "financebench_id_05915": "CVS 10-K: income-statement figures absent from the HTML",
}


@dataclass
class GoldQuestion:
    qid: str
    question: str
    answer: str
    justification: str
    doc_name: str
    company: str
    question_type: str
    shape: AnswerShape
    evidence_texts: list[str] = field(default_factory=list)
    evidence_full_pages: list[str] = field(default_factory=list)
    # Derived page_seq values that carry the gold evidence, once mapped.
    gold_page_seqs: list[int] = field(default_factory=list)
    mapping_score: float = 0.0

    @property
    def is_unanswerable(self) -> bool:
        return self.qid in UNANSWERABLE

    @property
    def unanswerable_reason(self) -> str | None:
        return UNANSWERABLE.get(self.qid)


def load_questions(path: Path) -> list[GoldQuestion]:
    out: list[GoldQuestion] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            evidence = row.get("evidence") or []
            out.append(
                GoldQuestion(
                    qid=row["financebench_id"],
                    question=row["question"],
                    answer=str(row.get("answer") or ""),
                    justification=str(row.get("justification") or ""),
                    doc_name=row["doc_name"],
                    company=row.get("company") or "",
                    question_type=row.get("question_type") or "",
                    shape=classify_answer(str(row.get("answer") or "")),
                    evidence_texts=[e.get("evidence_text", "") for e in evidence],
                    evidence_full_pages=[
                        e.get("evidence_text_full_page") or e.get("evidence_text", "")
                        for e in evidence
                    ],
                )
            )
    return out


def map_gold_pages(
    question: GoldQuestion,
    pages: list[tuple[int, str]],
    min_jaccard: float = 0.35,
) -> tuple[list[int], float]:
    """Map each gold evidence passage onto our derived pages.

    Jaccard rather than containment, because containment alone rewards a page
    that merely happens to be long. A question may legitimately have more than
    one gold page (income statement plus balance sheet), and every one of them
    counts as a correct location.
    """
    matched: list[int] = []
    best_overall = 0.0
    for gold_text in question.evidence_full_pages:
        gold = tokens(gold_text)
        if not gold:
            continue
        best = (0.0, None)
        for seq, text in pages:
            page = tokens(text)
            if not page:
                continue
            j = len(gold & page) / len(gold | page)
            if j > best[0]:
                best = (j, seq)
        best_overall = max(best_overall, best[0])
        if best[0] >= min_jaccard and best[1] is not None:
            matched.append(best[1])
    return sorted(set(matched)), best_overall

#!/usr/bin/env python
"""Score the real pipeline against the practice questions.

    python scripts/run_pipeline_eval.py --limit 20
    python scripts/run_pipeline_eval.py --split blind --workers 6

THE NUMBER THIS PRINTS IS THE BASELINE EVERY LATER CHANGE IS MEASURED
AGAINST. Without it, later changes are unmeasured guesses.

The 7 corpus-defect questions are excluded from the denominator - their
evidence is not in the supplied filing, so abstaining is the best possible
outcome and scoring them as failures would misdirect the build.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyst_copilot.config import load_settings                        # noqa: E402
from analyst_copilot.container import build_pipeline, load_corpus       # noqa: E402
from analyst_copilot.eval.gold import load_questions, map_gold_pages    # noqa: E402
from analyst_copilot.eval.judge import build_judge                      # noqa: E402
from analyst_copilot.eval.report import EvalReport, split_by_company    # noqa: E402
from analyst_copilot.eval.scorer import (                               # noqa: E402
    Citation, RubricScorer, SystemAnswer,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", choices=["all", "dev", "blind"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=".cache/pipeline_eval.json")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    corpus = load_corpus(settings)
    pipeline = build_pipeline(settings, corpus)
    print(f"corpus: {corpus.n_docs} filings, {corpus.n_pages} pages")

    pages_by_doc = {
        doc: sorted(pages.items()) for doc, pages in corpus.pages_by_doc.items()
    }

    questions = load_questions(settings.practice_questions)
    defects = [q for q in questions if q.is_unanswerable]
    questions = [q for q in questions if not q.is_unanswerable]

    for q in questions:
        q.gold_page_seqs, q.mapping_score = map_gold_pages(
            q, pages_by_doc.get(q.doc_name, []), settings.eval.gold_map_min_jaccard
        )

    if args.split != "all":
        dev, blind = split_by_company(questions)
        questions = dev if args.split == "dev" else blind
    if args.limit:
        questions = questions[: args.limit]

    scorer = RubricScorer(
        numeric_tolerance=settings.eval.numeric_tolerance,
        location_containment=settings.eval.location_containment,
        page_seq_slack=settings.eval.page_seq_slack,
        # Without a judge the 84 non-numeric answers all score wrong, which
        # would send the build chasing the 52 numeric ones.
        judge=None if args.no_judge else build_judge(settings),
    )
    report = EvalReport()
    report.unanswerable = [(q.qid, q.unanswerable_reason or "") for q in defects]

    def run(question):
        started = time.time()
        try:
            result = pipeline.answer(question.question)
        except Exception as exc:                       # never lose a row
            return question, SystemAnswer(abstained=True), f"error:{exc}", 0.0
        answer = SystemAnswer(
            text=result.answer or "",
            abstained=result.status == "abstained",
            clarified=result.status == "clarify",
            citations=[
                Citation(doc_id=c.doc_id, page_seq=c.page_seq, quote=c.quote)
                for c in result.citations
            ],
        )
        # The failing gate's detail carries the offending quote, which is what
        # makes an abstention diagnosable instead of merely counted.
        detail = ""
        for tier in (1, 2):
            for g in result.trace.get(f"tier{tier}_gates") or []:
                if not g["passed"]:
                    detail = f"tier{tier} {g['gate']}: {g['detail']}"
        answer.trace_detail = detail
        answer.errors = result.trace.get("errors") or []
        return question, answer, result.abstain_reason, time.time() - started

    rows = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (question, answer, reason, secs) in enumerate(
            pool.map(run, questions), start=1
        ):
            pages = dict(pages_by_doc.get(question.doc_name, []))
            scored = scorer.score(question, answer, pages)
            report.add(question, scored)
            rows.append(
                {
                    "qid": question.qid,
                    "shape": question.shape.value,
                    "score": scored.score,
                    "abstained": scored.abstained,
                    "abstain_reason": reason,
                    "gold": question.answer,
                    "got": answer.text,
                    "seconds": round(secs, 1),
                    "detail": getattr(answer, "trace_detail", ""),
                    "errors": getattr(answer, "errors", []),
                }
            )
            print(
                f"[{i}/{len(questions)}] {question.qid} {question.shape.value:<14} "
                f"score={scored.score:+d} {reason or ''} ({secs:.0f}s)",
                flush=True,
            )

    print("\n" + report.render())
    print(f"\nwall clock {time.time() - started:.0f}s over {args.workers} workers")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"per-question detail -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

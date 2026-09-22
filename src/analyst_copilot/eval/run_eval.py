"""Eval runner.

THIS EXISTS BEFORE ANY ANSWERING CODE, ON PURPOSE: a harness written
afterwards gets shaped to flatter the system.

Two modes:

  --baseline abstain   every question is refused. This is the zero-competence
                       floor and it MUST score exactly 0.000 with a 0% false
                       answer rate. It proves the harness is wired correctly
                       before any pipeline exists to flatter.

  --baseline oracle    the gold page is cited and the gold answer returned. This
                       is the ceiling the retrieval stack is aiming at, and it
                       verifies that the location predicate can actually be
                       satisfied by our derived pages.

Usage:
    python -m analyst_copilot.eval.run_eval --baseline abstain
    python -m analyst_copilot.eval.run_eval --baseline oracle --split dev
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from ..config import load_settings
from ..ingest.pages import split_pages
from .gold import GoldQuestion, load_questions, map_gold_pages
from .report import EvalReport, split_by_company
from .scorer import Citation, RubricScorer, SystemAnswer


def build_page_index(settings, cache: Path | None = None) -> dict[str, list[tuple[int, str]]]:
    """Split every filing into pages, with an on-disk cache.

    Re-splitting the corpus costs ~26 s, which is fine for ingest but painful in
    an eval loop that runs many times a day.
    """
    if cache and cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    index: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(settings.filings_dir.glob("*.htm")):
        pages = split_pages(path.read_bytes(), settings.ingest.min_page_chars)
        index[path.stem] = [(p.page_seq, p.raw_text) for p in pages]

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(index, fh)
    return index


def attach_gold_pages(
    questions: list[GoldQuestion],
    index: dict[str, list[tuple[int, str]]],
    min_jaccard: float,
) -> list[tuple[str, str, float]]:
    """Map gold evidence onto derived pages. Returns the UNMAPPED list."""
    unmapped: list[tuple[str, str, float]] = []
    for q in questions:
        pages = index.get(q.doc_name, [])
        seqs, best = map_gold_pages(q, pages, min_jaccard)
        q.gold_page_seqs, q.mapping_score = seqs, best
        if not seqs:
            unmapped.append((q.qid, q.doc_name, best))
    return unmapped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rubric eval harness")
    parser.add_argument(
        "--baseline",
        choices=["abstain", "oracle"],
        default="abstain",
        help="which stand-in system to score (no pipeline exists yet)",
    )
    parser.add_argument("--split", choices=["all", "dev", "blind"], default="all")
    parser.add_argument("--cache", default=".cache/pages.pkl")
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="score the 7 corpus-defect questions instead of excluding them",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    questions = load_questions(settings.practice_questions)
    index = build_page_index(settings, Path(args.cache) if args.cache else None)
    unmapped = attach_gold_pages(questions, index, settings.eval.gold_map_min_jaccard)

    report = EvalReport(unmapped=unmapped)

    if not args.include_unanswerable:
        defects = [q for q in questions if q.is_unanswerable]
        report.unanswerable = [(q.qid, q.unanswerable_reason or "") for q in defects]
        questions = [q for q in questions if not q.is_unanswerable]

    if args.split != "all":
        dev, blind = split_by_company(questions)
        questions = dev if args.split == "dev" else blind

    scorer = RubricScorer(
        numeric_tolerance=settings.eval.numeric_tolerance,
        location_containment=settings.eval.location_containment,
        page_seq_slack=settings.eval.page_seq_slack,
        judge=None,  # no LLM judge yet; free-text shapes score as incorrect
    )

    for q in questions:
        pages_by_seq = dict(index.get(q.doc_name, []))
        if args.baseline == "abstain":
            answer = SystemAnswer(text=settings.verification.abstain_string, abstained=True)
        else:
            answer = SystemAnswer(
                text=q.answer,
                citations=[
                    Citation(doc_id=q.doc_name, page_seq=seq) for seq in q.gold_page_seqs
                ],
            )
        report.add(q, scorer.score(q, answer, pages_by_seq))

    print(report.render())
    print()
    print(f"baseline={args.baseline}  split={args.split}  scored={report.overall.n}")

    if args.baseline == "abstain" and report.overall.total != 0:
        print("HARNESS BUG: the abstain baseline must score exactly 0", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

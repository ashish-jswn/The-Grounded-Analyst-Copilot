#!/usr/bin/env python
"""Measure gold-page recall of the retrieval stack.

Reproduces the numbers the plan was built on, from the real database rather
than from a notebook:

    anchors alone,   oracle document   ->  73.0%
    anchors + BM25@20, oracle document ->  85.7%
    anchors + BM25@20, router top-4    ->  81.0%

Recall is measured at the PAGE level, because the page is both the retrieval
unit and the citation unit. The 7 corpus-defect questions are excluded
from the denominator: their evidence is not in the supplied filing, so no
retriever could ever find it and counting them would understate the stack.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyst_copilot.config import load_settings                       # noqa: E402
from analyst_copilot.eval.gold import load_questions, map_gold_pages   # noqa: E402
from analyst_copilot.ingest.catalog import build_catalog               # noqa: E402
from analyst_copilot.query.router import DocumentRouter, load_aliases  # noqa: E402
from analyst_copilot.retrieval.anchors import build_from_rows          # noqa: E402
from analyst_copilot.retrieval.bm25 import BM25Index                   # noqa: E402
from analyst_copilot.retrieval.dense import DenseRetriever            # noqa: E402
from analyst_copilot.llm.registry import get_embedder                   # noqa: E402
from analyst_copilot.retrieval.fusion import union_preserving_order    # noqa: E402
from analyst_copilot.storage import repository as repo                 # noqa: E402
from analyst_copilot.storage.db import connect                         # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-k", type=int, default=0, help="default: config")
    args = parser.parse_args()

    settings = load_settings()
    bm25_k = args.bm25_k or settings.retrieval.bm25_top_k

    with connect(settings.database_url) as conn:
        t0 = time.time()
        rows = repo.pages_for_bm25(conn)
        print(f"loaded {len(rows):,} pages in {time.time() - t0:.1f}s")

        t0 = time.time()
        bm25 = BM25Index(rows)
        print(f"BM25 index built in {time.time() - t0:.1f}s")
        anchors = build_from_rows(rows, repo.load_narrative_spans(conn))
        dense = DenseRetriever(repo.pages_with_embeddings(conn), get_embedder(settings))
        print(f'dense index: {dense.coverage:,} embedded pages')

    pages_by_doc: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        pages_by_doc.setdefault(r["doc_id"], []).append((r["page_seq"], r["raw_text"]))

    catalog = build_catalog(settings.filings_dir)
    router = DocumentRouter(
        catalog,
        load_aliases(settings.data_dir / "company_aliases.yaml"),
        top_k=settings.routing.top_k_filings,
        prefer_coverage_years=settings.routing.prefer_coverage_years,
    )

    questions = load_questions(settings.practice_questions)
    scored = [q for q in questions if not q.is_unanswerable]

    for q in scored:
        q.gold_page_seqs, q.mapping_score = map_gold_pages(
            q, pages_by_doc.get(q.doc_name, []), settings.eval.gold_map_min_jaccard
        )

    evaluable = [q for q in scored if q.gold_page_seqs]
    print(
        f"questions: {len(questions)} total, {len(scored)} answerable, "
        f"{len(evaluable)} with a mapped gold page\n"
    )

    configs = [
        ("anchors only          (oracle doc)", True, True, False, 0),
        (f"dense@{settings.retrieval.dense_top_k} only        (oracle doc)", True, False, False, -1),
        (f"BM25@{bm25_k} only         (oracle doc)", True, False, True, bm25_k),
        (f"anchors + BM25@{bm25_k}     (oracle doc)", True, True, True, bm25_k),
        (f"anchors + BM25@{bm25_k}     (router top-4)", False, True, True, bm25_k),
        (
            f"escalated: BM25@{settings.retrieval.bm25_top_k_escalated} (router top-4)",
            False, True, True, settings.retrieval.bm25_top_k_escalated,
        ),
        ("anchors+BM25+DENSE    (oracle doc)", True, True, True, bm25_k),
        ("anchors+BM25+DENSE    (router top-4)", False, True, True, bm25_k),
    ]

    print(f"{'configuration':<40}{'recall':>9}{'pages':>9}")
    print("-" * 58)
    for label, oracle, use_anchor, use_bm25, k in configs:
        found = 0
        total_pages = 0
        for q in evaluable:
            scope = (
                [q.doc_name]
                if oracle
                else [c.doc_id for c in router.route(q.question).candidates]
            )
            groups = []
            dense_only = k == -1
            if dense_only:
                groups.append(dense.search(q.question, scope, settings.retrieval.dense_top_k))
            else:
                if use_anchor:
                    groups.append(anchors.search(q.question, scope, k=40))
                if use_bm25:
                    groups.append(bm25.search(q.question, scope, k, per_document=True))
                if "DENSE" in label:
                    groups.append(dense.search(q.question, scope, settings.retrieval.dense_top_k))
            hits = union_preserving_order(*groups)
            total_pages += len(hits)
            got = {(h.doc_id, h.page_seq) for h in hits}
            if any((q.doc_name, seq) in got for seq in q.gold_page_seqs):
                found += 1
        n = len(evaluable)
        print(
            f"{label:<40}{100 * found / n:>8.1f}%{total_pages / n:>9.1f}"
        )

    print("\nplan's measured targets: anchors 73.0% | +BM25 85.7% (oracle) | 81.0% (top-4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

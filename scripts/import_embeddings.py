#!/usr/bin/env python
"""Copy page embeddings from another database that already has them.

    python scripts/import_embeddings.py --from analyst_copilot

THIS IS A SHORTCUT AROUND AN EMBEDDING RUN, AND IT IS ONLY VALID BECAUSE
THE VECTOR SPACE WAS VERIFIED. Vectors from a different embedding model are not
merely worse - they are meaningless against a query embedded by ours, and
`<=>` would happily return the nearest of them. Before writing a single row the
script re-embeds a sample of the SOURCE text with OUR embedder and compares to
the stored vector; anything below `--min-cosine` aborts.

MEASURED on the source used here: cosine 1.0000 on three sampled pages, i.e.
the same model (text-embedding-3-small, 1024 dims).

Pages are matched by `page_id`, which is derived (`DOC#pN`) and therefore stable
across ingests of the same corpus. A page absent from the source is left NULL,
which `dense_search` filters out - it degrades, it does not corrupt.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import psycopg                                          # noqa: E402
from psycopg.rows import dict_row                       # noqa: E402

from analyst_copilot.config import load_settings        # noqa: E402
from analyst_copilot.llm.registry import get_embedder   # noqa: E402


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", required=True,
                    help="source database NAME on the same server")
    ap.add_argument("--sample", type=int, default=3, help="pages to verify")
    ap.add_argument("--min-cosine", type=float, default=0.98)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    settings = load_settings()
    target_url = settings.database_url
    source_url = re.sub(r"/([^/?]+)(\?|$)", rf"/{args.source}\2", target_url)
    if source_url == target_url:
        print("source and target are the same database", file=sys.stderr)
        return 2

    # ── 1. verify the vector space BEFORE writing anything ────────────────
    embedder = get_embedder(settings)
    with psycopg.connect(source_url, connect_timeout=30, row_factory=dict_row) as src:
        with src.cursor() as cur:
            cur.execute(
                """SELECT page_id, lexical_text, embedding::text AS v
                     FROM pages
                    WHERE embedding IS NOT NULL AND length(lexical_text) > 200
                    ORDER BY page_id LIMIT %s""",
                (args.sample,),
            )
            probes = cur.fetchall()
    if not probes:
        print(f"{args.source} has no embedded pages", file=sys.stderr)
        return 1

    print(f"verifying the vector space against {args.source} ...")
    worst = 1.0
    for p in probes:
        stored = [float(x) for x in p["v"].strip("[]").split(",")]
        mine = embedder.embed([p["lexical_text"]])[0]
        c = cosine(mine, stored)
        worst = min(worst, c)
        print(f"  {p['page_id']:<30} cosine {c:.4f}")
    if worst < args.min_cosine:
        print(
            f"\nABORT: lowest cosine {worst:.4f} < {args.min_cosine}. The source "
            f"was embedded by a DIFFERENT model; its vectors are meaningless "
            f"against queries embedded by ours.",
            file=sys.stderr,
        )
        return 1
    print(f"  OK - same model (worst {worst:.4f})\n")

    # ── 2. copy ───────────────────────────────────────────────────────────
    with psycopg.connect(source_url, connect_timeout=30, row_factory=dict_row) as src, \
         psycopg.connect(target_url, connect_timeout=30, row_factory=dict_row) as dst:
        with src.cursor() as scur:
            scur.execute(
                "SELECT page_id, embedding::text AS v FROM pages "
                "WHERE embedding IS NOT NULL ORDER BY page_id"
            )
            rows = scur.fetchall()
        print(f"copying {len(rows):,} vectors ...")
        written = 0
        with dst.cursor() as dcur:
            for i in range(0, len(rows), args.batch):
                chunk = rows[i: i + args.batch]
                dcur.executemany(
                    "UPDATE pages SET embedding = %s::vector WHERE page_id = %s",
                    [(r["v"], r["page_id"]) for r in chunk],
                )
                written += len(chunk)
                print(f"  {written:,}/{len(rows):,}", end="\r")
        dst.commit()
        print(f"  {written:,}/{len(rows):,} written")

        with dst.cursor() as dcur:
            dcur.execute(
                "SELECT count(*) n, count(embedding) e, "
                "count(*) FILTER (WHERE embedding IS NULL) missing FROM pages"
            )
            r = dcur.fetchone()
    pct = 100 * r["e"] / r["n"] if r["n"] else 0
    print(f"\ncoverage: {r['e']:,}/{r['n']:,} pages ({pct:.1f}%), {r['missing']:,} still NULL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

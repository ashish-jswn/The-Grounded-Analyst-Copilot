"""Connection management. THE ONLY MODULE THAT OPENS A DATABASE CONNECTION.

PostgreSQL + pgvector. The same schema runs on Azure Flexible Server and on
the local docker-compose instance, so anyone can reproduce the system with one
DATABASE_URL swap and no Azure-only feature.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row


@contextmanager
def connect(database_url: str, *, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Open one connection. The URL is passed in - never read from the
    environment here, which is config.py's sole job."""
    conn = psycopg.connect(database_url, row_factory=dict_row, autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def applied_migrations(conn: psycopg.Connection) -> list[str]:
    """Which migrations the database already has. Empty when the ledger table
    does not exist yet, which is the from-zero case of a fresh install."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public.schema_migrations') IS NOT NULL AS present"
        )
        row = cur.fetchone()
        if not row or not row["present"]:
            return []
        cur.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [r["version"] for r in cur.fetchall()]


def healthcheck(conn: psycopg.Connection) -> dict[str, object]:
    """Confirm the schema is present and the embedding width matches the code.

    A `vector(N)` that disagrees with EMBEDDING_DIMENSIONS fails EVERY page
    insert, and it fails at the END of a long ingest. Checking it up front turns
    a wasted ingest into a startup error.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT extname FROM pg_extension ORDER BY extname")
        extensions = [r["extname"] for r in cur.fetchall()]
        # `'pages'::regclass` RAISES when the table does not exist, and this
        # function's whole job is to be safe to call on a database that has not
        # been migrated yet. MEASURED: pointing at a fresh database produced
        # `relation "pages" does not exist` as an unhandled traceback, so
        # `ingest_all.py`'s helpful "schema not applied - run the migration"
        # message could never be reached. That is precisely the from-zero path
        # of a fresh install. `to_regclass` returns NULL instead of raising.
        cur.execute(
            """
            SELECT format_type(atttypid, atttypmod) AS t
            FROM pg_attribute
            WHERE attrelid = to_regclass('public.pages') AND attname = 'embedding'
            """
        )
        row = cur.fetchone()
        embedding_type = row["t"] if row else None
        cur.execute(
            "SELECT count(*) AS n FROM pg_tables WHERE schemaname = 'public'"
        )
        n_tables = cur.fetchone()["n"]
    return {
        "extensions": extensions,
        "embedding_type": embedding_type,
        "n_tables": n_tables,
        "migrations": applied_migrations(conn),
    }

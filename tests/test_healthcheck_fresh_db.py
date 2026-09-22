"""`healthcheck()` must survive a database with no schema.

MEASURED. Pointed at a freshly created database, it raised
`relation "pages" does not exist` — because `'pages'::regclass` RAISES rather
than returning NULL. The consequence was not cosmetic: `ingest_all.py` calls
`healthcheck()` first and branches on `health["migrations"]` to print

    schema not applied - run:
      psql "$DATABASE_URL" -f migrations/001_init.sql

so the one helpful message on the from-zero path — the path every fresh
install takes, since the README builds from an empty database — could never be
reached. A new user would get a traceback instead.
"""

from __future__ import annotations

import pytest

psycopg = pytest.importorskip("psycopg")

from analyst_copilot.storage.db import applied_migrations, healthcheck


class _Cursor:
    """A cursor over a database where public.pages does not exist.

    `to_regclass('public.pages')` returns NULL there, so the attribute lookup
    matches no rows — which is the behaviour being pinned. A cursor that raised
    on the query would be reproducing the bug, not testing the fix.
    """

    def __init__(self, tables: set[str]):
        self._tables = tables
        self._rows: list[dict] = []

    def execute(self, sql, params=None):
        text = " ".join(sql.split())
        if "'pages'::regclass" in text:
            raise psycopg.errors.UndefinedTable('relation "pages" does not exist')
        if "pg_extension" in text:
            self._rows = [{"extname": e} for e in ("pg_trgm", "unaccent", "vector")]
        elif "format_type" in text:
            self._rows = [] if "pages" not in self._tables else [{"t": "vector(1024)"}]
        elif "pg_tables" in text:
            self._rows = [{"n": len(self._tables)}]
        elif "to_regclass('public.schema_migrations')" in text:
            self._rows = [{"present": "schema_migrations" in self._tables}]
        elif "schema_migrations" in text:
            self._rows = [{"version": "001_init"}]
        else:
            self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Conn:
    def __init__(self, tables): self._tables = tables
    def cursor(self): return _Cursor(self._tables)


def test_healthcheck_on_an_unmigrated_database_reports_rather_than_raises():
    health = healthcheck(_Conn(set()))
    assert health["n_tables"] == 0
    assert health["embedding_type"] is None
    assert health["migrations"] == []          # what ingest_all.py branches on


def test_the_from_zero_path_reaches_its_guidance():
    """ingest_all.py: `if not health["migrations"]: print("schema not applied...")`."""
    assert not healthcheck(_Conn(set()))["migrations"]


def test_a_migrated_database_still_reports_the_embedding_width():
    """The width check exists because a vector(N) mismatch fails EVERY page
    insert, at the END of a long ingest."""
    health = healthcheck(_Conn({"pages", "filings", "schema_migrations"}))
    assert health["embedding_type"] == "vector(1024)"
    assert health["migrations"] == ["001_init"]


def test_applied_migrations_is_empty_before_the_ledger_exists():
    assert applied_migrations(_Conn(set())) == []

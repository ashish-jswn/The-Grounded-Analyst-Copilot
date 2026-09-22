#!/usr/bin/env python
"""Preflight check: is every dependency reachable from THIS network?

    python scripts/preflight.py

WHY THIS EXISTS. Azure Postgres allows connections by SOURCE IP. Moving to a
different network gives you a new public IP, and every database call then
fails with a timeout that looks nothing like a firewall rule. The failure
surfaces as a dead corpus panel and a dead "Add filing" button.

Each check is independent and prints what to do about it, because the useful
question is never "is it broken" but "which part".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "


def _line(state: str, name: str, detail: str) -> None:
    print(f"[{state}] {name:<22} {detail}")


def check_config() -> object | None:
    from analyst_copilot.config import load_settings

    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001
        _line(BAD, "config", f"{type(exc).__name__}: {exc}")
        print("        -> .env is missing or malformed. Copy .env.example and fill it in.")
        return None
    _line(OK, "config", "loaded .env + config.yaml")
    return settings


def check_database(settings) -> bool:
    """The check that a network change actually breaks."""
    from analyst_copilot.storage.db import connect
    from analyst_copilot.storage import repository as repo

    url = settings.database_url
    host = url.split("@")[-1].split("/")[0] if "@" in url else "(unparsed)"
    t0 = time.time()
    try:
        with connect(url) as conn:
            stats = repo.corpus_stats(conn)
    except Exception as exc:  # noqa: BLE001
        _line(BAD, "database", f"{host} - {type(exc).__name__}: {str(exc)[:90]}")
        print("        -> Most likely THIS NETWORK'S IP IS NOT ALLOWED. In the Azure")
        print("           portal: Postgres server -> Networking -> Firewall rules ->")
        print("           'Add current client IP address' -> Save. Takes ~1 minute.")
        print("        -> Questions still work if the backend is already warm; the")
        print("           corpus panel and Add filing do not.")
        return False
    _line(OK, "database", f"{host} - {stats['filings']} filings, "
                          f"{stats['pages']:,} pages ({time.time() - t0:.1f}s)")
    if stats["filings"] == 0:
        _line(WARN, "corpus", "schema is applied but EMPTY - run scripts/ingest_all.py")
    return True


def check_llm(settings) -> bool:
    """One real completion. A key that is present but rejected is not a key."""
    from analyst_copilot.llm.registry import get_provider

    t0 = time.time()
    try:
        provider = get_provider(settings, "composer")
        out = provider.complete(
            system="Reply with the single word: ready",
            user="ready?",
            schema=None,
            stage="composer",
        )
    except Exception as exc:  # noqa: BLE001
        _line(BAD, "llm", f"{type(exc).__name__}: {str(exc)[:90]}")
        print("        -> Check AZURE_OPENAI_ENDPOINT / _API_KEY / GPT_DEPLOYMENT,")
        print("           and that the deployment still has quota.")
        return False
    # DELIBERATELY NOT PRINTING THE REPLY. The composer stage expects a
    # schema, so a bare prompt comes back as an error-shaped JSON body - which
    # reads like a failure while proving exactly what this check is for. What
    # is being tested is that the endpoint resolves, the key is accepted and
    # the deployment has quota; a returned completion of ANY shape proves all
    # three, because the alternative is an exception.
    del out
    _line(OK, "llm", f"{settings.model('composer').deployment} accepted a call "
                     f"and returned a completion ({time.time() - t0:.1f}s)")
    return True


def check_backend() -> bool:
    """Optional: only meaningful if you already started uvicorn."""
    import urllib.request

    for port in (8300, 8000):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=10
            ) as r:
                if r.status == 200:
                    _line(OK, "backend", f"answering on :{port}")
                    return True
        except Exception:  # noqa: BLE001, S110
            continue
    _line(WARN, "backend", "not running on :8300 or :8000 (start uvicorn to use the UI)")
    return False


def main() -> int:
    print("preflight - checking every dependency from this network\n")
    settings = check_config()
    if settings is None:
        return 2

    db = check_database(settings)
    llm = check_llm(settings)
    check_backend()

    print()
    if db and llm:
        print("Ready. Warm the pipeline by asking one question -")
        print("the corpus loads into memory on the first question and stays there.")
        return 0
    print("NOT ready - fix the FAIL lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Enforces the project's governing constraint: generalisation.

No question ID, `doc_name`, or gold answer may be read by any module under
ingest/, retrieval/, query/, storage/ or api/. Benchmark knowledge is confined
to eval/. This is greppable and belongs in CI.

The 136 practice questions are TEST DATA, not the specification. Real users ask
different questions over filings the system has never seen, so a pipeline stage
that recognises a benchmark question or a specific document is a defect even
if it raises the practice score.

Only config.py may read the environment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "analyst_copilot"

# Stages that must be free of benchmark knowledge. `eval/` is deliberately absent.
PIPELINE_PACKAGES = ["ingest", "retrieval", "query", "storage", "api", "llm"]

# Signatures of benchmark leakage.
_FINANCEBENCH_ID = re.compile(r"financebench_id", re.I)
_PRACTICE_FILE = re.compile(r"practice[-_]questions", re.I)
_GOLD_FIELD = re.compile(r"\b(gold_answer|evidence_text|evidence_page_num)\b")

# A hardcoded document identity, e.g. "MICROSOFT_2016_10K" or "3M_2018_10K".
# The corpus names filings <COMPANY>_<YEAR>_<FORM>, so this catches any stage
# that special-cases one filing instead of describing a general property.
_DOC_ID_LITERAL = re.compile(r"""["'][A-Z0-9_]{2,}_(?:19|20)\d{2}(?:Q[1-4])?_(?:10K|10Q|8K)""")


_DOCSTRING = re.compile(r'("""|\'\'\').*?\1', re.S)


def _code_only(path: Path) -> str:
    """Strip docstrings and full-line comments.

    Prose may legitimately cite a measurement or show an example id - the
    provenance notes throughout this codebase depend on that. What must never
    appear is a CODE BRANCH on a specific document.
    """
    text = path.read_text(encoding="utf-8")
    text = _DOCSTRING.sub('""', text)
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _pipeline_files() -> list[Path]:
    files: list[Path] = []
    for package in PIPELINE_PACKAGES:
        files.extend(sorted((SRC / package).rglob("*.py")))
    return files


def test_pipeline_packages_exist():
    """Guard the guard: a renamed package must not silently disable this test."""
    present = [p for p in PIPELINE_PACKAGES if (SRC / p).is_dir()]
    assert present, "no pipeline packages found - the guard is not checking anything"


@pytest.mark.parametrize("path", _pipeline_files(), ids=lambda p: p.name)
def test_no_benchmark_knowledge_in_pipeline(path: Path):
    """A stage must never know about the benchmark."""
    text = path.read_text(encoding="utf-8")
    # Comments and docstrings may legitimately cite a measurement; only code is
    # checked. This strips full-line comments, which is where our provenance
    # notes live.
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    for pattern, why in [
        (_FINANCEBENCH_ID, "reads a benchmark question id"),
        (_PRACTICE_FILE, "reads the practice question file"),
        (_GOLD_FIELD, "reads a gold evidence field"),
    ]:
        assert not pattern.search(code), f"{path.name} {why}"


@pytest.mark.parametrize("path", _pipeline_files(), ids=lambda p: p.name)
def test_no_hardcoded_document_identity(path: Path):
    """No stage may branch on a specific filing.

    The <hr>-inside-<table> rule and the EDGAR envelope handling are general
    structural mechanisms; `if doc_id == 'MICROSOFT_2016_10K'` would not be.
    """
    hits = _DOC_ID_LITERAL.findall(_code_only(path))
    assert not hits, f"{path.name} hardcodes a document identity {hits}"


@pytest.mark.parametrize("path", _pipeline_files(), ids=lambda p: p.name)
def test_only_config_reads_the_environment(path: Path):
    """Only config.py reads os.environ; stages receive their settings."""
    code = path.read_text(encoding="utf-8")
    assert "os.environ" not in code, f"{path.name} reads os.environ"
    assert "getenv" not in code, f"{path.name} reads getenv"


def test_only_llm_package_imports_vendor_sdks():
    """No stage imports openai or anthropic directly, so swapping
    a model family stays a .env change rather than a rewrite."""
    for path in _pipeline_files():
        if path.parent.name == "llm":
            continue
        code = path.read_text(encoding="utf-8")
        for sdk in ("import openai", "from openai", "import anthropic", "from anthropic", "import cohere"):
            assert sdk not in code, f"{path.name} imports a vendor SDK ({sdk})"

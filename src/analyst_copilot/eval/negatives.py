"""The not-found / unanswerable evaluation set.

MANDATORY, AND NOT OPTIONAL POLISH. MEASURED: 0 of the 136 practice
questions has "not found" as its gold answer. The dev set therefore contains
almost no signal for the one behaviour the rubric rewards most - an honest
refusal - while a confident wrong answer is the only way to score -1.
Abstention is the component that earns the score and the component with no
benchmark coverage, so any claimed abstention rate without these is untested.

THE FOUR RULES:
  1. a SEPARATE set, held in `evaluation/negatives/`
  2. NOT hardcoded into the production pipeline - no module outside `eval/` may
     know these exist. The pipeline must refuse them because the evidence is
     absent, never because it recognised the question.
  3. NOT part of the benchmark's question distribution - never mixed into the
     136 for a headline number
  4. reported SEPARATELY from the 136-question accuracy results

Generated deterministically from the catalog, then HAND-REVIEWED. A negative
that turns out to be answerable poisons the calibration in the most damaging
direction: it teaches the system to refuse something it should have answered.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..ingest.catalog import FilingMeta

# Five categories, ~60 items.
WRONG_FILING = "wrong_filing_or_period"
MISSING_METRIC = "genuinely_missing_metric"
FALSE_PREMISE = "unsupported_fact_or_false_premise"
MISSING_OPERAND = "missing_operand"
OUTSIDE_FILING = "outside_the_filing"


@dataclass
class Negative:
    """One question that is genuinely unanswerable from the corpus."""

    nid: str
    question: str
    category: str
    rationale: str
    company_slug: str | None = None
    reviewed: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)


# Metrics a 10-K does not report as a GAAP line item. These are general facts
# about what filings contain, not facts about this corpus.
_ABSENT_METRICS = [
    "total employee headcount broken down by gender",
    "Scope 3 greenhouse gas emissions in tonnes CO2e",
    "average customer acquisition cost",
    "monthly active users",
    "the CEO's individual bonus target as a percentage of salary",
    "net promoter score",
    "the number of patents filed during the year",
    "average employee tenure in years",
    "customer churn rate",
    "the dollar value of the order backlog by region",
]

_FALSE_PREMISE_TEMPLATES = [
    "What was the goodwill impairment charge recorded by {company} in {year}?",
    "How much did {company} pay in litigation settlements related to its {year} product recall?",
    "What was the size of the special dividend {company} declared in {year}?",
    "How large was the restructuring reserve {company} released in {year}?",
    "What was the gain {company} recognised on the sale of its headquarters in {year}?",
]

_MISSING_OPERAND_TEMPLATES = [
    "What was {company}'s customer lifetime value to acquisition cost ratio in {year}?",
    "Calculate {company}'s revenue per employee for {year} using headcount disclosed in the filing.",
    "What was {company}'s cost per unit shipped in {year}?",
    "Compute {company}'s marketing spend as a percentage of revenue for {year}.",
    "What was {company}'s research spend per patent granted in {year}?",
]

_OUTSIDE_TEMPLATES = [
    "What is {company}'s share price today?",
    "Do analysts rate {company} a buy or a sell right now?",
    "How does {company}'s {year} revenue compare with its largest competitor's?",
    "What did {company} report in the quarter after this filing was published?",
    "What is {company}'s current market capitalisation?",
    "Has {company} announced any acquisitions since this filing?",
    "What is the consensus earnings estimate for {company} next year?",
]


def generate(
    catalog: list[FilingMeta],
    seed: int = 20260831,
    target: int = 60,
    aliases: dict[str, list[str]] | None = None,
) -> list[Negative]:
    """Build the negative set from the catalog. Deterministic given a seed.

    `aliases` supplies natural company names ("Coca-Cola", not "Cocacola"), so a
    negative reads like a question a person would actually ask. It also keeps
    the question ROUTABLE: the router must find candidate filings and then fail
    to support an answer, which is the behaviour under test. A question the
    router cannot route would take the clarify path and never exercise the
    gates.
    """
    rng = random.Random(seed)
    display = _display_names(aliases or {})
    tenk = [f for f in catalog if f.form_type == "10-K"]
    tenq = [f for f in catalog if f.form_type == "10-Q"]
    by_company: dict[str, list[FilingMeta]] = {}
    for f in catalog:
        by_company.setdefault(f.company_slug, []).append(f)

    out: list[Negative] = []

    def name(f: FilingMeta) -> str:
        return display.get(f.company_slug) or _fallback_name(f)

    def add(category: str, question: str, rationale: str, slug: str | None, tags=()):
        out.append(
            Negative(
                nid=f"neg_{len(out) + 1:03d}",
                question=question,
                category=category,
                rationale=rationale,
                company_slug=slug,
                tags=list(tags),
            )
        )

    # ── wrong filing / period (15) ────────────────────────────────────────
    # A year strictly beyond everything the company has on file. `coverage_years`
    # makes this exact rather than a guess.
    for f in rng.sample(tenk, min(10, len(tenk))):
        latest = max(x.fiscal_year for x in by_company[f.company_slug])
        year = latest + 2
        add(
            WRONG_FILING,
            f"What was {name(f)}'s total revenue in FY{year}?",
            f"no {name(f)} filing covers FY{year}; latest on file is FY{latest}",
            f.company_slug,
            tags=["future_period"],
        )
    for f in rng.sample(tenq, min(5, len(tenq))):
        other = 4 if f.fiscal_quarter != 4 else 1
        add(
            WRONG_FILING,
            f"What was {name(f)}'s net income in Q{other} FY{f.fiscal_year}?",
            f"only Q{f.fiscal_quarter} FY{f.fiscal_year} is on file for {name(f)}",
            f.company_slug,
            tags=["wrong_quarter"],
        )

    # ── genuinely missing metric (10) ─────────────────────────────────────
    for metric, f in zip(_ABSENT_METRICS, rng.sample(tenk, min(10, len(tenk)))):
        add(
            MISSING_METRIC,
            f"What was {name(f)}'s {metric} in FY{f.fiscal_year}?",
            "not a GAAP line item and not disclosed in the financial statements",
            f.company_slug,
        )

    # ── unsupported fact / false premise (10) ─────────────────────────────
    picks = rng.sample(tenk, min(10, len(tenk)))
    for i, f in enumerate(picks):
        template = _FALSE_PREMISE_TEMPLATES[i % len(_FALSE_PREMISE_TEMPLATES)]
        add(
            FALSE_PREMISE,
            template.format(company=name(f), year=f"FY{f.fiscal_year}"),
            "presupposes an event the filing does not record; the correct "
            "response is to decline, not to report zero",
            f.company_slug,
        )

    # ── missing operand (10) ──────────────────────────────────────────────
    picks = rng.sample(tenk, min(10, len(tenk)))
    for i, f in enumerate(picks):
        template = _MISSING_OPERAND_TEMPLATES[i % len(_MISSING_OPERAND_TEMPLATES)]
        add(
            MISSING_OPERAND,
            template.format(company=name(f), year=f"FY{f.fiscal_year}"),
            "the denominator is not disclosed, so the ratio cannot be computed "
            "from this filing",
            f.company_slug,
        )

    # ── outside the filing (15) ───────────────────────────────────────────
    picks = rng.sample(catalog, min(15, len(catalog)))
    for i, f in enumerate(picks):
        template = _OUTSIDE_TEMPLATES[i % len(_OUTSIDE_TEMPLATES)]
        add(
            OUTSIDE_FILING,
            template.format(company=name(f), year=f"FY{f.fiscal_year}"),
            "asks for information that is not in any filing (market data, "
            "opinion, or events after the filing date)",
            f.company_slug,
        )

    return out[:target] if target else out


def _display_names(aliases: dict[str, list[str]]) -> dict[str, str]:
    """Pick the most natural alias per company for question text."""
    out: dict[str, str] = {}
    for slug, values in aliases.items():
        # Prefer a multi-word/hyphenated alias ("coca-cola", "american express")
        # over an acronym or the run-together slug.
        spaced = [a for a in values if " " in a or "-" in a]
        best = max(spaced, key=len) if spaced else max(values, key=len)
        out[slug] = best.title()
    return out


def _fallback_name(f: FilingMeta) -> str:
    """Used only when a company has no alias entry - e.g. a new upload."""
    return f.company_slug.replace("_", " ").title()


def save(negatives: list[Negative], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for n in negatives:
            fh.write(json.dumps(asdict(n)) + "\n")


def load(path: Path) -> list[Negative]:
    with path.open(encoding="utf-8") as fh:
        return [Negative(**json.loads(line)) for line in fh if line.strip()]

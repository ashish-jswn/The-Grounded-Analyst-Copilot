"""Ablation runner - produces the approach-note table BY SCRIPT, not by hand.

Every component in this system claims a contribution. This measures them, one
switch at a time, so the approach note reports what each stage was actually
worth rather than what it was designed to be worth.

The seams already exist by design, which is why this file is short:
  * gates are a LIST OF OBJECTS, so any one can be disabled
  * `retrieval.use_reranker` / `use_dense` are config flags
  * `build_pipeline(with_verifiers=False)` drops the LLM verifiers

READ BOTH COLUMNS TOGETHER. A configuration that raises the score while
raising the false-answer rate is a REGRESSION, because a confident wrong answer
costs -1 while an honest refusal costs 0. Reporting accuracy alone would hide
exactly the trade the rubric punishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from ..config import Settings


@dataclass
class Variant:
    """One configuration to measure, and how it differs from the baseline."""

    name: str
    description: str
    mutate: Callable[[Settings], Settings] = lambda s: s
    with_verifiers: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)


def _gates_without(*disabled: str) -> Callable[[Settings], Settings]:
    def mutate(settings: Settings) -> Settings:
        kept = tuple(g for g in settings.verification.enabled_gates if g not in disabled)
        return replace(
            settings,
            verification=replace(settings.verification, enabled_gates=kept),
        )

    return mutate


def _retrieval(**changes: Any) -> Callable[[Settings], Settings]:
    def mutate(settings: Settings) -> Settings:
        return replace(settings, retrieval=replace(settings.retrieval, **changes))

    return mutate


def _routing(**changes: Any) -> Callable[[Settings], Settings]:
    def mutate(settings: Settings) -> Settings:
        return replace(settings, routing=replace(settings.routing, **changes))

    return mutate


def _verification(**changes: Any) -> Callable[[Settings], Settings]:
    def mutate(settings: Settings) -> Settings:
        return replace(
            settings, verification=replace(settings.verification, **changes)
        )

    return mutate


# The variants that answer a question someone will ask about this system.
VARIANTS: list[Variant] = [
    Variant("baseline", "everything on, as shipped"),
    # ── verification ──────────────────────────────────────────────────────
    Variant(
        "no_llm_verifiers",
        "deterministic gates only - measures what the LLM verifiers add over "
        "G1-G7, and what they cost in over-abstention",
        with_verifiers=False,
    ),
    Variant(
        "verifier_b_not_adversarial",
        "verifier B does plain entailment instead of refutation. Refutation is "
        "a SUBSTITUTE for a second family, not an addition, and 'default to "
        "REFUTED when uncertain' can reject correct answers",
        mutate=_verification(verifier_b_adversarial=False),
    ),
    Variant(
        "no_gates",
        "LLM verifiers only - the counterfactual for 'are the deterministic "
        "gates doing the work?'",
        mutate=_gates_without("G1", "G1b", "G2", "G3", "G4", "G5", "G6", "G7"),
    ),
    Variant(
        "no_G1",
        "drop the quote-on-page check. THE -1 PREVENTER: expect the false-answer "
        "rate to rise sharply",
        mutate=_gates_without("G1", "G1b"),
    ),
    Variant("no_G3", "drop the period check", mutate=_gates_without("G3")),
    Variant("no_G4", "answer even with missing slots", mutate=_gates_without("G4")),
    # ── retrieval ─────────────────────────────────────────────────────────
    Variant(
        "no_reranker",
        "RRF alone - isolates the reranker's lift",
        mutate=_retrieval(use_reranker=False),
    ),
    Variant(
        "bm25_only",
        "no structure anchors. Anchors alone measured 73.2% page recall, so "
        "this should hurt more than removing the reranker",
        mutate=_retrieval(use_dense=False),
        kwargs={"disable_anchors": True},
    ),
    Variant(
        "narrow_budget",
        "halve the assembly token budget",
        mutate=_retrieval(assembly_token_budget=21000),
    ),
    # ── routing ───────────────────────────────────────────────────────────
    Variant(
        "router_top1",
        "route to a single filing. Top-1 is 95.6% and top-4 is 98.5%, so this "
        "trades recall for a smaller context",
        mutate=_routing(top_k_filings=1),
    ),
    Variant(
        "router_top8",
        "escalate-by-depth counterfactual: top-4 and top-8 both hit 134/136, so "
        "widening should add nothing while costing context",
        mutate=_routing(top_k_filings=8),
    ),
]


def variant_by_name(name: str) -> Variant:
    for v in VARIANTS:
        if v.name == name:
            return v
    raise KeyError(f"unknown variant {name!r}; known: {[v.name for v in VARIANTS]}")

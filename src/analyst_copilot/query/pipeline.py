"""THE SPINE - the only place stage order lives.

    route -> navigate -> RETRIEVE -> extract -> COMPUTE -> verify -> ANSWER

One spine with switchable stages, NOT three pipelines. A misroute would send a
question down a path that structurally cannot answer it, and verification would
be duplicated and drift apart.

THIS IS A DETERMINISTIC WORKFLOW, NOT AN AGENT. Control flow lives
here, in code. The model chooses CONTENT - which evidence, which formula - never
what happens next. The honest label is "structure-aware RAG with an
evidence-first verification gate".

Escalation is by DEPTH, not breadth: MEASURED that the
gold filing is in router top-4 for 134/136 and in top-8 for the same 134/136, so
widening the document set buys nothing. One retry, deeper inside the same
candidates.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Settings
from ..llm import schemas
from ..llm.base import LLMError, LLMProvider
from ..llm.registry import load_prompt
from ..retrieval.anchors import AnchorRetriever
from ..retrieval.assemble import assemble
from ..retrieval.base import Hit
from ..retrieval.bm25 import BM25Index
from ..retrieval.rerank import maybe_reranker
from .compose import compose_answer
from .compute import (
    Computation, FormulaError, Operand, align_operands, compute,
    required_operand_names,
)
from .extract import Extraction, extract_evidence
from .formula_book import FormulaBook, FormulaChoice, FormulaSource
from .gates import AnswerCandidate, Citation, GateReport, run_gates
from .router import DocumentRouter, RouteResult, detect_intent


@dataclass
class QueryResult:
    """Mirrors `api/schemas.py::AnswerResponse`."""

    status: str                          # answered | abstained | clarify
    answer: str | None = None
    clarifying_question: str | None = None
    citations: list[Citation] = field(default_factory=list)
    computation: Computation | None = None
    definition: str | None = None
    formula_source: str | None = None
    abstain_reason: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)


class QueryPipeline:
    def __init__(
        self,
        settings: Settings,
        router: DocumentRouter,
        bm25: BM25Index,
        anchors: AnchorRetriever,
        pages_by_doc: dict[str, dict[int, str]],
        headers_by_page: dict[str, str],
        coverage_years: dict[str, list[int]],
        extractor: LLMProvider,
        verifier_a: LLMProvider | None = None,
        verifier_b: LLMProvider | None = None,
        formula_book: FormulaBook | None = None,
        router_llm: LLMProvider | None = None,
        composer: LLMProvider | None = None,
        dense=None,
    ) -> None:
        self.settings = settings
        self.router = router
        self.bm25 = bm25
        self.anchors = anchors
        self.pages_by_doc = pages_by_doc
        self.headers_by_page = headers_by_page
        self.coverage_years = coverage_years
        self.extractor = extractor
        self.verifier_a = verifier_a
        self.verifier_b = verifier_b
        self.book = formula_book or FormulaBook()
        self.router_llm = router_llm
        self.composer = composer
        self.dense = dense

    # ------------------------------------------------------------------
    def answer(self, question: str) -> QueryResult:
        started = time.time()
        trace: dict[str, Any] = {"query_id": str(uuid.uuid4())}

        # ── STAGE 1: route ────────────────────────────────────────────
        route = self.router.route(question)
        trace["candidate_docs"] = [c.doc_id for c in route.candidates]
        trace["company_named"] = route.company_named

        # A wrong document costs -1; a clarifying question is free.
        if route.needs_clarification or not route.candidates:
            return QueryResult(
                status="clarify",
                clarifying_question=(
                    "Which company's filing should I look at? I hold filings for "
                    "32 companies and the question does not name one."
                ),
                trace={**trace, "latency_ms": int((time.time() - started) * 1000)},
            )

        intent, question_years = self._route_metadata(question, route)
        trace["intent"] = intent

        # ── STAGES 2-6, tier 1 then one deeper retry ─────
        result = self._attempt(question, route, intent, question_years, trace, tier=1)
        if result.status == "answered":
            result.trace = {**trace, "tier": 1,
                            "latency_ms": int((time.time() - started) * 1000)}
            return result

        retry = self._attempt(question, route, intent, question_years, trace, tier=2)
        retry.trace = {**trace, "tier": 2,
                       "latency_ms": int((time.time() - started) * 1000),
                       "tier1_abstain_reason": result.abstain_reason}
        return retry

    # ------------------------------------------------------------------
    def _route_metadata(
        self, question: str, route: RouteResult
    ) -> tuple[str, list[int]]:
        """Intent and years, BOTH deterministic — no model call.

        `filing_period` != `question_period`: "What is Boeing forecasting for
        FY2023?" has no FY2023 filing, so forecast intent must search EARLIER
        filings for forward-looking statements. Years are never substituted.

        THIS USED TO COST AN LLM CALL FOR ONE FIELD. MEASURED at ~32 s per
        question — about 12% of end-to-end latency — on a ~400-token prompt, to
        decide a single historical/forecast flag that feeds exactly one thing:
        gate G3's exception permitting a question period after the filing
        period. At that price a keyword prior beats a cheap LLM call.

        MEASURED across all 136 practice questions: 3 classified forecast
        (2%), and all three genuinely are — the Boeing production-rate forecast,
        Pfizer's expected Upjohn spin-off cost, and Verizon's expected 2024
        retiree payments (asked of a FY2021 filing, which is precisely the G3
        exception). Zero false positives, and false positives are the risk that
        matters: one would relax G3's period check on a HISTORICAL question,
        weakening the gate that exists to catch wrong-period evidence.

        `router_llm` is still accepted by the constructor so the ablation runner
        can turn stages off, but it is no longer consulted here.
        """
        return detect_intent(question), sorted(route.years)

    def _retrieve(
        self, question: str, scope: list[str], tier: int
    ) -> tuple[list[list[Hit]], int]:
        k = (
            self.settings.retrieval.bm25_top_k
            if tier == 1
            else self.settings.retrieval.bm25_top_k_escalated
        )
        rankings = [
            self.anchors.search(question, scope, k=40),
            self.bm25.search(question, scope, k, per_document=True),
        ]
        # A THIRD RRF RANKING, NOT A REPLACEMENT. Dense is fused, never
        # substituted: it is scoped to the same routed candidates, and a dead
        # embedding endpoint returns [] so the question is still answered by
        # anchors and BM25. `dense` is None whenever retrieval.use_dense is off
        # or nothing is embedded, which keeps the ablation a config change.
        if self.dense is not None:
            dense_hits = self.dense.search(
                question, scope, k=self.settings.retrieval.dense_top_k
            )
            if dense_hits:
                rankings.append(dense_hits)
        return rankings, k

    def _attempt(
        self,
        question: str,
        route: RouteResult,
        intent: str,
        question_years: list[int],
        trace: dict[str, Any],
        *,
        tier: int,
    ) -> QueryResult:
        scope = [c.doc_id for c in route.candidates]
        rankings, k = self._retrieve(question, scope, tier)

        expand = self.settings.retrieval.neighbour_expand * tier
        context = assemble(
            rankings,
            self.pages_by_doc,
            token_budget=self.settings.retrieval.assembly_token_budget,
            rrf_k=self.settings.retrieval.rrf_k,
            neighbour_expand=expand,
            reranker=maybe_reranker(self.settings, question),
            rerank_top_n=self.settings.retrieval.rerank_top_n,
            headers_by_page=self.headers_by_page,
        )
        trace[f"tier{tier}_pages"] = context.page_count
        trace[f"tier{tier}_tokens"] = context.token_estimate
        trace[f"tier{tier}_bm25_k"] = k

        if not context.hits:
            return self._abstain("no_candidates")

        # ── STAGE 4: extract ──────────────────────────────────────────
        # THE FORMULA IS CHOSEN FROM THE QUESTION *BEFORE* EXTRACTING, so the
        # extractor can be told the operand names the calculator will demand.
        # Choosing it afterwards - as this originally did - meant those names
        # could never be communicated, the extractor invented its own, and every
        # computed answer died at G4 on a name mismatch that looked exactly like
        # missing evidence.
        planned = self.book.choose(question)
        wanted = required_operand_names(planned.formula) if planned else []
        trace[f"tier{tier}_required_operands"] = wanted
        try:
            extraction = extract_evidence(
                self.extractor, question, context.text, required_operands=wanted
            )
        except LLMError as exc:
            # A truncated or failed model call is NOT evidence of absence, but
            # we still cannot answer, so we abstain and record why. The MESSAGE
            # is kept, not just the type: an abstention caused by a deployment
            # quota limit must be distinguishable from one caused by the filing
            # genuinely lacking the evidence, or the abstention rate is
            # meaningless.
            trace.setdefault("errors", []).append(f"tier{tier}:{exc}")
            return self._abstain(f"extractor_error:{type(exc).__name__}")

        # ── STAGE 5: compute ──────────────────────────────────────────
        choice, computation, compute_error = self._compute(
            question, extraction, planned=planned
        )

        composed = compose_answer(
            extraction, computation, question, question_years, self.composer
        )
        trace[f"tier{tier}_supporting_quotes"] = composed.supporting_quote_indexes
        if composed.declined_in_prose:
            # The composer refused in prose while leaving the flag true. Counted,
            # because it means the prompt rule forbidding that is not landing -
            # and because shipping it would have cost a -1 instead of a 0.
            trace[f"tier{tier}_declined_in_prose"] = True
        if not composed.answerable or not composed.text:
            # The evidence did not settle the question. Declining here is the
            # correct outcome, not a failure.
            return self._abstain(
                "composer_declined_in_prose" if composed.declined_in_prose
                else "unanswerable_from_evidence"
            )
        answer_text = composed.text

        # ── STAGE 6: verify - deterministic gates first ───────────────
        candidate = AnswerCandidate(
            question=question,
            answer_text=answer_text,
            citations=extraction.citations,
            missing_slots=extraction.missing_slots + ([compute_error] if compute_error else []),
            operands=extraction.operands,
            computation=computation,
            page_texts=context.pages_by_doc,
            candidate_doc_ids=scope,
            coverage_years={d: self.coverage_years.get(d, []) for d in scope},
            evidence_years=self._evidence_years(extraction),
            intent=intent,
            question_years=question_years,
            answer_type=extraction.answer_type,
        )
        report = run_gates(candidate, enabled=self.settings.verification.enabled_gates)
        trace[f"tier{tier}_gates"] = [
            {"gate": r.gate_id, "passed": r.passed, "detail": r.detail}
            for r in report.results
        ]
        if not report.passed:
            return self._abstain(report.abstain_reason or "gates")

        # ── STAGE 6b: the two LLM verifiers, last net ─────────────────
        verdicts, verifier_reasons = self._verify(
            question, answer_text, extraction, computation
        )
        trace[f"tier{tier}_verifiers"] = verdicts
        trace[f"tier{tier}_verifier_reasons"] = verifier_reasons
        if verdicts and not all(verdicts.values()):
            failed = [k for k, v in verdicts.items() if not v]
            return self._abstain(f"verifier:{','.join(failed)}")

        return QueryResult(
            status="answered",
            answer=answer_text,
            citations=extraction.citations,
            computation=computation,
            definition=choice.definition if choice else None,
            formula_source=choice.source.value if choice else None,
        )

    # ------------------------------------------------------------------
    def _compute(
        self, question: str, extraction: Extraction,
        *, planned: FormulaChoice | None = None,
    ) -> tuple[FormulaChoice | None, Computation | None, str | None]:
        """Apply the precedence ladder, then evaluate in Decimal."""
        hint = extraction.metric_name or extraction.question_supplied_definition
        # `planned` is what the extractor was asked to name its slots for, so it
        # is the fallback when the hint resolves to nothing.
        choice = self.book.choose(question, hint) or planned
        if choice is None or not choice.formula:
            return choice, None, None
        # Safety net, not the mechanism: the prompt now asks for these names
        # directly, so this is usually a no-op. An operand that does NOT match
        # stays unmatched rather than being guessed at - a mis-assigned operand
        # computes a plausible wrong number, which is the -1 the whole system
        # exists to prevent, while an unresolved one fails G4 loudly.
        operands = align_operands(
            extraction.operands, required_operand_names(choice.formula)
        )
        try:
            computation = compute(
                choice.formula,
                operands,
                unit=choice.unit,
                render=choice.render,
                dp=choice.dp,
            )
        except FormulaError as exc:
            # A formula we cannot evaluate is a MISSING OPERAND, not a licence
            # to answer from the narrative.
            return choice, None, f"formula:{exc}"
        return choice, computation, None

    @staticmethod
    def _evidence_years(extraction: Extraction) -> list[int]:
        years: list[int] = []
        for slot in extraction.slots:
            for token in (slot.period or "").split():
                digits = "".join(ch for ch in token if ch.isdigit())
                if len(digits) == 4 and digits.startswith("20"):
                    years.append(int(digits))
        return sorted(set(years))

    @staticmethod
    def _computation_brief(computation: Computation | None) -> str:
        """Tell the verifier the arithmetic is already settled, and by what.

        MEASURED, AND IT COST A CORRECT ANSWER. On the Activision
        fixed-asset-turnover question the pipeline computed 24.26 - the gold
        answer exactly - and BOTH verifiers rejected it by re-deriving the ratio
        themselves from a revenue figure that was not FY2019's:
            "yielding 7,017/267.5 = 26.23, not 24.26, so the answer is incorrect"

        Three things are wrong with letting that happen:
          1. it is the MODEL DOING ARITHMETIC, the one thing the design forbids;
          2. it is REDUNDANT - gate G6 already re-evaluated the formula over
             these operands in Decimal, deterministically, to 1e-6;
          3. it is a NET NEGATIVE - the deterministic check is right and the
             model overrides it.

        So for a computed answer the verifier's job is narrowed to what it is
        actually good at: are these operands really in the quotes, for the
        period and unit the question asked for? The prompt's "mathematically
        entailed" wording is what invited the re-derivation, and this overrides
        it for exactly the case where a proof already exists.
        """
        if computation is None:
            return ""
        operands = ", ".join(
            f"{name}={value}" for name, value in computation.operands.items()
        )
        return (
            "\n\nCOMPUTED ANSWER — THE ARITHMETIC IS ALREADY PROVEN.\n"
            f"  formula:  {computation.formula}\n"
            f"  operands: {operands}\n"
            f"  result:   {computation.rendered()}\n"
            "Gate G6 has already re-evaluated this formula over these operands "
            "deterministically, in exact decimal arithmetic. DO NOT re-derive "
            "the result, and do not reject the answer because your own "
            "calculation differs — if it differs, your arithmetic is wrong.\n"
            "Check ONLY this: does each operand above appear in the quotes, for "
            "the period and in the unit the question asked for? Judge the "
            "EVIDENCE, not the sum."
        )

    # Cap per page: a cited page is normally a statement or a note, and the
    # header that carries units and year columns is near its top. This bounds
    # the added cost to a few thousand tokens per verifier call.
    _VERIFY_PAGE_CHARS = 6000

    def _cited_pages(self, extraction: Extraction) -> str:
        """Full text of each page the extractor quoted, de-duplicated.

        Ordered by (doc, page) rather than by slot so the same evidence always
        renders identically - two verifiers must see byte-identical input for
        their disagreement to mean anything.
        """
        wanted = sorted(
            {(s.doc_id, s.page_seq) for s in extraction.slots if s.doc_id}
        )
        blocks = []
        for doc_id, seq in wanted:
            text = (self.pages_by_doc.get(doc_id) or {}).get(seq)
            if not text:
                continue
            head = text[: self._VERIFY_PAGE_CHARS]
            blocks.append(f"--- {doc_id} page {seq} ---\n{head}")
        if not blocks:
            return ""
        return (
            "CITED PAGES IN FULL (the quotes above come from these; use them to "
            "resolve units, fiscal-year columns and line-item labels):\n"
            + "\n\n".join(blocks)
        )

    def _verify(
        self,
        question: str,
        answer_text: str,
        extraction: Extraction,
        computation: Computation | None = None,
    ) -> tuple[dict[str, bool], dict[str, str]]:
        """Verifiers A and B, each in ISOLATION.

        A verifier sees question + answer + quotes + THE CITED PAGES. Never
        verifier A's verdict, never the extractor's reasoning, never the
        retrieval trace. Isolation is about not inheriting another stage's
        conclusion; it was never about withholding the source document.

        THE CITED PAGE IS INCLUDED BECAUSE THE QUOTE ALONE IS UNVERIFIABLE.
        MEASURED on the full practice set: 48 of 90 refusals were verifier
        rejections, 40 of those (83%) objected to units, period or column
        labels - and 40 of the 48 had the GOLD page in context, meaning the
        evidence was there and a correct draft was thrown away.

        The cause is structural, not caution. A quote is one ROW of a financial
        table; "(in millions)" and the fiscal-year column headings live in the
        table HEADER, several lines above. Asked "does this quote state its
        units and period?", the honest answer for almost every table row is no
        - so a verifier told to reject anything ambiguous rejects correct
        answers as a matter of course. Example: gold $1,616.0m, quote reads
        "Trade receivables, net 1,615.9 1,864.3", rejected for "does not label
        which is FY2020 nor state the units".

        Giving it the page restores the header and the year columns, so the
        check it is asked to perform becomes possible. This does NOT relax the
        gate - the standard is unchanged - it supplies the evidence the
        standard needs.
        """
        if not (self.verifier_a or self.verifier_b):
            return {}, {}
        quotes = "\n".join(
            f"[{s.doc_id} p.{s.page_seq}] {s.quote}" for s in extraction.slots
        )
        pages = self._cited_pages(extraction)
        user = (
            f"QUESTION:\n{question}\n\nPROPOSED ANSWER:\n{answer_text}\n\n"
            f"QUOTES:\n{quotes}\n\n{pages}{self._computation_brief(computation)}"
        )

        jobs: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
        if self.verifier_a is not None:
            jobs.append(
                ("a", lambda: self._one_verdict(
                    self.verifier_a, "verifier_a", load_prompt("verify"),
                    user, schemas.VERIFY, ok="VALID",
                ))
            )
        if self.verifier_b is not None:
            adversarial = self.settings.verification.verifier_b_adversarial
            jobs.append(
                ("b", lambda: self._one_verdict(
                    self.verifier_b,
                    "verifier_b",
                    load_prompt("verify_adversarial" if adversarial else "verify"),
                    user,
                    schemas.VERIFY_ADVERSARIAL if adversarial else schemas.VERIFY,
                    ok="SUPPORTED" if adversarial else "VALID",
                ))
            )

        # THE VERIFIERS RUN CONCURRENTLY, AND ISOLATION IS WHY THAT IS SAFE.
        # Each sees only question + answer + quotes and never the other's
        # verdict, so there is no ordering between them to preserve - running
        # them in sequence was purely a lost ~17 s per question, on a call that
        # already takes 60-130 s.
        #
        # `copy_context()` carries the eval harness's per-question accounting
        # scope into the worker; without it a verifier's tokens would be billed
        # to no question at all. See CostLedger.
        verdicts: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        if len(jobs) < 2:
            for name, run in jobs:
                verdicts[name], reasons[name] = run()
            return verdicts, reasons

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [
                (name, pool.submit(contextvars.copy_context().run, run))
                for name, run in jobs
            ]
            # Results are collected in declaration order, so the verdict dict
            # is identical whichever verifier finishes first.
            for name, future in futures:
                verdicts[name], reasons[name] = future.result()
        return verdicts, reasons

    @staticmethod
    def _one_verdict(
        provider, stage, system, user, schema, *, ok: str
    ) -> tuple[bool, str]:
        """Verdict AND the one-sentence reason.

        THE REASON IS NOT DECORATION. Verifier rejection is the largest
        source of abstention in this system, and a bare boolean makes that
        undiagnosable: "verifier:a" tells you the score was lost but not
        whether the verifier was RIGHT. Calibration needs to separate a correct
        rejection from an over-strict one, and that distinction lives entirely
        in this sentence.
        """
        try:
            data = provider.complete(
                system=system, user=user, schema=schema, stage=stage
            ).data
        except LLMError as exc:
            # A verifier that cannot run has not approved anything. Fail closed -
            # but say WHY, so an outage is never mistaken for a rejection.
            return False, f"verifier unavailable: {exc}"
        return (
            (data.get("verdict") or "").upper() == ok,
            str(data.get("reasoning") or "").strip(),
        )

    def _abstain(self, reason: str) -> QueryResult:
        return QueryResult(
            status="abstained",
            answer=self.settings.verification.abstain_string,
            abstain_reason=reason,
        )

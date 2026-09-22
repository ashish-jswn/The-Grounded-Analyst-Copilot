#!/usr/bin/env python
"""Batched evaluation with an early stop, full evidence, and a spend guard.

    python scripts/run_batches.py                        # 5, 10, 20, 40, 40...
    python scripts/run_batches.py --sizes 5,10           # just the first 15
    python scripts/run_batches.py --stop-on any          # halt on abstentions too
    python scripts/run_batches.py --shapes numeric       # one answer shape
    python scripts/run_batches.py --resume               # skip what already ran
    python scripts/run_batches.py --max-tokens 2000000   # hard spend ceiling

Every question prints its gold answer, the system's answer, WHY it scored what
it scored, and the verbatim quote it cited — so a run can be read instead of
re-run.

`--stop-on` defaults to `wrong`. An abstention scores 0 and is the
designed-correct outcome when evidence does not verify, and this system
currently over-abstains, so stopping on every non-+1 would halt on question 1
and measure nothing. `--stop-on any` is there when you want the strict reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:                                    # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from analyst_copilot.config import load_settings                       # noqa: E402
from analyst_copilot.container import build_pipeline, load_corpus      # noqa: E402
from analyst_copilot.eval.batch import (                               # noqa: E402
    QuestionRun, StopOn, Tally, plan_batches, render_batch_summary,
    render_question, stop_reason, stratified_sample,
)
from analyst_copilot.eval.gold import load_questions, map_gold_pages   # noqa: E402
from analyst_copilot.eval.judge import build_judge                     # noqa: E402
from analyst_copilot.eval.metering import (                            # noqa: E402
    CostLedger, meter_judge, meter_pipeline,
)
from analyst_copilot.eval.report import EvalReport, split_by_company   # noqa: E402
from analyst_copilot.eval.scorer import (                              # noqa: E402
    Citation, RubricScorer, SystemAnswer,
)
from analyst_copilot.eval.shapes import AnswerShape                    # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="5,10,20,40",
                   help="escalating batch sizes; the last repeats (default 5,10,20,40)")
    p.add_argument("--stop-on", default="wrong",
                   help="none | wrong | wrong+location | any   (default wrong)")
    # MEASURED on this deployment, and it is NOT throttling: 4 concurrent
    # small calls all returned in <7 s, and two concurrent 63k-token extractor
    # calls returned in ~5 s each. Per-question latency is dominated by the
    # SEQUENCE of 5-6 calls (~60 s for a tier-1 question), and a question that
    # escalates to tier 2 runs that sequence twice.
    #
    # 8 workers produced 429s in a measured run, so the ceiling is real but higher
    # than 4. Default 4; raise it only against a measured run, because a 429
    # storm gets recorded as `extractor_error` and would look like the system
    # declining to answer.
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent questions (default 4; 8 produced 429s)")
    p.add_argument("--split", choices=["all", "dev", "blind"], default="all")
    p.add_argument("--limit", type=int, default=0,
                   help="take the FIRST n (ordered by company - rarely what you want)")
    p.add_argument("--sample", type=int, default=0,
                   help="deterministic stratified sample of n, preserving the "
                        "answer-shape mix and spreading companies")
    p.add_argument("--shapes", default="",
                   help="comma-separated: numeric,yes_no,phrase,multi_sentence")
    p.add_argument("--only", default="", help="comma-separated question ids")
    p.add_argument("--out", default=".cache/batch_eval.jsonl")
    p.add_argument("--resume", action="store_true",
                   help="skip questions already present in --out")
    p.add_argument("--no-judge", action="store_true",
                   help="skip the LLM judge; the 84 non-numeric answers then all score wrong")
    p.add_argument("--max-tokens", type=int, default=0, help="stop when exceeded")
    p.add_argument("--max-cost", type=float, default=0.0,
                   help="stop when exceeded (needs eval.pricing in config.yaml)")
    p.add_argument("--quote-chars", type=int, default=240)
    p.add_argument("--token-budget", type=int, default=0,
                   help="override retrieval.assembly_token_budget (0 = config)")
    # ── Calibration overrides ────────────────────────────────────────────
    # These are the knobs the verifier over-abstention lives behind. They are
    # CLI flags rather than config edits so an A/B is a repeatable command and
    # config.yaml keeps recording the SHIPPING configuration, not whatever the
    # last experiment happened to leave behind.
    p.add_argument("--verifier-b-adversarial", choices=["true", "false"], default=None,
                   help="override verification.verifier_b_adversarial for this run")
    p.add_argument("--verifier-policy", choices=["unanimous", "any"], default=None,
                   help="override verification.verifier_policy for this run")
    p.add_argument("--no-verifiers", action="store_true",
                   help="drop both LLM verifiers; deterministic gates G1-G7 still run")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    policy = StopOn.parse(args.stop_on)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    settings = load_settings()
    price_in = settings.eval.price_per_mtok_input
    price_out = settings.eval.price_per_mtok_output
    if args.max_cost and (price_in is None or price_out is None):
        print(
            "--max-cost needs eval.pricing.input_per_mtok / output_per_mtok in "
            "config.yaml. Tokens are measured exactly; dollars are only derived "
            "from a rate you supply. Use --max-tokens instead, or set the rates.",
            file=sys.stderr,
        )
        return 2

    # Apply calibration overrides before anything is built from Settings.
    if args.verifier_b_adversarial is not None or args.verifier_policy is not None:
        settings = replace(
            settings,
            verification=replace(
                settings.verification,
                verifier_b_adversarial=(
                    args.verifier_b_adversarial == "true"
                    if args.verifier_b_adversarial is not None
                    else settings.verification.verifier_b_adversarial
                ),
                verifier_policy=(
                    args.verifier_policy or settings.verification.verifier_policy
                ),
            ),
        )

    if args.token_budget:
        settings = replace(
            settings,
            retrieval=replace(
                settings.retrieval, assembly_token_budget=args.token_budget
            ),
        )

    corpus = load_corpus(settings)
    ledger = CostLedger()
    pipeline = meter_pipeline(
        build_pipeline(settings, corpus, with_verifiers=not args.no_verifiers), ledger
    )
    judge = None if args.no_judge else meter_judge(build_judge(settings), ledger)

    pages_by_doc = {d: sorted(p.items()) for d, p in corpus.pages_by_doc.items()}

    # ---- question set -------------------------------------------------
    questions = load_questions(settings.practice_questions)
    defects = [q for q in questions if q.is_unanswerable]
    questions = [q for q in questions if not q.is_unanswerable]
    for q in questions:
        q.gold_page_seqs, q.mapping_score = map_gold_pages(
            q, pages_by_doc.get(q.doc_name, []), settings.eval.gold_map_min_jaccard
        )

    if args.split != "all":
        dev, blind = split_by_company(questions)
        questions = dev if args.split == "dev" else blind
    if args.shapes:
        wanted = {AnswerShape(s.strip()) for s in args.shapes.split(",") if s.strip()}
        questions = [q for q in questions if q.shape in wanted]
    if args.only:
        ids = {s.strip() for s in args.only.split(",") if s.strip()}
        questions = [q for q in questions if q.qid in ids]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    already: set[str] = set()
    if args.resume and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    already.add(json.loads(line)["qid"])
                except Exception:
                    continue
        before = len(questions)
        questions = [q for q in questions if q.qid not in already]
        print(f"resuming: {before - len(questions)} already scored, {len(questions)} to go")
    elif out.exists():
        out.unlink()

    # Sample BEFORE resume-filtering would distort the mix, and before --limit.
    if args.sample:
        questions = stratified_sample(questions, args.sample)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("nothing to run")
        return 0

    scorer = RubricScorer(
        numeric_tolerance=settings.eval.numeric_tolerance,
        location_containment=settings.eval.location_containment,
        page_seq_slack=settings.eval.page_seq_slack,
        judge=judge,
    )

    # ---- header -------------------------------------------------------
    batches = plan_batches(len(questions), sizes)
    print("=" * 78)
    print("BATCHED EVALUATION")
    print("=" * 78)
    print(f"corpus     {corpus.n_docs} filings, {corpus.n_pages:,} pages")
    shape_mix = Counter(q.shape.value for q in questions)
    print(f"questions  {len(questions)} ({args.split} split)"
          + (f", sample of {args.sample}" if args.sample else "")
          + (f", shapes={args.shapes}" if args.shapes else ""))
    print(f"shape mix  {dict(sorted(shape_mix.items()))}")
    print(f"batches    {[b.size for b in batches]}")
    print(f"stop-on    {policy.value} — {policy.explanation}")
    print(f"judge      {'OFF (non-numeric answers will all score wrong)' if judge is None else 'on'}")
    v = settings.verification
    print(f"verifiers  {'OFF (gates only)' if args.no_verifiers else f'policy={v.verifier_policy}, b_adversarial={v.verifier_b_adversarial}'}")
    if price_in is None or price_out is None:
        print("pricing    unset — reporting TOKENS only "
              "(set eval.pricing in config.yaml for dollars)")
    else:
        print(f"pricing    ${price_in}/Mtok in, ${price_out}/Mtok out")
    print(f"excluded   {len(defects)} corpus-defect questions (evidence absent from the filing)")

    # ---- run ----------------------------------------------------------
    run_tally = Tally()
    report = EvalReport()
    report.unanswerable = [(q.qid, q.unanswerable_reason or "") for q in defects]
    # Gold pages that would not map onto our derived pages. Location cannot be
    # scored for these, so they are REPORTED rather than silently counted as
    # location failures - that would blame the system for a mapping gap.
    report.unmapped = [
        (q.qid, q.doc_name, q.mapping_score) for q in questions if not q.gold_page_seqs
    ]
    halted: str | None = None
    started = time.time()
    fh = out.open("a", encoding="utf-8")

    # Detailed reports print in SUBMISSION order so a run reads top-to-bottom and
    # the early stop is deterministic. But that means one slow question hides the
    # fact that others have finished - on a rate-limited deployment a single call
    # can sit in backoff for minutes, and a silent terminal is indistinguishable
    # from a hang. So completion is logged the moment it happens, out of order.
    done_lock = threading.Lock()
    done_count = [0]

    def note_done(qid: str, score: int, seconds: float) -> None:
        with done_lock:
            done_count[0] += 1
            print(f"    ... {done_count[0]}/{len(questions)} done  "
                  f"{qid} {score:+d} ({seconds:.0f}s)", flush=True)

    def run_one(question):
        ledger.begin()
        t0 = time.time()
        try:
            result = pipeline.answer(question.question)
        except Exception as exc:                       # never lose a row
            note_done(question.qid, 0, time.time() - t0)
            return QuestionRun(
                question=question,
                answer=SystemAnswer(abstained=True),
                scored=scorer.score(question, SystemAnswer(abstained=True), {}),
                abstain_reason=f"pipeline_error:{type(exc).__name__}: {exc}",
                usage=ledger.take(),
                seconds=time.time() - t0,
            )
        answer = SystemAnswer(
            text=result.answer or "",
            abstained=result.status == "abstained",
            clarified=result.status == "clarify",
            citations=[
                Citation(doc_id=c.doc_id, page_seq=c.page_seq, quote=c.quote)
                for c in result.citations
            ],
        )
        gate_detail = ""
        verifiers: dict[str, bool] = {}
        verifier_reasons: dict[str, str] = {}
        for tier in (1, 2):
            for g in result.trace.get(f"tier{tier}_gates") or []:
                if not g["passed"]:
                    gate_detail = f"tier{tier} {g['gate']}: {g['detail']}"
            verifiers = result.trace.get(f"tier{tier}_verifiers") or verifiers
            verifier_reasons = (
                result.trace.get(f"tier{tier}_verifier_reasons") or verifier_reasons
            )
        scored_result = scorer.score(
            question, answer, dict(pages_by_doc.get(question.doc_name, []))
        )
        note_done(question.qid, scored_result.score, time.time() - t0)
        return QuestionRun(
            question=question,
            answer=answer,
            scored=scored_result,
            abstain_reason=result.abstain_reason,
            gate_detail=gate_detail,
            verifiers=verifiers,
            verifier_reasons=verifier_reasons,
            errors=list(result.trace.get("errors") or []),
            usage=ledger.take(),
            seconds=time.time() - t0,
            trace=result.trace,
        )

    for batch in batches:
        if halted:
            break
        slice_ = questions[batch.start:batch.end]
        batch_tally = Tally()
        print(f"\n>>> {batch.label}  ({len(slice_)} questions, {args.workers} workers)")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, q) for q in slice_]
            for offset, future in enumerate(futures):
                if halted:
                    future.cancel()
                    continue
                run = future.result()
                position = f"[{batch.start + offset + 1}/{len(questions)}]"
                print(render_question(
                    run, position,
                    dict(pages_by_doc.get(run.question.doc_name, [])),
                    tolerance=settings.eval.numeric_tolerance,
                    containment=settings.eval.location_containment,
                    page_slack=settings.eval.page_seq_slack,
                    quote_chars=args.quote_chars,
                    price_in=price_in, price_out=price_out,
                ), flush=True)

                fh.write(json.dumps(run.as_row(price_in, price_out)) + "\n")
                fh.flush()
                batch_tally.add(run.scored)
                run_tally.add(run.scored)
                report.add(run.question, run.scored)

                reason = stop_reason(policy, run.scored)
                if reason:
                    halted = f"{run.question.qid}: {reason}"
                spent = ledger.total.cost(price_in, price_out)
                if args.max_tokens and ledger.total.total_tokens > args.max_tokens:
                    halted = (f"token ceiling reached "
                              f"({ledger.total.total_tokens:,} > {args.max_tokens:,})")
                if args.max_cost and spent is not None and spent > args.max_cost:
                    halted = f"cost ceiling reached (${spent:.2f} > ${args.max_cost:.2f})"

        print(render_batch_summary(
            batch, batch_tally, run_tally, ledger.total,
            price_in=price_in, price_out=price_out,
        ))

    fh.close()

    # ---- final ---------------------------------------------------------
    print()
    print(report.render())

    print("\n" + "=" * 78)
    print("SPEND")
    print("=" * 78)
    print(f"{'stage':<16}{'calls':>7}{'errors':>8}{'input':>12}{'output':>10}{'sec':>9}")
    for stage, usage in sorted(ledger.total.by_stage.items()):
        print(f"{stage:<16}{usage.calls:>7}{usage.errors:>8}"
              f"{usage.input_tokens:>12,}{usage.output_tokens:>10,}{usage.seconds:>9.0f}")
    total = ledger.total
    print(f"{'TOTAL':<16}{total.calls:>7}{total.errors:>8}"
          f"{total.input_tokens:>12,}{total.output_tokens:>10,}{total.seconds:>9.0f}")
    spent = total.cost(price_in, price_out)
    if spent is not None:
        per_q = spent / run_tally.n if run_tally.n else 0.0
        print(f"\nestimated ${spent:.2f} for {run_tally.n} questions "
              f"(${per_q:.4f} each -> ~${per_q * 136:.2f} for a full 136-question run)")
    else:
        per_q = total.total_tokens / run_tally.n if run_tally.n else 0
        print(f"\n{total.total_tokens:,} tokens for {run_tally.n} questions "
              f"({per_q:,.0f} each -> ~{per_q * 136:,.0f} for a full run)")

    print(f"\nwall clock {time.time() - started:.0f}s   detail -> {out}")
    if halted:
        print(f"\nHALTED EARLY: {halted}")
        print(f"remaining questions were not run. Re-run with --resume to continue, "
              f"or --stop-on none to run through.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

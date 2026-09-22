#!/usr/bin/env python
"""Score the not-found / unanswerable set.

    python scripts/run_negatives_eval.py --workers 3

REPORTED SEPARATELY from the 136-question accuracy results, never blended
into a single headline number. The two directions must be read
together:

  * a system that refuses everything has a 0% false-answer rate and is worthless
  * a system that answers everything maximises -1

The pair is the calibration curve, and the operating point is chosen on it.

The metric that matters here is the FALSE-ANSWER RATE: negatives answered
instead of refused. Target < 5%.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyst_copilot.config import load_settings                # noqa: E402
from analyst_copilot.container import build_pipeline, load_corpus  # noqa: E402
from analyst_copilot.eval.negatives import load                 # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="evaluation/negatives/negatives.jsonl")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=".cache/negatives_eval.json")
    # Same calibration overrides as scripts/run_batches.py, so BOTH axes of
    # the curve can be measured under one configuration without editing
    # config.yaml. Score alone is not an operating point; the pair is.
    parser.add_argument("--verifier-b-adversarial", choices=["true", "false"],
                        default=None)
    parser.add_argument("--verifier-policy", choices=["unanimous", "any"],
                        default=None)
    args = parser.parse_args()

    settings = load_settings()
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
    print(f"verifiers: policy={settings.verification.verifier_policy}, "
          f"b_adversarial={settings.verification.verifier_b_adversarial}")
    negatives = load(Path(args.path))
    if args.limit:
        negatives = negatives[: args.limit]

    reviewed = sum(1 for n in negatives if n.reviewed)
    if reviewed < len(negatives):
        print(
            f"WARNING: only {reviewed}/{len(negatives)} negatives are hand-reviewed. "
            "An item that is actually answerable teaches the system to refuse "
            "something it should have answered.\n"
        )

    pipeline = build_pipeline(settings, load_corpus(settings))
    abstain_string = settings.verification.abstain_string

    def run(negative):
        try:
            result = pipeline.answer(negative.question)
        except Exception as exc:
            return negative, "error", str(exc), None
        return negative, result.status, result.abstain_reason, result.answer

    rows = []
    by_category: dict[str, Counter] = defaultdict(Counter)
    gate_credit: Counter = Counter()
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (negative, status, reason, answer) in enumerate(
            pool.map(run, negatives), start=1
        ):
            # A clarifying question is NOT a false answer: the system declined
            # to guess, which is the behaviour we want.
            refused = status in ("abstained", "clarify")
            exact = (answer or "").strip() == abstain_string if status == "abstained" else True

            by_category[negative.category]["n"] += 1
            by_category[negative.category]["refused" if refused else "answered"] += 1
            if not exact:
                by_category[negative.category]["paraphrased_refusal"] += 1
            if refused and reason:
                gate_credit[reason] += 1

            rows.append(
                {
                    "nid": negative.nid,
                    "category": negative.category,
                    "question": negative.question,
                    "status": status,
                    "refused": refused,
                    "abstain_reason": reason,
                    "answer": answer,
                }
            )
            print(
                f"[{i}/{len(negatives)}] {negative.nid} {negative.category:34s} "
                f"{'REFUSED' if refused else 'ANSWERED <-- false answer'} "
                f"{reason or ''}",
                flush=True,
            )

    total = len(rows)
    answered = sum(1 for r in rows if not r["refused"])
    print("\n" + "=" * 78)
    print("NOT-FOUND / UNANSWERABLE SET  (reported separately from the 136)")
    print("=" * 78)
    print(f"{'category':36s}{'n':>5}{'refused':>9}{'answered':>10}{'false rate':>12}")
    print("-" * 78)
    for category in sorted(by_category):
        c = by_category[category]
        n = c["n"]
        print(
            f"{category:36s}{n:>5}{c['refused']:>9}{c['answered']:>10}"
            f"{100 * c['answered'] / n:>11.1f}%"
        )
    print("-" * 78)
    print(
        f"{'OVERALL':36s}{total:>5}{total - answered:>9}{answered:>10}"
        f"{100 * answered / max(total, 1):>11.1f}%"
    )
    print(f"\nFALSE-ANSWER RATE: {100 * answered / max(total, 1):.1f}%   target < 5%")
    print(f"correct-refusal rate: {100 * (total - answered) / max(total, 1):.1f}%")

    print("\nwhich check caught each refusal (drives the ablation table):")
    for reason, n in gate_credit.most_common():
        print(f"  {reason:34s} {n}")

    print(f"\nwall clock {time.time() - started:.0f}s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"per-question detail -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

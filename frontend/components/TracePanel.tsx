"use client";

/**
 * "How this was answered" — the trace told as steps a reader can follow, not
 * a JSON dump and not a wall of gate codes.
 *
 * The audience is an analyst deciding whether to trust the reply, so each
 * line says what the SYSTEM DID in plain words: which filings it considered,
 * how much it read, what was checked, who signed off. The most valuable line
 * is the failure one — when a reviewer rejects a draft answer, its stated
 * reason is why the user is looking at a decline, so it is shown in full.
 *
 * Every field is read defensively: the trace grows keys as stages are added,
 * and this panel must never be the reason a reply fails to render. The raw
 * JSON survives for developers, one quiet toggle deeper.
 */

import type { Gate, Verdicts } from "@/lib/api";

/* The deterministic checks, in words. Codes without an entry fall back to the
   code itself rather than an invented description. */
const GATE_WORDS: Record<string, string> = {
  G1: "every quote appears verbatim on its cited page",
  G1b: "the stated figure appears inside its own quote",
  G6: "the arithmetic was recomputed independently",
};

function num(t: Record<string, unknown>, k: string): number | null {
  const v = t[k];
  return typeof v === "number" ? v : null;
}

function strList(v: unknown): string[] {
  return Array.isArray(v) ? v.map(String) : [];
}

function gateList(v: unknown): Gate[] {
  if (!Array.isArray(v)) return [];
  return v.flatMap((g) =>
    g && typeof g === "object" && "gate" in g
      ? [
          {
            gate: String((g as Gate).gate),
            passed: Boolean((g as Gate).passed),
            detail: String((g as Gate).detail ?? ""),
          },
        ]
      : [],
  );
}

/** "COCACOLA_2022_10K" -> "COCACOLA 2022 10-K" — readable, still unambiguous. */
function docLabel(id: string): string {
  return id.replace(/_/g, " ").replace(/\b10(K|Q)\b/, "10-$1");
}

interface Step {
  icon: string;
  kind: "info" | "pass" | "fail";
  text: React.ReactNode;
  reason?: string;
}

function buildSteps(trace: Record<string, unknown>): Step[] {
  const steps: Step[] = [];

  const docs = strList(trace.candidate_docs);
  if (docs.length > 0) {
    steps.push({
      icon: "🔎",
      kind: "info",
      text: (
        <>
          Narrowed the corpus to <b>{docs.length} candidate filing{docs.length === 1 ? "" : "s"}</b>:{" "}
          {docs.map(docLabel).join(", ")}
        </>
      ),
    });
  }

  for (const tier of [1, 2]) {
    const pages = num(trace, `tier${tier}_pages`);
    if (pages == null) continue;

    if (tier === 2) {
      steps.push({
        icon: "↻",
        kind: "info",
        text: (
          <>
            The first pass did not produce a provable answer, so the search was{" "}
            <b>widened and run again</b>
          </>
        ),
      });
    }

    steps.push({
      icon: "📄",
      kind: "info",
      text: (
        <>
          Read <b>{pages} pages</b> chosen by document structure and search
        </>
      ),
    });

    const gates = gateList(trace[`tier${tier}_gates`]);
    if (gates.length > 0) {
      const failed = gates.filter((g) => !g.passed);
      if (failed.length === 0) {
        steps.push({
          icon: "✓",
          kind: "pass",
          text: (
            <>
              Passed all <b>{gates.length} automatic checks</b> — including that{" "}
              {GATE_WORDS.G1}
            </>
          ),
        });
      } else {
        for (const g of failed) {
          steps.push({
            icon: "✗",
            kind: "fail",
            text: (
              <>
                Failed check <b>{g.gate}</b>
                {GATE_WORDS[g.gate] ? <> — {GATE_WORDS[g.gate]}</> : null}
              </>
            ),
            reason: g.detail || undefined,
          });
        }
      }
    }

    const verifiers = trace[`tier${tier}_verifiers`] as Verdicts | undefined;
    if (verifiers && typeof verifiers === "object") {
      const entries = Object.entries(verifiers);
      const reasons = (trace[`tier${tier}_verifier_reasons`] ?? {}) as Record<string, string>;
      const rejected = entries.filter(([, ok]) => !ok);
      if (entries.length > 0 && rejected.length === 0) {
        steps.push({
          icon: "✓",
          kind: "pass",
          text: (
            <>
              <b>{entries.length === 1 ? "An independent reviewer" : `${entries.length} independent reviewers`}</b>{" "}
              read the draft against the quotes and accepted it
            </>
          ),
        });
      }
      for (const [name] of rejected) {
        steps.push({
          icon: "✗",
          kind: "fail",
          text: (
            <>
              An independent reviewer <b>rejected the draft answer</b>
            </>
          ),
          reason: reasons?.[name] ? String(reasons[name]) : undefined,
        });
      }
    }
  }

  return steps;
}

export default function TracePanel({
  trace,
  declined,
}: {
  trace: Record<string, unknown>;
  declined?: boolean;
}) {
  const steps = buildSteps(trace);
  const latency = num(trace, "latency_ms");

  if (steps.length === 0) {
    // Nothing narratable — offer only the developer view.
    return (
      <details className="dev">
        <summary>Details</summary>
        <pre className="dev-json mono">{JSON.stringify(trace, null, 2)}</pre>
      </details>
    );
  }

  return (
    <details className="how">
      <summary>{declined ? "Why it declined" : "How this was answered"}</summary>
      <ol className="how-steps">
        {steps.map((s, i) => (
          <li key={i} className={`how-step ${s.kind}`}>
            <span className="how-ico" aria-hidden="true">{s.icon}</span>
            <span>
              {s.text}
              {s.reason ? <div className="how-reason">{s.reason}</div> : null}
            </span>
          </li>
        ))}
        {declined ? (
          <li className="how-step">
            <span className="how-ico" aria-hidden="true">■</span>
            <span>
              Rather than guess, it answered <b>“Not found in this filing.”</b>
            </span>
          </li>
        ) : null}
      </ol>
      {latency != null ? (
        <div className="answer-meta" style={{ marginTop: 8 }}>
          {(latency / 1000).toFixed(0)}s end to end
        </div>
      ) : null}
      <details className="dev">
        <summary>Developer details</summary>
        <pre className="dev-json mono">{JSON.stringify(trace, null, 2)}</pre>
      </details>
    </details>
  );
}

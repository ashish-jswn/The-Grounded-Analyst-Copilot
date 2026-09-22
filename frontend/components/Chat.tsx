"use client";

/**
 * Core feature 2: the chat box.
 *
 * NO CLIENT-SIDE TIMEOUT. A question takes 60-130 s — the pipeline runs
 * 5-9 sequential model calls and may escalate to a second retrieval tier.
 * Cutting the request short would render identically to the system declining,
 * and telling those two apart is the entire point of this product.
 *
 * The transcript keeps the FULL response for every turn, not just the answer
 * text, so the sources stay readable after the next question.
 */

import { useEffect, useRef, useState } from "react";
import { api, type AnswerResponse } from "@/lib/api";
import AnswerCard from "./AnswerCard";

interface Turn {
  id: number;
  question: string;
  response?: AnswerResponse;
  error?: string;
  seconds?: number;
}

// THESE FOUR ARE MEASURED, NOT CHOSEN BY EYE. Each scores +1 - correct
// answer AND correct location - in the practice-set evaluation, and they span
// three answer shapes. The previous list included "What was Coca Cola's total
// revenue in FY2024?", which the corpus CANNOT answer (it holds Coca-Cola
// 2022, 2021 and 2017), so the first impression was a decline.
const EXAMPLES = [
  "If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?",
  "Does Adobe have an improving Free cashflow conversion as of FY2022?",
  "What are the major products and services that AMD sells as of FY22?",
  "What are the geographies that American Express primarily operates in as of 2022?",
];

/* The waiting label follows the pipeline's real stage order (route → retrieve
   → draft → verify). The elapsed thresholds are typical, not signals from the
   backend — which is why every label stays hedged with "…" rather than
   claiming a specific stage has completed. */
function stageLabel(elapsed: number): string {
  if (elapsed < 8) return "Finding the right filing…";
  if (elapsed < 35) return "Reading the relevant pages…";
  if (elapsed < 70) return "Drafting an answer from the evidence…";
  return "Verifying quotes and arithmetic…";
}

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const endRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, pending]);

  // A visible elapsed counter: two minutes of silence looks broken otherwise.
  useEffect(() => {
    if (!pending) return;
    setElapsed(0);
    const tick = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(tick);
  }, [pending]);

  async function ask(question: string) {
    const trimmed = question.trim();
    if (!trimmed || pending) return;

    const id = nextId.current++;
    setTurns((t) => [...t, { id, question: trimmed }]);
    setDraft("");
    setPending(true);
    const started = Date.now();

    try {
      const response = await api.ask(trimmed);
      setTurns((t) =>
        t.map((turn) =>
          turn.id === id
            ? { ...turn, response, seconds: (Date.now() - started) / 1000 }
            : turn,
        ),
      );
    } catch (e) {
      setTurns((t) =>
        t.map((turn) =>
          turn.id === id
            ? {
                ...turn,
                error: e instanceof Error ? e.message : String(e),
                seconds: (Date.now() - started) / 1000,
              }
            : turn,
        ),
      );
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <div className="transcript">
        {turns.length === 0 && !pending ? (
          <div className="empty">
            <h2>Ask about any filing</h2>
            <p>
              78 SEC filings, 33 companies. You don’t pick a document — the
              system finds the filing and the page, and every answer carries a
              verbatim quote, or it declines.
            </p>
            <div className="examples">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  className="example"
                  onClick={() => ask(example)}
                  disabled={pending}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="thread">
            {turns.map((turn) => (
              <div key={turn.id}>
                <div className="turn-question">{turn.question}</div>
                {turn.response || turn.error ? (
                  <div className="turn-answer" style={{ marginTop: 18 }}>
                    <AnswerCard response={turn.response} error={turn.error} />
                    {turn.seconds != null ? (
                      <div className="answer-meta">
                        {turn.seconds.toFixed(0)}s
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ))}
            {pending ? (
              <div className="thinking">
                <span className="dots">
                  <i />
                  <i />
                  <i />
                </span>
                {stageLabel(elapsed)}
                <span className="thinking-elapsed">{elapsed}s</span>
              </div>
            ) : null}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <div className="composer">
        <div className="composer-inner">
          <div className="composer-row">
            <textarea
              rows={1}
              value={draft}
              placeholder="Ask an analyst question…"
              disabled={pending}
              onChange={(e) => {
                setDraft(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(draft);
                }
              }}
            />
            <button
              className="btn-send"
              aria-label="Send"
              disabled={pending || !draft.trim()}
              onClick={() => ask(draft)}
            >
              <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true">
                <path
                  d="M8 13V3M3.5 7.5 8 3l4.5 4.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
          <div className="composer-foot">
            Answers take a minute or two — every one is verified against the
            filing before it’s shown, or it declines.
          </div>
        </div>
      </div>
    </>
  );
}

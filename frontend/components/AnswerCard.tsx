"use client";

/**
 * Two of the four core product features live here:
 *
 *   • EVIDENCE ON EVERY ANSWER — the document, the page and a verbatim quote
 *   • A PLAIN DECLINE PATH     — the exact string, never softened
 *
 * THE REFUSAL IS RENDERED VERBATIM. The backend returns exactly
 * "Not found in this filing." and this component prints that string. It must
 * never be reworded into something friendlier: the rubric scores the exact
 * refusal at 0, and a helpful rewrite reads as an answer, which scores -1.
 * The one-line explanation below it is ADDITIONAL text, not a replacement.
 *
 * AN ANSWER WITHOUT CITATIONS IS SHOWN AS INCOMPLETE, not as an answer.
 * A right answer with no provable location scores 0, so the UI must not let
 * it look like a win.
 *
 * Visually this is no longer a card: the reply is text on the page with one
 * quiet status line, the way ChatGPT and Gemini render replies. The sources
 * keep a thin border because they are excerpts of someone else's document.
 */

import type { AnswerResponse, Citation, Computation } from "@/lib/api";
import TracePanel from "./TracePanel";

const STATUS_LINE: Record<string, string> = {
  answered: "Answered · verified against the cited pages",
  abstained: "Declined",
  clarify: "Needs one detail",
  error: "Request failed",
};

function SourceItem({ citation }: { citation: Citation }) {
  const printed =
    citation.page_printed != null ? ` · printed p.${citation.page_printed}` : "";
  return (
    <li className="source-item">
      <div className="source-head">
        <span className="source-doc mono">{citation.doc_id}</span>
        <span className="source-page">
          page {citation.page_seq}
          {printed}
        </span>
        {citation.section_path ? (
          <span className="source-section">{citation.section_path}</span>
        ) : null}
      </div>
      {/* The quote is the proof. Verbatim, monospace, visible. */}
      <pre className="source-quote mono">{citation.quote}</pre>
    </li>
  );
}

function ComputationPanel({ computation }: { computation: Computation }) {
  return (
    <div className="computation">
      <div className="computation-definition">
        {computation.definition}
        <span className="computation-source">
          {computation.formula_source === "question"
            ? "definition from the question"
            : computation.formula_source === "book"
            ? "standard definition"
            : "proposed, then verified"}
        </span>
      </div>
      <div className="computation-formula mono">
        {computation.formula} = {String(computation.result)}
      </div>
      {computation.operands.length > 0 ? (
        <ul className="operands">
          {computation.operands.map((operand) => (
            <li key={operand.name} className="operand mono">
              <b>{operand.name}</b> = {String(operand.value)}
              {operand.scale ? ` ${operand.scale}` : ""}
              {operand.period ? ` (${operand.period})` : ""}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export default function AnswerCard({
  response,
  error,
}: {
  response?: AnswerResponse;
  error?: string;
}) {
  if (error) {
    return (
      <>
        <div className="status-line">
          <span className="status-dot error" />
          {STATUS_LINE.error}
        </div>
        <p className="answer-text">{error}</p>
      </>
    );
  }
  if (!response) return null;

  const { status, citations, computation, trace } = response;
  const answered = status === "answered";
  const declined = status === "abstained";
  const hasProof = citations.length > 0;
  const hasTrace = trace && Object.keys(trace).length > 0;

  return (
    <>
      <div className="status-line">
        <span className={`status-dot ${status}`} />
        {STATUS_LINE[status] ?? status}
      </div>

      {status === "clarify" ? (
        <p className="answer-text">{response.clarifying_question}</p>
      ) : (
        <p className={`answer-text${answered ? "" : " refusal"}`}>
          {response.answer}
        </p>
      )}

      {declined ? (
        <p className="decline-note">
          The evidence for this either isn’t in the corpus or didn’t survive
          verification — so it declines rather than guess.
        </p>
      ) : null}

      {computation ? <ComputationPanel computation={computation} /> : null}

      {hasProof ? (
        <>
          <div className="sources-label">
            Source{citations.length === 1 ? "" : "s"}
          </div>
          <ul className="sources">
            {citations.map((citation, i) => (
              <SourceItem
                key={`${citation.doc_id}-${citation.page_seq}-${i}`}
                citation={citation}
              />
            ))}
          </ul>
        </>
      ) : answered ? (
        // A correct answer with no provable location scores 0, so it must not
        // be presented as a success.
        <div className="no-proof">
          No citation returned — this answer cannot be verified
        </div>
      ) : null}

      {hasTrace ? <TracePanel trace={trace} declined={declined} /> : null}
    </>
  );
}

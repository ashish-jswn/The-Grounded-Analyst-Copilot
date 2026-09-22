/**
 * The API contract, mirrored from `src/analyst_copilot/api/schemas.py`.
 *
 * These types are a COPY of the pydantic models, not an interpretation of them.
 * If `schemas.py` changes, this file changes with it — that file's docstring is
 * explicit that it is the contract a separate UI binds against.
 */

/** One deterministic gate's outcome, as recorded in the trace. */
export interface Gate {
  gate: string;
  passed: boolean;
  detail?: string;
}

/** Verifier name -> whether it accepted the answer. */
export type Verdicts = Record<string, boolean>;

export type AnswerStatus = "answered" | "abstained" | "clarify";

export type IngestState =
  | "queued" | "parsing" | "tables" | "sections"
  | "xbrl" | "summaries" | "embedding" | "ready" | "failed";

/** doc + derived page + printed footer page + verbatim quote. */
export interface Citation {
  doc_id: string;
  company?: string | null;
  form_type?: string | null;
  period?: string | null;
  page_seq: number;
  /** The filing's own footer number. Nullable — Nike prints one on 1 of 102 pages. */
  page_printed?: number | null;
  section_path?: string | null;
  quote: string;
}

export interface Operand {
  name: string;
  value: string | number;
  unit?: string | null;
  scale?: string | null;
  period?: string | null;
  citation?: Citation | null;
}

export interface Computation {
  metric?: string | null;
  /** Always present: an answer that carries its own definition can be audited. */
  definition: string;
  formula: string;
  formula_source: "question" | "book" | "llm_proposed";
  operands: Operand[];
  result: string | number;
}

export interface AnswerResponse {
  status: AnswerStatus;
  answer?: string | null;
  clarifying_question?: string | null;
  citations: Citation[];
  computation?: Computation | null;
  /** The gate id that failed, e.g. "G1" — so a refusal is explainable. */
  abstain_reason?: string | null;
  trace: Record<string, unknown>;
}

export interface IngestStatus {
  doc_id: string;
  status: IngestState;
  progress: number;
  page_count?: number | null;
  error?: string | null;
}

export interface FilingSummary {
  doc_id: string;
  company_slug: string;
  form_type: string;
  period_label?: string | null;
  page_count?: number | null;
  ingest_status: string;
}

export interface CorpusStats {
  filings: number;
  pages: number;
  tables: number;
  table_cells: number;
  sections: number;
}

/**
 * The exact refusal string. NEVER paraphrase it, and never render a friendlier
 * variant: the rubric scores this string, and "I couldn't find that" scores as
 * a wrong answer rather than an honest decline.
 */
export const NOT_FOUND = "Not found in this filing.";

class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail ?? detail;
    } catch {
      /* a non-JSON error body is still an error */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  stats: () => request<CorpusStats>("/stats"),

  filings: () => request<FilingSummary[]>("/filings"),

  filingStatus: (docId: string) =>
    request<IngestStatus>(`/filings/${encodeURIComponent(docId)}`),

  /**
   * A question can take 60–130 s: the pipeline makes 5–9 sequential model calls
   * and may escalate to a second retrieval tier. There is deliberately NO
   * client-side timeout — cutting the request off would look identical to the
   * system declining, which is the one distinction this product exists to make.
   */
  ask: (question: string) =>
    request<AnswerResponse>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),

  addFiling: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    // No Content-Type header: the browser must set the multipart boundary.
    return request<IngestStatus>("/filings", { method: "POST", body: form });
  },
};

export { ApiError };

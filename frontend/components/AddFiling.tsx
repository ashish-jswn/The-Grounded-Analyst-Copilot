"use client";

/**
 * Core feature 1: "Add filing", with a VISIBLE PROCESSING STATUS.
 *
 * THE PROGRESS BAR IS REAL, NOT A SPINNER. It polls
 * GET /filings/{doc_id}, which reads `ingest_status` / `ingest_progress` as the
 * backend writes them through S1..S12. Ingest measures ~3-12 s per filing
 * (against a 10-minute target), so the honest thing to show is the stage the
 * parser is actually in.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type IngestStatus } from "@/lib/api";

const STAGE_LABEL: Record<string, string> = {
  queued: "Queued",
  parsing: "Splitting pages",
  tables: "Parsing tables",
  sections: "Building the section tree",
  xbrl: "Extracting XBRL facts",
  summaries: "Summarising pages",
  embedding: "Embedding",
  ready: "Ready",
  failed: "Failed",
};

/** The upload contract: the filename carries the catalog metadata. */
const NAME_HINT = "COMPANY_YEAR_FORM.htm";

export default function AddFiling({ onIngested }: { onIngested: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Polling must stop when the component goes away, or it leaks a timer that
  // keeps hitting the API after the page has moved on.
  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  const poll = useCallback(
    async (docId: string, attempt = 0) => {
      try {
        const next = await api.filingStatus(docId);
        setStatus(next);
        if (next.status === "ready") {
          setBusy(false);
          onIngested();
          return;
        }
        if (next.status === "failed") {
          setBusy(false);
          setError(next.error ?? "ingest failed");
          return;
        }
      } catch {
        // A transient poll failure is not an ingest failure — the row may not
        // be visible yet. Keep polling; the attempt cap ends it.
      }
      if (attempt > 300) {
        setBusy(false);
        setError("ingest did not finish within 5 minutes");
        return;
      }
      timer.current = setTimeout(() => poll(docId, attempt + 1), 1000);
    },
    [onIngested],
  );

  async function submit() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const started = await api.addFiling(file);
      setStatus(started);
      poll(started.doc_id);
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const pct = Math.round(Math.min(status?.progress ?? 0, 1) * 100);
  const done = status?.status === "ready";
  const failed = status?.status === "failed";

  return (
    <div>
      <label className="file-drop">
        <input
          type="file"
          accept=".htm,.html"
          disabled={busy}
          onChange={(e) => {
            setFile(e.target.files?.[0] ?? null);
            setStatus(null);
            setError(null);
          }}
        />
        {file ? "Choose a different file" : "Upload an SEC filing (.htm)"}
        {/* The naming contract lives on the control it applies to, not in a
            paragraph above it: the filename carries the catalog metadata. */}
        <div className="file-drop-sub mono">{NAME_HINT}</div>
        {file ? <div className="file-name mono">{file.name}</div> : null}
      </label>

      <button className="btn-primary" disabled={!file || busy} onClick={submit}>
        {busy ? "Processing…" : "Add filing"}
      </button>

      {status ? (
        <div className="progress">
          <div className="progress-head">
            <span className="progress-stage">
              {STAGE_LABEL[status.status] ?? status.status}
            </span>
            <span className="progress-pct">{pct}%</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill${done ? " done" : failed ? " failed" : ""}`}
              style={{ width: `${done ? 100 : pct}%` }}
            />
          </div>
          {done ? (
            <div className="progress-note ok">
              {status.doc_id} indexed
              {status.page_count ? ` · ${status.page_count} pages` : ""}
            </div>
          ) : (
            <div className="progress-note mono">{status.doc_id}</div>
          )}
        </div>
      ) : null}

      {error ? <div className="progress-note bad">{error}</div> : null}
    </div>
  );
}

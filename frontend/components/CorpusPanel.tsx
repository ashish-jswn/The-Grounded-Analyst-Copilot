"use client";

/** Corpus state, plus the "Add filing" control. Refreshes after an ingest so a
 *  newly uploaded document is visibly in the corpus, not just claimed to be.
 *
 *  The list is grouped by company: 78 raw doc_ids in one column is a wall,
 *  33 company rows that open on click is a directory. While a filter is
 *  typed, matching groups are forced open so the results are visible without
 *  a second click.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type CorpusStats, type FilingSummary } from "@/lib/api";
import AddFiling from "./AddFiling";

/** "ACTIVISIONBLIZZARD_2019_10K" -> label pieces a human can scan. */
function filingLabel(docId: string): string {
  const parts = docId.split("_");
  return parts
    .slice(1)
    .join(" ")
    .replace(/\b10K\b/, "10-K")
    .replace(/\b10Q\b/, "10-Q")
    .replace(/\b8K\b/, "8-K");
}

export default function CorpusPanel() {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [filings, setFilings] = useState<FilingSummary[]>([]);
  const [offline, setOffline] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [s, f] = await Promise.all([api.stats(), api.filings()]);
      setStats(s);
      setFilings(f);
      setOffline(null);
    } catch (e) {
      setOffline(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const needle = query.trim().toLowerCase();

  const groups = useMemo(() => {
    const byCompany = new Map<string, FilingSummary[]>();
    for (const f of filings) {
      if (needle && !f.doc_id.toLowerCase().includes(needle)) continue;
      const company = f.doc_id.split("_")[0] || f.doc_id;
      const bucket = byCompany.get(company);
      if (bucket) bucket.push(f);
      else byCompany.set(company, [f]);
    }
    return [...byCompany.entries()];
  }, [filings, needle]);

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" />
        <span className="brand-name">Analyst Copilot</span>
      </div>

      {offline ? (
        <div className="banner">
          Backend unreachable — start it with
          <br />
          <code className="mono">
            uvicorn analyst_copilot.api.main:app --port 8000
          </code>
        </div>
      ) : null}

      <div className="corpus-line">
        {stats ? (
          <>
            <b>{stats.filings.toLocaleString()}</b> filings ·{" "}
            <b>{stats.pages.toLocaleString()}</b> pages
          </>
        ) : (
          "loading corpus…"
        )}
      </div>

      <div className="section-label">Add filing</div>
      <AddFiling onIngested={refresh} />

      <div className="section-label">
        Filings
        <span className="section-count">{filings.length}</span>
      </div>
      <input
        className="filter"
        value={query}
        placeholder="Search filings…"
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="filings">
        {groups.map(([company, docs]) => (
          <details
            className="company"
            key={company}
            /* A typed filter forces matching groups open; clearing it lets the
               native toggle take over again. */
            open={needle ? true : undefined}
          >
            <summary>
              <span className="company-name">{company}</span>
              <span className="company-count">{docs.length}</span>
            </summary>
            {docs.map((filing) => (
              <div className="filing" key={filing.doc_id} title={filing.doc_id}>
                <span className="filing-id">{filingLabel(filing.doc_id)}</span>
                <span
                  className="filing-meta"
                  title={
                    filing.ingest_status === "ready"
                      ? `${filing.page_count} pages`
                      : filing.ingest_status
                  }
                >
                  {filing.ingest_status === "ready"
                    ? `${filing.page_count} pp`
                    : filing.ingest_status}
                </span>
              </div>
            ))}
          </details>
        ))}
        {needle && groups.length === 0 ? (
          <div className="filing-empty">no filing matches “{query}”</div>
        ) : null}
      </div>
    </aside>
  );
}

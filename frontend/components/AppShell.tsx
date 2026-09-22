"use client";

/**
 * The app frame: a collapsible sidebar beside the chat, as in ChatGPT/Claude,
 * plus the light/dark toggle.
 *
 * The collapse state lives here rather than in CorpusPanel because BOTH sides
 * need it — the sidebar to hide, the topbar to show the reopen control. Both
 * preferences are remembered per browser; reads and writes are wrapped because
 * storage throws outright in some embedded views rather than merely returning
 * null.
 *
 * The theme is read from document.documentElement, where an inline script in
 * layout.tsx already stamped it before first paint — asking matchMedia again
 * here could disagree with what is actually on screen.
 */

import { useEffect, useState } from "react";
import Chat from "./Chat";
import CorpusPanel from "./CorpusPanel";

const SIDEBAR_KEY = "ac.sidebar.collapsed";
const THEME_KEY = "ac.theme";

export default function AppShell() {
  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    try {
      if (window.localStorage.getItem(SIDEBAR_KEY) === "1") setCollapsed(true);
    } catch {
      /* storage unavailable — the default is fine */
    }
    const current = document.documentElement.dataset.theme;
    if (current === "dark") setTheme("dark");
  }, []);

  function toggleSidebar() {
    setCollapsed((c) => {
      const next = !c;
      try {
        window.localStorage.setItem(SIDEBAR_KEY, next ? "1" : "0");
      } catch {
        /* not worth failing a click over */
      }
      return next;
    });
  }

  function toggleTheme() {
    setTheme((t) => {
      const next = t === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        window.localStorage.setItem(THEME_KEY, next);
      } catch {
        /* the page is still themed for this visit */
      }
      return next;
    });
  }

  return (
    <div className={`shell${collapsed ? " collapsed" : ""}`}>
      {/* Kept mounted while collapsed: unmounting would refetch the corpus and
          drop any in-flight upload progress every time the panel is toggled. */}
      <CorpusPanel />

      {/* Closes the drawer when the sidebar is an overlay (narrow screens).
          Inert at desktop widths, where the sidebar is part of the grid. */}
      <button
        className="scrim"
        aria-hidden={collapsed}
        tabIndex={-1}
        onClick={() => setCollapsed(true)}
      />

      <section className="main">
        <header className="topbar">
          <button
            className="icon-btn"
            onClick={toggleSidebar}
            aria-label={collapsed ? "Show corpus panel" : "Hide corpus panel"}
            title={collapsed ? "Show corpus panel" : "Hide corpus panel"}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
              <rect
                x="1.25"
                y="2.25"
                width="13.5"
                height="11.5"
                rx="2.25"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.3"
              />
              <line
                x1="6.25"
                y1="2.25"
                x2="6.25"
                y2="13.75"
                stroke="currentColor"
                strokeWidth="1.3"
              />
            </svg>
          </button>
          <h1>The Analyst Copilot</h1>
          <button
            className="icon-btn theme-toggle"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? (
              /* sun */
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <circle cx="8" cy="8" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.3" />
                <g stroke="currentColor" strokeWidth="1.3" strokeLinecap="round">
                  <line x1="8" y1="0.9" x2="8" y2="2.5" />
                  <line x1="8" y1="13.5" x2="8" y2="15.1" />
                  <line x1="0.9" y1="8" x2="2.5" y2="8" />
                  <line x1="13.5" y1="8" x2="15.1" y2="8" />
                  <line x1="3" y1="3" x2="4.1" y2="4.1" />
                  <line x1="11.9" y1="11.9" x2="13" y2="13" />
                  <line x1="3" y1="13" x2="4.1" y2="11.9" />
                  <line x1="11.9" y1="4.1" x2="13" y2="3" />
                </g>
              </svg>
            ) : (
              /* moon */
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <path
                  d="M13.3 9.6a5.6 5.6 0 0 1-6.9-6.9 5.6 5.6 0 1 0 6.9 6.9z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  strokeLinejoin="round"
                />
              </svg>
            )}
          </button>
        </header>
        <Chat />
      </section>
    </div>
  );
}

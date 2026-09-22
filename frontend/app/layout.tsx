import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Analyst Copilot",
  description:
    "Question answering over SEC filings with an exact evidence location, " +
    "or an honest 'Not found in this filing.'",
};

/**
 * Stamps data-theme on <html> BEFORE first paint, so a dark-mode user never
 * sees a white flash. Runs synchronously at the top of <body>; the stored
 * choice wins, the OS preference is the default. Kept as a string because it
 * must execute before hydration, not as part of it.
 */
const THEME_BOOT = `try {
  var t = localStorage.getItem("ac.theme");
  if (t !== "dark" && t !== "light") {
    t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.documentElement.dataset.theme = t;
} catch (e) { document.documentElement.dataset.theme = "light"; }`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
        {children}
      </body>
    </html>
  );
}

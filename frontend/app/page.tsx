import AppShell from "@/components/AppShell";

/**
 * The four core product features, on one screen:
 *
 *   1. "Add filing" + visible processing status  -> CorpusPanel / AddFiling
 *   2. a chat box                                -> Chat
 *   3. evidence on every answer                  -> AnswerCard
 *   4. a plain decline path                      -> AnswerCard (verbatim string)
 *
 * AppShell is a client component only because the sidebar collapses; the
 * controls themselves are unchanged by it.
 */
export default function Page() {
  return (
    <main>
      <AppShell />
    </main>
  );
}

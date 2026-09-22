DATA
====

company_aliases.yaml
    Company name -> alias table the document router matches against
    (e.g. AMEX, JnJ, JPM). Seeded into Postgres by the migration.

practice-questions.jsonl
    136 analyst questions over the filings, each with its gold answer and the
    passage that proves it. Used ONLY by the evaluation harness (eval/, scripts/)
    - the pipeline never reads it; tests/test_generalisation_guard.py enforces that.

    Fields used:
        question   - the analyst question, as asked
        answer     - the gold answer
        evidence   - the supporting passage, and its page number
        doc_name   - which filing it refers to (matches a file in filings/)
        company, doc_period, doc_type - which company, which year, which form

filings/   (not committed - see .gitignore)
    One SEC filing per file, named COMPANY_YEAR_FORM.htm to match doc_name.
    Download them from EDGAR (https://www.sec.gov/edgar) and set FILINGS_DIR.

SOURCES AND LICENSES
--------------------
Questions: FinanceBench, by Patronus AI
    Islam et al., "FinanceBench: A New Benchmark for Financial Question
    Answering" (arXiv:2311.11944). Licensed CC BY-NC 4.0
    (creativecommons.org/licenses/by-nc/4.0) - non-commercial use only.
    Redistributed here unmodified, with attribution, for non-commercial use.

Filings: U.S. Securities and Exchange Commission, EDGAR (sec.gov/edgar) -
    public disclosure documents.

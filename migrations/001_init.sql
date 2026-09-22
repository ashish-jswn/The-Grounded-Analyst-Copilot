-- The Analyst Copilot — canonical schema
--
-- Must build from zero on an empty database, and re-running it is a no-op.
-- Run against Azure Postgres Flexible Server OR the local docker-compose instance;
-- the schema is identical and uses no Azure-only feature.

-- ─────────────────────────────────────────────────────────────────────────────
-- EXTENSIONS
-- On Azure Flexible Server these must first be added to the server parameter
-- `azure.extensions` (Portal → Server parameters), or CREATE EXTENSION fails.
-- Deliberately NOT used: azure_ai, pg_diskann — they would break local reproducibility.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector 0.8.2 — hnsw + ivfflat
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- fuzzy company / line-item matching
CREATE EXTENSION IF NOT EXISTS unaccent;


-- ═════════════════════════════════════════════════════════════════════════════
-- CATALOG — what the document router filters on
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS filings (
    doc_id          TEXT PRIMARY KEY,           -- '3M_2018_10K'
    file_path       TEXT        NOT NULL,
    content_hash    TEXT        NOT NULL,       -- sha256, makes re-ingest idempotent
    company_name    TEXT        NOT NULL,
    company_slug    TEXT        NOT NULL,
    ticker          TEXT,
    cik             TEXT,
    form_type       TEXT        NOT NULL,       -- '10-K' | '10-Q' | '8-K'
    fiscal_year     INTEGER,
    fiscal_quarter  INTEGER,                    -- NULL for 10-K
    period_label    TEXT,                       -- 'FY2018' | 'Q2 FY2024'
    period_start    DATE,
    period_end      DATE,
    filing_date     DATE,                       -- disambiguates same-company 8-K pairs
    coverage_years  INTEGER[],                  -- a FY2018 10-K covers {2018,2017,2016}
    page_count      INTEGER,
    has_xbrl        BOOLEAN     DEFAULT FALSE,
    ingest_status   TEXT        DEFAULT 'queued',
    ingest_progress REAL        DEFAULT 0.0,    -- 0..1 — drives the visible UI indicator
    ingest_error    TEXT,
    parser_version  TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT filings_status_ck CHECK (ingest_status IN
        ('queued','parsing','tables','sections','xbrl',
         'summaries','embedding','ready','failed'))
);
CREATE INDEX IF NOT EXISTS filings_lookup ON filings (company_slug, form_type, fiscal_year);

-- Closes the 14/136 questions naming a company only by alias (AMEX, JnJ, JPM).
-- Seeded from data/company_aliases.yaml.
CREATE TABLE IF NOT EXISTS company_aliases (
    company_slug TEXT NOT NULL,
    alias        TEXT NOT NULL,
    PRIMARY KEY (company_slug, alias)
);
CREATE INDEX IF NOT EXISTS company_aliases_trgm
    ON company_aliases USING gin (alias gin_trgm_ops);


-- ═════════════════════════════════════════════════════════════════════════════
-- PAGES — THE retrieval unit AND the citation unit
-- There is no `chunks` table: pages ARE the chunks.
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS pages (
    page_id        TEXT PRIMARY KEY,            -- '3M_2018_10K#p59'
    doc_id         TEXT NOT NULL REFERENCES filings ON DELETE CASCADE,
    section_id     TEXT,
    page_seq       INTEGER NOT NULL,            -- derived; ALWAYS present
    page_printed   INTEGER,                     -- filing's own footer; NULLABLE (Nike: 1/102)
    prev_page_id   TEXT,
    next_page_id   TEXT,
    char_start     INTEGER,
    char_end       INTEGER,

    -- The four representations.
    raw_text       TEXT NOT NULL,               -- (1) verbatim. Gate G1 checks quotes against THIS.
                                                --     The ONLY representation the generator ever reads.
    summary        TEXT,                        -- (2) de-noised gloss. NEVER shown to the generator.
    context_header TEXT,                        -- '[3M | 10-K | FY2018 | Item 8 > Cash Flows | p.59]'
    lexical_text   TEXT,                        -- (4) COMPOSITE: context_header + verbatim table/section
                                                --     headers + summary. Not the summary alone — BM25
                                                --     needs exact GAAP strings to survive indexing.
    embedding      vector(1024),                -- (3) over context_header + summary.
                                                --     MUST equal EMBEDDING_DIMENSIONS in .env.
                                                --     text-embedding-3-small is natively 1536 but
                                                --     supports Matryoshka truncation; 1024 is
                                                --     configured, and fits pgvector's 2000 hnsw cap.
                                                --     Changing the embedding model means
                                                --     re-embedding all pages, and a column
                                                --     migration if its dimension differs.

    has_tables     BOOLEAN DEFAULT FALSE,
    token_est      INTEGER,
    tsv            tsvector GENERATED ALWAYS AS
                       (to_tsvector('english', coalesce(lexical_text, ''))) STORED,
    UNIQUE (doc_id, page_seq)
);
CREATE INDEX IF NOT EXISTS pages_doc_seq  ON pages (doc_id, page_seq);
CREATE INDEX IF NOT EXISTS pages_tsv_idx  ON pages USING gin (tsv);
CREATE INDEX IF NOT EXISTS pages_trgm_idx ON pages USING gin (lexical_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS pages_vec_idx  ON pages USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ═════════════════════════════════════════════════════════════════════════════
-- SECTION TREE — what the reasoning navigator reads
-- Built from: anchors (72/78, primary) → SEC patterns → styling → TOC.
-- NOT from <h1>-<h6>: only 6/78 filings contain any, median 1 tag.
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS sections (
    section_id  TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES filings ON DELETE CASCADE,
    parent_id   TEXT,
    level       INTEGER,
    ordinal     INTEGER,
    raw_title   TEXT,
    clean_title TEXT,                           -- LLM-normalised: 'ITEM 1. B USINESS' → 'Item 1. Business'
    summary     TEXT,                           -- one line, same LLM call
    kind        TEXT,                           -- part|item|note|statement|mdna|exhibit|other
    stmt_type   TEXT,                           -- income|balance|cashflow|equity|compinc|segment|NULL
    source      TEXT,                           -- anchor|sec_pattern|styling|toc — node provenance
    page_start  INTEGER,
    page_end    INTEGER
);
CREATE INDEX IF NOT EXISTS sections_doc ON sections (doc_id, ordinal);


-- ═════════════════════════════════════════════════════════════════════════════
-- BLOCKS — children of a page; used for quoting and expansion, never for ranking
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS blocks (
    block_id      TEXT PRIMARY KEY,
    page_id       TEXT NOT NULL REFERENCES pages ON DELETE CASCADE,
    doc_id        TEXT NOT NULL,
    section_id    TEXT,
    order_idx     INTEGER NOT NULL,
    prev_block_id TEXT,
    next_block_id TEXT,
    block_type    TEXT,                         -- heading|paragraph|list|footnote|caption|table
    text          TEXT,
    dom_xpath     TEXT,
    char_start    INTEGER,
    char_end      INTEGER,
    table_id      TEXT
);
CREATE INDEX IF NOT EXISTS blocks_page ON blocks (page_id, order_idx);


-- ═════════════════════════════════════════════════════════════════════════════
-- TABLES
--   is_data_table : 46% of <table> elements are ≤2-row layout artifacts
--   alignment_ok  : header/value ordinal alignment validated. Typed cells exist
--                   ONLY when true — fail closed, a typed fact needs provable provenance.
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS tables (
    table_id        TEXT PRIMARY KEY,
    page_id         TEXT NOT NULL REFERENCES pages ON DELETE CASCADE,
    doc_id          TEXT NOT NULL,
    section_id      TEXT,
    order_idx       INTEGER,
    caption         TEXT,                       -- nearest preceding heading
    units_note      TEXT,                       -- '(In millions, except per share amounts)'
    n_rows          INTEGER,
    n_cols          INTEGER,
    markdown        TEXT NOT NULL,              -- ALWAYS present — what the model reads
    is_data_table   BOOLEAN NOT NULL,
    alignment_ok    BOOLEAN NOT NULL,
    header_tokens   TEXT[],
    value_col_count INTEGER,
    continues_from_table_id TEXT
);
CREATE INDEX IF NOT EXISTS tables_page ON tables (page_id, order_idx);

-- Populated ONLY when tables.alignment_ok = true.
CREATE TABLE IF NOT EXISTS table_cells (
    cell_id         TEXT PRIMARY KEY,
    table_id        TEXT NOT NULL REFERENCES tables ON DELETE CASCADE,
    row_idx         INTEGER,
    col_idx         INTEGER,
    row_header_path TEXT,                       -- 'Cash Flows from Investing > Purchases of PP&E'
    col_header_path TEXT,                       -- '2018'
    raw_text        TEXT,
    numeric_value   NUMERIC,                    -- NUMERIC, never float. (1,577) → -1577
    sign            INTEGER,
    scale           TEXT,
    unit            TEXT,
    is_header       BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS table_cells_tbl ON table_cells (table_id, row_idx);


-- ═════════════════════════════════════════════════════════════════════════════
-- XBRL FACTS — a first-class fact path, not an answer path
--   Query by CONCEPT + PERIOD, never by value (value-first matching produced a
--   468-way coincidental match in testing — it is a −1 generator).
--   has_dimensions : exclude segment/geography breakdowns by default.
--   row_label      : displayed label from the enclosing <tr>. The safety net that
--                    catches a mis-mapped concept, and the free citation quote.
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS facts (
    fact_id        TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES filings ON DELETE CASCADE,
    qname          TEXT NOT NULL,               -- 'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment'
    value          NUMERIC,
    unit           TEXT,
    scale          INTEGER,
    sign           INTEGER,
    period_start   DATE,
    period_end     DATE,
    is_instant     BOOLEAN,
    dimensions     JSONB,
    has_dimensions BOOLEAN DEFAULT FALSE,
    row_label      TEXT,
    page_id        TEXT REFERENCES pages ON DELETE CASCADE,
    block_id       TEXT
);
CREATE INDEX IF NOT EXISTS facts_lookup ON facts (doc_id, qname, period_end);
CREATE INDEX IF NOT EXISTS facts_nodim  ON facts (doc_id, qname)
    WHERE has_dimensions = FALSE;


-- ═════════════════════════════════════════════════════════════════════════════
-- TYPED EDGES — explicit, named relationships
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS edges (
    edge_id   BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    src_type  TEXT NOT NULL,                    -- page|block|table|section|fact
    src_id    TEXT NOT NULL,
    dst_type  TEXT NOT NULL,
    dst_id    TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    CONSTRAINT edges_type_ck CHECK (edge_type IN
        ('references','has_footnote','has_context',
         'continuation_of','same_table','parent_of'))
);
CREATE INDEX IF NOT EXISTS edges_src ON edges (src_type, src_id);
CREATE INDEX IF NOT EXISTS edges_dst ON edges (dst_type, dst_id);


-- ═════════════════════════════════════════════════════════════════════════════
-- OBSERVABILITY — feeds the ablation table in the approach note
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS query_log (
    query_id       UUID PRIMARY KEY,
    ts             TIMESTAMPTZ DEFAULT now(),
    question        TEXT,
    router          JSONB,
    candidate_docs  TEXT[],
    status          TEXT,                       -- answered|clarify|abstained
    abstain_reason  TEXT,                       -- the gate id that failed, e.g. 'G1'
    tier            INTEGER,                    -- 1 = first pass, 2 = escalated
    latency_ms      INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    CONSTRAINT query_log_status_ck CHECK (status IN ('answered','clarify','abstained'))
);

CREATE TABLE IF NOT EXISTS answer_log (
    answer_id      UUID PRIMARY KEY,
    query_id       UUID REFERENCES query_log ON DELETE CASCADE,
    answer_text    TEXT,
    formula        TEXT,
    formula_source TEXT,                        -- question|book|llm_proposed, in precedence order
    operands       JSONB,
    citations      JSONB,
    CONSTRAINT answer_log_fsrc_ck CHECK
        (formula_source IS NULL OR formula_source IN ('question','book','llm_proposed'))
);


-- ═════════════════════════════════════════════════════════════════════════════
-- MIGRATION BOOKKEEPING
-- ═════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT now()
);
INSERT INTO schema_migrations (version) VALUES ('001_init')
    ON CONFLICT (version) DO NOTHING;

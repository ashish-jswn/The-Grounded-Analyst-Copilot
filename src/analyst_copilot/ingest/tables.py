"""The 7-step table pipeline.

Roughly 75% of gold evidence lives in the three primary financial statements, so
tables are where the numeric answers are. They are also where a silently wrong
fact is easiest to manufacture, which is why the pipeline ends in a gate rather
than in a best effort.

THE CORE IDEA IS STEP 6. `len(header_tokens) == len(value_columns)` decides
whether typed `table_cells` exist at all:

    validates -> typed cells + markdown. Safe to compute on.
    fails     -> markdown ONLY, alignment_ok = False. No typed facts.

FAIL CLOSED: a typed fact exists only when its provenance is provable. Where
alignment fails the model reads the markdown and the verifier gates the result -
we lose a convenience, never correctness.

Grounded in 1,397 measured tables from the corpus:
  * 46% of <table> elements are <=2-row layout artifacts -> step 0
  * colspan is the critical path (JPM 31,121; J&J 11,171); rowspan is rare
  * ZERO nested tables anywhere -> no recursive logic needed
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from lxml import etree

# A period/year token in a header row: 2018, FY2018, Q2, "Three months ended".
_YEAR_TOK = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_PERIOD_TOK = re.compile(
    r"\b(q[1-4]|fy\d{2,4}|three|six|nine|twelve|months?|quarter|year|weeks?)\b", re.I
)

_QFY_TOK = re.compile(r"\b(q[1-4]|fy\s?\d{2,4})\b", re.I)
# "Year Ended June 30," / "Three Months Ended" - names the period for the WHOLE
# table rather than identifying one column, so it is not a column label.
_PERIOD_DESCRIPTOR = re.compile(
    r"\b(years?|three|six|nine|twelve|months?|quarters?|weeks?|periods?|fiscal)\b"
    r".{0,30}?\b(ended|ending)\b",
    re.I,
)

# A numeric cell. Accepts thousands separators, a leading currency symbol, a
# trailing percent, and the accounting convention of parentheses for negatives.
_NUMERIC = re.compile(
    r"^\(?\s*[-+]?[$€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?\s*\)?$"
)
_BARE_SYMBOL = re.compile(r"^[\s$€£%()\[\].,;:*+\-–—―]*$")

# Filings state scale in two shapes, and BOTH must be caught: the long form
# "(In millions, except per share amounts)" and the bare "(Millions)" that 3M
# uses on its cash-flow statement. Missing the bare form leaves `scale` NULL,
# and gate G5 (units-compatible) cannot then reconcile operands that are really
# in the same base - so the answer abstains for a fixable reason.
_UNITS_NOTE = re.compile(
    r"\(\s*(?:dollars\s+|amounts\s+)?(?:in\s+)?"
    r"(thousands|millions|billions)\b[^)]*\)",
    re.I,
)

# Column kinds produced by step 4.
VALUE, LABEL, SYMBOL, EMPTY = "value", "label", "symbol", "empty"


@dataclass
class TableCell:
    row_idx: int
    col_idx: int
    row_header_path: str
    col_header_path: str
    raw_text: str
    numeric_value: Decimal | None
    sign: int
    scale: str | None
    unit: str | None


@dataclass
class ParsedTable:
    """One table. `markdown` is ALWAYS present; typed cells only when it validates."""

    grid: list[list[str]]
    markdown: str
    is_data_table: bool
    alignment_ok: bool
    header_row_idx: int | None
    header_tokens: list[str] = field(default_factory=list)
    # THE ROW LABEL IS THE SEARCHABLE PART OF A FINANCIAL TABLE, AND IT WAS
    # NEVER INDEXED. `build_lexical_text`'s own docstring says the BM25 lift
    # depends on strings like "Purchases of property, plant and equipment"
    # surviving indexing - but only COLUMN headers were ever passed to it, and
    # a column header is "2022", "2021", "$". The line item, which is what an
    # analyst's question actually names, was dropped.
    #
    # Collected from the GRID, not from `cells`, so tables that fail the
    # alignment gate still contribute their labels. That is not a weakening of
    # fail-closed: the gate governs whether a typed FACT may exist; a row label
    # in the lexical index creates no fact, it only makes the page findable,
    # after which every gate and verifier applies unchanged.
    row_labels: list[str] = field(default_factory=list)
    value_col_count: int = 0
    col_kinds: list[str] = field(default_factory=list)
    cells: list[TableCell] = field(default_factory=list)
    caption: str | None = None
    units_note: str | None = None
    n_rows: int = 0
    n_cols: int = 0
    reject_reason: str | None = None


# ---------------------------------------------------------------------------
# Numeric normalisation
# ---------------------------------------------------------------------------
def normalise_number(text: str) -> tuple[Decimal | None, int]:
    """`(1,577)` -> (-1577, -1). Returns (value, sign).

    Accounting parentheses mean negative. Getting this wrong flips the sign of a
    cash-flow line item, which reads as a plausible number and is exactly the
    kind of error the verifier cannot catch from the digits alone.
    """
    t = (text or "").strip()
    if not t:
        return None, 1
    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()").strip()
    t = t.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    t = t.replace("%", "").strip()
    if t.startswith("-"):
        negative = True
        t = t[1:].strip()
    if not t:
        return None, 1
    try:
        value = Decimal(t)
    except InvalidOperation:
        return None, 1
    return (-value if negative else value), (-1 if negative else 1)


def _is_numeric(text: str) -> bool:
    t = (text or "").strip()
    if not t or _BARE_SYMBOL.match(t):
        return False
    return bool(_NUMERIC.match(t))


def _text_of(el: etree._Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).replace("\xa0", " ").strip()


# ---------------------------------------------------------------------------
# Step 1 - grid expansion
# ---------------------------------------------------------------------------
def expand_grid(table_el: etree._Element) -> list[list[str]]:
    """Expand colspan/rowspan into a dense rectangular grid.

    colspan is the critical path: without expansion a header spanning two
    columns collapses to one cell and every subsequent ordinal alignment is off
    by one. There are no nested tables in this corpus, so a direct row walk is
    safe - but a nested table's rows are excluded explicitly rather than by
    assumption, so an unseen upload cannot corrupt the grid.
    """
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}  # (row, col) -> text from a rowspan

    # Only rows belonging to THIS table, not to a nested one. `next(..., None)`
    # rather than `__next__()`: malformed EDGAR HTML can leave a <tr> with no
    # <table> ancestor after lxml's recovery, and StopIteration there would
    # abort the whole filing's ingest.
    rows = [
        tr
        for tr in table_el.iter("tr")
        if next(tr.iterancestors("table"), None) is table_el
    ]

    for r, tr in enumerate(rows):
        row: list[str] = []
        col = 0
        for cell in tr:
            if not isinstance(cell.tag, str) or cell.tag.lower() not in ("td", "th"):
                continue
            while (r, col) in pending:
                row.append(pending.pop((r, col)))
                col += 1
            text = _text_of(cell)
            try:
                cspan = max(1, int(cell.get("colspan") or 1))
                rspan = max(1, int(cell.get("rowspan") or 1))
            except ValueError:
                cspan = rspan = 1
            cspan, rspan = min(cspan, 64), min(rspan, 64)
            for c in range(cspan):
                row.append(text if c == 0 else "")
                for extra in range(1, rspan):
                    pending[(r + extra, col + c)] = text if c == 0 else ""
                col += 1
        while (r, col) in pending:
            row.append(pending.pop((r, col)))
            col += 1
        grid.append(row)

    width = max((len(r) for r in grid), default=0)
    return [r + [""] * (width - len(r)) for r in grid]


# ---------------------------------------------------------------------------
# Step 2 - drop all-empty columns
# ---------------------------------------------------------------------------
def drop_empty_columns(grid: list[list[str]]) -> list[list[str]]:
    if not grid:
        return grid
    width = len(grid[0])
    keep = [c for c in range(width) if any((row[c] or "").strip() for row in grid)]
    return [[row[c] for c in keep] for row in grid]


# ---------------------------------------------------------------------------
# Step 0 - the data-table gate
# ---------------------------------------------------------------------------
def is_data_table(grid: list[list[str]], min_rows: int, min_numeric: int) -> bool:
    """46% of <table> elements are layout artifacts. Reject them here.

    Requires enough rows, enough numeric cells, AND an identifiable label
    column - a block of figures with no captions is a layout grid, not a
    statement.
    """
    if len(grid) < min_rows:
        return False
    numeric = sum(1 for row in grid for cell in row if _is_numeric(cell))
    if numeric < min_numeric:
        return False
    return _label_column_index(grid) is not None


def _label_column_index(grid: list[list[str]]) -> int | None:
    """The leftmost column whose cells are mostly non-numeric prose."""
    if not grid:
        return None
    for c in range(len(grid[0])):
        vals = [(row[c] or "").strip() for row in grid]
        filled = [v for v in vals if v]
        if len(filled) < max(2, len(grid) // 3):
            continue
        wordy = sum(1 for v in filled if not _is_numeric(v) and not _BARE_SYMBOL.match(v))
        if wordy >= 0.6 * len(filled):
            return c
    return None


# ---------------------------------------------------------------------------
# Step 3 - detect the header row FIRST
# ---------------------------------------------------------------------------
def detect_header_row(grid: list[list[str]]) -> int | None:
    """The first row carrying >=2 year/period tokens.

    This runs BEFORE column classification, and the order matters: a year like
    `2018` matches any numeric regex, so classifying columns over the whole grid
    mislabels the header row's own cells as values. That was a real bug hit
    during measurement.
    """
    best: tuple[int, int] | None = None
    for r, row in enumerate(grid[:12]):  # a header is never deep in the table
        tokens = sum(
            1
            for cell in row
            if cell and (_YEAR_TOK.search(cell) or _PERIOD_TOK.search(cell))
        )
        if tokens >= 2 and (best is None or tokens > best[1]):
            best = (r, tokens)
    return best[0] if best else None


def header_tokens_of(
    grid: list[list[str]], header_row_idx: int, label_col: int | None = None
) -> list[str]:
    """The column labels in the header row, in left-to-right order.

    MEASURED: extracting "any cell matching a period regex" fails symmetrically,
    30 tables each way on MICROSOFT_2023_10K alone, and both directions break
    the step-6 gate:

      hdr > val: "Year Ended June 30," is a descriptor spanning the whole
                 header, not a column label. Counting it gave 4 tokens against
                 3 value columns.
      hdr < val: "Percentage Change" IS a real column but carries no period
                 token, so it was dropped. That gave 2 tokens against 3 columns.

    So a header token is any non-empty header cell that is not the label
    column's own heading, not a units note, and not a spanning period
    descriptor. That keeps "Percentage Change" and drops "Year Ended June 30,".
    """
    out = []
    for i, cell in enumerate(grid[header_row_idx]):
        t = (cell or "").strip()
        if not t or i == label_col:
            continue
        if _UNITS_NOTE.search(t):
            continue
        # A descriptor names the period for the whole table; a column label
        # identifies one column. "Year Ended June 30, 2023" is a label because
        # it pins a specific year; a bare "Year Ended June 30," is not.
        if _PERIOD_DESCRIPTOR.search(t) and not (
            _YEAR_TOK.search(t) or _QFY_TOK.search(t)
        ):
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Step 4 - classify columns using ONLY rows below the header
# ---------------------------------------------------------------------------
def classify_columns(grid: list[list[str]], header_row_idx: int | None) -> list[str]:
    start = (header_row_idx + 1) if header_row_idx is not None else 0
    body = grid[start:]
    if not body:
        return [EMPTY] * (len(grid[0]) if grid else 0)

    kinds: list[str] = []
    for c in range(len(grid[0])):
        vals = [(row[c] or "").strip() for row in body if c < len(row)]
        filled = [v for v in vals if v]
        if not filled:
            kinds.append(EMPTY)
            continue
        if all(_BARE_SYMBOL.match(v) for v in filled):
            # A currency symbol occupies its own column in EDGAR tables.
            kinds.append(SYMBOL)
            continue
        numeric = sum(1 for v in filled if _is_numeric(v))
        kinds.append(VALUE if numeric >= 0.6 * len(filled) else LABEL)
    return kinds


# ---------------------------------------------------------------------------
# Steps 5-6 - ordinal alignment and the validation gate
# ---------------------------------------------------------------------------
def align(
    header_tokens: list[str], col_kinds: list[str]
) -> tuple[bool, dict[int, str]]:
    """Map the k-th header token to the k-th VALUE column.

    Positional alignment is wrong, because the header row does not share the
    data rows' layout:

        header : ['(Millions)', '2018', '',        '2017', '',        '2016']
        data   : ['Purchases',  '$',    '(1,577)', '$',    '(1,373)', '$'  ]

    Index alignment yields `2018 -> "$"` - a silently wrong fact, and precisely
    a -1 generator. Ordinal alignment over value columns only yields
    `2018 -> (1,577)`.

    Returns (validates, {col_idx: header_token}).
    """
    value_cols = [i for i, k in enumerate(col_kinds) if k == VALUE]
    if not header_tokens or len(header_tokens) != len(value_cols):
        return False, {}
    return True, dict(zip(value_cols, header_tokens))


# ---------------------------------------------------------------------------
# Step 7 - enrichment
# ---------------------------------------------------------------------------
def _units_note(context: str) -> str | None:
    m = _UNITS_NOTE.search(context or "")
    return m.group(0) if m else None


def to_markdown(grid: list[list[str]]) -> str:
    """Markdown is ALWAYS produced, even when alignment fails - it is what the
    model reads when there are no typed cells."""
    if not grid:
        return ""
    width = len(grid[0])
    lines = ["| " + " | ".join(c.replace("|", r"\|") for c in grid[0]) + " |"]
    lines.append("|" + "|".join(["---"] * width) + "|")
    for row in grid[1:]:
        lines.append("| " + " | ".join(c.replace("|", r"\|") for c in row) + " |")
    return "\n".join(lines)


def parse_table(
    table_el: etree._Element,
    *,
    min_rows: int,
    min_numeric_cells: int,
    caption: str | None = None,
    context_text: str = "",
) -> ParsedTable:
    """Run the full 7-step pipeline over one <table> element."""
    grid = expand_grid(table_el)                       # step 1
    grid = drop_empty_columns(grid)                    # step 2
    n_rows, n_cols = len(grid), (len(grid[0]) if grid else 0)

    if not is_data_table(grid, min_rows, min_numeric_cells):   # step 0
        return ParsedTable(
            grid=grid,
            markdown=to_markdown(grid),
            is_data_table=False,
            alignment_ok=False,
            header_row_idx=None,
            n_rows=n_rows,
            n_cols=n_cols,
            caption=caption,
            reject_reason="not a data table (layout artifact)",
        )

    header_row_idx = detect_header_row(grid)                    # step 3
    col_kinds = classify_columns(grid, header_row_idx)          # step 4
    label_col = _label_column_index(grid)
    header_tokens = (
        header_tokens_of(grid, header_row_idx, label_col)
        if header_row_idx is not None
        else []
    )
    validates, col_map = align(header_tokens, col_kinds)        # steps 5-6

    units = _units_note(context_text) or _units_note(
        " ".join(c for row in grid[: (header_row_idx or 0) + 1] for c in row)
    )
    scale = None
    if units:
        low = units.lower()
        scale = (
            "thousands" if "thousand" in low
            else "millions" if "million" in low
            else "billions" if "billion" in low
            else None
        )

    table = ParsedTable(
        grid=grid,
        markdown=to_markdown(grid),
        is_data_table=True,
        alignment_ok=validates,
        header_row_idx=header_row_idx,
        header_tokens=header_tokens,
        value_col_count=sum(1 for k in col_kinds if k == VALUE),
        col_kinds=col_kinds,
        caption=caption,
        units_note=units,
        n_rows=n_rows,
        n_cols=n_cols,
        reject_reason=None if validates else "alignment gate failed",
    )

    body_start_all = (header_row_idx + 1) if header_row_idx is not None else 0
    if label_col is not None:
        seen: set[str] = set()
        for r in range(body_start_all, len(grid)):
            label = (grid[r][label_col] or "").strip()
            # A label is a line item, not a sentence, and not a bare number.
            if not (2 < len(label) <= 120) or label.lower() in seen:
                continue
            if not any(ch.isalpha() for ch in label):
                continue
            seen.add(label.lower())
            table.row_labels.append(label)

    # Typed cells exist ONLY when alignment validates (step 6, fail closed).
    if not validates:
        return table

    body_start = (header_row_idx + 1) if header_row_idx is not None else 0
    for r in range(body_start, len(grid)):
        row_label = (grid[r][label_col] if label_col is not None else "").strip()
        if not row_label:
            continue
        for c, token in col_map.items():
            raw = (grid[r][c] or "").strip()
            if not raw:
                continue
            value, sign = normalise_number(raw)
            if value is None:
                continue
            table.cells.append(
                TableCell(
                    row_idx=r,
                    col_idx=c,
                    row_header_path=row_label,
                    col_header_path=token,
                    raw_text=raw,
                    numeric_value=value,
                    sign=sign,
                    scale=scale,
                    unit="currency" if "$" in raw or scale else None,
                )
            )
    return table

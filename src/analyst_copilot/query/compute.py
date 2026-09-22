"""The calculator - Decimal in, Decimal out.

THE MODEL NEVER DOES ARITHMETIC. It may identify WHICH formula applies; the
numbers are evaluated here. `FinAgent-RAG` measured that program-of-thought
execution removes **88.0% of arithmetic errors**.

NEVER `eval()` MODEL-SUPPLIED TEXT. A formula reaches this module as a
string, and at level 3 of the precedence ladder that string was written by an
LLM. It is parsed to an AST and every node outside the whitelist is rejected, so
a formula can compute a number and nothing else - no attribute access, no calls
to anything but the named helpers, no imports, no comprehensions.

`Decimal`, never `float`: a binary float cannot represent 0.1, and a cent of
drift on a ratio is the difference between matching gold and scoring -1.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any, Callable

# The ONLY operators a formula may use.
_ALLOWED_BINOPS: dict[type, Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd,
    ast.Name, ast.Load, ast.Call, ast.Constant, ast.Tuple,
    *_ALLOWED_BINOPS,
)


class FormulaError(ValueError):
    """A formula that cannot be evaluated safely or completely.

    Always terminal: the caller abstains rather than guessing an operand.
    """


@dataclass
class Operand:
    name: str
    value: Decimal
    unit: str | None = None
    scale: str | None = None
    period: str | None = None
    citation: dict[str, Any] = field(default_factory=dict)


@dataclass
class Computation:
    formula: str
    result: Decimal
    operands: dict[str, Decimal]
    unit: str | None = None
    render: str | None = None
    dp: int | None = None

    def rendered(self) -> str:
        """Format for display. A ratio asked for as a percentage is multiplied
        HERE, once, rather than in a prompt where it can silently not happen."""
        value = self.result
        if self.render == "percent":
            value = value * 100
        if self.dp is not None:
            value = value.quantize(Decimal(1).scaleb(-self.dp), rounding=ROUND_HALF_UP)
        text = format(value, "f")
        if self.render == "percent":
            return f"{text}%"
        return text


# ---------------------------------------------------------------------------
# Period helpers - the named functions a formula may call
# ---------------------------------------------------------------------------
def _to_decimal(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    if isinstance(x, bool):
        raise FormulaError("boolean is not a number")
    if isinstance(x, int):
        return Decimal(x)
    if isinstance(x, float):
        # Never trust a float literal: route it through str so 0.1 stays 0.1.
        return Decimal(str(x))
    raise FormulaError(f"not a number: {x!r}")


class _Helpers:
    """`avg`, `delta`, `prev`, `sum_range`, `rank`, `compare` - the whitelist.

    Period-aware helpers resolve against the operand table, so
    `avg(inventory, prev, current)` needs `inventory` and `inventory__prev`.
    A missing prior period raises rather than silently averaging one value -
    that would produce a plausible number from incomplete evidence.
    """

    def __init__(self, values: dict[str, Decimal]) -> None:
        self._values = values

    def _lookup(self, name: str, period: str) -> Decimal:
        key = name if period == "current" else f"{name}__{period}"
        if key not in self._values:
            raise FormulaError(
                f"operand {key!r} is not available; "
                f"cannot evaluate the {period} period"
            )
        return self._values[key]

    def avg(self, name: str, *periods: str) -> Decimal:
        periods = periods or ("prev", "current")
        vals = [self._lookup(name, p) for p in periods]
        return sum(vals, Decimal(0)) / Decimal(len(vals))

    def delta(self, name: str, start: str = "prev", end: str = "current") -> Decimal:
        return self._lookup(name, end) - self._lookup(name, start)

    def prev(self, name: str) -> Decimal:
        return self._lookup(name, "prev")

    def sum_range(self, name: str, *periods: str) -> Decimal:
        return sum((self._lookup(name, p) for p in periods), Decimal(0))

    def rank(self, *values: Any) -> Decimal:
        """1-based position of the largest value. Used by "which segment ..."."""
        nums = [_to_decimal(v) for v in values]
        if not nums:
            raise FormulaError("rank() needs at least one value")
        return Decimal(nums.index(max(nums)) + 1)

    def compare(self, a: Any, b: Any) -> Decimal:
        """-1, 0 or 1. Keeps a yes/no judgement deterministic."""
        x, y = _to_decimal(a), _to_decimal(b)
        return Decimal(0) if x == y else (Decimal(1) if x > y else Decimal(-1))


# `prev` and the bare operand names are resolved by the evaluator, so the
# helper table only needs the callables.
_HELPER_NAMES = {"avg", "delta", "prev", "sum_range", "rank", "compare"}


def _bare_name(node: ast.AST) -> str:
    """A helper's first argument is an operand NAME, not its value."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise FormulaError("expected an operand name")


def evaluate(formula: str, values: dict[str, Decimal]) -> Decimal:
    """Evaluate `formula` over `values`. Raises FormulaError on anything unsafe."""
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"unparseable formula {formula!r}: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FormulaError(
                f"disallowed syntax {type(node).__name__} in formula {formula!r}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _HELPER_NAMES:
                raise FormulaError("only avg/delta/prev/sum_range/rank/compare may be called")
            if node.keywords:
                raise FormulaError("keyword arguments are not allowed in a formula")

    helpers = _Helpers(values)

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return _to_decimal(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise FormulaError(f"operand {node.id!r} is not available")
            return values[node.id]
        if isinstance(node, ast.UnaryOp):
            v = visit(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            op = _ALLOWED_BINOPS.get(type(node.op))
            if op is None:
                raise FormulaError(f"disallowed operator {type(node.op).__name__}")
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise FormulaError("division by zero")
            return op(left, right)
        if isinstance(node, ast.Call):
            name = node.func.id  # type: ignore[union-attr]
            fn = getattr(helpers, name)
            if name in ("avg", "delta", "prev", "sum_range"):
                args: list[Any] = [_bare_name(node.args[0])]
                args += [_bare_name(a) for a in node.args[1:]]
                return fn(*args)
            return fn(*[visit(a) for a in node.args])
        raise FormulaError(f"disallowed node {type(node).__name__}")

    # 28 digits is ample for financial magnitudes and keeps ** stable.
    with localcontext() as ctx:
        ctx.prec = 28
        try:
            return visit(tree)
        except (DivisionByZero, InvalidOperation) as exc:
            raise FormulaError(f"arithmetic error: {exc}") from exc


# ---------------------------------------------------------------------------
# Scale normalisation - gate G5 depends on this
# ---------------------------------------------------------------------------
_SCALE_FACTOR = {
    None: Decimal(1),
    "units": Decimal(1),
    "thousands": Decimal(10) ** 3,
    "millions": Decimal(10) ** 6,
    "billions": Decimal(10) ** 9,
}


def to_base_units(value: Decimal, scale: str | None) -> Decimal:
    """Bring an operand to absolute units before arithmetic.

    Mixing a figure reported in millions with one in thousands produces an
    answer wrong by 1000x that still looks like a number - the exact failure
    gate G5 exists to catch.
    """
    factor = _SCALE_FACTOR.get((scale or "").lower() or None)
    if factor is None:
        raise FormulaError(f"unknown scale {scale!r}")
    return value * factor


def units_compatible(operands: list[Operand]) -> bool:
    """G5: every operand must reduce to the same dimension."""
    units = {(o.unit or "").lower() for o in operands if o.unit}
    return len(units) <= 1


def required_operand_names(formula: str) -> list[str]:
    """The exact operand names `evaluate` will demand, period suffixes included.

    THIS CLOSES A CONTRACT GAP THAT DISABLED THE WHOLE CALCULATOR.
    MEASURED: `evaluate` resolves operands by EXACT name (`values[o.name]`),
    but nothing ever told the extractor what those names are, so it invented its
    own (`fy2019_revenue`, `capex_fy2018`). Every derived answer then died with
    "operand 'revenue' is not available" -> gate G4 -> abstain. That is the
    entire domain-relevant category plus most ratio questions, failing silently
    as if the evidence were missing.

    `avg(ppe_net, prev, current)` needs BOTH `ppe_net` and `ppe_net__prev`,
    because `_Helpers._lookup` keys a non-current period as `name__period`.
    """
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError:
        return []

    required: list[str] = []
    seen: set[str] = set()

    def want(name: str) -> None:
        if name not in seen:
            seen.add(name)
            required.append(name)

    period_words: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _HELPER_NAMES or not node.args:
                continue
            try:
                base = _bare_name(node.args[0])
            except FormulaError:
                continue
            periods = [
                a.id if isinstance(a, ast.Name) else
                (a.value if isinstance(a, ast.Constant) else None)
                for a in node.args[1:]
            ]
            periods = [p for p in periods if isinstance(p, str)]
            # `prev(x)` names no period explicitly and means the prior one.
            if node.func.id == "prev" and not periods:
                periods = ["prev"]
            period_words.update(periods)
            want(base)
            for period in periods:
                if period != "current":
                    want(f"{base}__{period}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _HELPER_NAMES:
            if node.id not in period_words and node.id not in {"current", "prev", "n"}:
                want(node.id)
    return required


_YEARish = re.compile(r"(?:fy)?\s*(?:19|20)\d{2}", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _canonical(name: str) -> str:
    """Strip period decoration so `fy2019_revenue` and `revenue` compare equal."""
    text = _YEARish.sub(" ", (name or "").lower())
    return _NON_ALNUM.sub(" ", text).strip()


def _period_key(operand: Operand) -> int:
    """Sort key: later period first. Undated operands sort last."""
    match = _YEARish.search(operand.period or "")
    if match:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) == 4:
            return -int(digits)
    return 1


def align_operands(operands: list[Operand], required: list[str]) -> list[Operand]:
    """Rename extractor slots onto the names the formula demands.

    The prompt now asks for these names directly, so this is the SAFETY NET, not
    the mechanism - a model that returns `total_revenue` or `revenue_fy2019`
    still resolves. Matching is on the canonical name (period decoration
    stripped), then by period, latest first: `x` takes the most recent period and
    `x__prev` the one before it.

    Anything that does not match is passed through UNCHANGED rather than guessed
    at. A mis-assigned operand computes a plausible wrong number, which is the
    -1 this whole system exists to avoid; leaving it unresolved fails G4 loudly.
    """
    if not required:
        return operands

    by_canonical: dict[str, list[Operand]] = {}
    for operand in operands:
        by_canonical.setdefault(_canonical(operand.name), []).append(operand)
    for group in by_canonical.values():
        group.sort(key=_period_key)

    claimed: set[int] = set()
    renamed: list[Operand] = []
    # Current-period names first, so they take the latest period before a
    # `__prev` sibling can claim it.
    for name in sorted(required, key=lambda n: n.endswith("__prev")):
        base, _, suffix = name.partition("__")
        wanted = _canonical(base)
        index = 1 if suffix else 0
        candidates = (
            by_canonical.get(wanted)
            # A looser fallback: the formula's word appears inside the slot name.
            or next(
                (g for key, g in by_canonical.items()
                 if wanted and (wanted in key or key in wanted)),
                None,
            )
        )
        if not candidates or index >= len(candidates):
            continue
        chosen = candidates[index]
        if id(chosen) in claimed:
            continue
        claimed.add(id(chosen))
        renamed.append(
            Operand(
                name=name,
                value=chosen.value,
                unit=chosen.unit,
                scale=chosen.scale,
                period=chosen.period,
                citation=chosen.citation,
            )
        )

    untouched = [o for o in operands if id(o) not in claimed]
    return renamed + untouched


def compute(
    formula: str,
    operands: list[Operand],
    *,
    unit: str | None = None,
    render: str | None = None,
    dp: int | None = None,
    normalise_scale: bool = True,
) -> Computation:
    """Evaluate a formula over typed operands, normalising scale first."""
    values: dict[str, Decimal] = {}
    for o in operands:
        values[o.name] = to_base_units(o.value, o.scale) if normalise_scale else o.value
    result = evaluate(formula, values)
    return Computation(
        formula=formula,
        result=result,
        operands=dict(values),
        unit=unit,
        render=render,
        dp=dp,
    )


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def recomputes(stated: str, computation: Computation, tolerance: Decimal) -> bool:
    """G6: re-evaluating the formula must reproduce the stated answer.

    This catches the case where the model computed correctly, then wrote a
    different number into its prose.

    TWO DEFECTS THIS GATE HAD, BOTH OF WHICH REJECTED CORRECT ANSWERS.

    1. IT COMPARED A ROUNDED CLAIM TO AN UNROUNDED RESULT. The answer states
       what `Computation.rendered()` produced - quantized to `dp` - while this
       compared against the raw quotient at 1e-6 relative. MEASURED: fixed
       asset turnover computed 24.2579..., rendered "24.26", and G6 rejected it
       at 8.6e-5. The question then abstained on an answer that matched gold
       (24.26) exactly. Every rounded metric in the book failed this way.

    2. IT READ ONLY THE FIRST NUMBER, so a prose answer naming its period first
       ("the FY2019 ratio is 24.26") was checked against 2019 - the same defect
       already fixed in the eval scorer.

    The gate still does its real job: a model that computes 24.26 and writes 27
    is rejected.
    """
    candidates: list[Decimal] = []
    for match in _NUM.finditer(stated or ""):
        try:
            candidates.append(Decimal(match.group(0).replace(",", "")))
        except InvalidOperation:
            continue
    if not candidates:
        return False

    actual = computation.result
    if computation.render == "percent":
        actual = actual * 100

    # The answer may state the exact value OR the value at its declared display
    # precision. Both are the same claim.
    accepted = [actual]
    if computation.dp is not None:
        accepted.append(
            actual.quantize(Decimal(1).scaleb(-computation.dp), rounding=ROUND_HALF_UP)
        )

    for target in accepted:
        for claimed in candidates:
            if target == 0:
                if abs(claimed) <= tolerance:
                    return True
            elif abs(target - claimed) / abs(target) <= tolerance:
                return True
    return False

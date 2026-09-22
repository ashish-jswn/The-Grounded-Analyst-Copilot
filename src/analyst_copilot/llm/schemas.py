"""JSON Schemas, one per prompt: every model call is structured.

The schema and its prompt must declare the same fields. Measured failure:
the router prompt listed `form_types` and `periods` while the schema omitted
them, and gpt-5-mini - which follows instructions with "surgical precision" -
tried to satisfy both by cramming the missing fields INTO a string field:

    metric_name: "revenue','form_types':[],"

A contradiction between two instructions is
reconciled at the model's discretion, expensively and unpredictably. Keeping the
schemas here, next to the prompt loader, is what stops them drifting apart.

Azure strict mode additionally requires `additionalProperties: false` and every
property listed in `required`, so an optional field is expressed as a nullable
type rather than by omission.
"""

from __future__ import annotations

from typing import Any


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


_STR = {"type": "string"}
_NULLABLE_STR = {"type": ["string", "null"]}
_STR_LIST = {"type": "array", "items": _STR}


ROUTER = _obj(
    {
        "companies": _STR_LIST,
        "form_types": {"type": "array", "items": {"type": "string", "enum": ["10-K", "10-Q", "8-K"]}},
        "periods": {
            "type": "array",
            "items": _obj(
                {
                    "fiscal_year": {"type": ["integer", "null"]},
                    "fiscal_quarter": {"type": ["integer", "null"]},
                }
            ),
        },
        "intent": {"type": "string", "enum": ["historical", "forecast"]},
        "company_named": {"type": "boolean"},
        "metric_name": _NULLABLE_STR,
    }
)

# One evidence slot. `quote` is checked character-by-character by gate G1, so
# the prompt and this schema both insist it is copied verbatim.
_SLOT = _obj(
    {
        "name": _STR,
        "value": _NULLABLE_STR,
        "unit": _NULLABLE_STR,
        "scale": _NULLABLE_STR,
        "period": _NULLABLE_STR,
        "doc_id": _STR,
        "page_seq": {"type": "integer"},
        "page_printed": {"type": ["integer", "null"]},
        "quote": _STR,
    }
)

EXTRACT = _obj(
    {
        "slots": {"type": "array", "items": _SLOT},
        "missing_slots": _STR_LIST,
        "answer_type": {
            "type": "string",
            "enum": ["scalar", "ratio", "comparison", "ranking", "qualitative"],
        },
        "metric_name": _NULLABLE_STR,
        "question_supplied_definition": _NULLABLE_STR,
    }
)

VERIFY = _obj(
    {
        "verdict": {"type": "string", "enum": ["VALID", "INVALID"]},
        "reasoning": _STR,
    }
)

VERIFY_ADVERSARIAL = _obj(
    {
        "verdict": {"type": "string", "enum": ["SUPPORTED", "REFUTED"]},
        "reasoning": _STR,
    }
)

FORMULA_PROPOSE = _obj(
    {
        "definition": _STR,
        "formula": _STR,
        "operands": {
            "type": "array",
            "items": _obj({"name": _STR, "concepts": _STR_LIST}),
        },
        "unit": _STR,
        "has_single_standard_definition": {"type": "boolean"},
    }
)

JUDGE = _obj(
    {
        "equivalent": {"type": "boolean"},
        "reasoning": _STR,
    }
)

# stage name -> (prompt file, schema). Keeps the pair together at the call site.
BY_STAGE: dict[str, tuple[str, dict[str, Any]]] = {
    "router": ("router", ROUTER),
    "extractor": ("extract", EXTRACT),
    "verifier_a": ("verify", VERIFY),
    "formula_pick": ("formula_propose", FORMULA_PROPOSE),
    "judge": ("judge", JUDGE),
}


COMPOSE = _obj(
    {
        "answer": _STR,
        "answerable": {"type": "boolean"},
        "supporting_quote_indexes": {"type": "array", "items": {"type": "integer"}},
    }
)

BY_STAGE["composer"] = ("compose", COMPOSE)

"""The verifier seam must survive a change of model family.

WHY THIS MATTERS. The generate->verify seam should span two independent
model families; today both verifiers are the same gpt-5-mini. Other families
served on Azure AI Foundry - Grok, DeepSeek, Llama, Mistral - use the same
`/openai/v1` route through the OpenAI SDK, so they need no new adapter.

What they DO need is tolerance for OpenAI-only request parameters. These tests
pin that, so switching families stays a config change instead of a debugging
session.
"""

from __future__ import annotations

import json

import pytest

from analyst_copilot.llm.azure_openai import _degrade, _is_unsupported_parameter

SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


def request_kwargs():
    return {
        "model": "grok-4.1-fast-reasoning",
        "messages": [
            {"role": "system", "content": "You are a verifier."},
            {"role": "user", "content": "Is this answer supported?"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verify_output", "strict": True, "schema": SCHEMA},
        },
        "max_completion_tokens": 4000,
        "reasoning_effort": "medium",
        "verbosity": "low",
    }


# ---------------------------------------------------------------------------
# Recognising a shape rejection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("message", [
    "Error code: 400 - unsupported parameter: 'reasoning_effort'",
    "400 Bad Request: Unrecognized request argument supplied: verbosity",
    "invalid_request_error: response_format json_schema is not supported",
    "400 - 'strict' is not supported for this model",
])
def test_a_rejected_request_shape_is_recognised(message):
    assert _is_unsupported_parameter(Exception(message))


@pytest.mark.parametrize("message", [
    "401 Unauthorized: invalid api key",
    "429 Too Many Requests",
    "500 Internal Server Error",
    "404 deployment not found",
    "The response was filtered due to the content management policy",
])
def test_real_errors_are_not_swallowed_as_shape_problems(message):
    """A credential, quota or deployment error must surface, not trigger a
    silent retry that hides it."""
    assert not _is_unsupported_parameter(Exception(message))


# ---------------------------------------------------------------------------
# The degraded retry
# ---------------------------------------------------------------------------
def test_degrade_drops_only_the_openai_specific_knobs():
    out = _degrade(request_kwargs(), SCHEMA)
    assert "reasoning_effort" not in out
    assert "verbosity" not in out
    # max_completion_tokens is standard OpenAI-compatible; keep the budget.
    assert out["max_completion_tokens"] == 4000
    assert out["model"] == "grok-4.1-fast-reasoning"


def test_degrade_keeps_the_output_structured():
    """Every stage parses structured output, so the shape cannot be abandoned -
    it moves from strict json_schema into the system prompt."""
    out = _degrade(request_kwargs(), SCHEMA)
    assert out["response_format"] == {"type": "json_object"}
    system = out["messages"][0]["content"]
    assert "JSON only" in system
    assert json.dumps(SCHEMA) in system


def test_degrade_does_not_mutate_the_original_request():
    """The caller may still need the original for logging or a real retry."""
    original = request_kwargs()
    _degrade(original, SCHEMA)
    assert original["reasoning_effort"] == "medium"
    assert original["response_format"]["type"] == "json_schema"
    assert original["messages"][0]["content"] == "You are a verifier."


def test_degrade_preserves_the_user_turn_verbatim():
    out = _degrade(request_kwargs(), SCHEMA)
    assert out["messages"][1] == {
        "role": "user", "content": "Is this answer supported?"
    }

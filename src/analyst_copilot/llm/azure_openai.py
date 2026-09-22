"""Azure OpenAI adapter - one of only two modules allowed to import a vendor SDK.

AZURE EXPOSES TWO INCOMPATIBLE CLIENT SHAPES AND MIXING THEM GIVES 404s:

    endpoint ending /openai/v1  ->  OpenAI(base_url=..., api_key=...)
    classic endpoint            ->  AzureOpenAI(azure_endpoint=, api_version=)

Our deployment is the `/openai/v1` shape, so passing it to the classic client
fails with 404s that look like a missing deployment. The client shape is
detected from the URL rather than configured, because getting it wrong produces
a misleading error.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import ModelSettings, Settings
from .base import (
    Embedder,
    LLMError,
    LLMProvider,
    LLMRateLimited,
    LLMResponse,
    LLMTruncated,
    with_retry,
)


def _is_rate_limit(exc: Exception) -> bool:
    """Throttling, in whatever shape the SDK surfaces it.

    Checked structurally where possible and textually as a fallback, because a
    429 misclassified as a hard error becomes a false abstention.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _is_unsupported_parameter(exc: Exception) -> bool:
    """True when the endpoint rejected a request SHAPE, not its content.

    Distinguished from a genuine error because the response is recoverable: drop
    the parameters this model family does not implement and try once more. A
    401, a quota error or a malformed schema must NOT be swallowed here.
    """
    text = str(exc).lower()
    if "400" not in text and "unsupported" not in text and "invalid_request" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "reasoning_effort", "verbosity", "max_completion_tokens",
            "response_format", "json_schema", "strict",
            "unsupported parameter", "unrecognized request argument",
        )
    )


def _degrade(kwargs: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Retry shape for a model that does not speak OpenAI's dialect.

    Drops the OpenAI-only knobs and swaps strict json_schema for plain JSON
    mode, moving the schema into the system prompt so the output shape is still
    specified. Every stage parses structured output, so the shape cannot simply
    be abandoned.
    """
    out = dict(kwargs)
    for key in ("reasoning_effort", "verbosity"):
        out.pop(key, None)
    out["response_format"] = {"type": "json_object"}
    messages = [dict(m) for m in out["messages"]]
    messages[0]["content"] = (
        f"{messages[0]['content']}\n\n"
        f"Reply with JSON only, conforming exactly to this schema:\n"
        f"{json.dumps(schema)}"
    )
    out["messages"] = messages
    return out


def _build_client(settings: Settings):
    from openai import AzureOpenAI, OpenAI  # imported here: see module docstring

    endpoint = (settings.azure_openai_endpoint or "").strip().rstrip("/")
    if not endpoint:
        raise LLMError("AZURE_OPENAI_ENDPOINT is not set")
    api_key = settings.azure_openai_api_key
    if not api_key:
        raise LLMError("AZURE_OPENAI_API_KEY is not set")

    # SET THE TIMEOUT EXPLICITLY. The SDK default is 600 s with no retries,
    # and a dropped connection does not raise - it BLOCKS.
    #
    # MEASURED: switching networks mid-run left an eval hung for 7+ minutes
    # with the process alive, zero CPU and no new output; the in-flight sockets
    # were dead and nothing timed out. In a live session that is
    # indistinguishable from "the system is thinking", for ten minutes.
    #
    # 180 s is far above the observed per-CALL cost (a 42k-token extraction
    # runs 30-60 s, a verifier ~20 s) and far below the point where a user
    # concludes the app has crashed. Two retries cover a transient blip,
    # which is the common case when a network changes underneath us.
    client_kwargs = {"timeout": 180.0, "max_retries": 2}
    if "/openai/v1" in endpoint:
        return OpenAI(base_url=endpoint, api_key=api_key, **client_kwargs)
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=settings.azure_openai_api_version or "2024-10-21",
        **client_kwargs,
    )


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, settings: Settings, model: ModelSettings) -> None:
        self._client = _build_client(settings)
        self._model = model

    def complete(
        self, *, system: str, user: str, schema: dict[str, Any], stage: str
    ) -> LLMResponse:
        return with_retry(lambda: self._complete_once(system, user, schema, stage))

    def _complete_once(
        self, system: str, user: str, schema: dict[str, Any], stage: str
    ) -> LLMResponse:
        m = self._model
        kwargs: dict[str, Any] = {
            "model": m.deployment,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Every call is structured; no stage parses free text.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{stage}_output",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if m.max_completion_tokens:
            kwargs["max_completion_tokens"] = m.max_completion_tokens
        if m.reasoning_effort:
            kwargs["reasoning_effort"] = m.reasoning_effort
        if m.verbosity:
            kwargs["verbosity"] = m.verbosity

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if _is_rate_limit(exc):
                raise LLMRateLimited(f"{stage}: {exc}") from exc
            if _is_unsupported_parameter(exc):
                # THIS IS WHAT MAKES A SECOND MODEL FAMILY A CONFIG CHANGE.
                # The `/openai/v1` Foundry route serves Grok, DeepSeek, Llama and
                # Mistral through this same OpenAI-compatible client, so a
                # second model family needs no new adapter. They do not all
                # accept OpenAI's `reasoning_effort` / `verbosity`, and not all
                # support strict json_schema.
                #
                # Rather than make the caller know which, drop the
                # OpenAI-specific parameters and fall back to plain JSON mode,
                # carrying the schema in the system prompt so the shape survives.
                # Retried ONCE - a second failure is a real error.
                resp = self._client.chat.completions.create(
                    **_degrade(kwargs, schema)
                )
            else:
                raise LLMError(f"{stage}: {type(exc).__name__}: {exc}") from exc

        choice = resp.choices[0]
        content = choice.message.content or ""
        finish = choice.finish_reason or ""

        usage = getattr(resp, "usage", None)
        reasoning = 0
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", 0) or 0

        if not content:
            # The reasoning-model trap: empty content with finish_reason
            # 'length' is a BUDGET error, not a model failure. Say so.
            raise LLMTruncated(
                f"{stage}: empty content (finish_reason={finish!r}, "
                f"reasoning_tokens={reasoning}, "
                f"max_completion_tokens={m.max_completion_tokens}). "
                "Raise max_completion_tokens for this stage in config.yaml."
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"{stage}: structured output was not valid JSON: {exc}") from exc

        return LLMResponse(
            data=data,
            raw=content,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=reasoning,
            model=m.deployment,
            stage=stage,
            finish_reason=finish,
        )


class AzureOpenAIEmbedder(Embedder):
    def __init__(self, settings: Settings, model: ModelSettings) -> None:
        self._client = _build_client(settings)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {"model": self._model.deployment, "input": texts}
        if self._model.dimensions:
            # MUST equal pages.embedding vector(N) or every insert fails, and it
            # fails at the END of a long ingest. config.py asserts this too.
            kwargs["dimensions"] = self._model.dimensions
        try:
            resp = self._client.embeddings.create(**kwargs)
        except Exception as exc:
            if _is_rate_limit(exc):
                raise LLMRateLimited(f"embeddings: {exc}") from exc
            raise LLMError(f"embeddings: {type(exc).__name__}: {exc}") from exc
        return [d.embedding for d in resp.data]

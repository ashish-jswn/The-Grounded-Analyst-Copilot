"""The LLM seam - one protocol, N adapters.

WHY THIS EXISTS. The generate->verify seam should ideally span two model
families, because a model checking its own output is not verification. Today
verifier A and verifier B are both `gpt-5-mini`, and independence comes from
adversarial framing plus context isolation.

The code therefore keeps a swap to a second family a matter of configuration.
Vendor SDKs differ in more than a base URL (call shape, where the system prompt
goes, how structured output is requested).

Scattering `if provider == ...` through the call sites would make that swap a
rewrite. So: EVERY stage calls only `complete()`, and no stage imports a vendor
SDK. Adding a family is one adapter file plus one registry entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Normalised failure from any provider, so retry/backoff lives in one place."""


class LLMRateLimited(LLMError):
    """A 429 / throttling response. Retryable, unlike a malformed request."""


class LLMTruncated(LLMError):
    """The model hit its completion budget before emitting content.

    THIS IS ITS OWN ERROR TYPE ON PURPOSE. `gpt-5-mini` is a REASONING model:
    it spends completion tokens on hidden reasoning BEFORE any content. With
    `max_completion_tokens` sized for the visible answer alone the call returns
    content='' with finish_reason='length' AND NO ERROR - it looks like a model
    failure rather than a config error. MEASURED: 64 reasoning tokens on a
    trivial prompt; at budget 64 the content was empty, at 512 it was correct.

    Raising a distinct error means a truncated call can never be mistaken for a
    refusal or an empty extraction, which would silently become an abstention.
    """


@dataclass
class LLMResponse:
    """One completion. `data` is the parsed structured output - every call is
    structured, so no stage ever parses free text."""

    data: dict[str, Any]
    raw: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Logged on every call so a truncation shows up as a budget problem rather
    # than a mystery, and so the approach note's cost table is measured.
    reasoning_tokens: int = 0
    model: str = ""
    stage: str = ""
    finish_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """The only interface a stage may use."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        stage: str,
    ) -> LLMResponse:
        """Return structured output conforming to `schema`.

        `stage` selects the per-stage settings bundle (reasoning_effort,
        verbosity, max_completion_tokens) from config.yaml - the caller passes a
        name, never a number.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]: ...


# ---------------------------------------------------------------------------
# Retry - ONE implementation, shared by every adapter
# ---------------------------------------------------------------------------
def with_retry(
    call: "Callable[[], LLMResponse]",
    *,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
):
    """Retry a throttled call with exponential backoff and jitter.

    MEASURED: running the eval at 8 concurrent workers turned most extractor
    calls into rate-limit failures, which the pipeline recorded as
    `extractor_error` and scored as abstentions. That is a DEPLOYMENT QUOTA
    ARTIFACT being mistaken for the system declining to answer - it would have
    made the abstention rate meaningless.

    Only throttling is retried. A malformed request or a truncated completion is
    deterministic and would fail identically every time.
    """
    import random
    import time as _time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except LLMRateLimited as exc:
            last = exc
            if attempt == attempts - 1:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            _time.sleep(delay * (0.5 + random.random()))
    raise last if last else LLMError("retry failed with no error recorded")

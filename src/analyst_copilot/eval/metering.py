"""Token and cost metering for eval runs.

WHY THIS EXISTS: the model budget is finite and a full 136-question run
makes 4-6 model calls per question. Running blind risks spending it on a
configuration that was already failing at question 3.

HOW IT AVOIDS TOUCHING THE PIPELINE: metering is a DECORATOR over `LLMProvider`,
applied by the harness after `build_pipeline()`. No stage knows it is being
measured, so this cannot change what is being measured — and `query/` keeps its
guarantee that nothing under it knows about evaluation.

TOKENS ARE MEASURED; DOLLARS ARE DERIVED. The per-token rates live in
`config.yaml` under `eval.pricing` and default to null, because a wrong hardcoded
price is worse than no price: it produces a confident budget number that is
silently incorrect. Set them from your own Azure pricing blade and the harness
will report cost; leave them and it reports tokens only.
"""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass, field

from ..llm.base import LLMProvider, LLMResponse


@dataclass
class StageUsage:
    """What one pipeline stage consumed."""

    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0

    def add(self, other: "StageUsage") -> None:
        self.calls += other.calls
        self.errors += other.errors
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.seconds += other.seconds


@dataclass
class Usage:
    """Consumption over some scope — one question, one batch, or a whole run."""

    by_stage: dict[str, StageUsage] = field(default_factory=dict)

    def record(
        self,
        stage: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        seconds: float = 0.0,
        error: bool = False,
    ) -> None:
        s = self.by_stage.setdefault(stage, StageUsage())
        s.calls += 1
        s.errors += int(error)
        s.input_tokens += input_tokens
        s.output_tokens += output_tokens
        s.reasoning_tokens += reasoning_tokens
        s.seconds += seconds

    def merge(self, other: "Usage") -> None:
        for stage, usage in other.by_stage.items():
            self.by_stage.setdefault(stage, StageUsage()).add(usage)

    # -- totals ---------------------------------------------------------
    @property
    def calls(self) -> int:
        return sum(s.calls for s in self.by_stage.values())

    @property
    def errors(self) -> int:
        return sum(s.errors for s in self.by_stage.values())

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.by_stage.values())

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.by_stage.values())

    @property
    def reasoning_tokens(self) -> int:
        return sum(s.reasoning_tokens for s in self.by_stage.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def seconds(self) -> float:
        """Wall-clock spent inside model calls. Sums across stages, so with
        concurrent workers it exceeds the run's elapsed time - that is the point:
        it shows how much of the run is model latency rather than our own work."""
        return sum(s.seconds for s in self.by_stage.values())

    def cost(self, price_in: float | None, price_out: float | None) -> float | None:
        """Dollars, or None when rates are unset. Never guesses a rate."""
        if price_in is None or price_out is None:
            return None
        return (
            self.input_tokens / 1_000_000 * price_in
            + self.output_tokens / 1_000_000 * price_out
        )


class CostLedger:
    """Thread-safe accumulator with per-question attribution.

    The eval harness runs questions concurrently, one question per worker
    thread, so the CURRENT question is tracked in ambient state. That gives
    exact per-question attribution without threading a context object through
    the pipeline, which would mean changing pipeline signatures for the sake of
    the harness.

    A CONTEXTVAR, NOT `threading.local`, AND THE DIFFERENCE IS LOAD-BEARING.
    The pipeline runs its two verifiers concurrently. With thread-local state
    those calls execute on pool threads that have no question scope, so their
    tokens vanished from the per-question total while still landing in the
    grand total - the two would silently disagree. A ContextVar is carried into
    a worker by `contextvars.copy_context()`, which is exactly what
    `pipeline._verify` submits, so a nested call attributes to the question
    that spawned it.
    """

    def __init__(self) -> None:
        self.total = Usage()
        self._lock = threading.Lock()
        self._current: contextvars.ContextVar[Usage | None] = contextvars.ContextVar(
            "analyst_copilot_question_usage", default=None
        )

    # -- per-question scope ---------------------------------------------
    def begin(self) -> None:
        """Start attributing this context's calls to a fresh question."""
        self._current.set(Usage())

    def take(self) -> Usage:
        """Return this question's usage and end the scope."""
        usage = self._current.get() or Usage()
        self._current.set(None)
        return usage

    # -- called by MeteredProvider --------------------------------------
    def record(self, stage: str, **kwargs) -> None:
        # Both accumulators move under ONE lock: concurrent verifiers would
        # otherwise race on the same per-question Usage.
        current = self._current.get()
        with self._lock:
            if current is not None:
                current.record(stage, **kwargs)
            self.total.record(stage, **kwargs)


class MeteredProvider:
    """Wraps an `LLMProvider`, recording usage, and changing nothing else.

    Implements the same protocol, so the pipeline cannot tell the difference —
    which is the point: a harness that perturbed the thing it measures would
    make every number it produced suspect.
    """

    def __init__(self, inner: LLMProvider, ledger: CostLedger, stage: str) -> None:
        self._inner = inner
        self._ledger = ledger
        self._stage = stage

    def complete(self, *, system: str, user: str, schema: dict, stage: str) -> LLMResponse:
        started = time.time()
        try:
            response = self._inner.complete(
                system=system, user=user, schema=schema, stage=stage
            )
        except Exception:
            # A failed call still consumed time, and often tokens. Recording it
            # as an error keeps a rate-limit storm visible instead of looking
            # like a cheap run.
            self._ledger.record(
                stage or self._stage, seconds=time.time() - started, error=True
            )
            raise
        self._ledger.record(
            stage or self._stage,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
            seconds=time.time() - started,
        )
        return response


def meter_pipeline(pipeline, ledger: CostLedger):
    """Wrap every provider a `QueryPipeline` holds. Returns the same pipeline.

    Applied after `build_pipeline()` so the composition root stays unaware of
    evaluation. Any provider left as None (the ablation runner drops verifiers)
    stays None.
    """
    for attribute in ("extractor", "verifier_a", "verifier_b", "router_llm", "composer"):
        provider = getattr(pipeline, attribute, None)
        if provider is not None and not isinstance(provider, MeteredProvider):
            setattr(pipeline, attribute, MeteredProvider(provider, ledger, attribute))
    return pipeline


def meter_judge(judge, ledger: CostLedger):
    """Wrap the eval judge too — it is a real cost, and it is not free to forget.

    The judge grades the 84 non-numeric answers, so at one call each it is a
    material share of a full run. Reporting it separately keeps the harness's
    own spend distinguishable from the system's.
    """
    if judge is None:
        return None
    inner = getattr(judge, "_provider", None)
    if inner is not None and not isinstance(inner, MeteredProvider):
        judge._provider = MeteredProvider(inner, ledger, "judge")
    return judge

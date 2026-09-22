"""The LLM judge for the two free-text answer shapes.

MEASURED: 84 of 136 gold answers (62%) are non-numeric. Without a judge the
harness scores every phrase and multi-sentence answer as WRONG, which is not
conservatism - it is a broken measurement that would send the whole build
chasing the 52 numeric questions.

The judge is used ONLY by `eval/`, never by the pipeline. It grades; it does not
help the system answer. Keeping it here is what stops evaluation logic leaking
into the thing being evaluated.

Cached on (question, gold, candidate) because ablation runs re-score the same
answers many times and a judge call is neither free nor deterministic.
"""

from __future__ import annotations

from functools import lru_cache

from ..llm import schemas
from ..llm.base import LLMError, LLMProvider
from ..llm.registry import load_prompt


class LLMJudge:
    """Implements the `Judge` protocol in `eval/scorer.py`."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @lru_cache(maxsize=4096)
    def _grade(self, question: str, gold: str, candidate: str) -> tuple[bool, str]:
        if not candidate.strip():
            return False, "empty answer"
        prompt_name, schema = schemas.BY_STAGE["judge"]
        user = (
            f"QUESTION:\n{question}\n\n"
            f"GOLD ANSWER:\n{gold}\n\n"
            f"CANDIDATE ANSWER:\n{candidate}"
        )
        try:
            data = self._provider.complete(
                system=load_prompt(prompt_name), user=user, schema=schema, stage="judge"
            ).data
        except LLMError as exc:
            # A judge that cannot run has not graded anything. Failing closed
            # UNDER-states the score, which is the safe direction for a harness:
            # it can never flatter the system.
            return False, f"judge unavailable: {exc}"
        return bool(data.get("equivalent")), str(data.get("reasoning") or "")

    def equivalent(self, question: str, gold: str, candidate: str) -> bool:
        return self._grade(question, gold, candidate)[0]

    def contradicted(self, question: str, gold: str, candidate: str) -> bool:
        """For yes/no answers, whose leading token already matched.

        Only the JUSTIFICATION is in question here, so a non-equivalent
        justification means contradicted.
        """
        return not self._grade(question, gold, candidate)[0]


def build_judge(settings) -> LLMJudge | None:
    """Construct the judge, or None if no judge stage is configured."""
    from ..llm.registry import get_provider

    try:
        return LLMJudge(get_provider(settings, "judge"))
    except Exception:
        return None

"""Stage -> provider resolution, and prompt loading.

Adding a model family is ONE adapter file plus ONE entry in `_ADAPTERS`.
Nothing else in the system changes, so moving verifier B to a different model
family is a `.env` edit (`VERIFIER_B_PROVIDER`, `VERIFIER_B_DEPLOYMENT`).

Prompts are versioned files, never inlined in code. A prompt is a tuned
artifact, so it belongs under review like any other measured component.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from ..config import Settings
from .base import Embedder, LLMProvider, Reranker

PROMPT_DIR = Path(__file__).parent / "prompts"


class UnknownProvider(RuntimeError):
    pass


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """Load `llm/prompts/<name>.txt`. Cached; prompts do not change at runtime."""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt {name!r} not found at {path}")
    return path.read_text(encoding="utf-8").strip()


def _azure_openai(settings: Settings, model) -> LLMProvider:
    from .azure_openai import AzureOpenAIProvider

    return AzureOpenAIProvider(settings, model)


_ADAPTERS: dict[str, Callable[[Settings, object], LLMProvider]] = {
    "azure_openai": _azure_openai,
}


def get_provider(settings: Settings, stage: str) -> LLMProvider:
    """Resolve config.yaml `models.<stage>` to a live adapter.

    stage in {router, extractor, composer, formula_pick, verifier_a,
              verifier_b, judge}
    """
    model = settings.model(stage)
    factory = _ADAPTERS.get(model.provider)
    if factory is None:
        raise UnknownProvider(
            f"stage {stage!r} wants provider {model.provider!r}; "
            f"known providers: {sorted(_ADAPTERS)}"
        )
    return factory(settings, model)


def get_embedder(settings: Settings) -> Embedder:
    from .azure_openai import AzureOpenAIEmbedder

    model = settings.model("embeddings")
    if model.provider != "azure_openai":
        raise UnknownProvider(f"no embedder adapter for {model.provider!r}")
    return AzureOpenAIEmbedder(settings, model)


def get_reranker(settings: Settings) -> Reranker:
    from .cohere_foundry import CohereFoundryReranker

    return CohereFoundryReranker(settings, settings.model("reranker"))


def verifier_prompt(settings: Settings) -> str:
    """Which verifier prompt verifier B gets, per `verification.independence`.

    Refutation framing is a substitute for independence, not an addition to
    it. With a genuine second family, set `verifier_b_adversarial: false` -
    running both over-abstains and costs answerable questions, and the
    abstention threshold must be re-calibrated after any such switch.
    """
    if settings.verification.verifier_b_adversarial:
        return load_prompt("verify_adversarial")
    return load_prompt("verify")

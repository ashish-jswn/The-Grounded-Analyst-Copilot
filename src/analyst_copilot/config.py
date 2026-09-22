"""Typed settings: config.yaml + .env -> Settings.

This is the only module that reads os.environ. Stages receive their settings;
they never fetch them. A grep for `os.environ` outside this file is a bug, and
so is a numeric literal in stage code.

Split of responsibility:
    .env         secrets + endpoints + deployment names ONLY
    config.yaml  everything tunable
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Exact refusal string required by the scoring rubric. Never paraphrase this.
NOT_FOUND = "Not found in this filing."


class ConfigError(RuntimeError):
    """Raised at startup for a missing or invalid setting. Fail fast, never
    fall back to a default that would silently change measured behaviour."""


# ---------------------------------------------------------------------------
# Per-stage bundles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSettings:
    """One row of config.yaml `models`, with env indirection already resolved."""

    stage: str
    provider: str
    deployment: str
    reasoning_effort: str | None = None
    verbosity: str | None = None
    max_completion_tokens: int | None = None
    dimensions: int | None = None
    endpoint: str | None = None


@dataclass(frozen=True)
class RoutingSettings:
    top_k_filings: int
    clarify_when_no_company: bool
    reroute_max: int
    fallback_next_4: bool
    prefer_coverage_years: bool


@dataclass(frozen=True)
class RetrievalSettings:
    bm25_top_k: int
    bm25_top_k_escalated: int
    dense_top_k: int
    rrf_k: int
    rerank_top_n: int
    assembly_token_budget: int
    neighbour_expand: int
    use_reranker: bool
    use_dense: bool


@dataclass(frozen=True)
class VerificationSettings:
    require_all_gates: bool
    verifier_policy: str
    independence: str
    verifier_b_adversarial: bool
    abstain_string: str
    # Gate toggles exist so eval/ablate.py can disable exactly one gate and
    # measure its contribution. Absent from config => all enabled.
    # Turning verifiers off is a scoring decision, not a performance one.
    # Measured on 25 stratified questions: gates-only answered 20/25 instead
    # of 12/25 and scored +4 against +6, because the 8 extra answers were 3
    # right and 5 wrong — 37% accuracy on exactly the questions verification
    # blocks, below the 50% break-even where answering (2p-1) beats refusing.
    # It is also ~20 s per question faster. Both are true; the trade is real.
    use_verifiers: bool = True
    enabled_gates: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")


@dataclass(frozen=True)
class IngestSettings:
    min_page_chars: int
    table_min_rows: int
    table_min_numeric_cells: int
    xbrl: bool
    xbrl_exclude_dimensional: bool


@dataclass(frozen=True)
class EvalSettings:
    numeric_tolerance: Decimal
    location_containment: float
    page_seq_slack: int
    gold_map_min_jaccard: float
    split_by: str
    report_by_answer_shape: bool
    # Token prices for the eval spend report. DEFAULT None ON PURPOSE: tokens
    # are measured exactly, dollars are only ever derived from a rate someone
    # confirmed. A hardcoded guess would produce a confident budget number that
    # is silently wrong, which is worse than no number at all.
    price_per_mtok_input: float | None = None
    price_per_mtok_output: float | None = None


@dataclass(frozen=True)
class Settings:
    profile: str
    database_url: str
    filings_dir: Path
    practice_questions: Path
    data_dir: Path
    models: dict[str, ModelSettings]
    routing: RoutingSettings
    retrieval: RetrievalSettings
    verification: VerificationSettings
    ingest: IngestSettings
    eval: EvalSettings
    # Endpoint/credential material, resolved once here so no stage reads env.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str | None = None
    cohere_rerank_endpoint: str | None = None
    cohere_rerank_api_key: str | None = None

    def model(self, stage: str) -> ModelSettings:
        try:
            return self.models[stage]
        except KeyError:
            raise ConfigError(
                f"no model binding for stage {stage!r}; "
                f"known stages: {sorted(self.models)}"
            ) from None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _req(env: dict[str, str], key: str) -> str:
    v = (env.get(key) or "").strip()
    if not v:
        raise ConfigError(f"{key} is not set in .env")
    return v


def _resolve_model(stage: str, raw: dict[str, Any], env: dict[str, str]) -> ModelSettings:
    """Resolve one `models.<stage>` entry.

    `provider` may be given literally or via `provider_env` (verifier_b uses the
    latter, which is what makes a model-family swap a .env change).
    """
    if "provider_env" in raw:
        provider = _req(env, raw["provider_env"])
    else:
        provider = raw.get("provider") or ""
    if not provider:
        raise ConfigError(f"models.{stage}: no provider or provider_env")

    if "deployment_env" in raw:
        deployment = _req(env, raw["deployment_env"])
    else:
        deployment = raw.get("deployment") or raw.get("model") or ""
    if not deployment:
        raise ConfigError(f"models.{stage}: no deployment, deployment_env or model")

    endpoint = None
    if "endpoint_env" in raw:
        endpoint = _req(env, raw["endpoint_env"])

    return ModelSettings(
        stage=stage,
        provider=provider,
        deployment=deployment,
        reasoning_effort=raw.get("reasoning_effort"),
        verbosity=raw.get("verbosity"),
        max_completion_tokens=raw.get("max_completion_tokens"),
        dimensions=raw.get("dimensions"),
        endpoint=endpoint,
    )


def load_settings(
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> Settings:
    """Read config.yaml + .env into a validated Settings. Fails fast."""
    config_path = config_path or PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise ConfigError(f"config.yaml not found at {config_path}")

    load_dotenv(env_path or PROJECT_ROOT / ".env")
    env = dict(os.environ)

    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    models = {
        stage: _resolve_model(stage, raw, env)
        for stage, raw in (cfg.get("models") or {}).items()
    }

    # The embedding width is asserted in three places (config.yaml, .env and the
    # vector(N) column). A mismatch makes EVERY page insert fail at the end of a
    # long ingest, so it is checked once, here, at startup.
    emb = models.get("embeddings")
    if emb is not None and emb.dimensions is not None:
        declared = env.get("EMBEDDING_DIMENSIONS")
        if declared and int(declared) != emb.dimensions:
            raise ConfigError(
                f"models.embeddings.dimensions={emb.dimensions} but "
                f"EMBEDDING_DIMENSIONS={declared}. These must match "
                f"pages.embedding vector(N) or every insert fails."
            )

    r = cfg["routing"]
    rt = cfg["retrieval"]
    v = cfg["verification"]
    ing = cfg["ingest"]
    ev = cfg["eval"]

    filings_dir = Path(_req(env, "FILINGS_DIR"))
    if not filings_dir.is_absolute():
        filings_dir = (PROJECT_ROOT / filings_dir).resolve()
    questions = Path(_req(env, "PRACTICE_QUESTIONS"))
    if not questions.is_absolute():
        questions = (PROJECT_ROOT / questions).resolve()
    data_dir = Path(env.get("ANALYST_COPILOT_DATA_DIR", "./data"))
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    return Settings(
        profile=cfg.get("profile", "prod"),
        database_url=_req(env, "DATABASE_URL"),
        filings_dir=filings_dir,
        practice_questions=questions,
        data_dir=data_dir,
        models=models,
        routing=RoutingSettings(
            top_k_filings=r["top_k_filings"],
            clarify_when_no_company=r["clarify_when_no_company"],
            reroute_max=r["reroute_max"],
            fallback_next_4=r["fallback_next_4"],
            prefer_coverage_years=r.get("prefer_coverage_years", True),
        ),
        retrieval=RetrievalSettings(
            bm25_top_k=rt["bm25_top_k"],
            bm25_top_k_escalated=rt["bm25_top_k_escalated"],
            dense_top_k=rt["dense_top_k"],
            rrf_k=rt["rrf_k"],
            rerank_top_n=rt["rerank_top_n"],
            assembly_token_budget=rt["assembly_token_budget"],
            neighbour_expand=rt["neighbour_expand"],
            use_reranker=rt.get("use_reranker", True),
            use_dense=rt.get("use_dense", True),
        ),
        verification=VerificationSettings(
            require_all_gates=v["require_all_gates"],
            verifier_policy=v["verifier_policy"],
            use_verifiers=bool(v.get("use_verifiers", True)),
            independence=v["independence"],
            verifier_b_adversarial=v["verifier_b_adversarial"],
            abstain_string=v["abstain_string"],
            enabled_gates=tuple(
                v.get("enabled_gates", ["G1", "G2", "G3", "G4", "G5", "G6", "G7"])
            ),
        ),
        ingest=IngestSettings(
            min_page_chars=ing["min_page_chars"],
            table_min_rows=ing["table_min_rows"],
            table_min_numeric_cells=ing["table_min_numeric_cells"],
            xbrl=ing["xbrl"],
            xbrl_exclude_dimensional=ing["xbrl_exclude_dimensional"],
        ),
        eval=EvalSettings(
            numeric_tolerance=Decimal(str(ev["numeric_tolerance"])),
            location_containment=ev["location_containment"],
            page_seq_slack=ev["page_seq_slack"],
            gold_map_min_jaccard=ev["gold_map_min_jaccard"],
            split_by=ev["split_by"],
            report_by_answer_shape=ev.get("report_by_answer_shape", True),
            price_per_mtok_input=(ev.get("pricing") or {}).get("input_per_mtok"),
            price_per_mtok_output=(ev.get("pricing") or {}).get("output_per_mtok"),
        ),
        azure_openai_endpoint=(env.get("AZURE_OPENAI_ENDPOINT") or "").strip() or None,
        azure_openai_api_key=(env.get("AZURE_OPENAI_API_KEY") or "").strip() or None,
        azure_openai_api_version=(env.get("AZURE_OPENAI_API_VERSION") or "").strip() or None,
        cohere_rerank_endpoint=(env.get("COHERE_RERANK_ENDPOINT") or "").strip() or None,
        cohere_rerank_api_key=(env.get("COHERE_RERANK_API_KEY") or "").strip() or None,
    )

"""The LLM seam: schema/prompt consistency and family-swap safety.

These run OFFLINE. They do not call a model - they check the contract that makes
a model call safe, which is the part that broke in practice.
"""

from __future__ import annotations

import re

import pytest

from analyst_copilot.config import load_settings
from analyst_copilot.llm import schemas
from analyst_copilot.llm.registry import PROMPT_DIR, load_prompt, verifier_prompt

# A field name as written in a prompt's "  name : description" listing.
_PROMPT_FIELD = re.compile(r"^\s{2}([a-z_]+)\s*:", re.M)


def _all_property_names(node: dict) -> set[str]:
    """Every property name at any depth.

    A prompt lists the fields of one ITEM ("clean_title", "kind"), while the
    schema may wrap them in an array ("sections"), so the comparison has to
    recurse or it reports false positives.
    """
    names: set[str] = set()
    if node.get("type") == "object":
        for name, child in node.get("properties", {}).items():
            names.add(name)
            names |= _all_property_names(child)
    elif node.get("type") == "array":
        names |= _all_property_names(node["items"])
    return names


def test_every_prompt_file_loads():
    for path in PROMPT_DIR.glob("*.txt"):
        assert load_prompt(path.stem).strip(), f"{path.name} is empty"


@pytest.mark.parametrize("stage", sorted(schemas.BY_STAGE))
def test_prompt_and_schema_declare_the_same_fields(stage):
    """MEASURED FAILURE: the router prompt listed `form_types` and `periods`
    while the schema omitted them, and gpt-5-mini crammed the missing fields
    into a string field rather than choose between two contradicting
    instructions."""
    prompt_name, schema = schemas.BY_STAGE[stage]
    prompt = load_prompt(prompt_name)
    declared = set(_PROMPT_FIELD.findall(prompt))
    if not declared:
        pytest.skip(f"{prompt_name} does not enumerate fields")
    missing = declared - _all_property_names(schema)
    assert not missing, (
        f"{prompt_name}.txt asks for {sorted(missing)} which "
        f"{stage}'s schema does not declare"
    )


@pytest.mark.parametrize("stage", sorted(schemas.BY_STAGE))
def test_schemas_are_azure_strict_mode_compatible(stage):
    """Strict mode requires additionalProperties:false and every property in
    `required`; an optional field must be a nullable type, not an absent one."""

    def check(node: dict) -> None:
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        elif node.get("type") == "array":
            check(node["items"])

    check(schemas.BY_STAGE[stage][1])


def test_the_extractor_schema_requires_a_quote_and_a_location():
    """Gate G1 checks `quote` against the cited page, so both must be present
    on every slot or the gate has nothing to check."""
    slot = schemas.EXTRACT["properties"]["slots"]["items"]
    for field in ("quote", "doc_id", "page_seq"):
        assert field in slot["properties"]
        assert field in slot["required"]


def test_the_extractor_can_report_missing_slots():
    """Our prompts invert OpenAI's escape hatch - the model needs a
    cheap, explicit way to STOP rather than proceed under uncertainty."""
    assert "missing_slots" in schemas.EXTRACT["properties"]
    prompt = load_prompt("extract")
    assert "missing_slots" in prompt
    assert "stop" in prompt.lower()


def test_extractor_prompt_forbids_using_model_knowledge():
    prompt = load_prompt("extract").lower()
    assert "verbatim" in prompt
    assert "own knowledge" in prompt


def test_verifier_prompt_follows_the_independence_setting():
    """Refutation framing is a SUBSTITUTE for a second family, not an addition.

    THIS TEST USED TO REQUIRE "Default to REFUTED when uncertain", AND THAT
    CLAUSE WAS REMOVED ON EVIDENCE. MEASURED on the full practice set: 48 of
    90 refusals were verifier rejections, and 40 of those (83%) objected to
    units, period or column labels - while 40 of the 48 HAD the gold page in
    context. A quote is one row of a financial table; its units and fiscal-year
    headings live in the table header, so "uncertain" was the default state for
    almost every correct answer, and the clause converted that into rejection.

    The adversarial FRAMING is what buys independence and it is still asserted
    below. What is no longer acceptable is rejecting without a nameable defect.
    """
    settings = load_settings()
    prompt = verifier_prompt(settings)
    if settings.verification.verifier_b_adversarial:
        assert "REFUTE" in prompt
        # A refutation must point at something, not merely feel unsure.
        assert "must name a defect" in prompt
        assert "Default to REFUTED when uncertain" not in prompt
    else:
        assert "VALID" in prompt


def test_verifier_prompts_direct_the_reviewer_to_the_cited_page():
    """Both prompts must tell the reviewer to resolve units from the page.

    The verifier is handed the full cited page precisely so that units and
    fiscal-year columns are resolvable; a prompt that does not say so leaves
    the reviewer objecting to the terseness of the quote instead of reading.
    """
    for name in ("verify", "verify_adversarial"):
        prompt = load_prompt(name)
        assert "table header" in prompt, name
        assert "in millions" in prompt, name


def test_verifier_b_is_configured_and_swappable():
    """The swap to a real second family must stay a .env change."""
    settings = load_settings()
    assert settings.model("verifier_b").provider
    assert settings.model("verifier_b").deployment
    assert settings.verification.independence in {
        "same_model_adversarial",
        "different_model",
        "different_family",
    }


def test_reasoning_truncation_has_its_own_error_type():
    """gpt-5-mini returns content='' with finish_reason='length' and NO error
    when the budget is spent on hidden reasoning. That must never be mistaken
    for a refusal, which would silently become an abstention."""
    from analyst_copilot.llm.base import LLMError, LLMTruncated

    assert issubclass(LLMTruncated, LLMError)


def test_verifiers_can_be_switched_off_without_touching_gates():
    """`use_verifiers: false` must disable ONLY the LLM verification.

    The deterministic gates G1-G7 do the grounding work and cost nothing: a
    quote must still appear verbatim on its cited page, the figure must appear
    inside its own quote, and the arithmetic must still recompute. Turning off
    the LLM reviewers must not quietly relax any of that.
    """
    settings = load_settings()
    v = settings.verification
    assert isinstance(v.use_verifiers, bool)
    # Whatever the switch is set to, the gates stay armed.
    assert v.require_all_gates is True
    assert "G1" in v.enabled_gates and "G1b" in v.enabled_gates
    assert "G6" in v.enabled_gates


def test_switching_verifiers_off_removes_both_providers():
    """Both must go: leaving one wired would make `unanimous` mean "A alone"."""
    from dataclasses import replace

    from analyst_copilot.container import Corpus, build_pipeline
    from analyst_copilot.retrieval.anchors import build_from_rows
    from analyst_copilot.retrieval.bm25 import BM25Index

    empty = Corpus(
        bm25=BM25Index([]),
        anchors=build_from_rows([]),
        pages_by_doc={},
        headers_by_page={},
        coverage_years={},
        n_pages=0,
        n_docs=0,
    )
    settings = load_settings()
    off = replace(
        settings, verification=replace(settings.verification, use_verifiers=False)
    )
    pipeline = build_pipeline(off, corpus=empty, with_router_llm=False)
    assert pipeline.verifier_a is None
    assert pipeline.verifier_b is None

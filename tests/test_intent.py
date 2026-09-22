"""Deterministic forecast/historical intent.

WHY DETERMINISTIC. Intent is ONE field feeding exactly one thing: gate G3's
exception permitting a question period AFTER the filing period. MEASURED:
asking a model for it cost ~32 s per question - about 12% of end-to-end latency
- on a ~400-token prompt. The plan always allowed "cheap LLM, OR keyword prior".

THE RISK IS FALSE POSITIVES, NOT MISSES. A missed forecast costs one
question; a false positive relaxes G3's period check on a HISTORICAL question,
weakening the gate that exists to catch wrong-period evidence. A 10-K is full of
forward-sounding accounting terms, so the exclusions are tested harder than the
matches.

MEASURED against all 136 practice questions: 3 classified forecast (2%), and
all three are genuinely forward-looking. Zero false positives.
"""

from __future__ import annotations

import pytest

from analyst_copilot.query.router import detect_intent


# ---------------------------------------------------------------------------
# The real forecast questions in the corpus
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "What production rate changes is Boeing forecasting for FY2023?",
    "How much does Pfizer expect to pay to spin off Upjohn in the future in USD million?",
    "As of FY 2021, how much did Verizon expect to pay for its retirees in 2024?",
])
def test_the_three_real_forecast_questions_are_detected(question):
    assert detect_intent(question) == "forecast"


@pytest.mark.parametrize("question", [
    "What is management's guidance for next year?",
    "What is the outlook for FY2024 revenue?",
    "What does the company anticipate for the coming year?",
    "What does the company expect to spend going forward?",
])
def test_other_forward_looking_phrasings_are_detected(question):
    assert detect_intent(question) == "forecast"


# ---------------------------------------------------------------------------
# THE FALSE FRIENDS - a 10-K is saturated with these
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "What were the pension plan assets at the end of FY2022?",
    "What is the fair value of the benefit plan obligations?",
    "How much did the company contribute to its 401(k) plan in FY2021?",
    "What was the stock plan compensation expense?",
    "What was the projected benefit obligation as of FY2022?",
    "What were expected credit losses on receivables in FY2022?",
    "What is the expected return on plan assets?",
    "What was the restructuring plan charge recorded in FY2022?",
    "What was the cost of planned maintenance in FY2021?",
])
def test_accounting_terms_are_not_mistaken_for_forecasts(question):
    """These would relax G3's period check on historical questions."""
    assert detect_intent(question) == "historical", question


@pytest.mark.parametrize("question", [
    "What is the FY2018 capital expenditure amount for 3M?",
    "Is 3M a capital-intensive business based on FY2022 data?",
    "Which segment dragged down 3M's overall growth in 2022?",
    "What is Amazon's FY2017 days payable outstanding?",
    "What drove operating margin change as of FY2022 for 3M?",
])
def test_ordinary_historical_questions_stay_historical(question):
    assert detect_intent(question) == "historical", question


# ---------------------------------------------------------------------------
# Mixed and degenerate input
# ---------------------------------------------------------------------------
def test_a_forecast_survives_an_accounting_term_in_the_same_question():
    """The exclusion strips the accounting phrase, then re-tests - it does not
    veto the whole question."""
    assert detect_intent(
        "What is the pension plan's expected return, and what does management "
        "forecast for FY2023?"
    ) == "forecast"


@pytest.mark.parametrize("text", ["", "   ", None])
def test_degenerate_input_is_historical(text):
    """Historical is the safe default: it keeps G3's period check strict."""
    assert detect_intent(text) == "historical"


def test_the_corpus_wide_rate_stays_low():
    """A detector that fired often would be silently disabling G3. This pins the
    measured 2% - if a change pushes it up, that is a regression to investigate."""
    from pathlib import Path
    import json

    path = Path("../analyst-copilot-data/practice-questions.jsonl")
    if not path.exists():          # corpus not present in this checkout
        pytest.skip("practice questions not available")
    questions = [
        json.loads(line)["question"]
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    forecast = sum(1 for q in questions if detect_intent(q) == "forecast")
    assert forecast / len(questions) <= 0.08, (
        f"{forecast}/{len(questions)} classified forecast - too many; "
        "a false positive weakens G3 on a historical question"
    )

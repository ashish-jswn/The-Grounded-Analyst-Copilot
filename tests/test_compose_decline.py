"""A refusal written as prose must reach the abstain path, not the answer path.

MEASURED (financebench_id_00678). The composer replied "I cannot determine
whether Boeing has an improving gross margin profile as of FY2022 because ... no
gross profit or cost of goods sold figures are provided" - the exact behaviour
this system exists to produce. It set `answerable: true`, the text shipped as
`status="answered"`, and the rubric scored it **-1 instead of 0**.

The system was right and the plumbing punished it, for two points.
"""

from __future__ import annotations

import pytest

from analyst_copilot.query.compose import ComposedAnswer, compose_answer, reads_as_decline
from analyst_copilot.query.extract import EvidenceSlot, Extraction


def slot(value="66,608", name="revenue"):
    return EvidenceSlot(
        name=name, value=value, unit="USD", scale="millions", period="FY2022",
        doc_id="BOEING_2022_10K", page_seq=60, page_printed=None,
        quote="Total revenues 66,608 62,286 58,158",
    )


class _Composer:
    """Returns whatever the model would have returned."""

    def __init__(self, answer: str, answerable: bool = True):
        self._data = {"answer": answer, "answerable": answerable}

    def complete(self, *, system, user, schema, stage):
        class R:
            data = self._data
        return R()


# ---------------------------------------------------------------------------
# The measured case
# ---------------------------------------------------------------------------
def test_the_boeing_refusal_is_routed_to_abstain():
    composed = compose_answer(
        Extraction(slots=[slot()]), None,
        "Does Boeing have an improving gross margin profile as of FY2022?",
        [2022],
        _Composer(
            "I cannot determine whether Boeing has an improving gross margin "
            "profile as of FY2022 because the only supplied data are total "
            "revenues and no gross profit or cost of goods sold figures are provided."
        ),
    )
    assert composed.answerable is False
    assert composed.text == ""


@pytest.mark.parametrize("text", [
    "I cannot determine the FY2022 gross margin.",
    "I am unable to answer from the supplied pages.",
    "Unable to determine the requested figure.",
    "It is not possible to determine the segment breakdown.",
    "The provided excerpts do not contain the requested line item.",
    "The supplied pages do not include a cost of goods sold figure.",
    "There is insufficient evidence in the quotes to answer.",
    "No information is provided about the restructuring charge.",
    "Cannot be determined from the given evidence.",
])
def test_declining_openings_are_detected(text):
    assert reads_as_decline(text), text


# ---------------------------------------------------------------------------
# It must NOT swallow real answers - that would trade a +1 for a 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "$1,577 million in FY2018 for purchases of property, plant and equipment.",
    "No, the company is not capital-intensive: CAPEX/Revenue is 5.1%.",
    "Yes. Gross profit improved from $3,017 million to $3,502 million.",
    "The consumer segment shrunk by 0.9% organically.",
    # A real answer may mention absence PARTWAY THROUGH - that is not a refusal.
    "Segment-level detail is not disclosed, but total revenue was $86,392 million.",
    "Revenue grew 8.7%. The company does not provide a regional breakdown.",
    "16.5%",
])
def test_real_answers_are_not_mistaken_for_refusals(text):
    assert not reads_as_decline(text), text


def test_a_normal_answer_still_composes():
    composed = compose_answer(
        Extraction(slots=[slot()]), None, "What was FY2022 revenue?", [2022],
        _Composer("Total revenues were $66,608 million in FY2022."),
    )
    assert composed.answerable is True
    assert composed.used_model is True
    assert "66,608" in composed.text


def test_the_explicit_answerable_flag_is_still_honoured():
    composed = compose_answer(
        Extraction(slots=[slot()]), None, "q", [2022],
        _Composer("anything at all", answerable=False),
    )
    assert composed.answerable is False


def test_empty_and_none_are_not_declines():
    """An empty answer is handled by the existing empty-slot path, not here."""
    assert not reads_as_decline("")
    assert not reads_as_decline(None)

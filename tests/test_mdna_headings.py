"""Enumerated headings survive a trailing full stop.

MEASURED. `_is_candidate` applied a prose heuristic — "ends in a full stop and
runs on" — ABOVE the structural test, so

    "Item 7. Management's Discussion and Analysis of Financial Condition and
     Results of Operations."

was discarded as prose: 13 words, trailing period. 86 such headings across 20 of
78 filings, including Item 7 on every one of them.

The consequence reached all the way to the answers. 3M_2022_10K ended up with NO
MD&A section — pages 18-47 absorbed into "PART II" — so MD&A anchoring covered 5
pages of 131. The word "organic", which the gold answer to
financebench_id_01865 turns on, appears on 12 pages, of which only 2 were
reachable by any anchor. The composer was handed segment operating income and
answered from it.

AND THE RULE COULD NEVER HAVE HELPED: `_is_candidate` only returns True via a
structural match, so a prose line returns False regardless. The full-stop test
could only ever discard genuine headings.
"""

from __future__ import annotations

import pytest

from analyst_copilot.ingest.sections import _is_candidate, build_tree, classify_title


@pytest.mark.parametrize("heading", [
    "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations.",
    "Item 7A. Quantitative and Qualitative Disclosures About Market Risk.",
    "Item 9. Changes in and Disagreements With Accountants on Accounting and Financial Disclosure.",
    "Item 10. Directors, Executive Officers and Corporate Governance.",
    "Item 13. Certain Relationships and Related Transactions, and Director Independence.",
    "Note 12. Long-Term Debt and Short-Term Borrowings.",
    "PART II.",
])
def test_enumerated_headings_survive_a_trailing_full_stop(heading):
    assert _is_candidate(heading), heading


def test_the_mdna_heading_is_classified_as_mdna():
    kind, stmt = classify_title(
        "Item 7. Management's Discussion and Analysis of Financial Condition "
        "and Results of Operations."
    )
    assert kind == "mdna"


@pytest.mark.parametrize("prose", [
    # The case the removed rule claimed to guard — still rejected, by LENGTH.
    "The following tables contain sales and operating income results by business "
    "segment for the fourth quarters of 2018 and 2017, and the related change.",
    "Additional information about results of operations and financial condition "
    "for 2021 and 2020 can be found in the sections referenced above.",
    "Refer to the consolidated statements of cash flows for further detail on this item.",
])
def test_prose_is_still_rejected(prose):
    assert not _is_candidate(prose), prose


def test_a_buried_statement_title_is_still_a_cross_reference():
    """A statement title must OPEN the line; inside a sentence it is a pointer,
    not the start of that statement."""
    assert not _is_candidate("as shown in the consolidated statements of cash flows")
    assert _is_candidate("Consolidated Statements of Cash Flows")


def test_mdna_now_becomes_a_section_of_its_own():
    """Before the fix this produced no MD&A node at all, and the pages fell into
    whatever section preceded them."""
    pages = [
        (1, "PART II.\nsome front matter"),
        (2, "Item 7. Management's Discussion and Analysis of Financial Condition "
            "and Results of Operations.\nOverview of results."),
        (3, "Organic sales growth by segment was mixed this year."),
        (4, "Consumer declined 0.9% organically during 2022."),
        (5, "Consolidated Statement of Income\nNet sales ..."),
    ]
    tree = build_tree(pages, {}, form_type="10-K")
    mdna = [s for s in tree if s.kind == "mdna"]
    assert mdna, "no MD&A section was created"

    section = mdna[0]
    assert section.page_start == 2
    # It must SPAN the narrative pages, not stop at its own title page —
    # otherwise the organic-growth discussion stays unreachable.
    assert section.page_end >= 4

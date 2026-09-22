"""Answer-shape classification.

THE MAJORITY CASE IS NOT NUMBER EXTRACTION. MEASURED: 84 of 136 gold
answers (62%) are non-numeric - judgements, phrases, explanations. A system that
is strong on the 52 numeric answers and weak on the other 84 looks fine in
aggregate right up until the live session, which is why the harness scores each
shape with its own predicate and REPORTS THEM SEPARATELY.

The shape is derived from the GOLD ANSWER, never from the FinanceBench
`question_type` label: MEASURED that label cannot separate `domain-relevant`
from `novel-generated`, and it does not exist at run time anyway.
"""

from __future__ import annotations

import re
from enum import Enum


class AnswerShape(str, Enum):
    NUMERIC = "numeric"
    YES_NO = "yes_no"
    PHRASE = "phrase"
    MULTI_SENTENCE = "multi_sentence"


# A leading yes/no token, optionally hedged ("Yes, but ...", "No. The company").
_YES_NO = re.compile(r"^\s*(yes|no)\b", re.I)

# A gold answer that is essentially one figure, possibly with a unit or a sign:
# "$1577.00", "12.4%", "-2.3", "$8.70 billion", "1.5x", "USD 3,222 million".
_NUMERIC_ANSWER = re.compile(
    r"^\s*(?:\$|usd|eur|£|€)?\s*[-+(]?\s*\d[\d,]*(?:\.\d+)?\s*\)?\s*"
    r"(?:%|x|bps|billion|million|thousand|bn|mm|k|usd|days?|years?)?\s*"
    r"(?:\$|usd)?\s*\.?\s*$",
    re.I,
)

# A sentence terminator. `(?=...)` rather than consuming the delimiter, because
# gold answers run a parenthetical straight onto the full stop:
#   "...between FY2023 and FY2022.(3.4% jump)"
_SENTENCE_END = re.compile(r"[.!?](?=\s|\(|\$|$)")

# A list is always more than one claim, however it is punctuated.
_LIST_MARKER = re.compile(r"[\n\r]|(?:^|\s)[-•*]\s")

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Abbreviations whose full stop is not a sentence end.
_ABBREV = re.compile(r"\b(?:U\.S|Inc|Corp|Ltd|No|approx|vs|e\.g|i\.e)\.")


def classify_answer(answer: str) -> AnswerShape:
    """Classify one gold answer into the shape whose predicate should score it.

    Order matters. A yes/no answer often continues into a numeric justification
    ("Yes, revenue grew 3.2%"), and it must be scored as a yes/no - the leading
    token is the claim, the rest is support.
    """
    text = (answer or "").strip()
    if not text:
        return AnswerShape.PHRASE

    if _YES_NO.match(text):
        return AnswerShape.YES_NO

    if _NUMERIC_ANSWER.match(text):
        return AnswerShape.NUMERIC

    # The phrase / multi-sentence boundary decides ONE thing: whether the
    # location predicate demands claim-level citations. So the test is "does
    # this answer assert more than one citable claim?", not "is it long?".
    if _LIST_MARKER.search(text):
        return AnswerShape.MULTI_SENTENCE

    stripped = _ABBREV.sub("", text)
    stripped = re.sub(r"(?<=\d)\.(?=\d)", "", stripped)   # 3.4 is not two sentences
    if len([s for s in _SENTENCE_END.split(stripped) if s.strip()]) >= 2:
        return AnswerShape.MULTI_SENTENCE

    # Two or more figures means two or more operands, and therefore two or more
    # citations. This reproduces the measured 26 phrase / 23 multi-sentence
    # split exactly; a length threshold does not.
    if len(_NUMBER.findall(text)) >= 2:
        return AnswerShape.MULTI_SENTENCE

    return AnswerShape.PHRASE

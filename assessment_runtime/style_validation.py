"""Bounded style/feedback checks, not a grammar or semantic accuracy oracle."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from collections.abc import Sequence

from assess_core.schemas import SchemaValidationError
from assessment_runtime.feedback_claims import contains_example, normalized

# Only explicit, scoped reassurance is removed. A subsequent clause such as
# "but this is wrong" is still inspected. Quoted learner/replacement content is
# treated as speech content, rather than as an assessor's diagnosis.
_REASSURANCE = re.compile(
    r"\b(?:not|isn't|is not|isn’t|aren't|wasn't) (?:a |an )?(?:grammar |grammatical )?(?:error|mistake|correction)\b(?!-)|"
    r"\b(?:not|isn't|is not) (?:incorrect\w*|wrong|ungrammatical\w*)\b|"
    r"\bno (?:grammar |grammatical )?(?:errors?|mistakes?|corrections?)\b|"
    r"\bwithout (?:grammar |grammatical )?(?:errors?|mistakes?)\b|"
    r"(?<!not )(?<!n't )(?<!non )(?<!nicht )\b(?:nothing wrong|error-free|mistake-free|free of errors|fehlerfrei|fehlerlos)\b|"
    r"\b(?:does not|doesn't) (?:need|require) (?:a )?correction\b|"
    r"\bnon (?:è |e' |sono )?(?:affatto )?(?:un |una |uno )?(?:error[ei]|sbagli[oa]|sbagliat[oaie]|scorrett[oaie]|errat[oaie]|correzione)\b|"
    r"\bnessun[oa]? (?:error[ei]|sbagli[oa]|correzion[ei])\b|"
    r"\bsenza (?:error[ei]|sbagli[oa])\b|"
    r"\bnicht (?:falsch|fehlerhaft)\b|\bkein[enrms]* (?:fehler\w*|grammatikfehler\w*|korrektur\w*)\b|"
    r"\bohne (?:fehler|grammatikfehler)\b|"
    r"\b(?:correct as written|correct and natural)\b",
    re.IGNORECASE,
)
_ERRORS = re.compile(
    r"\b(?:not (?:correct|right|proper)|non (?:è )?corrett\w*|nicht (?:korrekt|richtig)|inaccurate|improper|incorrect\w*|ungrammatical\w*|wrong|mistakes?|errors?|correction\w*|"
    r"error[ei]|sbagli[oa]|sbagliat[oaie]|scorrett[oaie]|errat[oaie]|corregg\w*|correzion\w*|"
    r"falsch|fehler\w*|grammatikfehler\w*|korrektur\w*|korrigier\w*)\b", re.IGNORECASE,
)
_ACTION = re.compile(
    r"(?:^|\b(?:please|to|should|must|need to|needs to|can)\s+)(?:correct|fix)\b|"
    r"\b(?:corrected|correcting|fixing|corregg\w*|korrigier\w*)\b", re.IGNORECASE,
)
_COACH_ACTION = re.compile(r"\b(?:replace|replacing|substitute|change|switch|sostituisc\w*|cambia\w*|ersetz\w*)\b", re.IGNORECASE)
_QUOTED = re.compile(r'(?<!\w)[\'‘]([^\'’\n]+)[\'’](?!\w)|["“«„]([^"”»“\n]+)["”»“]')


def quoted_fragments(text: str) -> list[str]:
    return [a or b for a, b in _QUOTED.findall(text)]


def without_content_quotes(text: str, quotes: Sequence[str]) -> str:
    checked = normalized(text)
    for quote in sorted((normalized(q) for q in quotes if q.strip()), key=len, reverse=True):
        checked = re.sub(r"['‘\"«„“]" + re.escape(quote) + r"['’\"»“”]", "[speech content]", checked)
    return checked



def correction_claim(text: str, *, content_quotes: Sequence[str] = ()) -> bool:
    checked = normalized(without_content_quotes(text, content_quotes))
    checked = _REASSURANCE.sub("[reassurance]", checked)
    return any(_ERRORS.search(sentence) or _ACTION.search(sentence.strip())
               for sentence in re.split(r"[.!?;\n]+", checked))


def validate_optional_explanation(text: str, field: str, *, content_quotes: Sequence[str]) -> None:
    if correction_claim(text, content_quotes=content_quotes):
        raise SchemaValidationError(f"{field}: optional style must not be described as an error or correction")


def references_style(text: str, original: str, suggestion: str) -> bool:
    # Multiword expressions can be referred to without quotation marks. A single
    # word such as "good" needs an explicit quote, otherwise unrelated praise
    # ("Good examples") would be mistaken for a reference to the optional item.
    alternatives = (original, suggestion)
    if any(len(normalized(value).split()) > 1 and contains_example(text, value) for value in alternatives):
        return True
    original_words, suggestion_words = (normalized(value).split() for value in alternatives)
    changed = []
    for tag, a, b, c, d in SequenceMatcher(None, original_words, suggestion_words, autojunk=False).get_opcodes():
        if tag != "equal":
            changed.extend([" ".join(original_words[a:b]), " ".join(suggestion_words[c:d])])
    return any(fragment.strip() and any(value and contains_example(value, fragment)
                                       for value in changed)
               for fragment in quoted_fragments(text))


def validate_style_coaching(text: str, field: str, styles: list[dict], *, priority: bool = False) -> None:
    for index, style in enumerate(styles):
        for sentence in re.split(r"[.!?;\n]+", text):
            if references_style(sentence, style["original"], style["suggestion"]) and (
                priority or correction_claim(sentence, content_quotes=[style["original"], style["suggestion"]])
                or _COACH_ACTION.search(normalized(without_content_quotes(sentence, [style["original"], style["suggestion"]])))
            ):
                raise SchemaValidationError(f"{field}: keep optional style suggestion {index} in its dedicated section")

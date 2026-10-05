"""Bounded en/it/de generation checks, not semantic accuracy certification.

Acoustic measurements remain in deterministic reports. Unknown paraphrases can
escape these rules. Only known learner quotations are exempt, never arbitrary
model quotation marks; historical report reading is unaffected.
"""
from collections.abc import Sequence
import re
import unicodedata
from assess_core.schemas import SchemaValidationError

FEEDBACK_CLAIM_POLICY = "bounded_text_claims_v3"

_TERM = (r"pronounc\w*|pronunc\w*|accent(?:s|ed|o|i)?|inton\w*|rhythm\w*|ritm\w*|"
         r"hesitat\w*|esit(?:az|an)\w*|(?:sprech)?paus\w*|aussprache\w*|akzent\w*|"
         r"sprechtempo|sprechrhythmus|sprechgeschwindigkeit|betonung|zöger\w*|"
         r"pacing|paced|pace|eloquio|speaking speed|speech speed|"
         r"velocità (?:di |della |del )?(?:eloquio|produzione|parlato|emissione)")
_ACOUSTIC = re.compile(r"\b(?:" + _TERM + r")\b", re.IGNORECASE)
_DELIVERY = re.compile(r"\b(?:(?:smooth|natural|hesitant) delivery|delivery (?:is|sounds|appears|feels) (?:smooth|natural|hesitant))\b", re.IGNORECASE)
_TOPIC = re.compile(r"\b(?:(?:la|della|alla|in) pace|accent marks?|accento grafico|accento su \w+|ritmo di vita|rhythm of (?:city )?life|pace of life|lunch pauses?|pausa pranzo|pause pranzo|pausen bei der arbeit)\b", re.IGNORECASE)
_ARTICLE = r"(?:(?:the|la|il|lo|le|gli|i|die|der|das|den)\s+|l')?"
_SUBJECT = _ARTICLE + r"(?:" + _TERM + r"|delivery)"
_SUBJECTS = _SUBJECT + r"(?:\s*(?:,\s*(?:(?:and|or|e|o|und|oder)\s+)?|(?:and|or|e|o|und|oder)\s+)" + _SUBJECT + r")*"
_SOURCE = r"(?:text(?: alone)?|the text|transcript|the transcript|audio|a recording)"
_LIMITATION = re.compile(
    r"(?:(?:i|we|it is) )?(?:cannot assess|can't assess|cannot evaluate|unable to assess|not possible to assess) " + _SUBJECTS + r" (?:from|without) " + _SOURCE + r"|"
    + _SUBJECTS + r" (?:cannot be assessed|cannot be evaluated|cannot be determined|is not assessable|is unknown|are unknown|is unavailable|are unavailable)(?: (?:from|without) " + _SOURCE + r")?|"
    r"non (?:è possibile|posso|si può) (?:valutare|determinare) " + _SUBJECTS + r" (?:dal testo|dalla trascrizione|senza audio)|"
    + _SUBJECTS + r" non (?:è|sono) (?:valutabile|valutabili|disponibile|disponibili)(?: dal testo)?|"
    + _SUBJECTS + r" (?:kann|können) (?:anhand des (?:textes|transkripts) )?nicht (?:bewertet|beurteilt) werden", re.IGNORECASE)
_FOLLOWUP = re.compile(r"^(?:(?:however|but|yet|ma|però|tuttavia|aber|jedoch)[, ]+)?(?:it|they|this|that|es|sie|è|sono)\b.*\b(?:natural|clear|smooth|good|fluent|chiar\w*|naturale|natürlich|gut)\b", re.IGNORECASE)
_NO_ERRORS = re.compile(
    r"\bno (?:(?:grammatical|grammar|obvious|noticeable|significant|major|serious) ){0,2}(?:errors?|mistakes?)\b|"
    r"\b(?:error-free|free of errors|without errors|flawless grammar|grammatically correct)\b|"
    r"\b(?:grammar (?:is|was)|grammatica [èe]|die grammatik ist) (?:(?:fully|entirely|sempre) )?(?:correct|corretta|korrekt|einwandfrei)\b|"
    r"\b(?:non (?:ci sono(?: stati)?|sono presenti|sono stati rilevati)|nessun[oa]?) error[ei]\b|"
    r"\b(?:senza errori|grammaticalmente corrett[oa]|keine (?:grammati(?:kalischen|schen) )?fehler|fehlerfrei|fehlerlos|ohne fehler)\b", re.IGNORECASE)
_EXCEPTION = re.compile(r"\b(?:except|apart from|besides|other than|aside from|tranne|eccetto|außer|ausser)\b", re.IGNORECASE)
_GOAL = re.compile(r"^(?:aim|try|record|repeat|say|practice|speak|cerca|prova|ripeti|registra|versuche|übe)\b", re.IGNORECASE)


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).replace('’', "'").casefold().split())


def contains_example(text: str, example: str) -> bool:
    example = normalized(example)
    if not example:
        return False
    pattern = (r"(?<!\w)" if example[0].isalnum() else '') + re.escape(example)
    pattern += r"(?!\w)" if example[-1].isalnum() else ''
    return re.search(pattern, normalized(text)) is not None


def validate_claim_text(text: str, field: str, *, grammar_errors: bool = False,
                        known_quotes: Sequence[str] = (), grammar_examples: Sequence[str] = ()) -> None:
    original = normalized(text)
    checked = original
    for quote in sorted((normalized(q) for q in known_quotes if q.strip()), key=len, reverse=True):
        if not _ACOUSTIC.search(quote) or len(quote.split()) == 1:
            continue
        checked = re.sub(r"['‘\"«„“]" + re.escape(quote) + r"['\"»“”]", '[learner quote]', checked)
    checked = _TOPIC.sub('[topic phrase]', checked)
    limitation_seen = False
    for sentence in re.split(r"[.!?;\n]+", checked):
        sentence = sentence.strip()
        if not sentence:
            continue
        if limitation_seen and _FOLLOWUP.search(sentence):
            raise SchemaValidationError(f"{field}: unsupported acoustic follow-up after a limitation")
        if _ACOUSTIC.search(sentence) or _DELIVERY.search(sentence):
            if not _LIMITATION.fullmatch(sentence):
                raise SchemaValidationError(f"{field}: unsupported acoustic/delivery claim; discuss text only")
        limitation_seen = bool(_LIMITATION.fullmatch(sentence))
    if grammar_errors:
        for sentence in re.split(r"[.!?;\n]+", checked):
            if not _NO_ERRORS.search(sentence):
                continue
            # A goal exempts its own clause, never a later assertion.
            contradiction = any(not _GOAL.search(clause.strip()) and _NO_ERRORS.search(clause)
                                for clause in re.split(r"[:,]", sentence))
            qualified = _EXCEPTION.search(sentence) and any(contains_example(sentence, q) for q in grammar_examples)
            if contradiction and not qualified:
                raise SchemaValidationError(f"{field}: contradicts the retained grammar findings")

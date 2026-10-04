"""Generation-time checks; historical report loading intentionally stays lenient.

Grounding proves membership in ASR text, not grammatical correctness. Matching
allows NFC, case, whitespace and typographic apostrophe differences only; it
does not remove punctuation or use fuzzy matching.
"""

from __future__ import annotations

import math
import re
import unicodedata

from assess_core.coaching_taxonomy import (
    COACHING_CONFIDENCE_LEVELS, COHERENCE_ISSUE_CATEGORIES,
    GRAMMAR_ERROR_CATEGORIES, LEXICAL_GAP_CATEGORIES,
)
from assess_core.schemas import CoachingSummary, RubricResult, SchemaValidationError
from assessment_runtime.transcript_quality import finite_number
from assessment_runtime.feedback_claims import contains_example, validate_claim_text
from assessment_runtime.style_validation import validate_optional_explanation, validate_style_coaching

ABSENCE_COHERENCE_CATEGORIES = {
    "missing_sequence_markers", "insufficient_linking", "underdeveloped_detail",
}


class UncertainTranscriptEvidenceError(SchemaValidationError):
    """A claimed issue relies on observed uncertain or unaligned ASR evidence."""


def _bounded_text_pattern(text: str) -> re.Pattern:
    pattern = (r"(?<!\w)" if text[0].isalnum() else "") + re.escape(text)
    pattern += r"(?!\w)" if text[-1].isalnum() else ""
    return re.compile(pattern)


def _asr_quote_guard(source: str | None, quote: str, words: list[dict] | None, field: str) -> None:
    if not words:
        return
    probabilities = [finite_number(word.get("probability")) for word in words]
    if not any(value is not None and 0 <= value <= 1 for value in probabilities):
        return  # Missing confidence is unknown; it is not proof of uncertainty.
    if source is None:
        raise UncertainTranscriptEvidenceError(f"{field}: ASR evidence cannot be aligned without the transcript")
    spans = []
    cursor = 0
    for word, probability in zip(words, probabilities):
        token = normalize_evidence(str(word.get("text") or "").replace('"""', "'''").strip())
        if not token:
            continue
        match = _bounded_text_pattern(token).search(source, cursor)
        if match is None:
            # An unrelated alignment miss must not invalidate a confidently
            # aligned quote elsewhere. Per-quote coverage remains fail-closed.
            continue
        start, end = match.start(), match.end()
        spans.append((start, end, probability))
        cursor = end
    normalized_quote = normalize_evidence(quote)
    quote_pattern = _bounded_text_pattern(normalized_quote)
    quote_match = quote_pattern.search(source)
    if quote_match is None:
        raise UncertainTranscriptEvidenceError(f"{field}: quoted ASR fragment has no complete-word match")
    while quote_match is not None:
        quote_start, quote_end = quote_match.start(), quote_match.end()
        overlap = [(start, end, probability) for start, end, probability in spans
                   if start < quote_end and end > quote_start]
        # Every quoted lexical character needs an aligned word. Word confidence
        # can remain unknown, but gaps in the mapping cannot be called verified.
        if any(source[position].isalnum() and not any(start <= position < end for start, end, _ in overlap)
               for position in range(quote_start, quote_end)):
            raise UncertainTranscriptEvidenceError(f"{field}: quoted ASR evidence alignment is incomplete")
        if any(probability is not None and 0 <= probability < 0.5 for _, _, probability in overlap):
            raise UncertainTranscriptEvidenceError(f"{field}: claimed issue overlaps an observed low-confidence ASR word")
        # Repeated quotations are ambiguous: do not choose a confident occurrence
        # while silently ignoring an equally matching uncertain occurrence.
        quote_match = quote_pattern.search(source, quote_start + 1)
_DURATION = re.compile(
    r"(?<![\w.])(?P<number>\d+(?:[.,]\d+)?)\s*[-–]?\s*"
    r"(?P<unit>seconds?|secs?|second[oi]|secondes?|segundos?|sekunden?|sek|s|"
    r"minutes?|mins?|minut[oi]|minutos?|minuten?)\b\.?",
    re.IGNORECASE,
)
_COMPOUND_SEPARATOR = re.compile(r"\s*(?:(?:and|e|und|y|et)\s+)?", re.IGNORECASE)


def _instruction_durations(instruction: str) -> list[float]:
    matches = list(_DURATION.finditer(instruction))
    durations = []
    index = 0
    while index < len(matches):
        match = matches[index]
        value = float(match.group("number").replace(",", "."))
        minutes = match.group("unit").lower().startswith("min")
        seconds = value * 60 if minutes else value
        # Aggregate only an adjacent minutes + sub-minute seconds expression.
        # Separate instructions or '90 seconds ... 15 seconds' remain conflicts.
        if minutes and index + 1 < len(matches):
            following = matches[index + 1]
            following_value = float(following.group("number").replace(",", "."))
            gap = instruction[match.end():following.start()]
            if (not following.group("unit").lower().startswith("min")
                    and following_value < 60 and _COMPOUND_SEPARATOR.fullmatch(gap)):
                seconds += following_value
                index += 1
        durations.append(seconds)
        index += 1
    return durations


def generation_json_schema(kind: str, *, target_duration_sec: float | None = None) -> dict:
    """Provider-side shape constraints complement authoritative local validation."""
    text = {"type": "string", "minLength": 1}
    texts = {"type": "array", "items": text}

    def object_schema(properties: dict) -> dict:
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    if kind == "rubric":
        properties = {field: {"type": "integer", "minimum": 1, "maximum": 5}
                      for field in ("fluency", "cohesion", "accuracy", "range", "overall", "topic_relevance_score")}
        properties.update({field: text for field in ("comments_fluency", "comments_cohesion",
                                                    "comments_accuracy", "comments_range", "overall_comment")})
        properties.update(on_topic={"type": "boolean"}, language_ok={"type": "boolean"},
                          evidence_quotes=texts, confidence={"type": "string", "enum": list(COACHING_CONFIDENCE_LEVELS)})
        properties["style_suggestions"] = {"type": "array", "items": object_schema({
            "original": text, "suggestion": text, "explanation": text,
        })}
        for field, categories in (("recurring_grammar_errors", GRAMMAR_ERROR_CATEGORIES),
                                  ("coherence_issues", COHERENCE_ISSUE_CATEGORIES),
                                  ("lexical_gaps", LEXICAL_GAP_CATEGORIES)):
            properties[field] = {"type": "array", "items": object_schema({
                "category": {"type": "string", "enum": list(categories)},
                "explanation": text, "examples": texts,
            })}
    elif kind == "coaching":
        properties = {field: text for field in ("next_focus", "next_exercise", "coach_summary")}
        properties.update(strengths=texts, top_3_priorities={**texts, "minItems": 3, "maxItems": 3})
        if target_duration_sec is not None:
            properties.update(next_attempt_instruction=text,
                              retry_duration_sec={"type": "number", "enum": [float(target_duration_sec)]})
    else:
        raise ValueError(f"Unknown generation schema '{kind}'")
    return object_schema(properties)


def _nonblank(text: str, field: str) -> None:
    if not text.strip():
        raise SchemaValidationError(f"{field}: must contain nonblank text")


def normalize_evidence(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).replace("’", "'").casefold().split())


def validate_rubric_generation(rubric: RubricResult, transcript: str | None, asr_words: list[dict] | None = None) -> None:
    grammar_examples = [q for issue in rubric.recurring_grammar_errors for q in issue.examples]
    error_examples = grammar_examples + [q for issue in rubric.lexical_gaps for q in issue.examples]
    known_quotes = rubric.evidence_quotes + [style.original for style in rubric.style_suggestions or []] + [q for name in ("recurring_grammar_errors", "coherence_issues", "lexical_gaps") for issue in getattr(rubric, name) for q in issue.examples]
    for field in ("comments_fluency", "comments_cohesion", "comments_accuracy",
                  "comments_range", "overall_comment"):
        _nonblank(getattr(rubric, field), field)
        validate_claim_text(getattr(rubric, field), field, grammar_errors=bool(rubric.recurring_grammar_errors), known_quotes=known_quotes, grammar_examples=grammar_examples)
    # This is the same escaping used in rubric_prompt, not a fuzzy alternative.
    source = normalize_evidence(transcript.replace('"""', "'''").strip()) if transcript is not None else None

    def quote_check(quote: str, field: str) -> None:
        _nonblank(quote, field)
        if source is not None and _bounded_text_pattern(normalize_evidence(quote)).search(source) is None:
            raise SchemaValidationError(f"{field}: quote is not present as complete-word evidence in the transcript")

    for index, style in enumerate(rubric.style_suggestions or []):
        path = f"style_suggestions[{index}]"
        quote_check(style.original, f"{path}.original")
        _asr_quote_guard(source, style.original, asr_words, f"{path}.original")
        for field in ("suggestion", "explanation"):
            _nonblank(getattr(style, field), f"{path}.{field}")
        # Replacement wording is speech/topic content, not an acoustic diagnosis.
        validate_claim_text(style.explanation, f"{path}.explanation", known_quotes=[style.original, style.suggestion])
        validate_optional_explanation(style.explanation, f"{path}.explanation", content_quotes=[style.original, style.suggestion])
        if normalize_evidence(style.original) == normalize_evidence(style.suggestion):
            raise SchemaValidationError(f"{path}.suggestion: must offer a different optional expression")
        original = normalize_evidence(style.original)
        if source is not None and _bounded_text_pattern(original).search(source) is None:
            raise SchemaValidationError(f"{path}.original: quote must use complete words")
        overlaps = any(contains_example(original, q) or contains_example(q, original) for q in error_examples)
        if source is not None:
            style_spans = list(_bounded_text_pattern(original).finditer(source))
            error_spans = [match for quote in error_examples
                           if normalize_evidence(quote)
                           for match in _bounded_text_pattern(normalize_evidence(quote)).finditer(source)]
            overlaps = overlaps or any(a.start() < b.end() and b.start() < a.end() for a in style_spans for b in error_spans)
        if overlaps:
            raise SchemaValidationError(f"{path}.original: cannot label overlapping evidence as both a grammar error or lexical gap and optional style")
    for index, quote in enumerate(rubric.evidence_quotes):
        quote_check(quote, f"evidence_quotes[{index}]")
    for field in ("recurring_grammar_errors", "coherence_issues", "lexical_gaps"):
        for index, issue in enumerate(getattr(rubric, field)):
            path = f"{field}[{index}]"
            _nonblank(issue.explanation, f"{path}.explanation")
            validate_claim_text(issue.explanation, f"{path}.explanation", known_quotes=known_quotes)
            may_describe_absence = field == "coherence_issues" and issue.category in ABSENCE_COHERENCE_CATEGORIES
            if not issue.examples and not may_describe_absence:
                raise SchemaValidationError(f"{path}.examples: at least one transcript quote is required")
            for quote_index, quote in enumerate(issue.examples):
                quote_check(quote, f"{path}.examples[{quote_index}]")
                _asr_quote_guard(source, quote, asr_words, f"{path}.examples[{quote_index}]")


def validate_coaching_generation(coaching: CoachingSummary, target_duration_sec: float | None, rubric: dict | None = None) -> None:
    for field in ("next_focus", "next_exercise", "coach_summary"):
        _nonblank(getattr(coaching, field), field)
    for field in ("strengths", "top_3_priorities"):
        for index, text in enumerate(getattr(coaching, field)):
            _nonblank(text, f"{field}[{index}]")
    if coaching.next_attempt_instruction is not None:
        _nonblank(coaching.next_attempt_instruction, "next_attempt_instruction")
    issues = (rubric or {}).get("recurring_grammar_errors") or []
    texts = [coaching.next_focus, coaching.next_exercise, coaching.coach_summary,
             *coaching.strengths, *coaching.top_3_priorities]
    if coaching.next_attempt_instruction:
        texts.append(coaching.next_attempt_instruction)
    grammar_examples = [q for issue in issues for q in issue.get("examples", [])]
    known_quotes = (rubric or {}).get("evidence_quotes", []) + [q for name in ("recurring_grammar_errors", "coherence_issues", "lexical_gaps") for issue in (rubric or {}).get(name, []) for q in issue.get("examples", [])]
    for index, text in enumerate(texts):
        validate_claim_text(text, f"coaching[{index}]", grammar_errors=bool(issues), known_quotes=known_quotes, grammar_examples=grammar_examples)
    priorities = [normalize_evidence(text) for text in coaching.top_3_priorities]
    styles = (rubric or {}).get("style_suggestions") or []
    for index, text in enumerate(texts):
        validate_style_coaching(text, f"coaching[{index}]", styles)
    for index, text in enumerate(coaching.top_3_priorities):
        validate_style_coaching(text, f"top_3_priorities[{index}]", styles, priority=True)
    for index, issue in enumerate(issues):
        examples = issue.get("examples") or []
        if not any(contains_example(priority, example) for example in examples for priority in priorities):
            raise SchemaValidationError(f"top_3_priorities: reference a quoted example from grammar finding {index}")
    if target_duration_sec is None:
        return
    target = float(target_duration_sec)
    if not math.isfinite(target) or target <= 0:
        raise SchemaValidationError("target_duration_sec: must be finite and positive")
    duration = coaching.retry_duration_sec
    if duration is None or not math.isclose(duration, target, abs_tol=0.01):
        raise SchemaValidationError("retry_duration_sec: must equal the configured task duration")
    instruction = coaching.next_attempt_instruction
    if instruction is None:
        raise SchemaValidationError("next_attempt_instruction: explicit full-task retry is required")
    _nonblank(instruction, "next_attempt_instruction")
    durations = _instruction_durations(instruction)
    if not durations or any(not math.isclose(value, target, abs_tol=0.01) for value in durations):
        raise SchemaValidationError("next_attempt_instruction: state only the configured retry duration using digits and seconds/minutes")

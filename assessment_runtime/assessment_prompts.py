"""Prompt templates for rubric and coaching generation."""

from __future__ import annotations

import json

from assess_core.coaching_taxonomy import (
    COACHING_CONFIDENCE_LEVELS,
    COHERENCE_ISSUE_CATEGORIES,
    GRAMMAR_ERROR_CATEGORIES,
    LEXICAL_GAP_CATEGORIES,
)

RUBRIC_PROMPT_VERSION = "rubric_multilingual_v1"
COACHING_PROMPT_VERSION = "coaching_multilingual_v1"
PROMPT_VERSION = RUBRIC_PROMPT_VERSION

SUPPORTED_LANGUAGE_CODES = ("de", "en", "it")
LANGUAGE_DISPLAY_NAMES_EN = {
    "de": "German",
    "en": "English",
    "it": "Italian",
}
LANGUAGE_DISPLAY_NAMES_LOCALIZED = {
    "de": {"de": "Deutsch", "en": "Englisch", "it": "Italienisch"},
    "en": {"de": "German", "en": "English", "it": "Italian"},
    "it": {"de": "tedesco", "en": "inglese", "it": "italiano"},
}


def normalize_language_code(language_code: str | None, *, fallback: str = "en") -> str:
    normalized = str(language_code or "").strip().lower()
    if normalized in SUPPORTED_LANGUAGE_CODES:
        return normalized
    return fallback


def language_name(language_code: str | None, *, fallback: str = "English") -> str:
    normalized = str(language_code or "").strip().lower()
    if normalized in LANGUAGE_DISPLAY_NAMES_EN:
        return LANGUAGE_DISPLAY_NAMES_EN[normalized]
    if normalized:
        if len(normalized) <= 8:
            return f"the language identified by code '{normalized}'"
        return normalized.title()
    return fallback


def localized_language_name(language_code: str | None, *, locale: str = "en") -> str:
    normalized_language = str(language_code or "").strip().lower()
    normalized_locale = normalize_language_code(locale, fallback="en")
    if normalized_language in LANGUAGE_DISPLAY_NAMES_EN:
        return LANGUAGE_DISPLAY_NAMES_LOCALIZED.get(
            normalized_locale,
            LANGUAGE_DISPLAY_NAMES_LOCALIZED["en"],
        ).get(normalized_language, language_name(normalized_language))
    if normalized_language:
        return {
            "de": f"die Sprache mit dem Code '{normalized_language}'",
            "en": f"the language identified by code '{normalized_language}'",
            "it": f"la lingua con codice '{normalized_language}'",
        }.get(normalized_locale, f"the language identified by code '{normalized_language}'")
    return language_name(normalized_language)


def rubric_prompt(
    transcript: str,
    metrics: dict,
    theme: str = "free topic",
    *,
    expected_language: str = "it",
    feedback_language: str | None = None,
) -> str:
    safe_transcript = transcript.replace('"""', "'''").strip()
    grammar_categories = ", ".join(GRAMMAR_ERROR_CATEGORIES)
    coherence_categories = ", ".join(COHERENCE_ISSUE_CATEGORIES)
    lexical_categories = ", ".join(LEXICAL_GAP_CATEGORIES)
    confidence_levels = ", ".join(COACHING_CONFIDENCE_LEVELS)
    expected_language_name = language_name(expected_language)
    feedback_language_name = language_name(feedback_language or expected_language)
    return f"""
You are a CEFR examiner for spoken {expected_language_name} as a target language. Evaluate only the spoken production based on the transcript and metrics.
The required theme is: "{theme}".
Write every natural-language string value in {feedback_language_name}.

Rules:
- Reply ONLY with valid JSON (no prose before or after).
- Translate comment, explanation, and summary strings into {feedback_language_name}.
- Do NOT translate `examples` or `evidence_quotes`; those must stay as exact substrings from the original spoken response.
- Do NOT translate `category` or `confidence` values; those enum values must remain exactly as listed.
- Scores must be integers from 1 to 5.
- `on_topic` must be true only if the response is clearly on theme.
- `topic_relevance_score` must be an integer from 1 to 5.
- `language_ok` must be true only if the spoken response is clearly in {expected_language_name}.
- `evidence_quotes` must contain exact, untranslated substrings copied from the TRANSCRIPT.
- For recurring errors, use ONLY the allowed categories.

OBJECTIVE METRICS:
- Duration: {metrics['duration_sec']} s
- Speaking time: {metrics['speaking_time_sec']} s
- Total pause time: {metrics['pause_total_sec']} s
- Pause count: {metrics['pause_count']}
- Words: {metrics['word_count']}
- WPM: {metrics['wpm']}
- Fillers: {metrics['fillers']}
- Cohesion markers: {metrics['cohesion_markers']}
- Complexity index: {metrics['complexity_index']}

TRANSCRIPT:
\"\"\"{safe_transcript}\"\"\"

Note:
- The transcript comes from automatic ASR and may contain transcription errors.

Required JSON schema:
{{
  "fluency": 1-5,
  "cohesion": 1-5,
  "accuracy": 1-5,
  "range": 1-5,
  "overall": 1-5,
  "comments_fluency": "string",
  "comments_cohesion": "string",
  "comments_accuracy": "string",
  "comments_range": "string",
  "overall_comment": "string",
  "on_topic": true/false,
  "topic_relevance_score": 1-5,
  "language_ok": true/false,
  "recurring_grammar_errors": [
    {{
      "category": "one of: {grammar_categories}",
      "explanation": "string",
      "examples": ["string"]
    }}
  ],
  "coherence_issues": [
    {{
      "category": "one of: {coherence_categories}",
      "explanation": "string",
      "examples": ["string"]
    }}
  ],
  "lexical_gaps": [
    {{
      "category": "one of: {lexical_categories}",
      "explanation": "string",
      "examples": ["string"]
    }}
  ],
  "evidence_quotes": ["string"],
  "confidence": "one of: {confidence_levels}"
}}
"""


def rubric_prompt_it(transcript: str, metrics: dict, theme: str = "tema libero") -> str:
    return rubric_prompt(
        transcript,
        metrics,
        theme,
        expected_language="it",
        feedback_language="it",
    )


def selftest_prompt_it() -> str:
    fake_metrics = {
        "duration_sec": 75.0,
        "speaking_time_sec": 63.0,
        "pause_total_sec": 12.0,
        "pause_count": 8,
        "word_count": 140,
        "wpm": 133.3,
        "fillers": 5,
        "cohesion_markers": 4,
        "complexity_index": 3,
    }
    transcript = (
        "Oggi parlo della mia città. Negli ultimi anni il trasporto pubblico è migliorato, "
        "tuttavia i costi sono ancora alti e molte persone preferiscono l'auto."
    )
    return rubric_prompt_it(transcript, fake_metrics, "la mia città")


def coaching_prompt(
    metrics: dict,
    rubric: dict,
    theme: str,
    target_duration_sec: float,
    *,
    expected_language: str = "it",
    feedback_language: str | None = None,
) -> str:
    rubric_json = json.dumps(rubric, ensure_ascii=False, indent=2)
    expected_language_name = language_name(expected_language)
    feedback_language_name = language_name(feedback_language or expected_language)
    return f"""
You are a speaking coach for learners of {expected_language_name}. Use ONLY the metrics and the already validated rubric to give practical next-step advice.
The task was to speak in {expected_language_name} for {target_duration_sec:.0f} seconds on the theme "{theme}".
Write every natural-language string value in {feedback_language_name}.

Rules:
- Reply ONLY with valid JSON.
- Write all generated text fields (`strengths`, `top_3_priorities`, `next_focus`, `next_exercise`, `coach_summary`) in {feedback_language_name}.
- If you quote the learner's original speech from the validated rubric, do NOT translate the quote.
- Do NOT translate `category` or `confidence` values if they appear in the validated rubric.
- `top_3_priorities` must contain EXACTLY 3 items.
- `next_exercise` must be a concrete practice activity, not an internal ID or an invented link.
- Do not change facts from the already validated rubric.

METRICS:
- Speaking time: {metrics['speaking_time_sec']} s
- Total pause time: {metrics['pause_total_sec']} s
- Pause count: {metrics['pause_count']}
- Words: {metrics['word_count']}
- WPM: {metrics['wpm']}
- Fillers: {metrics['fillers']}
- Cohesion markers: {metrics['cohesion_markers']}
- Complexity index: {metrics['complexity_index']}

VALIDATED RUBRIC:
{rubric_json}

Required JSON schema:
{{
  "strengths": ["string"],
  "top_3_priorities": ["string", "string", "string"],
  "next_focus": "string",
  "next_exercise": "string",
  "coach_summary": "string"
}}
"""


def coaching_prompt_it(
    metrics: dict,
    rubric: dict,
    theme: str,
    target_duration_sec: float,
) -> str:
    return coaching_prompt(
        metrics,
        rubric,
        theme,
        target_duration_sec,
        expected_language="it",
        feedback_language="it",
    )

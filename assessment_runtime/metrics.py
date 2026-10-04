"""Deterministic metric extraction from transcript tokens and pauses."""

from __future__ import annotations

import re
import unicodedata

from assess_core.language_profiles import LanguageProfile

from assess_core.language_profiles import fallback_language_profile, resolve_language_profile


def _normalized_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    # Apostrophes belong to words: Italian elisions must not match a substring.
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", normalized)


def _count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    tokens = _normalized_tokens(text)
    candidates = sorted(
        {tuple(_normalized_tokens(phrase)) for phrase in phrases} - {()},
        key=lambda phrase: (-len(phrase), phrase),
    )
    hits = 0
    index = 0
    while index < len(tokens):
        match = next(
            (phrase for phrase in candidates if tuple(tokens[index:index + len(phrase)]) == phrase),
            None,
        )
        if match:
            hits += 1
            index += len(match)
        else:
            index += 1
    return hits


def _historical_count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    return sum(
        len(re.findall(r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b", text, re.IGNORECASE))
        for phrase in phrases
    )


def _is_current_live_profile(profile: LanguageProfile) -> bool:
    return profile.scorer_version in {"language_profile_en_v3_live", "language_profile_it_v2_live"}


def metrics_from(
    words: list[dict],
    audio_feats: dict,
    *,
    language_code: str = "it",
    language_profile_key: str | None = None,
) -> dict:
    duration = audio_feats["duration_sec"]
    pause_total = sum(p[2] for p in audio_feats["pauses"])
    speaking_time = max(0.001, duration - pause_total)
    profile = (
        resolve_language_profile(language_code, profile_key=language_profile_key)
        if language_profile_key is not None
        else None
    )
    if profile is None:
        profile = fallback_language_profile(language_code)
    if _is_current_live_profile(profile):
        tokens = _normalized_tokens(" ".join(str(w["text"]) for w in words))
        count_phrases = _count_phrases
    else:
        # Explicit historical/benchmark profiles reproduce their original metrics.
        tokens = [re.sub(r"[^a-zà-ù’']", "", str(w["text"]).lower()) for w in words]
        tokens = [token for token in tokens if token]
        count_phrases = _historical_count_phrases
        filler_set = set(profile.fillers)
    word_count = len(tokens)
    wpm = word_count / (speaking_time / 60.0)
    text = " " + " ".join(tokens) + " "
    fillers = (
        _count_phrases(text, profile.fillers)
        if _is_current_live_profile(profile)
        else sum(1 for token in tokens if token in filler_set)
    )
    cohesion_hits = count_phrases(text, profile.discourse_markers)
    rel_markers = count_phrases(text, profile.relative_markers)
    cond_markers = count_phrases(text, profile.conditional_markers)
    complexity = rel_markers + cond_markers
    return {
        "duration_sec": round(duration, 2),
        "pause_count": len(audio_feats["pauses"]),
        "pause_total_sec": round(pause_total, 2),
        "speaking_time_sec": round(speaking_time, 2),
        "word_count": word_count,
        "wpm": round(wpm, 1),
        "fillers": fillers,
        "cohesion_markers": cohesion_hits,
        "complexity_index": complexity,
    }

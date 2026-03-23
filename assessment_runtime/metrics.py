"""Deterministic metric extraction from transcript tokens and pauses."""

from __future__ import annotations

import re

from assess_core.language_profiles import fallback_language_profile, resolve_language_profile


def _count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    hits = 0
    for phrase in phrases:
        parts = [re.escape(part) for part in phrase.split()]
        pattern = r"\b" + r"\s+".join(parts) + r"\b"
        hits += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return hits


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
    tokens = [re.sub(r"[^a-zà-ù’']", "", str(w["text"]).lower()) for w in words]
    tokens = [token for token in tokens if token]
    filler_set = set(profile.fillers)
    word_count = len(tokens)
    wpm = word_count / (speaking_time / 60.0)
    fillers = sum(1 for token in tokens if token in filler_set)
    text = " " + " ".join(tokens) + " "
    cohesion_hits = _count_phrases(text, profile.discourse_markers)
    rel_markers = _count_phrases(text, profile.relative_markers)
    cond_markers = _count_phrases(text, profile.conditional_markers)
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

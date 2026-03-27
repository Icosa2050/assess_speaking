"""Provisional dimension-first scoring built on language profiles."""

from __future__ import annotations

from typing import Any

from assess_core.language_profiles import LanguageProfile
from assess_core.schemas import RubricResult


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _bounded_score(weighted_zero_to_one: float) -> float:
    return round(1.0 + 4.0 * _clip(weighted_zero_to_one, 0.0, 1.0), 2)


def _round_score(value: float) -> float:
    return round(_clip(value, 1.0, 5.0), 2)


def score_dimensions(
    *,
    metrics: dict[str, Any],
    rubric: RubricResult | None,
    checks: dict[str, Any],
    profile: LanguageProfile,
    detected_language_probability: float | None,
) -> dict[str, dict[str, Any]]:
    word_count = max(1, int(metrics.get("word_count", 0)))
    duration = max(0.001, float(metrics.get("duration_sec", 0)))
    pause_total = float(metrics.get("pause_total_sec", 0))
    fillers = int(metrics.get("fillers", 0))
    wpm = float(metrics.get("wpm", 0))
    cohesion_markers = float(metrics.get("cohesion_markers", 0))
    complexity = float(metrics.get("complexity_index", 0))
    pause_ratio = _clip(pause_total / duration, 0.0, 1.0)
    filler_ratio = _clip(fillers / word_count, 0.0, 1.0)

    pace_tolerance = max(0.001, float(profile.pace_tolerance_wpm))
    pause_ceiling = max(0.001, float(profile.pause_ratio_ceiling))
    filler_ceiling = max(0.001, float(profile.filler_ratio_ceiling))

    wpm_score = _clip(1.0 - abs(wpm - profile.pace_center_wpm) / pace_tolerance, 0.0, 1.0)
    pause_score = _clip(1.0 - pause_ratio / pause_ceiling, 0.0, 1.0)
    filler_score = _clip(1.0 - filler_ratio / filler_ceiling, 0.0, 1.0)
    marker_score = _clip(cohesion_markers / max(1.0, profile.cohesion_marker_target), 0.0, 1.0)
    complexity_score = _clip(complexity / max(1.0, profile.grammar_complexity_target), 0.0, 1.0)
    lexical_support_score = _clip(word_count / max(1, profile.lexical_word_target), 0.0, 1.0)

    acoustic_fluency = _bounded_score((0.45 * wpm_score) + (0.35 * pause_score) + (0.20 * filler_score))
    if rubric is not None:
        fluency = _round_score((0.55 * acoustic_fluency) + (0.45 * rubric.fluency))
    else:
        fluency = acoustic_fluency

    lang_prob = _clip(float(detected_language_probability or 0.0), 0.0, 1.0)
    intelligibility_proxy = _bounded_score(
        (0.55 * _clip((lang_prob - 0.5) / 0.5, 0.0, 1.0))
        + (0.25 * (1.0 if checks.get("asr_pause_consistent") is not False else 0.4))
        + (0.20 * (1.0 if checks.get("min_words_pass") is True else 0.5))
    )
    if checks.get("language_pass") is False:
        intelligibility_proxy = min(intelligibility_proxy, 2.0)

    grammar_base = float(rubric.accuracy) if rubric is not None else 3.0
    grammar = _round_score((0.85 * grammar_base) + (0.15 * _bounded_score(complexity_score)))

    lexicon_base = float(rubric.range) if rubric is not None else 3.0
    lexicon = _round_score((0.85 * lexicon_base) + (0.15 * _bounded_score(lexical_support_score)))

    coherence_base = float(rubric.cohesion) if rubric is not None else 3.0
    topic_gate = checks.get("topic_pass")
    topic_relevance = (
        float(rubric.topic_relevance_score)
        if rubric is not None
        else (5.0 if topic_gate is True else 3.0 if topic_gate is None else 2.0)
    )
    coherence = _round_score(
        (0.55 * coherence_base)
        + (0.25 * _bounded_score(marker_score))
        + (0.20 * topic_relevance)
    )

    task_fulfillment = _round_score(
        (0.50 * (5.0 if topic_gate is True else 3.0 if topic_gate is None else topic_relevance))
        + (0.25 * (5.0 if checks.get("duration_pass") else 2.5))
        + (0.25 * (5.0 if checks.get("min_words_pass") else 2.5))
    )

    return {
        "fluency": {
            "score": fluency,
            "confidence": "medium" if rubric is not None else "low",
            "method": "acoustic_plus_rubric_v1",
            "signals": {
                "wpm": round(wpm, 1),
                "pace_score": round(wpm_score, 3),
                "pause_ratio": round(pause_ratio, 3),
                "pause_score": round(pause_score, 3),
                "filler_ratio": round(filler_ratio, 3),
                "filler_score": round(filler_score, 3),
            },
        },
        "pronunciation_intelligibility": {
            "score": intelligibility_proxy,
            "confidence": "low",
            "method": "asr_proxy_v1",
            "signals": {
                "detected_language_probability": round(lang_prob, 4),
                "language_pass": checks.get("language_pass"),
                "asr_pause_consistent": checks.get("asr_pause_consistent"),
            },
        },
        "grammar": {
            "score": grammar,
            "confidence": "medium" if rubric is not None else "low",
            "method": "rubric_plus_complexity_v1",
            "signals": {
                "rubric_accuracy": rubric.accuracy if rubric is not None else None,
                "complexity_index": complexity,
                "complexity_score": round(complexity_score, 3),
            },
        },
        "lexicon": {
            "score": lexicon,
            "confidence": "medium" if rubric is not None else "low",
            "method": "rubric_plus_support_v1",
            "signals": {
                "rubric_range": rubric.range if rubric is not None else None,
                "word_count": word_count,
                "lexical_support_score": round(lexical_support_score, 3),
            },
        },
        "coherence": {
            "score": coherence,
            "confidence": "medium" if rubric is not None else "low",
            "method": "rubric_plus_discourse_v1",
            "signals": {
                "rubric_cohesion": rubric.cohesion if rubric is not None else None,
                "topic_relevance_score": rubric.topic_relevance_score if rubric is not None else None,
                "cohesion_markers": cohesion_markers,
                "marker_score": round(marker_score, 3),
            },
        },
        "task_fulfillment": {
            "score": task_fulfillment,
            "confidence": "medium" if rubric is not None else "low",
            "method": "task_checks_plus_topic_v1",
            "signals": {
                "topic_pass": checks.get("topic_pass"),
                "duration_pass": checks.get("duration_pass"),
                "min_words_pass": checks.get("min_words_pass"),
                "topic_relevance_score": rubric.topic_relevance_score if rubric is not None else None,
            },
        },
    }


def aggregate_dimension_scores(
    dimensions: dict[str, dict[str, Any]],
    *,
    profile: LanguageProfile,
) -> dict[str, Any]:
    weighted_total = 0.0
    weight_sum = 0.0
    low_confidence = False
    for key, weight in profile.dimension_weights.items():
        dimension = dimensions.get(key)
        if not isinstance(dimension, dict):
            continue
        score = dimension.get("score")
        if not isinstance(score, (int, float)):
            continue
        weighted_total += float(score) * float(weight)
        weight_sum += float(weight)
        if dimension.get("confidence") == "low":
            low_confidence = True

    continuous = round(weighted_total / weight_sum, 2) if weight_sum else None
    level = None
    c2_cut = profile.cefr_cut_scores.get("C2")
    c1_cut = profile.cefr_cut_scores.get("C1")
    b2_cut = profile.cefr_cut_scores.get("B2")
    if continuous is not None and any(boundary is not None for boundary in (b2_cut, c1_cut, c2_cut)):
        if c2_cut is not None and continuous >= float(c2_cut):
            level = "C2"
        elif c1_cut is not None and continuous >= float(c1_cut):
            level = "C1"
        elif b2_cut is not None and continuous >= float(b2_cut):
            level = "B2"
        else:
            level = "B1"

    nearest_margin = None
    if continuous is not None:
        margins = [
            abs(continuous - float(boundary))
            for boundary in profile.cefr_cut_scores.values()
            if boundary is not None
        ]
        nearest_margin = min(margins) if margins else None
    confidence = (
        "low"
        if not weight_sum or low_confidence or (nearest_margin is not None and nearest_margin < 0.15)
        else "medium"
    )

    return {
        "continuous": continuous,
        "level": level,
        "confidence": confidence,
        "calibrated": False,
        "method": profile.scorer_version,
        "weights": dict(profile.dimension_weights),
    }

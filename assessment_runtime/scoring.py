"""Deterministic scoring and gate logic."""

from __future__ import annotations

from typing import Optional

from assess_core.schemas import RubricResult

SCORE_MIN = 1.0
SCORE_MAX = 5.0
DETERMINISTIC_TARGET_WPM = 130.0
DETERMINISTIC_WPM_TOLERANCE = 90.0
DETERMINISTIC_PAUSE_RATIO_CEILING = 0.45
DETERMINISTIC_FILLER_RATIO_CEILING = 0.12
DETERMINISTIC_COHESION_MARKER_TARGET = 3.0
DETERMINISTIC_COMPLEXITY_TARGET = 4.0
DETERMINISTIC_COMPONENT_WEIGHTS = {
    "wpm": 0.35,
    "pause_ratio": 0.25,
    "filler_ratio": 0.20,
    "cohesion_markers": 0.10,
    "complexity_index": 0.10,
}
FINAL_SCORE_WEIGHTS = {
    "deterministic": 0.40,
    "rubric": 0.60,
}


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def deterministic_score(metrics: dict) -> float:
    word_count = max(1, int(metrics.get("word_count", 0)))
    duration = max(0.001, float(metrics.get("duration_sec", 0)))
    wpm = float(metrics.get("wpm", 0))
    pause_total = float(metrics.get("pause_total_sec", 0))
    fillers = int(metrics.get("fillers", 0))
    cohesion_markers = float(metrics.get("cohesion_markers", 0))
    complexity = float(metrics.get("complexity_index", 0))

    pause_ratio = _clip(pause_total / duration, 0.0, 1.0)
    filler_ratio = _clip(fillers / word_count, 0.0, 1.0)

    wpm_score = _clip(1.0 - abs(wpm - DETERMINISTIC_TARGET_WPM) / DETERMINISTIC_WPM_TOLERANCE, 0.0, 1.0)
    pause_score = _clip(1.0 - pause_ratio / DETERMINISTIC_PAUSE_RATIO_CEILING, 0.0, 1.0)
    filler_score = _clip(1.0 - filler_ratio / DETERMINISTIC_FILLER_RATIO_CEILING, 0.0, 1.0)
    cohesion_score = _clip(cohesion_markers / DETERMINISTIC_COHESION_MARKER_TARGET, 0.0, 1.0)
    complexity_score = _clip(complexity / DETERMINISTIC_COMPLEXITY_TARGET, 0.0, 1.0)

    weighted = (
        DETERMINISTIC_COMPONENT_WEIGHTS["wpm"] * wpm_score
        + DETERMINISTIC_COMPONENT_WEIGHTS["pause_ratio"] * pause_score
        + DETERMINISTIC_COMPONENT_WEIGHTS["filler_ratio"] * filler_score
        + DETERMINISTIC_COMPONENT_WEIGHTS["cohesion_markers"] * cohesion_score
        + DETERMINISTIC_COMPONENT_WEIGHTS["complexity_index"] * complexity_score
    )
    return round(SCORE_MIN + (SCORE_MAX - SCORE_MIN) * weighted, 2)


def rubric_score(rubric: RubricResult | None) -> Optional[float]:
    if rubric is None:
        return None
    values = [rubric.fluency, rubric.cohesion, rubric.accuracy, rubric.range, rubric.overall]
    return round(sum(values) / len(values), 2)


def compute_checks(
    metrics: dict,
    rubric: RubricResult | None,
    target_duration_sec: float,
    min_word_count: int,
    duration_pass_ratio: float,
    language_pass: bool,
) -> dict:
    speaking_time = float(metrics.get("speaking_time_sec", 0))
    word_count = int(metrics.get("word_count", 0))
    duration_pass = speaking_time >= (target_duration_sec * duration_pass_ratio)
    min_words_pass = word_count >= min_word_count
    topic_pass = rubric.on_topic if rubric is not None else None
    return {
        "duration_pass": duration_pass,
        "topic_pass": topic_pass,
        "min_words_pass": min_words_pass,
        "language_pass": language_pass,
    }


def final_scores(
    deterministic: float,
    llm: Optional[float],
    topic_pass: Optional[bool],
    topic_fail_cap_score: float,
) -> dict:
    if llm is None:
        final = deterministic
        mode = "deterministic_only"
    else:
        final = (FINAL_SCORE_WEIGHTS["deterministic"] * deterministic) + (FINAL_SCORE_WEIGHTS["rubric"] * llm)
        mode = "hybrid"
    if topic_pass is False:
        final = min(final, topic_fail_cap_score)
    final = round(_clip(final, SCORE_MIN, SCORE_MAX), 2)
    band = int(round(final))
    band = int(_clip(band, int(SCORE_MIN), int(SCORE_MAX)))
    return {
        "deterministic": round(deterministic, 2),
        "llm": None if llm is None else round(llm, 2),
        "final": final,
        "band": band,
        "mode": mode,
    }

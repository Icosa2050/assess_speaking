"""Deterministic scoring and gate logic."""

from __future__ import annotations

from typing import Optional

from schemas import RubricResult


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

    wpm_score = _clip(1.0 - abs(wpm - 130.0) / 90.0, 0.0, 1.0)
    pause_score = _clip(1.0 - pause_ratio / 0.45, 0.0, 1.0)
    filler_score = _clip(1.0 - filler_ratio / 0.12, 0.0, 1.0)
    cohesion_score = _clip(cohesion_markers / 3.0, 0.0, 1.0)
    complexity_score = _clip(complexity / 4.0, 0.0, 1.0)

    weighted = (
        0.35 * wpm_score
        + 0.25 * pause_score
        + 0.20 * filler_score
        + 0.10 * cohesion_score
        + 0.10 * complexity_score
    )
    return round(1.0 + 4.0 * weighted, 2)


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
    topic_pass = rubric.on_topic if rubric is not None else True
    return {
        "duration_pass": duration_pass,
        "topic_pass": topic_pass,
        "min_words_pass": min_words_pass,
        "language_pass": language_pass,
    }


def final_scores(
    deterministic: float,
    llm: Optional[float],
    topic_pass: bool,
    topic_fail_cap_score: float,
) -> dict:
    if llm is None:
        final = deterministic
        mode = "deterministic_only"
    else:
        final = (0.4 * deterministic) + (0.6 * llm)
        mode = "hybrid"
    if not topic_pass:
        final = min(final, topic_fail_cap_score)
    final = round(_clip(final, 1.0, 5.0), 2)
    band = int(round(final))
    band = int(_clip(band, 1, 5))
    return {
        "deterministic": round(deterministic, 2),
        "llm": None if llm is None else round(llm, 2),
        "final": final,
        "band": band,
        "mode": mode,
    }

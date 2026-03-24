from __future__ import annotations

from typing import Any

from assess_core.language_profiles import require_language_profile
from assess_core.settings import Settings
from assessment_runtime.scoring import (
    DETERMINISTIC_COHESION_MARKER_TARGET,
    DETERMINISTIC_COMPLEXITY_TARGET,
    DETERMINISTIC_COMPONENT_WEIGHTS,
    DETERMINISTIC_FILLER_RATIO_CEILING,
    DETERMINISTIC_PAUSE_RATIO_CEILING,
    DETERMINISTIC_TARGET_WPM,
    DETERMINISTIC_WPM_TOLERANCE,
    FINAL_SCORE_WEIGHTS,
    SCORE_MAX,
    SCORE_MIN,
)


def build_scoring_guide_data(settings: Settings | None = None) -> dict[str, Any]:
    runtime_settings = settings or Settings()
    english_profile = require_language_profile("en")
    italian_profile = require_language_profile("it")

    return {
        "score_scale": {
            "min": SCORE_MIN,
            "max": SCORE_MAX,
        },
        "formula": {
            "deterministic_weight_pct": int(FINAL_SCORE_WEIGHTS["deterministic"] * 100),
            "rubric_weight_pct": int(FINAL_SCORE_WEIGHTS["rubric"] * 100),
            "topic_fail_cap_score": runtime_settings.topic_fail_cap_score,
        },
        "deterministic_signals": [
            {
                "id": "wpm",
                "weight_pct": int(DETERMINISTIC_COMPONENT_WEIGHTS["wpm"] * 100),
                "target": DETERMINISTIC_TARGET_WPM,
                "tolerance": DETERMINISTIC_WPM_TOLERANCE,
            },
            {
                "id": "pause_ratio",
                "weight_pct": int(DETERMINISTIC_COMPONENT_WEIGHTS["pause_ratio"] * 100),
                "ceiling": DETERMINISTIC_PAUSE_RATIO_CEILING,
            },
            {
                "id": "filler_ratio",
                "weight_pct": int(DETERMINISTIC_COMPONENT_WEIGHTS["filler_ratio"] * 100),
                "ceiling": DETERMINISTIC_FILLER_RATIO_CEILING,
            },
            {
                "id": "cohesion_markers",
                "weight_pct": int(DETERMINISTIC_COMPONENT_WEIGHTS["cohesion_markers"] * 100),
                "target": DETERMINISTIC_COHESION_MARKER_TARGET,
            },
            {
                "id": "complexity_index",
                "weight_pct": int(DETERMINISTIC_COMPONENT_WEIGHTS["complexity_index"] * 100),
                "target": DETERMINISTIC_COMPLEXITY_TARGET,
            },
        ],
        "rubric_dimensions": [
            "fluency",
            "cohesion",
            "accuracy",
            "range",
            "overall",
        ],
        "gates": [
            {
                "id": "language_pass",
            },
            {
                "id": "topic_pass",
                "topic_fail_cap_score": runtime_settings.topic_fail_cap_score,
            },
            {
                "id": "duration_pass",
                "duration_pass_ratio_pct": int(runtime_settings.duration_pass_ratio * 100),
            },
            {
                "id": "min_words_pass",
                "min_word_count": runtime_settings.min_word_count,
            },
        ],
        "cefr_dimensions": [
            "fluency",
            "pronunciation_intelligibility",
            "grammar",
            "lexicon",
            "coherence",
            "task_fulfillment",
        ],
        "cefr_thresholds": [
            {
                "code": english_profile.code,
                "label": english_profile.label,
                "B2": english_profile.cefr_cut_scores.get("B2"),
                "C1": english_profile.cefr_cut_scores.get("C1"),
                "C2": english_profile.cefr_cut_scores.get("C2"),
            },
            {
                "code": italian_profile.code,
                "label": italian_profile.label,
                "B2": italian_profile.cefr_cut_scores.get("B2"),
                "C1": italian_profile.cefr_cut_scores.get("C1"),
                "C2": italian_profile.cefr_cut_scores.get("C2"),
            },
        ],
    }

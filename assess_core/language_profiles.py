"""Language profiles for feature extraction and provisional CEFR mapping."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    label: str
    fillers: tuple[str, ...]
    discourse_markers: tuple[str, ...]
    relative_markers: tuple[str, ...]
    conditional_markers: tuple[str, ...]
    pace_center_wpm: float
    pace_tolerance_wpm: float
    pause_ratio_ceiling: float
    filler_ratio_ceiling: float
    cohesion_marker_target: float
    grammar_complexity_target: float
    lexical_word_target: int
    dimension_weights: dict[str, float]
    cefr_cut_scores: dict[str, float]
    scorer_version: str


def _normalize_language_code(code: str | None) -> str:
    return str(code or "").strip().lower()


def _normalize_profile_key(profile_key: str | None) -> str:
    return str(profile_key or "").strip().lower()


ENGLISH_PROFILE_BENCHMARK = LanguageProfile(
    code="en",
    label="English",
    fillers=("uh", "um", "erm", "hmm"),
    discourse_markers=(
        "however",
        "therefore",
        "moreover",
        "in addition",
        "for example",
        "for instance",
        "as a result",
        "on the other hand",
        "in conclusion",
        "overall",
        "in my opinion",
        "because of this",
    ),
    relative_markers=("that", "which", "who", "whose", "where"),
    conditional_markers=("if", "unless", "provided", "supposing"),
    pace_center_wpm=135.0,
    pace_tolerance_wpm=55.0,
    pause_ratio_ceiling=0.35,
    filler_ratio_ceiling=0.08,
    cohesion_marker_target=3.0,
    grammar_complexity_target=3.0,
    lexical_word_target=180,
    dimension_weights={
        "fluency": 0.24,
        "pronunciation_intelligibility": 0.12,
        "grammar": 0.20,
        "lexicon": 0.18,
        "coherence": 0.16,
        "task_fulfillment": 0.10,
    },
    cefr_cut_scores={
        "B1": 2.8,
        "B2": 4.05,
        "C1": 4.65,
        "C2": 4.85,
    },
    scorer_version="language_profile_en_v2",
)

ENGLISH_PROFILE_LIVE_SHADOW = replace(
    ENGLISH_PROFILE_BENCHMARK,
    scorer_version="language_profile_en_v2_live_shadow",
)

ITALIAN_PROFILE_BENCHMARK = LanguageProfile(
    code="it",
    label="Italian",
    fillers=("eh", "ehm", "mmm", "cioè", "allora", "dunque", "tipo", "insomma"),
    discourse_markers=(
        "inoltre",
        "per quanto riguarda",
        "tuttavia",
        "ciò nonostante",
        "in definitiva",
        "da un lato",
        "dall’altro",
        "a mio avviso",
        "tenuto conto di",
        "a quanto pare",
        "presumibilmente",
        "parrebbe che",
        "pertanto",
        "quindi",
        "invece",
        "comunque",
    ),
    relative_markers=("che", "cui", "nella quale", "nei quali"),
    conditional_markers=("se", "qualora"),
    pace_center_wpm=130.0,
    pace_tolerance_wpm=55.0,
    pause_ratio_ceiling=0.35,
    filler_ratio_ceiling=0.08,
    cohesion_marker_target=3.0,
    grammar_complexity_target=3.0,
    lexical_word_target=180,
    dimension_weights={
        "fluency": 0.24,
        "pronunciation_intelligibility": 0.12,
        "grammar": 0.22,
        "lexicon": 0.16,
        "coherence": 0.16,
        "task_fulfillment": 0.10,
    },
    cefr_cut_scores={
        "B1": 2.8,
        "B2": 3.45,
        "C1": 4.10,
        "C2": 4.65,
    },
    scorer_version="language_profile_it_v1",
)

ITALIAN_PROFILE_LIVE_SHADOW = replace(
    ITALIAN_PROFILE_BENCHMARK,
    scorer_version="language_profile_it_v1_live_shadow",
)

LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "en": ENGLISH_PROFILE_BENCHMARK,
    "en_benchmark": ENGLISH_PROFILE_BENCHMARK,
    "en_live_shadow": ENGLISH_PROFILE_LIVE_SHADOW,
    "it": ITALIAN_PROFILE_LIVE_SHADOW,
    "it_benchmark": ITALIAN_PROFILE_BENCHMARK,
    "it_live_shadow": ITALIAN_PROFILE_LIVE_SHADOW,
}

LANGUAGE_PROFILE_DEFAULTS: dict[str, str] = {
    "en": "en",
    "it": "it",
}

DEFAULT_LANGUAGE_PROFILE = LanguageProfile(
    code="generic",
    label="Generic",
    fillers=(),
    discourse_markers=(),
    relative_markers=(),
    conditional_markers=(),
    pace_center_wpm=130.0,
    pace_tolerance_wpm=55.0,
    pause_ratio_ceiling=0.35,
    filler_ratio_ceiling=0.08,
    cohesion_marker_target=3.0,
    grammar_complexity_target=3.0,
    lexical_word_target=180,
    dimension_weights={},
    cefr_cut_scores={},
    scorer_version="language_profile_generic_v1",
)


def get_language_profile_by_key(profile_key: str | None) -> LanguageProfile | None:
    normalized = _normalize_profile_key(profile_key)
    return LANGUAGE_PROFILES.get(normalized)


def require_language_profile_by_key(profile_key: str | None) -> LanguageProfile:
    normalized = _normalize_profile_key(profile_key)
    profile = get_language_profile_by_key(normalized)
    if profile is None:
        raise KeyError(f"Unsupported language profile key: {profile_key!r}")
    return profile


def default_language_profile_key(code: str | None) -> str | None:
    normalized = _normalize_language_code(code)
    return LANGUAGE_PROFILE_DEFAULTS.get(normalized)


def get_language_profile(code: str | None) -> LanguageProfile | None:
    profile_key = default_language_profile_key(code)
    return get_language_profile_by_key(profile_key)


def resolve_language_profile(
    code: str | None,
    *,
    profile_key: str | None = None,
) -> LanguageProfile | None:
    normalized_code = _normalize_language_code(code)
    normalized_profile_key = _normalize_profile_key(profile_key)
    if normalized_profile_key:
        profile = require_language_profile_by_key(normalized_profile_key)
        if normalized_code and profile.code != normalized_code:
            raise KeyError(
                f"Language profile key {profile_key!r} does not match expected language {code!r}"
            )
        return profile
    return get_language_profile(normalized_code)


def require_language_profile(code: str | None) -> LanguageProfile:
    profile = get_language_profile(code)
    if profile is None:
        raise KeyError(f"Unsupported language profile: {code!r}")
    return profile


def require_resolved_language_profile(
    code: str | None,
    *,
    profile_key: str | None = None,
) -> LanguageProfile:
    profile = resolve_language_profile(code, profile_key=profile_key)
    if profile is None:
        raise KeyError(f"Unsupported language profile: {code!r}")
    return profile


def fallback_language_profile(code: str | None) -> LanguageProfile:
    return get_language_profile(code) or DEFAULT_LANGUAGE_PROFILE

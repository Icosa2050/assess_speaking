from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from app_core.app_data import build_app_data_paths
from app_core.runtime_providers import DEFAULT_PROVIDER

APP_NAME = "Vostavo"
DEFAULT_UI_LOCALE = "en"
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_OPENROUTER_HTTP_REFERER = "http://localhost:8503"
DEFAULT_OPENROUTER_APP_TITLE = APP_NAME
SUPPORTED_UI_LOCALES = ("en", "de", "it")
CEFR_LEVELS = ("B1", "B2", "C1")
DURATION_OPTIONS = (60, 90, 120, 180)
TASK_FAMILY_OPTIONS = (
    "travel_narrative",
    "personal_experience",
    "opinion_monologue",
    "picture_description",
    "free_monologue",
)


def _is_valid_openrouter_http_referer(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_openrouter_http_referer(value: str | None, *, require_valid: bool = False) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return DEFAULT_OPENROUTER_HTTP_REFERER
    if _is_valid_openrouter_http_referer(candidate):
        return candidate
    if require_valid:
        raise ValueError("OpenRouter HTTP Referer must be a valid URL starting with http:// or https://.")
    return DEFAULT_OPENROUTER_HTTP_REFERER


def normalize_openrouter_app_title(value: str | None) -> str:
    return str(value or DEFAULT_OPENROUTER_APP_TITLE).strip() or DEFAULT_OPENROUTER_APP_TITLE


def _default_log_dir() -> str:
    return str(build_app_data_paths().reports_dir)


def _default_whisper_cache_dir() -> str:
    return str(build_app_data_paths().whisper_cache_dir)


class RecordingStatus(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    READY = "ready"
    ASSESSING = "assessing"
    SUBMITTED = "submitted"


@dataclass
class ProviderConnection:
    connection_id: str = ""
    provider_kind: Literal["openrouter", "ollama", "lmstudio", "openai_compatible"] = DEFAULT_PROVIDER
    label: str = ""
    base_url: str = ""
    default_model: str = DEFAULT_MODEL
    auth_mode: Literal["none", "bearer"] = "none"
    secret_ref: str = ""
    is_default: bool = False
    is_local: bool = False
    last_test_status: str = ""
    last_tested_at: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppPreferences:
    ui_locale: str = DEFAULT_UI_LOCALE
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    llm_base_url: str = ""
    llm_api_key: str = ""
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_cache_dir: str = field(default_factory=_default_whisper_cache_dir)
    openrouter_http_referer: str = DEFAULT_OPENROUTER_HTTP_REFERER
    openrouter_app_title: str = DEFAULT_OPENROUTER_APP_TITLE
    active_connection_id: str = ""
    connections: list[ProviderConnection] = field(default_factory=list)
    setup_complete: bool = False
    log_dir: str = field(default_factory=_default_log_dir)


@dataclass
class DraftSession:
    session_id: str = ""
    speaker_id: str = ""
    learning_language: str = "it"
    learning_language_label: str = "Italiano"
    cefr_level: str = "B1"
    theme_id: str = ""
    theme_label: str = ""
    task_family: str = "free_monologue"
    duration_sec: int = 90
    prompt_id: str = ""
    prompt_text: str = ""


@dataclass
class AssessmentJobState:
    assessment_id: str = ""
    status: str = ""
    phase: str = ""
    progress: float = 0.0
    error: str = ""
    report_path: str = ""


@dataclass
class RecordingState:
    status: RecordingStatus = RecordingStatus.IDLE
    audio_path: str = ""
    duration_sec: int = 0
    input_digest: str = ""
    input_method: str = ""
    error: str = ""
    label_input: str = ""
    notes_input: str = ""
    job: AssessmentJobState = field(default_factory=AssessmentJobState)


@dataclass
class ReviewState:
    report_id: str = ""
    transcript: str = ""
    score_overall: float | None = None
    band: str = ""
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationState:
    current_page: str = "home"
    return_to: str = "home"


@dataclass
class AppState:
    prefs: AppPreferences = field(default_factory=AppPreferences)
    draft: DraftSession = field(default_factory=DraftSession)
    recording: RecordingState = field(default_factory=RecordingState)
    review: ReviewState = field(default_factory=ReviewState)
    nav: NavigationState = field(default_factory=NavigationState)


def ensure_session_id(state: AppState) -> None:
    if not state.draft.session_id:
        state.draft.session_id = f"draft-{uuid4().hex[:8]}"


def build_default_state() -> AppState:
    state = AppState()
    ensure_session_id(state)
    return state


def has_setup(state: AppState) -> bool:
    return bool(
        state.draft.speaker_id
        and state.draft.theme_id
        and state.draft.prompt_text
        and state.draft.cefr_level
    )


def has_recording(state: AppState) -> bool:
    return bool(state.recording.audio_path)


def has_review(state: AppState) -> bool:
    return bool(state.review.report_id)


__all__ = [
    "APP_NAME",
    "CEFR_LEVELS",
    "DEFAULT_MODEL",
    "DEFAULT_OPENROUTER_APP_TITLE",
    "DEFAULT_OPENROUTER_HTTP_REFERER",
    "DEFAULT_PROVIDER",
    "DEFAULT_UI_LOCALE",
    "DEFAULT_WHISPER_MODEL",
    "DURATION_OPTIONS",
    "SUPPORTED_UI_LOCALES",
    "TASK_FAMILY_OPTIONS",
    "AppPreferences",
    "AppState",
    "AssessmentJobState",
    "DraftSession",
    "NavigationState",
    "ProviderConnection",
    "RecordingState",
    "RecordingStatus",
    "ReviewState",
    "build_default_state",
    "ensure_session_id",
    "has_recording",
    "has_review",
    "has_setup",
    "normalize_openrouter_app_title",
    "normalize_openrouter_http_referer",
]

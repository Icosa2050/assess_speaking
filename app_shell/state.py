from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

import streamlit as st
from app_shell.runtime_providers import DEFAULT_PROVIDER, SUPPORTED_PROVIDERS

APP_SHELL_STATE_KEY = "app_shell_state"
APP_NAME = "Speaking Studio"
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
    whisper_cache_dir: str = ""
    openrouter_http_referer: str = DEFAULT_OPENROUTER_HTTP_REFERER
    openrouter_app_title: str = DEFAULT_OPENROUTER_APP_TITLE
    active_connection_id: str = ""
    connections: list[ProviderConnection] = field(default_factory=list)
    setup_complete: bool = False
    log_dir: str = "reports"


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
class RecordingState:
    status: RecordingStatus = RecordingStatus.IDLE
    audio_path: str = ""
    duration_sec: int = 0
    input_digest: str = ""
    input_method: str = ""
    error: str = ""
    label_input: str = ""
    notes_input: str = ""


@dataclass
class ReviewState:
    report_id: str = ""
    transcript: str = ""
    score_overall: Optional[float] = None
    band: str = ""
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationState:
    current_page: str = "home"
    return_to: str = "home"


@dataclass
class AppShellState:
    prefs: AppPreferences = field(default_factory=AppPreferences)
    draft: DraftSession = field(default_factory=DraftSession)
    recording: RecordingState = field(default_factory=RecordingState)
    review: ReviewState = field(default_factory=ReviewState)
    nav: NavigationState = field(default_factory=NavigationState)


def build_default_state() -> AppShellState:
    state = AppShellState()
    ensure_session_id(state)
    return state


def get_app_state() -> AppShellState:
    if APP_SHELL_STATE_KEY not in st.session_state:
        st.session_state[APP_SHELL_STATE_KEY] = build_default_state()
    state = st.session_state[APP_SHELL_STATE_KEY]
    ensure_session_id(state)
    return state


def ensure_session_id(state: AppShellState) -> None:
    if not state.draft.session_id:
        state.draft.session_id = f"draft-{uuid4().hex[:8]}"


def set_current_page(page_id: str) -> AppShellState:
    state = get_app_state()
    state.nav.current_page = page_id
    st.session_state["_page_id"] = page_id
    return state


def set_return_to(page_id: str) -> AppShellState:
    state = get_app_state()
    state.nav.return_to = page_id
    return state


def begin_new_session(*, preserve_preferences: bool = True) -> AppShellState:
    current = get_app_state()
    prefs = current.prefs if preserve_preferences else AppPreferences()
    state = AppShellState(prefs=prefs)
    ensure_session_id(state)
    st.session_state[APP_SHELL_STATE_KEY] = state
    st.session_state["_page_id"] = "home"
    _clear_widget_state("speak_label", "speak_notes")
    return state


def apply_setup(
    *,
    speaker_id: str,
    learning_language: str,
    learning_language_label: str,
    cefr_level: str,
    theme_id: str,
    theme_label: str,
    task_family: str,
    duration_sec: int,
    prompt_text: str,
) -> AppShellState:
    state = get_app_state()
    state.draft.speaker_id = speaker_id
    state.draft.learning_language = learning_language
    state.draft.learning_language_label = learning_language_label
    state.draft.cefr_level = cefr_level
    state.draft.theme_id = theme_id
    state.draft.theme_label = theme_label
    state.draft.task_family = task_family
    state.draft.duration_sec = int(duration_sec)
    state.draft.prompt_id = f"{theme_id}-{cefr_level.lower()}"
    state.draft.prompt_text = prompt_text
    state.recording = RecordingState()
    state.review = ReviewState()
    _clear_widget_state("speak_label", "speak_notes")
    ensure_session_id(state)
    return state


def update_recording(*, audio_path: str, duration_sec: int = 0, input_digest: str = "", input_method: str = "") -> AppShellState:
    state = get_app_state()
    state.recording.status = RecordingStatus.READY
    state.recording.audio_path = audio_path
    state.recording.duration_sec = duration_sec
    state.recording.input_digest = input_digest
    state.recording.input_method = input_method
    state.recording.error = ""
    return state


def update_recording_inputs(*, label_input: str, notes_input: str) -> AppShellState:
    state = get_app_state()
    state.recording.label_input = label_input
    state.recording.notes_input = notes_input
    return state


def set_recording_error(message: str) -> AppShellState:
    state = get_app_state()
    state.recording.status = RecordingStatus.IDLE
    state.recording.error = message
    return state


def set_recording_assessing() -> AppShellState:
    state = get_app_state()
    state.recording.status = RecordingStatus.ASSESSING
    state.recording.error = ""
    return state


def clear_recording(*, preserve_inputs: bool = True) -> AppShellState:
    state = get_app_state()
    state.recording = RecordingState(
        label_input=state.recording.label_input if preserve_inputs else "",
        notes_input=state.recording.notes_input if preserve_inputs else "",
    )
    return state


def apply_review_payload(
    *,
    payload: dict[str, Any],
    report_id: str,
    transcript: str,
    score_overall: Optional[float],
    band: str,
    summary: str,
) -> AppShellState:
    state = get_app_state()
    state.recording.status = RecordingStatus.SUBMITTED
    state.review.report_id = report_id or f"report-{uuid4().hex[:8]}"
    state.review.transcript = transcript
    state.review.score_overall = score_overall
    state.review.band = band
    state.review.summary = summary
    state.review.payload = payload
    return state


def clear_attempt(keep_setup: bool = True) -> AppShellState:
    state = get_app_state()
    preserved_label = state.recording.label_input if keep_setup else ""
    preserved_notes = state.recording.notes_input if keep_setup else ""
    state.recording = RecordingState(
        label_input=preserved_label,
        notes_input=preserved_notes,
    )
    state.review = ReviewState()
    if not keep_setup:
        state.draft = DraftSession(
            session_id=state.draft.session_id,
            speaker_id=state.draft.speaker_id,
            learning_language=state.draft.learning_language,
            learning_language_label=state.draft.learning_language_label,
        )
        _clear_widget_state("speak_label", "speak_notes")
    return state


def has_setup(state: Optional[AppShellState] = None) -> bool:
    state = state or get_app_state()
    return bool(
        state.draft.speaker_id
        and state.draft.theme_id
        and state.draft.prompt_text
        and state.draft.cefr_level
    )


def has_recording(state: Optional[AppShellState] = None) -> bool:
    state = state or get_app_state()
    return bool(state.recording.audio_path)


def has_review(state: Optional[AppShellState] = None) -> bool:
    state = state or get_app_state()
    return bool(state.review.report_id)


def serialize_state(state: Optional[AppShellState] = None) -> dict:
    state = state or get_app_state()
    return asdict(state)


def _clear_widget_state(*keys: str) -> None:
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

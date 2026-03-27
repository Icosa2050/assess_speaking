from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from assessment_runtime.asr import describe_model_availability, ensure_model_downloaded, recommend_model_choice
from assessment_runtime.llm_client import health_check as llm_health_check, test_connection as test_llm_connection
import assessment_runtime.theme_library as theme_library_store
from scripts import progress_dashboard
from app_shell.runtime_connections import deserialize_connections, ensure_single_default_connection, serialize_connections
from app_shell.runtime_providers import (
    connection_secret_ref,
    default_base_url,
    default_connection_label,
    default_setup_base_url,
    normalize_provider,
    provider_kind_from_choice,
    resolved_base_url,
    runtime_base_url,
    service_base_url,
)
from app_shell.runtime_resolver import active_connection, sync_runtime_fields
from app_shell.secret_store import SecretStoreStatus, delete_secret, get_secret, secret_store_status, set_secret
from app_shell.state import (
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_APP_TITLE,
    DEFAULT_OPENROUTER_HTTP_REFERER,
    DEFAULT_PROVIDER,
    DEFAULT_UI_LOCALE,
    DEFAULT_WHISPER_MODEL,
    ProviderConnection,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSESS_SCRIPT = PROJECT_ROOT / "assess_speaking.py"
DEFAULT_LOG_DIR = PROJECT_ROOT / "reports"
DEFAULT_WHISPER_OPTIONS = ("tiny", "base", "small", "medium", "large-v3")
NEW_LANGUAGE_OPTION = "__new_language__"
BOOTSTRAP_KEY = "_app_shell_bootstrapped"


def _secret_env_var_names(provider: str) -> tuple[str, ...]:
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return ("OPENROUTER_API_KEY", "LLM_API_KEY")
    if normalized == "ollama":
        return ("OLLAMA_API_KEY", "LLM_API_KEY")
    return ("LLM_API_KEY",)


def _env_api_key_for_provider(provider: str) -> str:
    normalized = normalize_provider(provider)
    if os.getenv("LLM_API_KEY"):
        return str(os.getenv("LLM_API_KEY") or "")
    if normalized == "openrouter":
        return str(os.getenv("OPENROUTER_API_KEY") or "")
    if normalized == "ollama":
        return str(os.getenv("OLLAMA_API_KEY") or "")
    return ""


def _persist_connection_secret(connection: ProviderConnection, api_key: str) -> SecretStoreStatus:
    connection.secret_ref = str(connection.secret_ref or connection_secret_ref(connection.connection_id)).strip()
    if not str(api_key or "").strip():
        return delete_secret(connection.secret_ref, env_var_names=_secret_env_var_names(connection.provider_kind))
    return set_secret(connection.secret_ref, str(api_key).strip(), env_var_names=_secret_env_var_names(connection.provider_kind))


def resolve_log_dir(log_dir: str | Path | None = None) -> Path:
    if log_dir is None:
        return DEFAULT_LOG_DIR
    return Path(log_dir).expanduser().resolve()


def load_theme_library(log_dir: str | Path | None = None) -> dict:
    return theme_library_store.load_theme_library(resolve_log_dir(log_dir))


def load_dashboard_prefs(log_dir: str | Path | None = None) -> dict:
    return theme_library_store.load_dashboard_prefs(resolve_log_dir(log_dir))


def save_dashboard_prefs(log_dir: str | Path | None, prefs: dict) -> None:
    theme_library_store.save_dashboard_prefs(resolve_log_dir(log_dir), prefs)


def save_theme_library(log_dir: str | Path | None, library: dict) -> None:
    theme_library_store.save_theme_library(resolve_log_dir(log_dir), library)


def needs_runtime_setup(state) -> bool:
    prefs = getattr(state, "prefs", state)
    connections = list(getattr(prefs, "connections", []) or [])
    return not bool(connections) and not bool(getattr(prefs, "setup_complete", False))


def build_provider_connection(
    *,
    provider_choice: str,
    label: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    openrouter_http_referer: str = "",
    openrouter_app_title: str = "",
    existing_connection: ProviderConnection | None = None,
) -> ProviderConnection:
    connection = existing_connection or ProviderConnection(connection_id=uuid4().hex)
    connection.connection_id = str(connection.connection_id or uuid4().hex)
    connection.provider_kind = provider_kind_from_choice(provider_choice)
    connection.label = str(label or default_connection_label(provider_choice)).strip() or default_connection_label(provider_choice)
    connection.default_model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    sanitized_base_url = sanitize_setup_base_url(provider_choice, base_url)
    connection.base_url = str(
        sanitized_base_url or default_setup_base_url(provider_choice) or default_base_url(connection.provider_kind)
    ).strip()
    connection.secret_ref = str(connection.secret_ref or connection_secret_ref(connection.connection_id)).strip()
    connection.is_local = provider_choice in {"ollama_local", "lmstudio_local"} or "localhost" in connection.base_url or "127.0.0.1" in connection.base_url
    connection.auth_mode = "bearer" if connection.provider_kind == "openrouter" or str(api_key or "").strip() else "none"
    metadata: dict[str, Any] = {}
    if connection.provider_kind == "openrouter":
        metadata = {
            "http_referer": str(openrouter_http_referer or DEFAULT_OPENROUTER_HTTP_REFERER).strip(),
            "app_title": str(openrouter_app_title or DEFAULT_OPENROUTER_APP_TITLE).strip(),
        }
    elif connection.provider_kind == "ollama":
        metadata = {"deployment": "local" if provider_choice == "ollama_local" or connection.is_local else "cloud"}
    elif connection.provider_kind == "lmstudio":
        metadata = {"deployment": "local", "token_optional": True}
    elif connection.provider_kind == "openai_compatible":
        metadata = {"deployment": "custom"}
    connection.provider_metadata = metadata
    return connection


def provider_choice_for_connection(connection: ProviderConnection | None, fallback_provider: str = DEFAULT_PROVIDER) -> str:
    if connection is None:
        provider = normalize_provider(fallback_provider)
        if provider == "ollama":
            return "ollama_local"
        if provider == "lmstudio":
            return "lmstudio_local"
        return provider
    if connection.provider_kind == "ollama":
        deployment = str(connection.provider_metadata.get("deployment") or "").strip().lower()
        if deployment == "cloud" or not connection.is_local:
            return "ollama_cloud"
        return "ollama_local"
    if connection.provider_kind == "lmstudio":
        return "lmstudio_local"
    return connection.provider_kind


def save_provider_connection(
    state,
    connection: ProviderConnection,
    *,
    api_key: str = "",
    persist_draft: bool = False,
) -> SecretStoreStatus:
    existing = list(getattr(state.prefs, "connections", []) or [])
    updated: list[ProviderConnection] = []
    replaced = False
    for item in existing:
        if item.connection_id == connection.connection_id:
            updated.append(connection)
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append(connection)
    state.prefs.connections, state.prefs.active_connection_id = ensure_single_default_connection(
        updated,
        active_connection_id=connection.connection_id,
    )
    state.prefs.setup_complete = bool(state.prefs.connections)
    sync_runtime_fields(state.prefs)
    if connection.provider_kind == "openrouter":
        state.prefs.openrouter_http_referer = str(
            connection.provider_metadata.get("http_referer") or DEFAULT_OPENROUTER_HTTP_REFERER
        ).strip()
        state.prefs.openrouter_app_title = str(
            connection.provider_metadata.get("app_title") or DEFAULT_OPENROUTER_APP_TITLE
        ).strip()
    state.prefs.llm_api_key = str(api_key or state.prefs.llm_api_key or "").strip()
    secret_status = _persist_connection_secret(connection, state.prefs.llm_api_key)
    save_state_preferences(state, persist_draft=persist_draft)
    return secret_status


def set_default_provider_connection(
    state,
    connection_id: str,
    *,
    persist_draft: bool = False,
) -> bool:
    connection_id = str(connection_id or "").strip()
    existing = list(getattr(state.prefs, "connections", []) or [])
    if not connection_id or not any(item.connection_id == connection_id for item in existing):
        return False
    state.prefs.connections, state.prefs.active_connection_id = ensure_single_default_connection(
        existing,
        active_connection_id=connection_id,
    )
    state.prefs.setup_complete = bool(state.prefs.connections)
    sync_runtime_fields(state.prefs)
    save_state_preferences(state, persist_draft=persist_draft)
    return True


def delete_provider_connection(
    state,
    connection_id: str,
    *,
    persist_draft: bool = False,
) -> bool:
    connection_id = str(connection_id or "").strip()
    existing = list(getattr(state.prefs, "connections", []) or [])
    if not connection_id or not existing:
        return False

    remaining: list[ProviderConnection] = []
    removed: ProviderConnection | None = None
    for connection in existing:
        if connection.connection_id == connection_id and removed is None:
            removed = connection
        else:
            remaining.append(connection)
    if removed is None:
        return False

    delete_secret(
        str(removed.secret_ref or connection_secret_ref(removed.connection_id)).strip(),
        env_var_names=_secret_env_var_names(removed.provider_kind),
    )

    next_active_id = state.prefs.active_connection_id
    if next_active_id == connection_id:
        next_active_id = ""

    state.prefs.connections, state.prefs.active_connection_id = ensure_single_default_connection(
        remaining,
        active_connection_id=next_active_id,
    )
    state.prefs.setup_complete = bool(state.prefs.connections)

    if state.prefs.connections:
        sync_runtime_fields(state.prefs)
    else:
        state.prefs.active_connection_id = ""
        state.prefs.llm_api_key = ""

    save_state_preferences(state, persist_draft=persist_draft)
    return True


def add_theme(
    library: dict,
    *,
    language_code: str,
    language_label: str,
    title: str,
    level: str,
    task_family: str,
) -> dict:
    return theme_library_store.add_theme(
        library,
        language_code=language_code,
        language_label=language_label,
        title=title,
        level=level,
        task_family=task_family,
    )


def language_codes(library: dict) -> list[str]:
    return theme_library_store.language_options(library)


def language_label(library: dict, language_code: str) -> str:
    return theme_library_store.language_label(library, language_code)


def themes_for_language_and_level(library: dict, language_code: str, level: str) -> list[dict]:
    return theme_library_store.themes_for_language_and_level(library, language_code, level)


def theme_option_label(theme_entry: dict) -> str:
    level = str(theme_entry.get("level") or "").upper()
    title = str(theme_entry.get("title") or "").strip()
    return f"{level} - {title}" if level else title


def theme_entry_id(theme_entry: dict) -> str:
    title = str(theme_entry.get("title") or "").strip().lower()
    title = re.sub(r"[^a-z0-9]+", "-", title).strip("-") or "theme"
    level = str(theme_entry.get("level") or "").strip().lower() or "b1"
    return f"{level}-{title}"


def validate_theme_submission(
    *,
    manage_mode: str,
    language_code: str,
    language_label_text: str,
    theme_title: str,
) -> dict[str, str]:
    errors: dict[str, str] = {}
    if manage_mode == NEW_LANGUAGE_OPTION:
        if not language_code.strip():
            errors["language_code"] = "language_code"
        if not language_label_text.strip():
            errors["language_label"] = "language_label"
    if not theme_title.strip():
        errors["theme_title"] = "theme_title"
    return errors


def build_practice_brief(
    *,
    task_family: str,
    theme: str,
    target_duration_sec: int,
    language_code: str,
) -> dict[str, Any]:
    theme = (theme or "").strip()
    minutes = round(float(target_duration_sec) / 60.0, 1)
    prompts = {
        "it": {
            "travel_narrative": f"Parla del tema '{theme}' seguendo una sequenza chiara: inizio, sviluppo, conclusione.",
            "personal_experience": f"Racconta il tema '{theme}' come un'esperienza personale con dettagli concreti.",
            "opinion_monologue": f"Esprimi la tua opinione sul tema '{theme}' con almeno due argomenti distinti.",
            "free_monologue": f"Parla in italiano del tema '{theme}' con una struttura semplice ma chiara.",
            "picture_description": f"Descrivi il tema '{theme}', interpreta il contesto e ipotizza cosa succede dopo.",
            "default_duration": f"Punta a parlare per circa {minutes} minuti." if minutes >= 1 else f"Punta a parlare per circa {target_duration_sec} secondi.",
            "success_focus": [
                "Usa connettivi chiari tra le idee.",
                "Chiudi con una breve riflessione personale.",
            ],
        },
        "en": {
            "travel_narrative": f"Speak about '{theme}' in a clear sequence: beginning, development, and ending.",
            "personal_experience": f"Explain '{theme}' as a personal experience with concrete details.",
            "opinion_monologue": f"Give your opinion about '{theme}' with at least two distinct arguments.",
            "free_monologue": f"Speak in English about '{theme}' with a simple but clear structure.",
            "picture_description": f"Describe '{theme}', explain the context, and suggest what could happen next.",
            "default_duration": f"Aim to speak for about {minutes} minutes." if minutes >= 1 else f"Aim to speak for about {target_duration_sec} seconds.",
            "success_focus": [
                "Link ideas with clear connectors.",
                "Finish with one personal takeaway.",
            ],
        },
        "de": {
            "travel_narrative": f"Sprich ueber '{theme}' in einer klaren Reihenfolge: Anfang, Entwicklung, Schluss.",
            "personal_experience": f"Erzaehle '{theme}' als persoenliche Erfahrung mit konkreten Details.",
            "opinion_monologue": f"Beziehe zu '{theme}' Stellung und stuetze deine Meinung mit mindestens zwei Argumenten.",
            "free_monologue": f"Sprich auf Deutsch ueber '{theme}' mit einer einfachen, klaren Struktur.",
            "picture_description": f"Beschreibe '{theme}', erklaere den Kontext und vermute, was danach passiert.",
            "default_duration": f"Sprich etwa {minutes} Minuten." if minutes >= 1 else f"Sprich etwa {target_duration_sec} Sekunden.",
            "success_focus": [
                "Verbinde deine Ideen mit klaren Uebergaengen.",
                "Beende den Beitrag mit einer kurzen Reflexion.",
            ],
        },
    }
    localized = prompts.get(language_code, prompts["en"])
    prompt = localized.get(task_family) or localized["free_monologue"]
    return {
        "prompt": prompt,
        "success_focus": [localized["default_duration"], *localized["success_focus"]],
    }


def _explicit_last_setup(prefs: dict[str, Any]) -> dict[str, Any]:
    payload = prefs.get("last_setup")
    return payload if isinstance(payload, dict) else {}


def _legacy_draft_preferences(prefs: dict[str, Any]) -> dict[str, Any]:
    return {
        "speaker_id": str(prefs.get("speaker_id") or "").strip(),
        "learning_language": str(prefs.get("learning_language") or prefs.get("language") or "").strip().lower(),
        "cefr_level": str(prefs.get("cefr_level") or "").strip().upper(),
        "theme": str(prefs.get("theme") or "").strip(),
        "task_family": str(prefs.get("task_family") or "").strip(),
        "target_duration_sec": prefs.get("target_duration_sec"),
    }


def _draft_preferences_from_state(state) -> dict[str, Any]:
    return {
        "speaker_id": str(state.draft.speaker_id or "").strip(),
        "learning_language": str(state.draft.learning_language or "").strip().lower(),
        "language": str(state.draft.learning_language or "").strip().lower(),
        "cefr_level": str(state.draft.cefr_level or "").strip().upper(),
        "theme": str(state.draft.theme_label or "").strip(),
        "task_family": str(state.draft.task_family or "").strip(),
        "target_duration_sec": int(state.draft.duration_sec or 90),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _speaker_profiles(prefs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = prefs.get("speaker_profiles")
    if not isinstance(payload, dict):
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for raw_speaker_id, raw_profile in payload.items():
        speaker_id = str(raw_speaker_id or "").strip()
        if not speaker_id or not isinstance(raw_profile, dict):
            continue
        profiles[speaker_id] = {
            "speaker_id": speaker_id,
            "learning_language": str(raw_profile.get("learning_language") or raw_profile.get("language") or "").strip().lower(),
            "cefr_level": str(raw_profile.get("cefr_level") or "").strip().upper(),
            "theme": str(raw_profile.get("theme") or "").strip(),
            "task_family": str(raw_profile.get("task_family") or "").strip(),
            "target_duration_sec": raw_profile.get("target_duration_sec"),
            "updated_at": str(raw_profile.get("updated_at") or "").strip(),
        }
    return profiles


def _history_draft_preferences(record: object) -> dict[str, Any]:
    payload = load_report_payload(getattr(record, "report_path", ""))
    report = payload.get("report") if isinstance(payload, dict) and isinstance(payload.get("report"), dict) else {}
    report_input = report.get("input") if isinstance(report.get("input"), dict) else {}
    baseline = payload.get("baseline_comparison") if isinstance(payload, dict) and isinstance(payload.get("baseline_comparison"), dict) else {}
    scores = report.get("scores") if isinstance(report.get("scores"), dict) else {}
    cefr_estimate = scores.get("cefr_estimate") if isinstance(scores.get("cefr_estimate"), dict) else {}
    return {
        "speaker_id": str(getattr(record, "speaker_id", "") or report_input.get("speaker_id") or "").strip(),
        "learning_language": str(
            getattr(record, "learning_language", "")
            or report_input.get("learning_language")
            or report_input.get("expected_language")
            or report_input.get("language_profile_key")
            or ""
        ).strip().lower(),
        "cefr_level": str(baseline.get("level") or cefr_estimate.get("level") or "").strip().upper(),
        "theme": str(getattr(record, "theme", "") or report_input.get("theme") or "").strip(),
        "task_family": str(getattr(record, "task_family", "") or report_input.get("task_family") or "").strip(),
        "target_duration_sec": report_input.get("target_duration_sec") or getattr(record, "target_duration_sec", ""),
        "updated_at": str(getattr(getattr(record, "timestamp", None), "isoformat", lambda: "")() or "").strip(),
    }


def _latest_history_draft_preferences(log_dir: str | Path | None, *, speaker_id: str = "") -> dict[str, Any]:
    selected_speaker = str(speaker_id or "").strip()
    for record in reversed(load_history_records(log_dir)):
        record_speaker = str(getattr(record, "speaker_id", "") or "").strip()
        if selected_speaker and record_speaker != selected_speaker:
            continue
        draft = _history_draft_preferences(record)
        if draft.get("speaker_id") or draft.get("learning_language") or draft.get("theme"):
            return draft
    return {}


def _resolved_draft_preferences(
    prefs: dict[str, Any],
    *,
    log_dir: str | Path | None,
    current_speaker_id: str = "",
) -> dict[str, Any]:
    def _timestamp(value: dict[str, Any]) -> tuple[int, float]:
        raw = str(value.get("updated_at") or "").strip()
        if not raw:
            return (0, 0.0)
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return (0, 0.0)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = parsed.astimezone(UTC)
        return (1, parsed.timestamp())

    def _pick_more_recent(*candidates: dict[str, Any]) -> dict[str, Any]:
        populated = [candidate for candidate in candidates if candidate]
        if not populated:
            return {}
        return max(populated, key=_timestamp)

    explicit_last_setup = _explicit_last_setup(prefs)
    legacy = _legacy_draft_preferences(prefs)
    selected_speaker = (
        str(current_speaker_id or "").strip()
        or str(explicit_last_setup.get("speaker_id") or "").strip()
        or str(legacy.get("speaker_id") or "").strip()
    )
    profiles = _speaker_profiles(prefs)
    history_fallback = _latest_history_draft_preferences(log_dir, speaker_id=selected_speaker)
    if selected_speaker and selected_speaker in profiles:
        return _pick_more_recent(profiles[selected_speaker], history_fallback)
    if explicit_last_setup:
        explicit_speaker = str(explicit_last_setup.get("speaker_id") or "").strip()
        if not selected_speaker or explicit_speaker == selected_speaker:
            return _pick_more_recent(explicit_last_setup, history_fallback)
    if history_fallback:
        return history_fallback
    if explicit_last_setup:
        return explicit_last_setup
    return legacy


def _load_runtime_connections(prefs: dict[str, Any]) -> tuple[list[ProviderConnection], str]:
    connections = deserialize_connections(prefs.get("connections"))
    active_connection_id = str(prefs.get("active_connection_id") or "").strip()
    connections, active_connection_id = ensure_single_default_connection(connections, active_connection_id)
    return connections, active_connection_id


def hydrate_state_from_storage(state) -> Any:
    if os.environ.get("APP_SHELL_SKIP_BOOTSTRAP") == "1":
        return state
    if hasattr(state, "__dict__") and getattr(state, BOOTSTRAP_KEY, False):
        return state
    prefs = load_dashboard_prefs(getattr(state.prefs, "log_dir", "") or DEFAULT_LOG_DIR)
    library = load_theme_library(prefs.get("log_dir") or getattr(state.prefs, "log_dir", "") or DEFAULT_LOG_DIR)
    state.prefs.ui_locale = str(prefs.get("ui_locale") or state.prefs.ui_locale or DEFAULT_UI_LOCALE)
    state.prefs.provider = normalize_provider(prefs.get("provider") or state.prefs.provider or DEFAULT_PROVIDER)
    state.prefs.model = str(prefs.get("model") or state.prefs.model or DEFAULT_MODEL)
    state.prefs.llm_base_url = resolved_base_url(
        state.prefs.provider,
        prefs.get("llm_base_url") or getattr(state.prefs, "llm_base_url", ""),
    )
    state.prefs.whisper_model = str(prefs.get("whisper_model") or state.prefs.whisper_model or DEFAULT_WHISPER_MODEL)
    state.prefs.whisper_cache_dir = str(prefs.get("whisper_cache_dir") or state.prefs.whisper_cache_dir or "").strip()
    state.prefs.llm_api_key = _env_api_key_for_provider(state.prefs.provider)
    state.prefs.openrouter_http_referer = str(
        prefs.get("openrouter_http_referer")
        or state.prefs.openrouter_http_referer
        or DEFAULT_OPENROUTER_HTTP_REFERER
    ).strip()
    loaded_app_title = str(
        prefs.get("openrouter_app_title")
        or state.prefs.openrouter_app_title
        or DEFAULT_OPENROUTER_APP_TITLE
    ).strip()
    if not loaded_app_title:
        loaded_app_title = DEFAULT_OPENROUTER_APP_TITLE
    state.prefs.openrouter_app_title = loaded_app_title
    state.prefs.connections, state.prefs.active_connection_id = _load_runtime_connections(prefs)
    state.prefs.setup_complete = bool(prefs.get("setup_complete")) or bool(state.prefs.connections)
    state.prefs.log_dir = str(resolve_log_dir(prefs.get("log_dir") or state.prefs.log_dir or DEFAULT_LOG_DIR))
    if state.prefs.connections:
        sync_runtime_fields(state.prefs)

    draft_prefs = _resolved_draft_preferences(
        prefs,
        log_dir=state.prefs.log_dir,
        current_speaker_id=state.draft.speaker_id,
    )
    state.draft.speaker_id = str(draft_prefs.get("speaker_id") or state.draft.speaker_id or "")
    state.draft.learning_language = str(
        draft_prefs.get("learning_language")
        or draft_prefs.get("language")
        or state.draft.learning_language
        or "it"
    ).strip().lower()
    state.draft.learning_language_label = language_label(library, state.draft.learning_language)
    state.draft.cefr_level = str(draft_prefs.get("cefr_level") or state.draft.cefr_level or "B1").upper()
    state.draft.duration_sec = int(float(draft_prefs.get("target_duration_sec") or state.draft.duration_sec or 90))
    state.draft.theme_label = str(draft_prefs.get("theme") or state.draft.theme_label or "").strip()
    state.draft.task_family = str(draft_prefs.get("task_family") or state.draft.task_family or "free_monologue")
    if state.draft.theme_label:
        matching = next(
            (
                item
                for item in themes_for_language_and_level(library, state.draft.learning_language, state.draft.cefr_level)
                if str(item.get("title") or "").strip() == state.draft.theme_label
            ),
            None,
        )
        if matching:
            state.draft.theme_id = theme_entry_id(matching)
            state.draft.task_family = str(matching.get("task_family") or state.draft.task_family)
        elif not state.draft.theme_id:
            state.draft.theme_id = theme_entry_id({"title": state.draft.theme_label, "level": state.draft.cefr_level})
        state.draft.prompt_text = build_practice_brief(
            task_family=state.draft.task_family,
            theme=state.draft.theme_label,
            target_duration_sec=state.draft.duration_sec,
            language_code=state.draft.learning_language,
        )["prompt"]
    setattr(state, BOOTSTRAP_KEY, True)
    return state


def save_state_preferences(state, *, persist_draft: bool = True) -> SecretStoreStatus:
    if state.prefs.connections:
        state.prefs.connections, state.prefs.active_connection_id = ensure_single_default_connection(
            list(state.prefs.connections or []),
            active_connection_id=state.prefs.active_connection_id,
        )
        sync_runtime_fields(state.prefs)
    active = active_connection(state.prefs)
    current_api_key = str(getattr(state.prefs, "llm_api_key", "") or "").strip()
    if active is not None and current_api_key:
        secret_status = _persist_connection_secret(active, current_api_key)
    else:
        secret_status = secret_store_status(env_var_names=_secret_env_var_names(state.prefs.provider))
    stored = load_dashboard_prefs(state.prefs.log_dir or DEFAULT_LOG_DIR)
    prefs = {
        **(stored if isinstance(stored, dict) else {}),
        "ui_locale": state.prefs.ui_locale,
        "provider": normalize_provider(state.prefs.provider),
        "model": state.prefs.model,
        "llm_base_url": resolved_base_url(state.prefs.provider, getattr(state.prefs, "llm_base_url", "")),
        "whisper_model": state.prefs.whisper_model,
        "whisper_cache_dir": state.prefs.whisper_cache_dir,
        "openrouter_http_referer": state.prefs.openrouter_http_referer,
        "openrouter_app_title": state.prefs.openrouter_app_title,
        "active_connection_id": state.prefs.active_connection_id,
        "connections": serialize_connections(list(state.prefs.connections or [])),
        "setup_complete": bool(state.prefs.setup_complete or state.prefs.connections),
        "log_dir": str(resolve_log_dir(state.prefs.log_dir or DEFAULT_LOG_DIR)),
    }
    prefs.pop("openrouter_api_key", None)
    prefs.pop("llm_api_key", None)
    if persist_draft:
        draft_prefs = _draft_preferences_from_state(state)
        prefs.update(draft_prefs)
        prefs["last_setup"] = draft_prefs
        profiles = _speaker_profiles(prefs)
        if draft_prefs["speaker_id"]:
            profiles[draft_prefs["speaker_id"]] = draft_prefs
        prefs["speaker_profiles"] = profiles
    save_dashboard_prefs(state.prefs.log_dir or DEFAULT_LOG_DIR, prefs)
    if state.prefs.connections:
        sync_runtime_fields(state.prefs)
    state.prefs.setup_complete = bool(state.prefs.setup_complete or state.prefs.connections)
    return secret_status


def whisper_model_status(model_size: str) -> dict[str, Any]:
    availability = describe_model_availability(model_size)
    recommendation = recommend_model_choice()
    availability["recommended"] = recommendation.get("model") == model_size
    availability["recommendation_reason"] = recommendation.get("reason") or ""
    return availability


def download_whisper_model(model_size: str, *, progress_callback=None) -> dict[str, Any]:
    ensure_model_downloaded(model_size, progress_callback=progress_callback)
    return whisper_model_status(model_size)


def sanitize_setup_base_url(provider_choice: str = "", base_url: str = "") -> str:
    candidate = str(base_url or "").strip().rstrip("/")
    normalized_choice = str(provider_choice or "").strip().lower()
    if not candidate:
        return ""
    if normalized_choice == "ollama_local":
        for suffix in ("/api/v1", "/v1", "/api"):
            if candidate.lower().endswith(suffix):
                candidate = candidate[: -len(suffix)].rstrip("/")
                break
    elif normalized_choice == "lmstudio_local" and candidate.lower().endswith("/v1"):
        candidate = f"{candidate[:-len('/v1')].rstrip('/')}/v1"
    return candidate.rstrip("/")


def _runtime_setup_test_timeout(provider_choice: str = "", base_url: str = "", timeout_sec: float = 10.0) -> float:
    selected_choice = str(provider_choice or "").strip().lower()
    candidate = str(base_url or "").strip().lower()
    is_local_setup_provider = selected_choice in {"ollama_local", "lmstudio_local"}
    is_localhost_endpoint = "localhost" in candidate or "127.0.0.1" in candidate
    if is_local_setup_provider or is_localhost_endpoint:
        return max(float(timeout_sec), 30.0)
    return float(timeout_sec)


def _health_payload_models(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    discovered: list[str] = []
    for collection_key in ("data", "models"):
        items = payload.get(collection_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for value_key in ("id", "name", "model"):
                candidate = str(item.get(value_key) or "").strip()
                if candidate:
                    if candidate not in discovered:
                        discovered.append(candidate)
                    break
    return discovered


def _first_model_from_health_payload(payload: dict[str, Any] | None) -> str:
    models = _health_payload_models(payload)
    return models[0] if models else ""


def discover_runtime_models(
    *,
    provider: str,
    provider_choice: str = "",
    base_url: str = "",
    api_key: str = "",
    openrouter_http_referer: str = "",
    openrouter_app_title: str = "",
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    normalized_provider = provider_kind_from_choice(provider_choice or provider)
    sanitized_base_url = sanitize_setup_base_url(provider_choice, base_url)
    if provider_choice and not sanitized_base_url:
        sanitized_base_url = default_setup_base_url(provider_choice)
    health_result = llm_health_check(
        provider=normalized_provider,
        base_url=sanitized_base_url or runtime_base_url(normalized_provider, sanitized_base_url),
        api_key=api_key,
        timeout_sec=timeout_sec,
        openrouter_http_referer=openrouter_http_referer,
        openrouter_app_title=openrouter_app_title,
    )
    health_payload = health_result.get("payload") or {}
    return {
        "provider": normalized_provider,
        "base_url": runtime_base_url(normalized_provider, sanitized_base_url),
        "service_base_url": service_base_url(normalized_provider, sanitized_base_url),
        "health_endpoint": str(health_result.get("endpoint") or ""),
        "health_payload": health_payload,
        "models": _health_payload_models(health_payload),
    }


def test_runtime_connection(
    *,
    provider: str,
    model: str,
    provider_choice: str = "",
    base_url: str = "",
    api_key: str = "",
    openrouter_http_referer: str = "",
    openrouter_app_title: str = "",
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    normalized_provider = provider_kind_from_choice(provider_choice or provider)
    sanitized_base_url = sanitize_setup_base_url(provider_choice, base_url)
    if provider_choice and not sanitized_base_url:
        sanitized_base_url = default_setup_base_url(provider_choice)
    timeout_sec = _runtime_setup_test_timeout(provider_choice, sanitized_base_url or base_url, timeout_sec)
    resolved_url = runtime_base_url(normalized_provider, sanitized_base_url)
    health_result = llm_health_check(
        provider=normalized_provider,
        base_url=sanitized_base_url or resolved_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        openrouter_http_referer=openrouter_http_referer,
        openrouter_app_title=openrouter_app_title,
    )
    health_payload = health_result.get("payload") or {}
    resolved_model = str(model or "").strip() or _first_model_from_health_payload(health_payload)
    if not resolved_model:
        raise ValueError("Enter a model name or use a provider that exposes models in the health check.")
    test_payload = test_llm_connection(
        provider=normalized_provider,
        model=resolved_model,
        base_url=resolved_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        openrouter_http_referer=openrouter_http_referer,
        openrouter_app_title=openrouter_app_title,
    )
    return {
        "provider": normalized_provider,
        "base_url": resolved_url,
        "service_base_url": service_base_url(normalized_provider, sanitized_base_url),
        "health_endpoint": str(health_result.get("endpoint") or ""),
        "health_payload": health_payload,
        "models_payload": health_payload,
        "test_payload": test_payload,
    }


# Keep the public helper name for call sites, but prevent pytest from
# mis-collecting it as a module-level test function.
test_runtime_connection.__test__ = False


def parse_cli_json(stdout: str) -> dict | None:
    stdout = stdout.strip()
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = stdout[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def load_latest_report_payload(log_dir: str | Path | None, *, label: str = "") -> dict | None:
    resolved = resolve_log_dir(log_dir)
    history_path = resolved / "history.csv"
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in reversed(rows):
            if label and row.get("label") != label:
                continue
            payload = load_report_payload(row.get("report_path") or "")
            if payload is not None:
                return payload
    for report_path in sorted(resolved.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        payload = load_report_payload(report_path)
        if payload is not None:
            return payload
    return None


def load_report_payload(report_path: str | Path | None) -> dict | None:
    if not report_path:
        return None
    path = Path(report_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def create_assessment_request(
    *,
    audio_path: Path,
    log_dir: str | Path | None,
    whisper: str,
    provider: str,
    llm_model: str,
    expected_language: str,
    feedback_language: str,
    speaker_id: str,
    task_family: str,
    theme: str,
    target_duration_sec: int,
    label: str = "",
    notes: str = "",
    target_cefr: str | None = None,
    language_profile_key: str | None = None,
    llm_base_url: str = "",
    llm_api_key: str = "",
    openrouter_http_referer: str = "",
    openrouter_app_title: str = "",
) -> dict[str, Any]:
    return {
        "audio_path": str(audio_path),
        "log_dir": str(resolve_log_dir(log_dir)),
        "whisper": whisper,
        "provider": normalize_provider(provider),
        "llm_model": llm_model,
        "llm_base_url": resolved_base_url(provider, llm_base_url),
        "expected_language": expected_language,
        "language_profile_key": language_profile_key,
        "feedback_language": feedback_language,
        "llm_api_key": llm_api_key,
        "openrouter_http_referer": openrouter_http_referer,
        "openrouter_app_title": openrouter_app_title,
        "speaker_id": speaker_id,
        "task_family": task_family,
        "theme": theme,
        "target_duration_sec": int(target_duration_sec),
        "label": label,
        "notes": notes,
        "target_cefr": target_cefr,
    }


def execute_assessment_request(request: dict[str, Any]) -> tuple[dict | None, str | None]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    request_api_key = str(request.get("llm_api_key") or "")
    if request_api_key:
        env["LLM_API_KEY"] = request_api_key
        if request.get("provider") == "openrouter":
            env["OPENROUTER_API_KEY"] = request_api_key
        if request.get("provider") == "ollama":
            env["OLLAMA_API_KEY"] = request_api_key
    if request.get("openrouter_http_referer"):
        env["OPENROUTER_HTTP_REFERER"] = str(request["openrouter_http_referer"])
    if request.get("openrouter_app_title"):
        env["OPENROUTER_APP_TITLE"] = str(request["openrouter_app_title"])
    cmd = [
        sys.executable,
        str(ASSESS_SCRIPT),
        request["audio_path"],
        "--whisper",
        request["whisper"],
        "--provider",
        request["provider"],
        "--llm-base-url",
        request.get("llm_base_url") or resolved_base_url(request.get("provider"), ""),
        "--expected-language",
        request["expected_language"],
        "--feedback-language",
        request.get("feedback_language") or request["expected_language"],
        "--llm-model",
        request["llm_model"],
        "--log-dir",
        request["log_dir"],
        "--theme",
        request["theme"],
        "--task-family",
        request["task_family"],
        "--target-duration-sec",
        str(float(request["target_duration_sec"])),
        "--speaker-id",
        request["speaker_id"],
    ]
    if request.get("language_profile_key"):
        cmd.extend(["--language-profile-key", str(request["language_profile_key"])])
    if request.get("label"):
        cmd.extend(["--label", request["label"]])
    if request.get("notes"):
        cmd.extend(["--notes", request["notes"]])
    if request.get("target_cefr"):
        cmd.extend(["--target-cefr", request["target_cefr"]])
    if os.getenv("ASSESS_SPEAKING_DRY_RUN") == "1":
        cmd.append("--dry-run")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        return None, (result.stderr or result.stdout or "Unknown assessment error").strip()
    payload = parse_cli_json(result.stdout)
    if payload is not None and not payload.get("transcript_full"):
        logged_payload = load_latest_report_payload(request["log_dir"], label=request.get("label", ""))
        if logged_payload is not None:
            payload = logged_payload
    elif payload is None:
        payload = load_latest_report_payload(request["log_dir"], label=request.get("label", ""))
    if payload is None:
        return None, "Assessment completed but no report payload could be loaded."
    return payload, None


def store_uploaded_audio(
    uploaded_file,
    *,
    target_dir: str | Path | None,
    filename: str | None = None,
    previous_digest: str = "",
    previous_path: str = "",
) -> tuple[Path | None, str]:
    try:
        data = uploaded_file.getvalue()
    except Exception:
        data = uploaded_file.getbuffer()
    digest = hashlib.sha1(bytes(data)).hexdigest()
    if digest == previous_digest and previous_path and Path(previous_path).exists():
        return Path(previous_path), digest
    suffix = Path(filename or getattr(uploaded_file, "name", "") or "audio.wav").suffix or ".wav"
    resolved = resolve_log_dir(target_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=resolved) as tmp:
        tmp.write(data)
        return Path(tmp.name), digest


def cleanup_temp_audio(path_str: str, *, allowed_root: str | Path | None = None) -> None:
    if not path_str:
        return
    path = Path(path_str)
    try:
        root = resolve_log_dir(allowed_root or DEFAULT_LOG_DIR)
        if path.exists() and root in path.resolve().parents:
            path.unlink()
    except OSError:
        return


def load_history_records(log_dir: str | Path | None = None) -> list[object]:
    resolved = resolve_log_dir(log_dir)
    history_path = resolved / "history.csv"
    if not history_path.exists():
        return []
    return progress_dashboard.load_history(history_path)


def history_rows(log_dir: str | Path | None = None) -> list[dict[str, Any]]:
    rows = []
    for record in load_history_records(log_dir):
        rows.append(
            {
                "timestamp": getattr(record, "timestamp", ""),
                "session_id": getattr(record, "session_id", ""),
                "speaker_id": getattr(record, "speaker_id", ""),
                "learning_language": getattr(record, "learning_language", ""),
                "theme": getattr(record, "theme", ""),
                "task_family": getattr(record, "task_family", ""),
                "final_score": getattr(record, "final_score", ""),
                "band": getattr(record, "band", ""),
            }
        )
    return rows


def review_summary(payload: dict | None) -> dict[str, Any]:
    payload = payload or {}
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
    scores = report.get("scores") if isinstance(report.get("scores"), dict) else {}
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    coaching = report.get("coaching") if isinstance(report.get("coaching"), dict) else {}
    rubric = report.get("rubric") if isinstance(report.get("rubric"), dict) else {}
    progress_delta = report.get("progress_delta") if isinstance(report.get("progress_delta"), dict) else {}
    failed_gates = [
        gate
        for gate in ("language_pass", "topic_pass", "duration_pass", "min_words_pass")
        if checks.get(gate) is False
    ]
    return {
        "report_id": str(report.get("session_id") or payload.get("report_path") or ""),
        "transcript": str(payload.get("transcript_full") or payload.get("transcript_preview") or report.get("transcript_preview") or ""),
        "notes": str(payload.get("notes") or ""),
        "score_overall": scores.get("final"),
        "band": scores.get("band") or "",
        "mode": str(scores.get("mode") or ""),
        "llm_score": scores.get("llm"),
        "deterministic_score": scores.get("deterministic"),
        "coach_summary": str(coaching.get("coach_summary") or ""),
        "strengths": [str(item) for item in coaching.get("strengths", []) if str(item).strip()],
        "priorities": [str(item) for item in coaching.get("top_3_priorities", []) if str(item).strip()],
        "next_focus": str(coaching.get("next_focus") or ""),
        "next_exercise": str(coaching.get("next_exercise") or ""),
        "warnings": [str(item) for item in report.get("warnings", []) if str(item).strip()],
        "requires_human_review": bool(report.get("requires_human_review")),
        "failed_gates": failed_gates,
        "gates": {
            "language_pass": _gate_value(checks, "language_pass"),
            "topic_pass": _gate_value(checks, "topic_pass"),
            "duration_pass": _gate_value(checks, "duration_pass"),
            "min_words_pass": _gate_value(checks, "min_words_pass"),
        },
        "recurring_grammar": [
            _humanize_issue_name(item.get("type") or item.get("category") or "")
            for item in rubric.get("recurring_grammar_errors", [])
            if isinstance(item, dict) and str(item.get("type") or item.get("category") or "").strip()
        ],
        "recurring_coherence": [
            _humanize_issue_name(item.get("type") or item.get("category") or "")
            for item in rubric.get("coherence_issues", [])
            if isinstance(item, dict) and str(item.get("type") or item.get("category") or "").strip()
        ],
        "progress_items": _build_progress_delta_items(progress_delta),
        "baseline": payload.get("baseline_comparison") if isinstance(payload.get("baseline_comparison"), dict) else None,
        "report": report,
        "payload": payload,
    }


def _humanize_issue_name(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    return value.replace("_", " ").capitalize()


def _build_progress_delta_items(progress_delta: dict[str, Any]) -> list[dict[str, Any]]:
    if not progress_delta:
        return []
    items: list[dict[str, Any]] = []
    score_delta = progress_delta.get("score_delta") if isinstance(progress_delta.get("score_delta"), dict) else {}
    if progress_delta.get("previous_session_id"):
        items.append({"kind": "previous_session", "value": str(progress_delta["previous_session_id"])})
    for key in ("final", "overall", "wpm"):
        value = score_delta.get(key)
        if isinstance(value, (int, float)) and value != 0:
            items.append({"kind": f"delta_{key}", "value": float(value)})
    for key in ("new_priorities", "resolved_priorities"):
        values = [str(item) for item in progress_delta.get(key, []) if str(item).strip()]
        if values:
            items.append({"kind": key, "value": values})
    repeating_grammar = [str(item) for item in progress_delta.get("repeating_grammar_categories", []) if str(item).strip()]
    if repeating_grammar:
        items.append({"kind": "repeating_grammar", "value": [_humanize_issue_name(item) for item in repeating_grammar]})
    repeating_coherence = [str(item) for item in progress_delta.get("repeating_coherence_categories", []) if str(item).strip()]
    if repeating_coherence:
        items.append({"kind": "repeating_coherence", "value": [_humanize_issue_name(item) for item in repeating_coherence]})
    return items


def _gate_value(checks: dict[str, Any], key: str) -> bool | None:
    value = checks.get(key)
    if value is True or value is False:
        return value
    return None

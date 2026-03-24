#!/usr/bin/env python3
"""Legacy Streamlit compatibility dashboard for assess_speaking results."""
from __future__ import annotations

import asyncio
import argparse
import csv
import hashlib
import io
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from assessment_runtime import asr as asr_backend
from assessment_runtime import progress_analysis
from assessment_runtime import theme_library as theme_library_store
from assess_core.settings import Settings
from scripts import progress_dashboard
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
try:
    from streamlit_webrtc.component import compile_state, generate_frontend_component_key
except Exception:  # pragma: no cover - optional internal API
    compile_state = None
    generate_frontend_component_key = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--log-dir")
_known_args, _ = _parser.parse_known_args()

DEFAULT_LOG_DIR = Path(_known_args.log_dir).expanduser().resolve() if _known_args.log_dir else PROJECT_ROOT / "reports"
ASSESS_SCRIPT = PROJECT_ROOT / "assess_speaking.py"
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "prompts.json"
DEFAULT_SETTINGS = Settings.from_env()
DEFAULT_PROVIDER = DEFAULT_SETTINGS.provider
DEFAULT_WHISPER_MODEL = asr_backend.recommend_model_choice()["model"]
DEFAULT_LLM_MODEL = (
    DEFAULT_SETTINGS.openrouter_rubric_model
    if DEFAULT_PROVIDER == "openrouter"
    else DEFAULT_SETTINGS.ollama_model
)
DEFAULT_TASK_FAMILY = "travel_narrative" if DEFAULT_SETTINGS.task_family == "generic" else DEFAULT_SETTINGS.task_family
DEFAULT_THEME = "Il mio ultimo viaggio all'estero"
DEFAULT_TARGET_DURATION_SEC = 180.0
DEFAULT_LEARNING_LANGUAGE = (DEFAULT_SETTINGS.expected_language or "it").lower()
DEFAULT_UI_LOCALE = os.getenv("ASSESS_UI_LOCALE", "de").strip().lower() or "de"
HAS_NATIVE_AUDIO_INPUT = callable(getattr(st, "audio_input", None))
WHISPER_MODEL_OPTIONS = list(asr_backend.KNOWN_WHISPER_MODELS)
PRACTICE_TASK_FAMILIES = [
    "travel_narrative",
    "personal_experience",
    "opinion_monologue",
    "picture_description",
    "free_monologue",
]
NEW_LANGUAGE_OPTION = "__new_language__"
PRACTICE_MODE_RECORD = "record"
PRACTICE_MODE_UPLOAD = "upload"
RECORDER_TRANSLATIONS = {
    "start": "Aufnahme starten",
    "stop": "Aufnahme beenden",
    "select_device": "Mikrofon wählen",
    "media_api_not_available": "Dieser Browser unterstützt die Mikrofonaufnahme nicht.",
    "device_ask_permission": "Bitte Mikrofonzugriff erlauben.",
    "device_not_available": "Kein Mikrofon gefunden.",
    "device_access_denied": "Mikrofonzugriff wurde verweigert.",
}
RECORDER_DEBUG_ENABLED = os.getenv("ASSESS_SPEAKING_RECORDER_DEBUG", "0") == "1"

UI_LOCALE_NAMES = {
    "de": "Deutsch",
    "en": "English",
    "it": "Italiano",
}

LEGACY_DASHBOARD_NOTICE = (
    "Legacy compatibility surface: this dashboard remains available during the "
    "migration to the multipage app shell. Prefer `streamlit run streamlit_app.py` "
    "for current product work. This dashboard will be archived once shell parity is signed off."
)

LEARNING_LANGUAGE_NAMES = {
    "it": "Italiano",
    "en": "English",
}

UI_STRINGS = {
    "de": {
        "log_dir": "Log-Verzeichnis",
        "ui_locale": "UX-Sprache",
        "learning_language": "Lernsprache",
        "workflow_title": "Workflow",
        "workflow_steps": (
            "Aufgabe definieren",
            "Direkt im Browser sprechen",
            "Aufnahme prüfen und auswerten",
            "Coaching lesen und direkt erneut versuchen",
        ),
        "training_heading": "Sprechtraining",
        "hero_title": "Sprich zuerst. Aufnahme und Aufgabenfokus stehen im Mittelpunkt.",
        "hero_subtitle": "Wähle ein Thema, hole dir einen klaren Sprechauftrag und versuche die Aufgabe direkt noch einmal nach dem Feedback.",
        "speaker_id": "Speaker ID",
        "theme_suggestion": "Themenvorschlag",
        "target_duration": "Zielsprechdauer (Sekunden)",
        "theme": "Thema",
        "task_settings": "Aufgabe anpassen",
        "secondary_tools": "Weitere Werkzeuge und Verlauf",
        "language_status": "Lernsprache: {language}. Die Gates prüfen Sprache, Thema, Dauer und Wortmenge.",
        "feedback_heading": "Deine Rückmeldung",
        "strengths_heading": "Das gelingt dir schon",
        "priorities_heading": "Darauf solltest du als Nächstes achten",
        "next_focus": "Konkreter Fokus für den nächsten Versuch:",
        "next_exercise": "Nächste Übung für dich:",
        "progress_heading": "Vergleich zum letzten Versuch",
        "recurring_grammar": "Wiederkehrende Grammatikmuster",
        "recurring_structure": "Wiederkehrende Strukturprobleme",
        "evaluation_details": "So wurde bewertet",
        "baseline_heading": "CEFR-Baseline",
        "warnings": "Warnungen",
        "raw_json": "Rohdaten und JSON",
        "retry_task": "Gleiche Aufgabe erneut versuchen",
        "gate_language": "Sprache",
        "gate_theme": "Thema",
        "gate_duration": "Dauer",
        "gate_words": "Wortmenge",
        "status_review": "Manuelle Prüfung empfohlen",
        "status_unstable": "Aufgabe noch nicht stabil erfüllt",
        "status_done": "Aufgabe erfüllt",
    },
    "en": {
        "log_dir": "Log directory",
        "ui_locale": "UX language",
        "learning_language": "Learning language",
        "workflow_title": "Workflow",
        "workflow_steps": (
            "Define the task",
            "Speak directly in the browser",
            "Review and evaluate the recording",
            "Read the coaching and try again",
        ),
        "training_heading": "Speaking practice",
        "hero_title": "Speak first. The recording and the task focus come first.",
        "hero_subtitle": "Choose a topic, get a clear speaking task, and try it again right after the feedback.",
        "speaker_id": "Speaker ID",
        "theme_suggestion": "Suggested theme",
        "target_duration": "Target speaking duration (seconds)",
        "theme": "Theme",
        "task_settings": "Adjust task",
        "secondary_tools": "More tools and progress",
        "language_status": "Learning language: {language}. The gates check language, topic, duration, and word count.",
        "feedback_heading": "Your feedback",
        "strengths_heading": "What is already working",
        "priorities_heading": "What to focus on next",
        "next_focus": "Specific focus for the next attempt:",
        "next_exercise": "Next exercise for you:",
        "progress_heading": "Comparison with your last attempt",
        "recurring_grammar": "Recurring grammar patterns",
        "recurring_structure": "Recurring structure issues",
        "evaluation_details": "How this was evaluated",
        "baseline_heading": "CEFR baseline",
        "warnings": "Warnings",
        "raw_json": "Raw data and JSON",
        "retry_task": "Try the same task again",
        "gate_language": "Language",
        "gate_theme": "Topic",
        "gate_duration": "Duration",
        "gate_words": "Word count",
        "status_review": "Manual review recommended",
        "status_unstable": "Task not yet completed reliably",
        "status_done": "Task completed",
    },
    "it": {
        "log_dir": "Cartella dei log",
        "ui_locale": "Lingua dell'interfaccia",
        "learning_language": "Lingua di apprendimento",
        "workflow_title": "Flusso",
        "workflow_steps": (
            "Definisci il compito",
            "Parla direttamente nel browser",
            "Controlla e valuta la registrazione",
            "Leggi il coaching e riprova subito",
        ),
        "training_heading": "Allenamento orale",
        "hero_title": "Prima parla. Registrazione e compito vengono al primo posto.",
        "hero_subtitle": "Scegli un tema, ottieni un compito orale chiaro e riprovalo subito dopo il feedback.",
        "speaker_id": "Speaker ID",
        "theme_suggestion": "Tema suggerito",
        "target_duration": "Durata target (secondi)",
        "theme": "Tema",
        "task_settings": "Personalizza il compito",
        "secondary_tools": "Altri strumenti e progressi",
        "language_status": "Lingua di apprendimento: {language}. I controlli verificano lingua, tema, durata e quantità di parole.",
        "feedback_heading": "Il tuo feedback",
        "strengths_heading": "Cosa sta già funzionando",
        "priorities_heading": "Su cosa concentrarti adesso",
        "next_focus": "Focus concreto per il prossimo tentativo:",
        "next_exercise": "Prossimo esercizio per te:",
        "progress_heading": "Confronto con l'ultimo tentativo",
        "recurring_grammar": "Pattern grammaticali ricorrenti",
        "recurring_structure": "Problemi strutturali ricorrenti",
        "evaluation_details": "Come è stata fatta la valutazione",
        "baseline_heading": "Baseline CEFR",
        "warnings": "Avvisi",
        "raw_json": "Dati grezzi e JSON",
        "retry_task": "Prova di nuovo lo stesso compito",
        "gate_language": "Lingua",
        "gate_theme": "Tema",
        "gate_duration": "Durata",
        "gate_words": "Parole",
        "status_review": "Revisione manuale consigliata",
        "status_unstable": "Compito non ancora svolto in modo stabile",
        "status_done": "Compito svolto",
    },
}


def _supported_ui_locale(locale_code: str | None) -> str:
    code = (locale_code or DEFAULT_UI_LOCALE).strip().lower()
    return code if code in UI_STRINGS else "de"


def ui_text(locale_code: str, key: str, **kwargs) -> str:
    locale_bucket = UI_STRINGS.get(_supported_ui_locale(locale_code), UI_STRINGS["de"])
    template = locale_bucket.get(key, UI_STRINGS["de"].get(key, key))
    if isinstance(template, str):
        return template.format(**kwargs)
    raise TypeError(f"UI text '{key}' is not a string")


def workflow_steps(locale_code: str) -> tuple[str, ...]:
    locale_bucket = UI_STRINGS.get(_supported_ui_locale(locale_code), UI_STRINGS["de"])
    return tuple(locale_bucket.get("workflow_steps", UI_STRINGS["de"]["workflow_steps"]))


def normalize_practice_mode(raw_mode: str | None) -> str:
    mode = (raw_mode or PRACTICE_MODE_RECORD).strip()
    if mode == PRACTICE_MODE_UPLOAD or mode == "Audiodatei hochladen":
        return PRACTICE_MODE_UPLOAD
    return PRACTICE_MODE_RECORD


def validate_theme_library_submission(
    *,
    manage_mode: str,
    language_code: str,
    language_label: str,
    theme_title: str,
) -> dict[str, str]:
    errors: dict[str, str] = {}
    if manage_mode == NEW_LANGUAGE_OPTION:
        if not language_code.strip():
            errors["language_code"] = "Bitte gib einen Sprachcode ein."
        if not language_label.strip():
            errors["language_label"] = "Bitte gib einen Sprachname ein."
    if not theme_title.strip():
        errors["theme_title"] = "Bitte gib ein Thema ein."
    return errors


def format_whisper_model_option(model_name: str) -> str:
    availability = asr_backend.describe_model_availability(model_name)
    suffix = "lokal" if availability["cached"] else "Download nötig"
    return f"{model_name} ({suffix})"


def build_rtc_configuration() -> RTCConfiguration:
    stun_urls = [url.strip() for url in os.getenv("ASSESS_SPEAKING_STUN_URLS", "").split(",") if url.strip()]
    if not stun_urls:
        # Local browser-to-local Streamlit works with host candidates only and avoids flaky external STUN retries.
        return RTCConfiguration(iceServers=[])
    return RTCConfiguration(iceServers=[{"urls": stun_urls}])


RTC_CONFIGURATION = build_rtc_configuration()


def _transport_is_usable(transport) -> bool:
    if transport is None:
        return False
    is_closing = getattr(transport, "is_closing", None)
    if callable(is_closing) and is_closing():
        return False
    sock = getattr(transport, "_sock", object())
    if sock is None:
        return False
    loop = getattr(transport, "_loop", None)
    if loop is not None and getattr(loop, "is_closed", lambda: False)():
        return False
    return True


def patch_aioice_closed_transport_bug() -> bool:
    try:
        import aioice.ice as aioice_ice
        import aioice.stun as aioice_stun
    except Exception:
        return False

    current_send = aioice_ice.StunProtocol.send_stun
    if getattr(current_send, "__assess_speaking_patched__", False):
        return True

    original_send_stun = current_send
    original_retry = aioice_stun.Transaction._Transaction__retry

    def safe_send_stun(self, message, addr) -> None:
        transport = getattr(self, "transport", None)
        if not _transport_is_usable(transport):
            return
        try:
            return original_send_stun(self, message, addr)
        except AttributeError as exc:
            if "sendto" in str(exc) or "call_exception_handler" in str(exc):
                LOGGER.debug("Ignoring aioice send_stun on closed transport: %s", exc)
                return
            raise
        except RuntimeError as exc:
            if "closed" in str(exc).lower():
                LOGGER.debug("Ignoring aioice send_stun on closed runtime: %s", exc)
                return
            raise

    def safe_retry(self) -> None:
        future = getattr(self, "_Transaction__future")
        if future.done():
            return
        tries = getattr(self, "_Transaction__tries")
        tries_max = getattr(self, "_Transaction__tries_max")
        if tries >= tries_max:
            future.set_exception(aioice_stun.TransactionTimeout())
            return

        protocol = getattr(self, "_Transaction__protocol")
        transport = getattr(protocol, "transport", None)
        if not _transport_is_usable(transport):
            future.set_exception(aioice_stun.TransactionTimeout())
            return

        try:
            original_send_stun(protocol, getattr(self, "_Transaction__request"), getattr(self, "_Transaction__addr"))
        except AttributeError as exc:
            if "sendto" in str(exc) or "call_exception_handler" in str(exc):
                future.set_exception(aioice_stun.TransactionTimeout())
                return
            raise
        except RuntimeError as exc:
            if "closed" in str(exc).lower():
                future.set_exception(aioice_stun.TransactionTimeout())
                return
            raise

        loop = asyncio.get_event_loop()
        if loop.is_closed():
            future.set_exception(aioice_stun.TransactionTimeout())
            return
        timeout_delay = getattr(self, "_Transaction__timeout_delay")
        setattr(self, "_Transaction__timeout_handle", loop.call_later(timeout_delay, self._Transaction__retry))
        setattr(self, "_Transaction__timeout_delay", timeout_delay * 2)
        setattr(self, "_Transaction__tries", tries + 1)

    safe_send_stun.__assess_speaking_patched__ = True
    safe_retry.__assess_speaking_patched__ = True
    aioice_ice.StunProtocol.send_stun = safe_send_stun
    aioice_stun.Transaction._Transaction__retry = safe_retry
    return True


AIOICE_PATCHED = patch_aioice_closed_transport_bug()

MODE_LABELS = {
    "hybrid": "Vollbewertung",
    "deterministic_only": "Basisbewertung",
}

ISSUE_LABELS = {
    "gender_agreement": "Genus und Angleichung",
    "number_agreement": "Singular und Plural",
    "article_usage": "Artikelgebrauch",
    "preposition_choice": "Präpositionen",
    "verb_conjugation_present": "Verbformen im Präsens",
    "verb_conjugation_past": "Verbformen in der Vergangenheit",
    "auxiliary_choice": "Hilfsverbwahl",
    "tense_consistency": "Zeitformen konsistent halten",
    "mood_selection": "Moduswahl",
    "word_order": "Wortstellung",
    "pronoun_usage": "Pronomen",
    "clitic_placement": "Pronomenstellung",
    "subject_omission_or_redundancy": "Subjekt fehlt oder wird doppelt gesetzt",
    "lexical_repetition": "Wortwiederholungen",
    "false_friend_or_wrong_word_choice": "Unpassende Wortwahl",
    "missing_sequence_markers": "Fehlende Reihenfolge-Marker",
    "weak_narrative_order": "Unklare Erzählreihenfolge",
    "abrupt_topic_shift": "Sprunghafter Themenwechsel",
    "insufficient_linking": "Zu wenig Verknüpfungen",
    "underdeveloped_detail": "Zu wenig konkrete Details",
    "repetition_without_progress": "Wiederholung ohne neue Information",
    "unclear_reference": "Unklare Bezüge",
}


@st.cache_resource(show_spinner=False)
def load_history_records(log_dir: Path):
    return progress_dashboard.load_history(log_dir / "history.csv")


@st.cache_data(show_spinner=False)
def load_history_df(log_dir: Path) -> pd.DataFrame:
    records = load_history_records(log_dir)
    if not records:
        return pd.DataFrame()
    data = {
        "timestamp": [r.timestamp for r in records],
        "session_id": [r.session_id for r in records],
        "speaker_id": [r.speaker_id for r in records],
        "task_family": [r.task_family for r in records],
        "theme": [r.theme for r in records],
        "label": [r.label for r in records],
        "audio": [r.audio for r in records],
        "whisper": [r.whisper for r in records],
        "llm": [r.llm for r in records],
        "target_duration_sec": [r.target_duration_sec for r in records],
        "duration_sec": [r.duration_sec for r in records],
        "wpm": [r.wpm for r in records],
        "word_count": [r.word_count for r in records],
        "overall": [r.overall for r in records],
        "final_score": [r.final_score for r in records],
        "band": [r.band for r in records],
        "requires_human_review": [r.requires_human_review for r in records],
        "top_priorities": [" | ".join(r.top_priorities) for r in records],
        "grammar_error_categories": [" | ".join(r.grammar_error_categories) for r in records],
        "coherence_issue_categories": [" | ".join(r.coherence_issue_categories) for r in records],
        "report_path": [r.report_path for r in records],
    }
    return pd.DataFrame(data)


def rerun_history(log_dir: Path):
    load_history_records.clear()
    load_history_df.clear()
    load_history_df(log_dir)


@st.cache_data(show_spinner=False)
def load_prompts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        item["learning_language"] = str(item.get("learning_language") or item.get("language") or "it").strip().lower()
        audio_path = Path(item["audio"])
        if not audio_path.is_absolute():
            item["audio_path"] = str((PROJECT_ROOT / audio_path).resolve())
        else:
            item["audio_path"] = str(audio_path)
    return data


def create_prompt_attempt(prompt: dict, now: float | None = None) -> dict:
    now = now or time.time()
    return {
        "id": prompt["id"],
        "start": now,
        "deadline": now + prompt["response_seconds"],
        "plays_remaining": prompt["max_playbacks"],
        "audio": prompt["audio_path"],
        "cefr": prompt["cefr_target"],
        "learning_language": str(prompt.get("learning_language") or "it").strip().lower(),
        "label": f"prompt:{prompt['id']}",
        **create_recording_attempt(),
    }


def create_recording_attempt() -> dict:
    return {
        "chunks": [],
        "sample_rate": None,
        "channels": None,
        "sample_width": 2,
        "is_recording": False,
        "is_signalling": False,
        "status": "idle",
        "saved_path": None,
        "saved_duration_sec": 0.0,
        "saved_chunk_count": 0,
        "save_error": "",
        "signalling_started_at": None,
        "connection_requested_at": None,
        "recording_started_at": None,
        "show_saved_notice": False,
        "input_digest": None,
    }


def remaining_time(attempt: dict, now: float | None = None) -> float:
    now = now or time.time()
    return attempt["deadline"] - now


def attempt_expired(attempt: dict, now: float | None = None) -> bool:
    return remaining_time(attempt, now=now) <= 0


def can_play_prompt(attempt: dict) -> bool:
    return attempt.get("plays_remaining", 0) > 0


def decrement_playback(attempt: dict) -> None:
    if attempt.get("plays_remaining", 0) <= 0:
        raise ValueError("No playbacks remaining")
    attempt["plays_remaining"] -= 1


def append_audio_bytes(
    attempt: dict,
    chunk: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int = 2,
) -> None:
    attempt.setdefault("chunks", []).append(chunk)
    attempt["sample_rate"] = sample_rate
    attempt["channels"] = channels
    attempt["sample_width"] = sample_width


def attempt_duration_sec(attempt: dict) -> float:
    chunks = attempt.get("chunks") or []
    sample_rate = attempt.get("sample_rate") or 0
    channels = attempt.get("channels") or 1
    sample_width = attempt.get("sample_width") or 2
    if not chunks or sample_rate <= 0:
        return 0.0
    total_bytes = sum(len(chunk) for chunk in chunks)
    bytes_per_second = sample_rate * channels * sample_width
    if bytes_per_second <= 0:
        return 0.0
    return round(total_bytes / bytes_per_second, 2)


def display_duration_sec(attempt: dict) -> float:
    measured = attempt_duration_sec(attempt)
    if attempt.get("status") != "recording":
        return measured
    started_at = attempt.get("recording_started_at")
    if not started_at:
        return measured
    return round(max(measured, time.time() - float(started_at)), 2)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def write_attempt_audio(attempt: dict, output_path: Path) -> None:
    chunks = attempt.get("chunks") or []
    if not chunks:
        raise ValueError("No audio chunks recorded")
    sample_rate = attempt.get("sample_rate") or 16000
    channels = attempt.get("channels") or 1
    sample_width = attempt.get("sample_width") or 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(chunks))


def save_recording_attempt(attempt: dict, target_dir: Path, prefix: str) -> Path:
    chunks = attempt.get("chunks") or []
    saved_path = attempt.get("saved_path")
    saved_chunk_count = int(attempt.get("saved_chunk_count") or 0)
    if saved_path and saved_chunk_count == len(chunks):
        path = Path(saved_path)
        if path.exists():
            attempt["saved_duration_sec"] = attempt_duration_sec(attempt)
            return path
    filename = f"{prefix}_{int(time.time())}.wav"
    output_path = target_dir / filename
    write_attempt_audio(attempt, output_path)
    attempt["saved_path"] = str(output_path)
    attempt["saved_chunk_count"] = len(chunks)
    attempt["saved_duration_sec"] = attempt_duration_sec(attempt)
    attempt["save_error"] = ""
    return output_path


def sync_recording_state(
    attempt: dict,
    webrtc_ctx,
    *,
    component_key: str | None = None,
    target_dir: Path,
    prefix: str,
    connection_timeout_sec: float = 8.0,
) -> dict:
    state = resolve_webrtc_state(component_key, webrtc_ctx)
    is_recording = bool(state and state.playing)
    is_signalling = bool(state and state.signalling)
    was_recording = bool(attempt.get("is_recording"))
    connection_requested_at = attempt.get("connection_requested_at")

    if is_recording and not was_recording and attempt.get("saved_path"):
        attempt = create_recording_attempt()
        connection_requested_at = None

    if is_signalling or is_recording:
        connection_requested_at = connection_requested_at or time.time()
    if is_signalling and not is_recording:
        attempt["signalling_started_at"] = attempt.get("signalling_started_at") or time.time()

    attempt["is_recording"] = is_recording
    attempt["is_signalling"] = is_signalling
    attempt["connection_requested_at"] = connection_requested_at

    if is_recording:
        if not attempt.get("recording_started_at"):
            attempt["recording_started_at"] = time.time()
        attempt["status"] = "recording"
        attempt["save_error"] = ""
        attempt["show_saved_notice"] = False
    elif was_recording:
        if attempt.get("chunks"):
            try:
                save_recording_attempt(attempt, target_dir, prefix)
                attempt["status"] = "ready"
                attempt["show_saved_notice"] = True
            except Exception as exc:  # pragma: no cover - defensive
                attempt["status"] = "error"
                attempt["save_error"] = f"Audio konnte nicht gespeichert werden: {exc}"
        else:
            attempt["status"] = "error"
            attempt["save_error"] = "Es wurde kein Audio aufgezeichnet. Bitte versuche es erneut oder nutze den Upload."
        attempt["signalling_started_at"] = None
        attempt["connection_requested_at"] = None
        attempt["recording_started_at"] = None
    elif attempt.get("saved_path"):
        attempt["status"] = "ready"
        attempt["signalling_started_at"] = None
        attempt["connection_requested_at"] = None
        attempt["recording_started_at"] = None
    elif is_signalling:
        waiting_since = attempt.get("connection_requested_at") or time.time()
        if time.time() - waiting_since >= connection_timeout_sec:
            attempt["status"] = "error"
            attempt["save_error"] = (
                "Die Mikrofonverbindung konnte nicht stabil aufgebaut werden. "
                "Bitte prüfe den Browser-Mikrofonzugriff oder nutze den Upload."
            )
            attempt["signalling_started_at"] = None
            attempt["connection_requested_at"] = None
            attempt["recording_started_at"] = None
        else:
            attempt["status"] = "connecting"
    elif connection_requested_at and not attempt.get("saved_path"):
        if time.time() - connection_requested_at >= connection_timeout_sec:
            attempt["status"] = "error"
            attempt["save_error"] = (
                "Die Aufnahme wurde nicht gestartet. Bitte erlaube den Mikrofonzugriff oder wechsle zur Datei-Option."
            )
            attempt["signalling_started_at"] = None
            attempt["connection_requested_at"] = None
            attempt["recording_started_at"] = None
        else:
            attempt["status"] = "connecting"
    else:
        attempt["status"] = "idle"
        attempt["show_saved_notice"] = False
        attempt["recording_started_at"] = None
    return attempt


def get_frontend_component_value(component_key: str | None):
    if not component_key or not generate_frontend_component_key:
        return None
    frontend_key = generate_frontend_component_key(component_key)
    component_value = st.session_state.get(frontend_key)
    if isinstance(component_value, str):
        try:
            component_value = json.loads(component_value)
        except json.JSONDecodeError:
            return {"_raw": component_value, "_invalid_json": True}
    if isinstance(component_value, dict):
        return component_value
    return None


def resolve_webrtc_state(component_key: str | None, webrtc_ctx):
    state = getattr(webrtc_ctx, "state", None)
    if not component_key or not compile_state or not generate_frontend_component_key:
        return state
    component_value = get_frontend_component_value(component_key)
    if not isinstance(component_value, dict):
        return state
    if component_value.get("_invalid_json"):
        return state
    frontend_state = compile_state(component_value)
    if webrtc_ctx is not None and state != frontend_state:
        setter = getattr(webrtc_ctx, "_set_state", None)
        if callable(setter):
            setter(frontend_state)
    return frontend_state


def build_recorder_debug_snapshot(
    attempt: dict,
    webrtc_ctx,
    *,
    component_key: str | None,
    audio_frames_count: int = 0,
) -> dict:
    ctx_state = getattr(webrtc_ctx, "state", None)
    frontend_value = get_frontend_component_value(component_key)
    resolved_state = resolve_webrtc_state(component_key, webrtc_ctx)
    return {
        "ts": round(time.time(), 3),
        "component_key": component_key,
        "status": attempt.get("status"),
        "ctx_playing": bool(ctx_state and getattr(ctx_state, "playing", False)),
        "ctx_signalling": bool(ctx_state and getattr(ctx_state, "signalling", False)),
        "resolved_playing": bool(resolved_state and getattr(resolved_state, "playing", False)),
        "resolved_signalling": bool(resolved_state and getattr(resolved_state, "signalling", False)),
        "frontend_value": frontend_value,
        "audio_receiver": bool(webrtc_ctx and getattr(webrtc_ctx, "audio_receiver", None)),
        "audio_frames_count": int(audio_frames_count),
        "chunk_count": len(attempt.get("chunks") or []),
        "measured_duration_sec": attempt_duration_sec(attempt),
        "display_duration_sec": display_duration_sec(attempt),
        "saved_path": attempt.get("saved_path"),
        "connection_requested_at": attempt.get("connection_requested_at"),
        "recording_started_at": attempt.get("recording_started_at"),
        "save_error": attempt.get("save_error"),
    }


def log_recorder_snapshot(session_key: str, snapshot: dict) -> None:
    serialized = json.dumps(snapshot, sort_keys=True, default=str)
    digest_key = f"{session_key}_debug_digest"
    if st.session_state.get(digest_key) == serialized:
        return
    st.session_state[digest_key] = serialized
    log_key = f"{session_key}_debug_log"
    entries = st.session_state.get(log_key) or []
    entries = list(entries)
    entries.append(snapshot)
    st.session_state[log_key] = entries[-25:]
    LOGGER.warning("RECORDER_DEBUG %s", serialized)


def render_recorder_debug(session_key: str, snapshot: dict) -> None:
    if not RECORDER_DEBUG_ENABLED:
        return
    log_key = f"{session_key}_debug_log"
    with st.expander("Recorder-Diagnose", expanded=True):
        st.caption("Rohzustand aus Streamlit und streamlit-webrtc. Relevant, wenn der Browser 'Running...' zeigt, aber die UI stillsteht.")
        st.json(snapshot)
        history = st.session_state.get(log_key) or []
        if history:
            st.caption("Letzte Zustandswechsel")
            st.json(history[-10:])


def mark_recording_connecting(attempt: dict) -> dict:
    attempt = dict(attempt)
    attempt["connection_requested_at"] = attempt.get("connection_requested_at") or time.time()
    if not attempt.get("saved_path"):
        attempt["status"] = "connecting"
    return attempt


def flag_recording_requested(session_key: str) -> None:
    attempt = st.session_state.get(session_key) or create_recording_attempt()
    st.session_state[session_key] = mark_recording_connecting(attempt)


def render_whisper_model_controls(
    *,
    select_key: str,
    notice_key: str,
    label: str = "Whisper-Modell",
    compact: bool = False,
    show_cache_details: bool = False,
) -> tuple[str, dict]:
    recommendation = asr_backend.recommend_model_choice()
    if select_key not in st.session_state:
        st.session_state[select_key] = recommendation["model"]
    selected_model = st.selectbox(
        label,
        options=WHISPER_MODEL_OPTIONS,
        key=select_key,
        format_func=format_whisper_model_option,
    )
    availability = asr_backend.describe_model_availability(selected_model)
    st.caption(f"Empfehlung: `{recommendation['model']}`. {recommendation['reason']}")

    notice = st.session_state.get(notice_key)
    if notice:
        level, message = notice
        getattr(st, level, st.info)(message)

    if availability["cached"]:
        status_method = st.caption if compact else st.success
        status_method(f"`{selected_model}` ist lokal vorhanden und sofort nutzbar.")
        if show_cache_details and availability.get("cached_path"):
            st.caption(f"Cache: `{availability['cached_path']}`")
        return selected_model, availability

    st.warning(f"`{selected_model}` ist noch nicht lokal vorhanden. Lade es zuerst herunter oder wähle ein bereits vorhandenes Modell.")
    action_cols = st.columns([1, 1])
    if action_cols[0].button(f"`{selected_model}` herunterladen", key=f"{notice_key}_download"):
        with st.spinner(f"Lade `{selected_model}` herunter..."):
            try:
                availability = asr_backend.ensure_model_downloaded(selected_model)
                st.session_state[notice_key] = (
                    "success",
                    f"`{selected_model}` wurde heruntergeladen und ist jetzt einsatzbereit.",
                )
            except Exception as exc:  # pragma: no cover - network / local environment
                st.session_state[notice_key] = (
                    "error",
                    f"Download von `{selected_model}` fehlgeschlagen: {exc}",
                )
        dashboard_rerun()
    if selected_model != recommendation["model"] and action_cols[1].button(
        f"Empfehlung `{recommendation['model']}` wählen",
        key=f"{notice_key}_recommend",
    ):
        st.session_state[select_key] = recommendation["model"]
        dashboard_rerun()
    return selected_model, availability


def run_assessment(
    audio_path: Path,
    log_dir: Path,
    whisper: str,
    llm: str,
    label: str,
    notes: str,
    target_cefr: str | None = None,
    *,
    provider: str = DEFAULT_PROVIDER,
    expected_language: str = DEFAULT_LEARNING_LANGUAGE,
    language_profile_key: str | None = None,
    feedback_language: str = DEFAULT_UI_LOCALE,
    speaker_id: str = "",
    task_family: str = DEFAULT_TASK_FAMILY,
    theme: str = DEFAULT_THEME,
    target_duration_sec: float = DEFAULT_TARGET_DURATION_SEC,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ASSESS_SCRIPT),
        str(audio_path),
        "--whisper",
        whisper,
        "--provider",
        provider,
        "--expected-language",
        expected_language,
        "--feedback-language",
        feedback_language,
        "--llm-model",
        llm,
        "--log-dir",
        str(log_dir),
        "--theme",
        theme,
        "--task-family",
        task_family,
        "--target-duration-sec",
        str(float(target_duration_sec)),
    ]
    if language_profile_key:
        cmd.extend(["--language-profile-key", language_profile_key])
    if speaker_id:
        cmd.extend(["--speaker-id", speaker_id])
    if label:
        cmd.extend(["--label", label])
    if notes:
        cmd.extend(["--notes", notes])
    if target_cefr:
        cmd.extend(["--target-cefr", target_cefr])
    if os.getenv("ASSESS_SPEAKING_DRY_RUN") == "1":
        cmd.append("--dry-run")

    return subprocess.run(cmd, capture_output=True, text=True)


def create_assessment_request(
    *,
    audio_path: Path,
    log_dir: Path,
    whisper: str,
    llm: str,
    label: str,
    notes: str,
    provider: str,
    expected_language: str,
    feedback_language: str,
    speaker_id: str,
    task_family: str,
    theme: str,
    target_duration_sec: float,
    target_cefr: str | None = None,
    language_profile_key: str | None = None,
) -> dict:
    return {
        "audio_path": str(audio_path),
        "log_dir": str(log_dir),
        "whisper": whisper,
        "llm": llm,
        "label": label,
        "notes": notes,
        "provider": provider,
        "expected_language": expected_language,
        "language_profile_key": language_profile_key,
        "feedback_language": feedback_language,
        "speaker_id": speaker_id,
        "task_family": task_family,
        "theme": theme,
        "target_duration_sec": float(target_duration_sec),
        "target_cefr": target_cefr,
    }


def execute_assessment_request(request: dict) -> subprocess.CompletedProcess:
    return run_assessment(
        Path(request["audio_path"]),
        Path(request["log_dir"]),
        request["whisper"],
        request["llm"],
        request.get("label", ""),
        request.get("notes", ""),
        target_cefr=request.get("target_cefr"),
        provider=request.get("provider", DEFAULT_PROVIDER),
        expected_language=request.get("expected_language", DEFAULT_LEARNING_LANGUAGE),
        language_profile_key=request.get("language_profile_key"),
        feedback_language=request.get("feedback_language", request.get("expected_language", DEFAULT_LEARNING_LANGUAGE)),
        speaker_id=request.get("speaker_id", ""),
        task_family=request.get("task_family", DEFAULT_TASK_FAMILY),
        theme=request.get("theme", DEFAULT_THEME),
        target_duration_sec=float(request.get("target_duration_sec", DEFAULT_TARGET_DURATION_SEC)),
    )


def build_prompt_assessment_request(
    *,
    attempt: dict,
    prompt: dict,
    response_path: Path,
    log_dir: Path,
    whisper: str,
    llm: str,
    notes: str,
    provider: str,
    speaker_id: str,
    ui_locale: str,
    target_cefr: str | None = None,
) -> dict:
    learning_language = str(
        attempt.get("learning_language")
        or prompt.get("learning_language")
        or DEFAULT_LEARNING_LANGUAGE
    ).strip().lower()
    return create_assessment_request(
        audio_path=response_path,
        log_dir=log_dir,
        whisper=whisper,
        llm=llm,
        label=attempt["label"],
        notes=notes,
        provider=provider,
        expected_language=learning_language,
        feedback_language=ui_locale,
        speaker_id=speaker_id,
        task_family="prompt_trainer",
        theme=prompt["title"],
        target_duration_sec=float(prompt["response_seconds"]),
        target_cefr=target_cefr or attempt.get("cefr"),
    )


def store_uploaded_audio(uploaded_file: io.BytesIO, original_name: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=target_dir) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def persist_audio_input(uploaded_audio, *, session_key: str, target_dir: Path, prefix: str) -> dict:
    attempt = st.session_state.get(session_key) or create_recording_attempt()
    if uploaded_audio is None:
        return attempt
    audio_bytes = uploaded_audio.getvalue()
    digest = hashlib.sha1(audio_bytes).hexdigest()
    saved_path = attempt.get("saved_path")
    if digest == attempt.get("input_digest") and saved_path and Path(saved_path).exists():
        attempt["status"] = "ready"
        return attempt
    try:
        output_path = store_uploaded_audio(
            uploaded_audio,
            getattr(uploaded_audio, "name", f"{prefix}.wav") or f"{prefix}.wav",
            target_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive
        errored = create_recording_attempt()
        errored["status"] = "error"
        errored["save_error"] = f"Audio konnte nicht gespeichert werden: {exc}"
        return errored
    stored = create_recording_attempt()
    stored["saved_path"] = str(output_path)
    stored["saved_chunk_count"] = 1
    stored["show_saved_notice"] = True
    stored["status"] = "ready"
    stored["input_digest"] = digest
    return stored


def reset_audio_input_recorder(*, session_key: str, version_key: str) -> None:
    st.session_state[session_key] = create_recording_attempt()
    st.session_state[version_key] = int(st.session_state.get(version_key, 0)) + 1


def parse_cli_json(stdout: str) -> dict | None:
    stdout = stdout.strip()
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = stdout[start:end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def load_latest_report_payload(log_dir: Path, *, label: str = "") -> dict | None:
    history_path = log_dir / "history.csv"
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in reversed(rows):
            if label and row.get("label") != label:
                continue
            report_path = row.get("report_path") or ""
            if report_path and Path(report_path).exists():
                try:
                    return json.loads(Path(report_path).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
    report_files = sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for report_path in report_files:
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def build_trend_chart_df(records: list[object]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    data = pd.DataFrame(
        {
            "timestamp": [r.timestamp for r in records],
            "wpm": [r.wpm for r in records],
            "overall": [r.overall for r in records],
            "final_score": [r.final_score for r in records],
        }
    )
    return data.set_index("timestamp").dropna(how="all")


def build_issue_count_df(records: list[object], attribute: str) -> pd.DataFrame:
    counts = progress_analysis.recurring_issue_counts(records, attribute)
    if not counts:
        return pd.DataFrame(columns=["category", "count"])
    return pd.DataFrame(
        [{"category": format_issue_label(category), "count": count} for category, count in counts.most_common()]
    )


def dashboard_rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def format_issue_label(issue: str) -> str:
    issue = str(issue or "").strip()
    if not issue:
        return "–"
    return ISSUE_LABELS.get(issue, issue.replace("_", " ").capitalize())


def format_language_option(library: dict, language_code: str) -> str:
    return theme_library_store.language_label(library, language_code)


def theme_option_label(theme_entry: dict) -> str:
    level = str(theme_entry.get("level") or "").upper()
    title = str(theme_entry.get("title") or "").strip()
    return f"{level} – {title}" if level else title


def generate_practice_brief(
    task_family: str,
    theme: str,
    target_duration_sec: float,
    *,
    language_code: str = DEFAULT_LEARNING_LANGUAGE,
    variant_index: int = 0,
) -> dict:
    theme = (theme or DEFAULT_THEME).strip()
    templates_by_language = {
        "it": {
            "travel_narrative": [
                {
                    "title": "Racconta il tema come una storia chiara",
                    "prompt": f"Parla in italiano del tema '{theme}'. Porta l'ascoltatore dall'inizio alla fine senza saltare passaggi importanti.",
                    "cover_points": ["Dove e con chi eri", "Che cosa è successo prima, poi e alla fine", "Che cosa ti è rimasto dell'esperienza"],
                    "starter_phrases": ["Prima di partire...", "La cosa più memorabile è stata...", "Alla fine ho capito che..."],
                },
                {
                    "title": "Rendi il racconto più concreto",
                    "prompt": f"Parla di '{theme}' aggiungendo dettagli precisi: luogo, persone, imprevisti e sensazioni.",
                    "cover_points": ["Un dettaglio visivo o pratico", "Un piccolo problema o sorpresa", "Una riflessione personale finale"],
                    "starter_phrases": ["Appena sono arrivato...", "A un certo punto...", "Se ci ripenso oggi..."],
                },
            ],
            "personal_experience": [
                {
                    "title": "Spiega un'esperienza personale con ordine",
                    "prompt": f"Parla del tema '{theme}' come se stessi raccontando un episodio importante a un amico.",
                    "cover_points": ["Contesto iniziale", "Momento decisivo", "Cosa hai imparato"],
                    "starter_phrases": ["All'inizio...", "Il momento chiave è stato...", "Da allora..."],
                }
            ],
            "opinion_monologue": [
                {
                    "title": "Prendi posizione e sostienila",
                    "prompt": f"Esprimi la tua opinione sul tema '{theme}' con almeno due argomenti chiari e un esempio.",
                    "cover_points": ["La tua posizione", "Due argomenti distinti", "Un esempio concreto"],
                    "starter_phrases": ["Secondo me...", "Il punto principale è...", "Per esempio..."],
                }
            ],
            "picture_description": [
                {
                    "title": "Descrivi e interpreta",
                    "prompt": f"Usa il tema '{theme}' per descrivere quello che si vede, spiegare il contesto e ipotizzare cosa succede dopo.",
                    "cover_points": ["Che cosa si vede", "Che atmosfera c'è", "Che cosa potrebbe succedere dopo"],
                    "starter_phrases": ["In primo piano...", "Mi sembra che...", "Probabilmente..."],
                }
            ],
            "free_monologue": [
                {
                    "title": "Parla liberamente, ma con una struttura",
                    "prompt": f"Parla in italiano del tema '{theme}' mantenendo una struttura semplice: apertura, sviluppo, chiusura.",
                    "cover_points": ["Introduzione breve", "Due sviluppi chiari", "Chiusura con opinione o lezione"],
                    "starter_phrases": ["Vorrei parlare di...", "Un aspetto importante è...", "In conclusione..."],
                }
            ],
            "success_focus": [
                "Usa connettivi per legare gli eventi o le idee.",
                "Chiudi con una riflessione personale invece di fermarti bruscamente.",
            ],
            "summary_caption": "Ziel: ruhig sprechen, einen roten Faden halten und sauber abschließen.",
        },
        "en": {
            "travel_narrative": [
                {
                    "title": "Tell the story in a clear sequence",
                    "prompt": f"Speak in English about '{theme}'. Lead the listener from the beginning to the end without skipping important steps.",
                    "cover_points": ["Where you were and who was with you", "What happened first, next, and in the end", "What stayed with you afterwards"],
                    "starter_phrases": ["Before I left...", "The most memorable part was...", "In the end I realised that..."],
                },
                {
                    "title": "Make the story more concrete",
                    "prompt": f"Talk about '{theme}' with specific details: place, people, surprises, and feelings.",
                    "cover_points": ["One vivid detail", "One problem or surprise", "A final personal reflection"],
                    "starter_phrases": ["As soon as I arrived...", "At one point...", "Looking back now..."],
                },
            ],
            "personal_experience": [
                {
                    "title": "Explain a personal experience in order",
                    "prompt": f"Talk about '{theme}' as if you were telling an important experience to a friend.",
                    "cover_points": ["The initial context", "The key moment", "What you learned"],
                    "starter_phrases": ["At the beginning...", "The key moment was...", "Since then..."],
                }
            ],
            "opinion_monologue": [
                {
                    "title": "Take a position and support it",
                    "prompt": f"Give your opinion on '{theme}' with at least two clear arguments and one example.",
                    "cover_points": ["Your position", "Two distinct arguments", "One concrete example"],
                    "starter_phrases": ["In my view...", "The main point is...", "For example..."],
                }
            ],
            "picture_description": [
                {
                    "title": "Describe and interpret",
                    "prompt": f"Use the theme '{theme}' to describe what can be seen, explain the context, and suggest what might happen next.",
                    "cover_points": ["What can be seen", "What atmosphere is present", "What may happen next"],
                    "starter_phrases": ["In the foreground...", "It seems to me that...", "Probably..."],
                }
            ],
            "free_monologue": [
                {
                    "title": "Speak freely, but keep a structure",
                    "prompt": f"Speak in English about '{theme}' with a simple structure: opening, development, closing.",
                    "cover_points": ["A short introduction", "Two clear developments", "A closing thought or lesson"],
                    "starter_phrases": ["I’d like to talk about...", "One important aspect is...", "To sum up..."],
                }
            ],
            "success_focus": [
                "Link ideas with connectors instead of short isolated sentences.",
                "Finish with a personal reflection instead of stopping abruptly.",
            ],
            "summary_caption": "Goal: speak calmly, keep a clear thread, and finish cleanly.",
        },
    }
    templates = templates_by_language.get(language_code) or templates_by_language["en"]
    options = templates.get(task_family) or templates["free_monologue"]
    chosen = options[variant_index % len(options)]
    target_minutes = round(float(target_duration_sec) / 60.0, 1)
    duration_focus = (
        f"Punta a parlare per circa {target_minutes} minuti."
        if language_code == "it" and target_minutes >= 1
        else f"Punta a parlare per circa {int(target_duration_sec)} secondi."
        if language_code == "it"
        else f"Aim to speak for about {target_minutes} minutes."
        if target_minutes >= 1
        else f"Aim to speak for about {int(target_duration_sec)} seconds."
    )
    return {
        "title": chosen["title"],
        "prompt": chosen["prompt"],
        "cover_points": chosen["cover_points"],
        "starter_phrases": chosen["starter_phrases"],
        "success_focus": [
            duration_focus,
            *templates["success_focus"],
        ],
        "summary_caption": templates["summary_caption"],
    }


def render_practice_brief(brief: dict) -> None:
    card_cols = st.columns([1.4, 1, 1])
    with card_cols[0]:
        st.markdown("### Dein Sprechauftrag")
        st.markdown(f"**{brief['title']}**")
        st.write(brief["prompt"])
        st.caption(brief.get("summary_caption", ""))
    with card_cols[1]:
        st.markdown("### Darüber solltest du sprechen")
        for item in brief["cover_points"]:
            st.markdown(f"- {item}")
    with card_cols[2]:
        st.markdown("### Hilfreiche Satzanfänge")
        for item in brief["starter_phrases"]:
            st.markdown(f"- {item}")
        st.markdown("### Worauf du heute achten solltest")
        for item in brief["success_focus"]:
            st.markdown(f"- {item}")


def render_step_strip(current_step: int, *, ui_locale: str) -> None:
    localized_steps = workflow_steps(ui_locale)
    step_cols = st.columns(len(localized_steps))
    for idx, label in enumerate(localized_steps, start=1):
        text = f"{idx}. {label}"
        if idx < current_step:
            step_cols[idx - 1].success(text)
        elif idx == current_step:
            step_cols[idx - 1].info(text)
        else:
            step_cols[idx - 1].caption(text)


def render_live_timer_widget(attempt: dict, *, target_duration_sec: float, key: str) -> None:
    measured = attempt_duration_sec(attempt)
    started_at = attempt.get("recording_started_at")
    status = attempt.get("status") or "idle"
    target_sec = max(0, int(round(float(target_duration_sec))))
    started_at_js = "null" if not started_at else str(float(started_at))
    html = f"""
    <div id="{key}" style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0.25rem 0 0.75rem 0;">
      <div style="padding:12px;border-radius:12px;background:#0f172a;color:#e5e7eb;">
        <div style="font-size:0.85rem;opacity:0.8;">Gesprochen</div>
        <div data-role="spoken" style="font-size:2rem;font-weight:700;">00:00</div>
      </div>
      <div style="padding:12px;border-radius:12px;background:#111827;color:#e5e7eb;">
        <div style="font-size:0.85rem;opacity:0.8;">Ziel</div>
        <div data-role="target" style="font-size:2rem;font-weight:700;">00:00</div>
      </div>
      <div style="padding:12px;border-radius:12px;background:#111827;color:#e5e7eb;">
        <div style="font-size:0.85rem;opacity:0.8;">Rest</div>
        <div data-role="remaining" style="font-size:2rem;font-weight:700;">00:00</div>
      </div>
    </div>
    <script>
    (function() {{
      const root = document.getElementById({json.dumps(key)});
      if (!root) return;
      const spokenEl = root.querySelector('[data-role="spoken"]');
      const targetEl = root.querySelector('[data-role="target"]');
      const remainingEl = root.querySelector('[data-role="remaining"]');
      const target = {target_sec};
      const status = {json.dumps(status)};
      const measured = {float(measured)};
      const startedAt = {started_at_js};
      function fmt(total) {{
        total = Math.max(0, Math.round(total));
        const m = String(Math.floor(total / 60)).padStart(2, '0');
        const s = String(total % 60).padStart(2, '0');
        return `${{m}}:${{s}}`;
      }}
      function render() {{
        const now = Date.now() / 1000;
        const spoken = (status === 'recording' && startedAt) ? Math.max(measured, now - startedAt) : measured;
        const remaining = Math.max(0, target - spoken);
        spokenEl.textContent = fmt(spoken);
        targetEl.textContent = fmt(target);
        remainingEl.textContent = fmt(remaining);
      }}
      render();
      if (status === 'recording') {{
        const id = setInterval(render, 1000);
        window.addEventListener('beforeunload', () => clearInterval(id), {{ once: true }});
      }}
    }})();
    </script>
    """
    components.html(html, height=118)


def render_recorder_status(
    attempt: dict,
    *,
    target_duration_sec: float,
    evaluation_running: bool = False,
    title: str = "Status der Aufnahme",
) -> None:
    if evaluation_running:
        st.info("Auswertung läuft. Das dauert meist 15 bis 30 Sekunden.")
        return

    status = attempt.get("status", "idle")
    st.markdown(f"### {title}")
    render_live_timer_widget(attempt, target_duration_sec=target_duration_sec, key=f"timer_{title}_{attempt.get('saved_path')}_{attempt.get('recording_started_at')}")
    recorded_sec = display_duration_sec(attempt)
    target_sec = float(target_duration_sec)
    progress = 0.0 if target_sec <= 0 else min(recorded_sec / target_sec, 1.0)
    st.progress(progress)

    if status == "idle":
        st.info("Bereit. Klicke auf 'Aufnahme starten'. Nach dem Stoppen wird die Datei automatisch gespeichert.")
    elif status == "connecting":
        st.info("Mikrofon wird verbunden. Wenn nach einigen Sekunden nichts passiert, prüfe den Browser-Mikrofonzugriff oder nutze den Upload darunter.")
    elif status == "recording":
        st.success("Aufnahme läuft. Wenn du fertig bist, beende die Aufnahme direkt im Recorder.")
        if recorded_sec >= target_sec and target_sec > 0:
            st.caption("Zielzeit erreicht. Du kannst jetzt ruhig abschließen.")
    elif status == "ready":
        st.success(f"Audio erfolgreich gespeichert ({format_duration(attempt.get('saved_duration_sec') or recorded_sec)}).")
        st.markdown("1. Kurz anhören\n2. Aufnahme auswerten\n3. Bei Bedarf neu aufnehmen")
    elif status == "error":
        st.error(attempt.get("save_error") or "Mit der Aufnahme ist ein Fehler aufgetreten.")


def render_native_recorder_status(
    attempt: dict,
    *,
    title: str,
    target_duration_sec: float,
    evaluate_label: str,
) -> None:
    st.markdown(f"### {title}")
    if attempt.get("saved_path"):
        st.success("Aufnahme gespeichert. Du kannst sie jetzt anhören oder direkt auswerten.")
        saved_path = Path(str(attempt["saved_path"]))
        if saved_path.exists():
            st.audio(saved_path.read_bytes(), format="audio/wav")
        st.caption(f"Nächster Schritt: Klicke rechts auf `{evaluate_label}`.")
        return
    if attempt.get("status") == "error":
        st.error(attempt.get("save_error") or "Mit der Aufnahme ist ein Fehler aufgetreten.")
        return
    target_minutes = round(float(target_duration_sec) / 60.0, 1)
    st.info(
        "Der Browser-Recorder übernimmt Start, Stop und Laufzeit direkt im Widget oben."
    )
    st.caption(
        f"Ziel: etwa {target_minutes} Minuten sprechen. Nach dem Stoppen wird die Datei automatisch übernommen."
    )


def current_practice_step(attempt: dict, *, evaluation_running: bool = False) -> int:
    if evaluation_running:
        return 3
    status = attempt.get("status")
    if status in {"recording", "connecting", "ready", "error"}:
        return 2
    return 1


def render_practice_recorder_fragment(log_dir: Path, target_duration_sec: float) -> None:
    practice_attempt = st.session_state.get("practice_attempt") or create_recording_attempt()

    webrtc_ctx = webrtc_streamer(
        key="practice_recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},
        translations=RECORDER_TRANSLATIONS,
        on_change=lambda: flag_recording_requested("practice_attempt"),
    )
    audio_frames = []
    if webrtc_ctx and webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            audio_frames = []
        if audio_frames:
            for frame in audio_frames:
                audio = frame.to_ndarray()
                sample_rate = getattr(frame, "sample_rate", 48000)
                if audio.ndim == 1:
                    channels = 1
                    data = audio
                else:
                    channels = audio.shape[0]
                    data = audio.T
                if data.dtype != np.int16:
                    data = np.clip(data, -1.0, 1.0)
                    data = (data * 32767).astype(np.int16)
                else:
                    data = data.astype(np.int16, copy=False)
                append_audio_bytes(practice_attempt, data.tobytes(), sample_rate, channels)

    practice_attempt = sync_recording_state(
        practice_attempt,
        webrtc_ctx,
        component_key="practice_recorder",
        target_dir=log_dir / "recordings",
        prefix="practice",
    )
    st.session_state["practice_attempt"] = practice_attempt
    debug_snapshot = build_recorder_debug_snapshot(
        practice_attempt,
        webrtc_ctx,
        component_key="practice_recorder",
        audio_frames_count=len(audio_frames),
    )
    log_recorder_snapshot("practice_attempt", debug_snapshot)

    render_recorder_status(
        practice_attempt,
        target_duration_sec=target_duration_sec,
        evaluation_running=st.session_state.get("manual_assessment_running", False),
        title="Status deiner Aufnahme",
    )
    render_recorder_debug("practice_attempt", debug_snapshot)
    if practice_attempt.get("saved_path"):
        saved_path = Path(practice_attempt["saved_path"])
        if saved_path.exists():
            st.audio(saved_path.read_bytes(), format="audio/wav")
        recorder_actions = st.columns(2)
        if recorder_actions[0].button("Neue Aufnahme starten", key="reset_practice_recording"):
            st.session_state["practice_attempt"] = create_recording_attempt()
            st.session_state["manual_payload"] = None
            st.rerun()
        recorder_actions[1].markdown("**Nächster Schritt:** Klicke rechts auf `Aufnahme auswerten`.")
    elif practice_attempt.get("status") == "error":
        if st.button("Erneut aufnehmen", key="retry_practice_recording"):
            st.session_state["practice_attempt"] = create_recording_attempt()
            st.rerun()



def build_result_summary(payload: dict, *, ui_locale: str) -> dict:
    report = payload.get("report") if isinstance(payload, dict) else None
    report = report if isinstance(report, dict) else {}
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    scores = report.get("scores") if isinstance(report.get("scores"), dict) else {}
    coaching = report.get("coaching") if isinstance(report.get("coaching"), dict) else {}
    rubric = report.get("rubric") if isinstance(report.get("rubric"), dict) else {}
    progress_delta = report.get("progress_delta") if isinstance(report.get("progress_delta"), dict) else None
    warnings = [str(item) for item in report.get("warnings", []) if str(item).strip()]
    recurring_grammar = [
        issue.get("type")
        for issue in rubric.get("recurring_grammar_errors", [])
        if isinstance(issue, dict) and issue.get("type")
    ]
    recurring_coherence = [
        issue.get("type")
        for issue in rubric.get("coherence_issues", [])
        if isinstance(issue, dict) and issue.get("type")
    ]
    gates = [
        (ui_text(ui_locale, "gate_language"), bool(checks.get("language_pass"))),
        (ui_text(ui_locale, "gate_theme"), bool(checks.get("topic_pass"))),
        (ui_text(ui_locale, "gate_duration"), bool(checks.get("duration_pass"))),
        (ui_text(ui_locale, "gate_words"), bool(checks.get("min_words_pass"))),
    ]
    failed_gates = [label for label, passed in gates if not passed]
    requires_review = bool(report.get("requires_human_review"))
    if requires_review:
        status_level = "warning"
        status_title = ui_text(ui_locale, "status_review")
    elif failed_gates:
        status_level = "info"
        status_title = ui_text(ui_locale, "status_unstable")
    else:
        status_level = "success"
        status_title = ui_text(ui_locale, "status_done")
    return {
        "status_level": status_level,
        "status_title": status_title,
        "requires_review": requires_review,
        "failed_gates": failed_gates,
        "gates": [{"label": label, "passed": passed} for label, passed in gates],
        "final_score": scores.get("final"),
        "band": scores.get("band"),
        "mode": scores.get("mode"),
        "mode_label": MODE_LABELS.get(scores.get("mode"), "Unbekannt"),
        "llm_score": scores.get("llm"),
        "deterministic_score": scores.get("deterministic"),
        "strengths": [str(item) for item in coaching.get("strengths", []) if str(item).strip()],
        "priorities": [str(item) for item in coaching.get("top_3_priorities", []) if str(item).strip()],
        "next_focus": str(coaching.get("next_focus") or ""),
        "next_exercise": str(coaching.get("next_exercise") or ""),
        "coach_summary": str(coaching.get("coach_summary") or ""),
        "warnings": warnings,
        "progress_delta": progress_delta,
        "progress_lines": build_progress_delta_lines(progress_delta),
        "recurring_grammar": [format_issue_label(issue) for issue in recurring_grammar],
        "recurring_coherence": [format_issue_label(issue) for issue in recurring_coherence],
        "baseline": payload.get("baseline_comparison") if isinstance(payload, dict) else None,
    }


def build_progress_delta_lines(progress_delta: dict | None) -> list[str]:
    if not isinstance(progress_delta, dict):
        return []
    score_delta = progress_delta.get("score_delta") if isinstance(progress_delta.get("score_delta"), dict) else {}
    lines: list[str] = []
    if progress_delta.get("previous_session_id"):
        lines.append(
            f"Verglichen mit deinem letzten gespeicherten Versuch ({progress_delta['previous_session_id']})."
        )
    final_delta = score_delta.get("final")
    if isinstance(final_delta, (int, float)) and final_delta != 0:
        lines.append(f"Gesamtwert: {final_delta:+.2f}.")
    overall_delta = score_delta.get("overall")
    if isinstance(overall_delta, (int, float)) and overall_delta != 0:
        lines.append(f"Gesamteindruck: {overall_delta:+.2f}.")
    wpm_delta = score_delta.get("wpm")
    if isinstance(wpm_delta, (int, float)) and wpm_delta != 0:
        lines.append(f"Sprechtempo: {wpm_delta:+.2f} WPM.")
    new_priorities = [item for item in progress_delta.get("new_priorities", []) if item]
    if new_priorities:
        lines.append("Neue Schwerpunkte: " + ", ".join(new_priorities) + ".")
    repeating_grammar = [item for item in progress_delta.get("repeating_grammar_categories", []) if item]
    if repeating_grammar:
        lines.append(
            "Wiederkehrende Grammatik: "
            + ", ".join(format_issue_label(item) for item in repeating_grammar)
            + "."
        )
    repeating_coherence = [item for item in progress_delta.get("repeating_coherence_categories", []) if item]
    if repeating_coherence:
        lines.append(
            "Wiederkehrende Strukturprobleme: "
            + ", ".join(format_issue_label(item) for item in repeating_coherence)
            + "."
        )
    return lines


def render_assessment_feedback(payload: dict, *, key_prefix: str, ui_locale: str) -> None:
    summary = build_result_summary(payload, ui_locale=ui_locale)
    status_text = summary["status_title"]
    if summary["failed_gates"]:
        status_text += " – offen: " + ", ".join(summary["failed_gates"])
    if summary["status_level"] == "success":
        st.success(status_text)
    elif summary["status_level"] == "warning":
        st.warning(status_text)
    else:
        st.info(status_text)

    st.markdown(f"### {ui_text(ui_locale, 'feedback_heading')}")
    score_cols = st.columns(2)
    score_cols[0].metric("Gesamtwert", summary["final_score"] if summary["final_score"] is not None else "–")
    score_cols[1].metric("Niveau", summary["band"] if summary["band"] is not None else "–")
    if summary["coach_summary"]:
        st.caption(summary["coach_summary"])

    gate_cols = st.columns(4)
    for idx, gate in enumerate(summary["gates"]):
        gate_cols[idx].metric(gate["label"], "OK" if gate["passed"] else "Offen")

    left, right = st.columns(2)
    with left:
        st.subheader(ui_text(ui_locale, "strengths_heading"))
        if summary["strengths"]:
            for item in summary["strengths"]:
                st.markdown(f"- {item}")
        else:
            st.caption("Hier erscheinen die Dinge, die bereits stabil wirken.")
    with right:
        st.subheader(ui_text(ui_locale, "priorities_heading"))
        if summary["priorities"]:
            for idx, item in enumerate(summary["priorities"], start=1):
                st.markdown(f"{idx}. {item}")
        else:
            st.caption("Noch keine Prioritäten vorhanden.")
        if summary["next_focus"]:
            st.markdown(f"**{ui_text(ui_locale, 'next_focus')}** {summary['next_focus']}")

    if summary["next_exercise"]:
        st.info(f"**{ui_text(ui_locale, 'next_exercise')}** {summary['next_exercise']}")

    if summary["progress_lines"]:
        st.subheader(ui_text(ui_locale, "progress_heading"))
        for line in summary["progress_lines"]:
            st.markdown(f"- {line}")

    issue_cols = st.columns(2)
    with issue_cols[0]:
        st.caption(ui_text(ui_locale, "recurring_grammar"))
        if summary["recurring_grammar"]:
            st.write(", ".join(summary["recurring_grammar"]))
        else:
            st.write("–")
    with issue_cols[1]:
        st.caption(ui_text(ui_locale, "recurring_structure"))
        if summary["recurring_coherence"]:
            st.write(", ".join(summary["recurring_coherence"]))
        else:
            st.write("–")

    with st.expander(ui_text(ui_locale, "evaluation_details"), expanded=False):
        detail_cols = st.columns(3)
        detail_cols[0].metric("Bewertungsart", summary["mode_label"])
        detail_cols[1].metric("Sprachurteil", summary["llm_score"] if summary["llm_score"] is not None else "–")
        detail_cols[2].metric(
            "Messwerte",
            summary["deterministic_score"] if summary["deterministic_score"] is not None else "–",
        )

    baseline = summary["baseline"]
    if isinstance(baseline, dict):
        st.subheader(ui_text(ui_locale, "baseline_heading"))
        st.markdown(f"**Baseline {baseline['level']}** – {baseline.get('comment', '')}")
        rows = [
            {
                "Metrik": metric,
                "Soll": entry["expected"],
                "Ist": entry["actual"],
                "OK": "✅" if entry["ok"] else "⚠️",
            }
            for metric, entry in baseline["targets"].items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")

    if summary["warnings"]:
        st.caption(ui_text(ui_locale, "warnings") + ": " + ", ".join(summary["warnings"]))

    with st.expander(ui_text(ui_locale, "raw_json")):
        st.json(payload)
    if st.button(ui_text(ui_locale, "retry_task"), key=f"{key_prefix}_retry"):
        st.session_state[f"{key_prefix}_payload"] = None
        dashboard_rerun()


def main() -> None:
    st.set_page_config(page_title="Speaking Studio Dashboard", layout="wide")
    st.title("Speaking Studio - Interactive Dashboard")
    st.warning(LEGACY_DASHBOARD_NOTICE)
    st.caption("Primary app surface: `streamlit run streamlit_app.py`")
    st.markdown(
        """
        <style>
        .practice-hero {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 18px;
            padding: 1.2rem 1.2rem 0.9rem 1.2rem;
            background: linear-gradient(135deg, rgba(227, 240, 255, 0.65), rgba(250, 245, 232, 0.75));
            margin-bottom: 1rem;
        }
        .practice-hero strong {
            font-size: 1.05rem;
        }
        .practice-subtle {
            color: rgba(49, 51, 63, 0.72);
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    initial_ui_locale = _supported_ui_locale(st.session_state.get("ui_locale_select", DEFAULT_UI_LOCALE))
    log_dir_str = st.sidebar.text_input(ui_text(initial_ui_locale, "log_dir"), str(DEFAULT_LOG_DIR))
    log_dir = Path(log_dir_str).expanduser().resolve()
    prompts = load_prompts(PROMPTS_FILE)
    theme_library = theme_library_store.load_theme_library(log_dir)
    dashboard_prefs = theme_library_store.load_dashboard_prefs(log_dir)
    if "prompt_attempt" not in st.session_state:
        st.session_state["prompt_attempt"] = None
    st.session_state.setdefault("practice_attempt", create_recording_attempt())
    st.session_state.setdefault("manual_payload", None)
    st.session_state.setdefault("manual_assessment_running", False)
    st.session_state.setdefault("manual_assessment_request", None)
    st.session_state.setdefault("prompt_payload", None)
    st.session_state.setdefault("prompt_assessment_running", False)
    st.session_state.setdefault("prompt_assessment_request", None)
    st.session_state.setdefault("practice_prompt_variant", 0)
    st.session_state.setdefault("practice_mode", PRACTICE_MODE_RECORD)
    st.session_state["practice_mode"] = normalize_practice_mode(st.session_state.get("practice_mode"))
    st.session_state.setdefault("practice_audio_input_version", 0)
    st.session_state.setdefault("prompt_audio_input_version", 0)
    st.session_state.setdefault("practice_whisper_model_select", DEFAULT_WHISPER_MODEL)
    st.session_state.setdefault("prompt_whisper_model_select", DEFAULT_WHISPER_MODEL)
    st.session_state.setdefault(
        "ui_locale_select",
        _supported_ui_locale(dashboard_prefs.get("ui_locale", initial_ui_locale)),
    )
    st.session_state.setdefault("speaker_id_input", dashboard_prefs.get("speaker_id", DEFAULT_SETTINGS.speaker_id or ""))
    st.session_state.setdefault(
        "learning_language_select",
        dashboard_prefs.get("learning_language", dashboard_prefs.get("language", DEFAULT_LEARNING_LANGUAGE)),
    )
    st.session_state.setdefault("practice_level_select", dashboard_prefs.get("cefr_level", "B1"))
    st.session_state.setdefault("practice_theme_text", dashboard_prefs.get("theme", DEFAULT_THEME))
    st.session_state.setdefault("practice_theme_choice", dashboard_prefs.get("theme_choice", ""))
    st.session_state.setdefault("practice_task_family", dashboard_prefs.get("task_family", DEFAULT_TASK_FAMILY))
    st.session_state.setdefault("target_duration_sec", float(dashboard_prefs.get("target_duration_sec", DEFAULT_TARGET_DURATION_SEC)))
    pending_practice_selection = st.session_state.pop("pending_practice_selection", None)
    if isinstance(pending_practice_selection, dict):
        if pending_practice_selection.get("learning_language"):
            st.session_state["learning_language_select"] = pending_practice_selection["learning_language"]
        if pending_practice_selection.get("cefr_level"):
            st.session_state["practice_level_select"] = pending_practice_selection["cefr_level"]
        if pending_practice_selection.get("theme_choice"):
            st.session_state["practice_theme_choice"] = pending_practice_selection["theme_choice"]
            st.session_state["practice_theme_choice_previous"] = ""
        if pending_practice_selection.get("theme"):
            st.session_state["practice_theme_text"] = pending_practice_selection["theme"]
        if pending_practice_selection.get("task_family"):
            st.session_state["practice_task_family"] = pending_practice_selection["task_family"]
    theme_library_success = st.session_state.pop("theme_library_success", None)
    ui_locale = _supported_ui_locale(st.session_state["ui_locale_select"])

    st.sidebar.selectbox(
        ui_text(ui_locale, "ui_locale"),
        options=tuple(UI_LOCALE_NAMES),
        key="ui_locale_select",
        format_func=lambda code: UI_LOCALE_NAMES.get(code, code),
    )
    ui_locale = _supported_ui_locale(st.session_state["ui_locale_select"])
    st.sidebar.markdown(
        "\n".join(
            [
                f"**{ui_text(ui_locale, 'workflow_title')}**",
                *[f"{idx}. {step}" for idx, step in enumerate(workflow_steps(ui_locale), start=1)],
            ]
        )
    )

    history_df = pd.DataFrame()
    history_records = []
    history_status: tuple[str, str] | None = None
    try:
        history_records = load_history_records(log_dir)
        history_df = load_history_df(log_dir)
    except FileNotFoundError:
        history_status = ("info", "Noch keine `history.csv` gefunden – führe zuerst eine Bewertung aus.")
    except ValueError as exc:  # pragma: no cover - defensive
        history_status = ("error", f"Konnte history.csv nicht lesen: {exc}")

    st.header(ui_text(ui_locale, "training_heading"))
    st.markdown(
        f"""
        <div class="practice-hero">
          <strong>{ui_text(ui_locale, "hero_title")}</strong>
          <div class="practice-subtle">
            {ui_text(ui_locale, "hero_subtitle")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    language_codes = theme_library_store.language_options(theme_library)
    if st.session_state["learning_language_select"] not in language_codes and language_codes:
        st.session_state["learning_language_select"] = language_codes[0]
    context_row_one = st.columns([1, 1, 1])
    with context_row_one[0]:
        speaker_id = st.text_input(
            ui_text(ui_locale, "speaker_id"),
            key="speaker_id_input",
            help="Pflichtfeld für Timeline und Vergleich gleicher Sprecher.",
        ).strip()
    with context_row_one[1]:
        selected_learning_language = st.selectbox(
            ui_text(ui_locale, "learning_language"),
            options=language_codes,
            key="learning_language_select",
            format_func=lambda code: format_language_option(theme_library, code),
        )
    with context_row_one[2]:
        selected_level = st.selectbox(
            "CEFR-Stufe",
            options=["B1", "B2", "C1"],
            key="practice_level_select",
        )

    available_themes = theme_library_store.themes_for_language_and_level(
        theme_library,
        selected_learning_language,
        selected_level,
    )
    theme_labels = [theme_option_label(entry) for entry in available_themes]
    theme_options = theme_labels + ["Eigenes Thema"]
    if st.session_state["practice_theme_choice"] not in theme_options:
        st.session_state["practice_theme_choice"] = theme_labels[0] if theme_labels else "Eigenes Thema"

    with st.expander(ui_text(ui_locale, "task_settings"), expanded=True):
        context_row_two = st.columns([1.1, 1.4])
        with context_row_two[0]:
            selected_theme_label = st.selectbox(
                ui_text(ui_locale, "theme_suggestion"),
                options=theme_options,
                key="practice_theme_choice",
            )
            selected_theme_entry = next((entry for entry in available_themes if theme_option_label(entry) == selected_theme_label), None)
            previous_theme_label = st.session_state.get("practice_theme_choice_previous")
            if selected_theme_entry and selected_theme_label != previous_theme_label:
                st.session_state["practice_theme_text"] = selected_theme_entry["title"]
                st.session_state["practice_task_family"] = selected_theme_entry.get("task_family", DEFAULT_TASK_FAMILY)
            st.session_state["practice_theme_choice_previous"] = selected_theme_label
            target_duration_sec = st.number_input(
                ui_text(ui_locale, "target_duration"),
                min_value=30.0,
                max_value=600.0,
                step=30.0,
                key="target_duration_sec",
            )
        with context_row_two[1]:
            theme = st.text_area(
                ui_text(ui_locale, "theme"),
                key="practice_theme_text",
                help="Pflichtfeld. Du kannst einen Vorschlag direkt anpassen oder ein eigenes Thema schreiben.",
            ).strip()

    task_family = st.session_state.get("practice_task_family", DEFAULT_TASK_FAMILY)
    st.session_state["speaker_id"] = speaker_id
    st.session_state["theme"] = theme
    st.session_state["task_family"] = task_family
    st.caption(
        ui_text(
            ui_locale,
            "language_status",
            language=format_language_option(theme_library, selected_learning_language),
        )
    )

    with st.sidebar.expander("Sprachen und Themen verwalten", expanded=False):
        theme_library_errors = st.session_state.get("theme_library_form_errors", {})
        if theme_library_success:
            st.success(theme_library_success)
        st.caption("Hier kannst du weitere Sprachen und eigene Themen dauerhaft im lokalen Dashboard speichern.")
        with st.form("theme_library_form"):
            manage_cols = st.columns([1, 1])
            with manage_cols[0]:
                known_languages = theme_library_store.language_options(theme_library)
                manage_mode = st.selectbox(
                    "Sprache wählen oder neu anlegen",
                    options=known_languages + [NEW_LANGUAGE_OPTION],
                    key="manage_language_mode",
                    format_func=lambda code: "Neue Sprache anlegen" if code == NEW_LANGUAGE_OPTION else format_language_option(theme_library, code),
                )
                if manage_mode == NEW_LANGUAGE_OPTION:
                    manage_language_code = st.text_input("Sprachcode", key="manage_language_code", placeholder="z. B. de")
                    if theme_library_errors.get("language_code"):
                        st.error(theme_library_errors["language_code"])
                    manage_language_label = st.text_input("Sprachname", key="manage_language_label", placeholder="z. B. Deutsch")
                    if theme_library_errors.get("language_label"):
                        st.error(theme_library_errors["language_label"])
                else:
                    manage_language_code = manage_mode
                    manage_language_label = theme_library_store.language_label(theme_library, manage_mode)
                    st.caption(f"Speichert unter `{manage_language_code}` · {manage_language_label}")
            with manage_cols[1]:
                new_theme_title = st.text_input("Neues Thema", key="manage_theme_title", placeholder="z. B. Ein Gespräch, das mich beeindruckt hat")
                if theme_library_errors.get("theme_title"):
                    st.error(theme_library_errors["theme_title"])
                new_theme_level = st.selectbox("CEFR-Stufe (Thema)", options=["B1", "B2", "C1"], key="manage_theme_level")
                new_theme_family = st.selectbox("Task-Family (Thema)", options=PRACTICE_TASK_FAMILIES, key="manage_theme_family")
            save_theme_clicked = st.form_submit_button("Thema speichern")
        if save_theme_clicked:
            validation_errors = validate_theme_library_submission(
                manage_mode=manage_mode,
                language_code=manage_language_code,
                language_label=manage_language_label,
                theme_title=new_theme_title,
            )
            if validation_errors:
                st.session_state["theme_library_form_errors"] = validation_errors
                dashboard_rerun()
            try:
                normalized_code = manage_language_code.strip().lower()
                normalized_label = manage_language_label.strip()
                normalized_title = new_theme_title.strip()
                updated_library = theme_library_store.add_theme(
                    theme_library,
                    language_code=normalized_code,
                    language_label=normalized_label,
                    title=normalized_title,
                    level=new_theme_level,
                    task_family=new_theme_family,
                )
                theme_library_store.save_theme_library(log_dir, updated_library)
                st.session_state["theme_library_form_errors"] = {}
                st.session_state["theme_library_success"] = f"Thema '{normalized_title}' wurde gespeichert."
                st.session_state["pending_practice_selection"] = {
                    "learning_language": normalized_code,
                    "cefr_level": new_theme_level,
                    "theme_choice": theme_option_label({"title": normalized_title, "level": new_theme_level}),
                    "theme": normalized_title,
                    "task_family": new_theme_family,
                }
                dashboard_rerun()
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Thema konnte nicht gespeichert werden: {exc}")

    try:
        theme_library_store.save_dashboard_prefs(
            log_dir,
            {
                "ui_locale": ui_locale,
                "speaker_id": speaker_id,
                "learning_language": selected_learning_language,
                "language": selected_learning_language,
                "cefr_level": selected_level,
                "theme": theme,
                "theme_choice": st.session_state.get("practice_theme_choice", ""),
                "task_family": task_family,
                "target_duration_sec": float(target_duration_sec),
            },
        )
    except Exception:
        pass
    control_cols = st.columns([1, 1])
    with control_cols[1]:
        if st.button("Neue Aufgabenfassung", key="rotate_practice_prompt"):
            st.session_state["practice_prompt_variant"] += 1
            dashboard_rerun()
    practice_mode = normalize_practice_mode(st.session_state.get("practice_mode", PRACTICE_MODE_RECORD))
    with control_cols[0]:
        if practice_mode == PRACTICE_MODE_RECORD:
            st.caption("Primärer Weg: direkt sprechen. Upload und lokaler Dateipfad sind nur Ausweichwege.")
        else:
            st.warning("Du nutzt gerade eine vorhandene Aufnahme. Für echtes Sprechtraining ist die Browseraufnahme der bevorzugte Weg.")
            if st.button("Zur Direktaufnahme zurückkehren", key="switch_back_to_capture"):
                st.session_state["practice_mode"] = PRACTICE_MODE_RECORD
                dashboard_rerun()

    practice_brief = generate_practice_brief(
        task_family=task_family,
        theme=theme,
        target_duration_sec=target_duration_sec,
        language_code=selected_learning_language,
        variant_index=st.session_state.get("practice_prompt_variant", 0),
    )
    render_practice_brief(practice_brief)
    practice_attempt = st.session_state.get("practice_attempt") or create_recording_attempt()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded = None
        render_step_strip(
            2
            if practice_mode != PRACTICE_MODE_RECORD
            else current_practice_step(
                practice_attempt,
                evaluation_running=st.session_state.get("manual_assessment_running", False),
            ),
            ui_locale=ui_locale,
        )
        st.markdown("### Jetzt sprechen")
        st.caption("Die Aufnahme wird nach dem Stoppen automatisch gespeichert. Erst danach ist die Auswertung aktiv.")
        if practice_mode != PRACTICE_MODE_RECORD:
            with st.expander("Alternative: vorhandene Aufnahme nutzen", expanded=True):
                st.session_state["practice_mode"] = PRACTICE_MODE_UPLOAD
                practice_mode = PRACTICE_MODE_UPLOAD
                uploaded = st.file_uploader(
                    "Audio-Datei hinzufügen",
                    type=["wav", "mp3", "m4a", "flac", "ogg"],
                    key="practice_upload_alt",
                )
        else:
            with st.expander("Stattdessen eine vorhandene Aufnahme nutzen", expanded=False):
                if st.button("Alternative aktivieren", key="activate_alternative_mode"):
                    st.session_state["practice_mode"] = PRACTICE_MODE_UPLOAD
                    dashboard_rerun()
        if practice_mode == PRACTICE_MODE_RECORD:
            if HAS_NATIVE_AUDIO_INPUT:
                st.caption("Direktaufnahme aktiv. Dein Browser zeigt Start, Stop und Laufzeit direkt im Recorder.")
                widget_key = f"practice_audio_input_{st.session_state.get('practice_audio_input_version', 0)}"
                recorded_audio = st.audio_input("Direkt im Browser aufnehmen", key=widget_key)
                practice_attempt = persist_audio_input(
                    recorded_audio,
                    session_key="practice_attempt",
                    target_dir=log_dir / "recordings",
                    prefix="practice",
                )
                st.session_state["practice_attempt"] = practice_attempt
                render_native_recorder_status(
                    practice_attempt,
                    title="Status deiner Aufnahme",
                    target_duration_sec=float(target_duration_sec),
                    evaluate_label="Aufnahme auswerten",
                )
                if practice_attempt.get("saved_path"):
                    recorder_actions = st.columns(2)
                    if recorder_actions[0].button("Neue Aufnahme starten", key="reset_practice_recording_native"):
                        reset_audio_input_recorder(
                            session_key="practice_attempt",
                            version_key="practice_audio_input_version",
                        )
                        st.session_state["manual_payload"] = None
                        dashboard_rerun()
                    recorder_actions[1].markdown("**Nächster Schritt:** Klicke rechts auf `Aufnahme auswerten`.")
            else:
                st.warning("Dieser Streamlit-Build hat noch keinen nativen Recorder. Fallback auf die ältere WebRTC-Aufnahme.")
                render_practice_recorder_fragment(log_dir, float(target_duration_sec))
            practice_attempt = st.session_state.get("practice_attempt") or create_recording_attempt()
        elif practice_mode == PRACTICE_MODE_UPLOAD:
            st.caption("Sekundärer Weg: eine bereits vorhandene Aufnahme auswerten.")
    with col_right:
        st.markdown("### Auswertung")
        st.caption("Wenn die Aufnahme gespeichert ist, kannst du sie hier direkt bewerten.")
        with st.expander("Technik und Notizen", expanded=False):
            label = st.text_input("Label", "")
            st.markdown("#### Transkription (Whisper, lokal)")
            whisper_model, whisper_availability = render_whisper_model_controls(
                select_key="practice_whisper_model_select",
                notice_key="practice_whisper_model_notice",
                label="Whisper-Modell (lokal)",
                compact=True,
            )
            notes = st.text_area("Notiz", "", height=100)
            provider = st.selectbox(
                "Bewertungsanbieter (LLM)",
                options=["openrouter", "ollama"],
                index=0 if DEFAULT_PROVIDER == "openrouter" else 1,
            )
            llm_default = DEFAULT_LLM_MODEL if provider == DEFAULT_PROVIDER else (
                DEFAULT_SETTINGS.openrouter_rubric_model if provider == "openrouter" else DEFAULT_SETTINGS.ollama_model
            )
            llm_model = st.text_input("Bewertungsmodell (LLM)", value=llm_default)
            if not RTC_CONFIGURATION.get("iceServers"):
                st.caption("Recorder läuft lokal ohne externen STUN-Server. Für Fernzugriff kannst du `ASSESS_SPEAKING_STUN_URLS` setzen.")
        st.caption(f"Whisper: `{st.session_state.get('practice_whisper_model_select', DEFAULT_WHISPER_MODEL)}`")
        run_label = (
            "Auswertung läuft..."
            if st.session_state.get("manual_assessment_running")
            else "Aufnahme auswerten"
            if practice_mode == PRACTICE_MODE_RECORD
            else "Datei auswerten"
        )
        run_disabled = (
            (practice_mode == PRACTICE_MODE_RECORD and not bool(practice_attempt.get("saved_path")))
            or not whisper_availability.get("cached")
            or not speaker_id
            or not theme
            or st.session_state.get("manual_assessment_running", False)
        )
        run_button = st.button(run_label, type="primary", disabled=run_disabled)
        if not whisper_availability.get("cached"):
            st.caption("Die Auswertung bleibt gesperrt, bis ein lokales Whisper-Modell bereitsteht.")
        elif not speaker_id or not theme:
            st.caption("Speaker ID und Thema sind Pflichtfelder, bevor die Auswertung starten kann.")
        elif run_disabled:
            st.caption("Die Auswertung wird erst aktiv, wenn die Aufnahme beendet und erfolgreich gespeichert wurde.")

    if run_button:
        audio_path: Path | None = None
        if practice_mode == PRACTICE_MODE_RECORD:
            attempt = st.session_state.get("practice_attempt") or {}
            if not attempt.get("saved_path"):
                st.warning("Bitte beende zuerst die Aufnahme. Danach wird sie automatisch gespeichert.")
            else:
                audio_path = Path(attempt["saved_path"])
        elif uploaded:
            try:
                audio_path = store_uploaded_audio(uploaded, uploaded.name, log_dir / "uploads")
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Upload fehlgeschlagen: {exc}")
        else:
            st.warning("Bitte nimm Audio auf oder lade eine Datei hoch.")

        if audio_path and not whisper_availability.get("cached"):
            st.error("Das gewählte Whisper-Modell ist noch nicht lokal verfügbar. Lade es zuerst herunter oder wähle ein vorhandenes Modell.")
        elif audio_path and (not speaker_id or not theme):
            st.error("Speaker ID und Thema sind Pflichtfelder.")
        elif audio_path:
            st.session_state["manual_assessment_request"] = create_assessment_request(
                audio_path=audio_path,
                log_dir=log_dir,
                whisper=whisper_model,
                llm=llm_model,
                label=label,
                notes=notes,
                provider=provider,
                expected_language=selected_learning_language,
                feedback_language=ui_locale,
                speaker_id=speaker_id,
                task_family=task_family,
                theme=theme,
                target_duration_sec=target_duration_sec,
            )
            st.session_state["manual_assessment_running"] = True
            dashboard_rerun()
        else:
            st.session_state["manual_assessment_running"] = False
            st.session_state["manual_payload"] = None

    manual_request = st.session_state.get("manual_assessment_request")
    if manual_request and st.session_state.get("manual_assessment_running"):
        with st.spinner("Auswertung läuft..."):
            result = execute_assessment_request(manual_request)
        st.session_state["manual_assessment_running"] = False
        st.session_state["manual_assessment_request"] = None
        if result.returncode != 0:
            st.error("Bewertung fehlgeschlagen. Siehe Log unten.")
            st.code(result.stderr or result.stdout)
        else:
            payload = parse_cli_json(result.stdout) or load_latest_report_payload(
                Path(manual_request["log_dir"]),
                label=manual_request.get("label", ""),
            )
            if payload:
                st.session_state["manual_payload"] = payload
            st.success("Bewertung abgeschlossen – Verlauf aktualisiert.")
            rerun_history(Path(manual_request["log_dir"]))
            history_df = load_history_df(Path(manual_request["log_dir"]))
            if practice_mode == PRACTICE_MODE_RECORD:
                st.session_state["practice_attempt"] = create_recording_attempt()
        dashboard_rerun()

    if st.session_state.get("manual_payload"):
        render_assessment_feedback(st.session_state["manual_payload"], key_prefix="manual", ui_locale=ui_locale)

    with st.expander(ui_text(ui_locale, "secondary_tools"), expanded=False):
        trainer_tab, chart_tab, table_tab, detail_tab = st.tabs(["Prompt-Trainer", "Trend", "Tabelle", "Details"])

        with trainer_tab:
            existing_prompt_payload = st.session_state.get("prompt_payload")
            if isinstance(existing_prompt_payload, dict):
                st.subheader("Letztes Prompt-Ergebnis")
                render_assessment_feedback(existing_prompt_payload, key_prefix="prompt", ui_locale=ui_locale)
            if not prompts:
                st.info("Keine Übungsprompts gefunden (`prompts/prompts.json`).")
            else:
                titles = [f"{p['title']} ({p['cefr_target']})" for p in prompts]
                current_idx = st.selectbox(
                    "Prompt auswählen", options=range(len(prompts)), format_func=lambda i: titles[i]
                )
                selected_prompt = prompts[current_idx]
                st.markdown(
                    f"**Ziel CEFR:** {selected_prompt['cefr_target']} – Antwortzeit: {selected_prompt['response_seconds']} s – Wiedergaben: {selected_prompt['max_playbacks']}"
                )
                st.write(selected_prompt["prompt_text"])

                with st.expander("Advanced Prompt-Einstellungen", expanded=False):
                    prompt_provider = st.selectbox(
                        "Bewertungsanbieter (Prompt-LLM)",
                        options=["openrouter", "ollama"],
                        index=0 if provider == "openrouter" else 1,
                        key="prompt_provider",
                    )
                    prompt_whisper, prompt_whisper_availability = render_whisper_model_controls(
                        select_key="prompt_whisper_model_select",
                        notice_key="prompt_whisper_model_notice",
                        label="Whisper-Modell (Prompt, lokal)",
                    )
                    prompt_llm_default = (
                        DEFAULT_SETTINGS.openrouter_rubric_model
                        if prompt_provider == "openrouter"
                        else DEFAULT_SETTINGS.ollama_model
                    )
                    prompt_llm = st.text_input(
                        "Bewertungsmodell (Prompt-LLM)", value=prompt_llm_default, key="prompt_llm_model"
                    )
                prompt_notes = st.text_input("Notiz (optional)", key="prompt_notes")

                attempt = st.session_state.get("prompt_attempt")
                if attempt and attempt["id"] != selected_prompt["id"]:
                    st.warning("Es läuft gerade ein Versuch für einen anderen Prompt.")
                    if st.button("Aktuellen Versuch verwerfen"):
                        st.session_state["prompt_attempt"] = None
                    st.write("Wähle den ursprünglichen Prompt oder verwerfe den Versuch.")
                elif attempt is None:
                    if st.button("Übung starten", key=f"start_{selected_prompt['id']}"):
                        st.session_state["prompt_attempt"] = create_prompt_attempt(selected_prompt)
                        attempt = st.session_state["prompt_attempt"]
                else:
                    attempt = st.session_state["prompt_attempt"]

                attempt = st.session_state.get("prompt_attempt")
                if attempt and attempt["id"] == selected_prompt["id"]:
                    remaining = remaining_time(attempt)
                    attempt_is_expired = attempt_expired(attempt)
                    st.info(
                        f"Verbleibende Zeit: {max(0, int(remaining))}s von {selected_prompt['response_seconds']}s"
                    )
                    if attempt_is_expired:
                        st.error("Zeitlimit überschritten – starte die Übung neu, bevor du eine Antwort einreichst.")
                    if st.button("Übung abbrechen", key=f"cancel_{selected_prompt['id']}"):
                        st.session_state["prompt_attempt"] = None
                        dashboard_rerun()
                    if HAS_NATIVE_AUDIO_INPUT:
                        st.caption("Direktantwort aktiv. Der Browser zeigt Aufnahme und Laufzeit im Recorder.")
                        widget_key = f"prompt_audio_input_{st.session_state.get('prompt_audio_input_version', 0)}"
                        prompt_audio = st.audio_input("Antwort direkt im Browser aufnehmen", key=widget_key)
                        attempt = persist_audio_input(
                            prompt_audio,
                            session_key="prompt_attempt",
                            target_dir=log_dir / "prompt_responses",
                            prefix=f"prompt_{selected_prompt['id']}",
                        )
                        st.session_state["prompt_attempt"] = attempt
                        render_native_recorder_status(
                            attempt,
                            title="Status deiner Antwort",
                            target_duration_sec=float(selected_prompt["response_seconds"]),
                            evaluate_label="Antwort auswerten",
                        )
                    else:
                        webrtc_ctx = webrtc_streamer(
                            key=f"recorder_{selected_prompt['id']}",
                            mode=WebRtcMode.SENDONLY,
                            audio_receiver_size=256,
                            rtc_configuration=RTC_CONFIGURATION,
                            media_stream_constraints={"audio": True, "video": False},
                            translations=RECORDER_TRANSLATIONS,
                            on_change=lambda: flag_recording_requested("prompt_attempt"),
                        )

                        if attempt.get("chunks") is None:
                            attempt["chunks"] = []

                        audio_frames = []
                        if webrtc_ctx and webrtc_ctx.audio_receiver:
                            try:
                                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                            except queue.Empty:
                                audio_frames = []
                            if audio_frames:
                                for frame in audio_frames:
                                    audio = frame.to_ndarray()
                                    sample_rate = getattr(frame, "sample_rate", 48000)
                                    if audio.ndim == 1:
                                        channels = 1
                                        data = audio
                                    else:
                                        channels = audio.shape[0]
                                        data = audio.T
                                    if data.dtype != np.int16:
                                        data = np.clip(data, -1.0, 1.0)
                                        data = (data * 32767).astype(np.int16)
                                    else:
                                        data = data.astype(np.int16, copy=False)
                                    append_audio_bytes(attempt, data.tobytes(), sample_rate, channels)
                        attempt = sync_recording_state(
                            attempt,
                            webrtc_ctx,
                            component_key=f"recorder_{selected_prompt['id']}",
                            target_dir=log_dir / "prompt_responses",
                            prefix=f"prompt_{selected_prompt['id']}",
                        )
                        st.session_state["prompt_attempt"] = attempt
                        debug_snapshot = build_recorder_debug_snapshot(
                            attempt,
                            webrtc_ctx,
                            component_key=f"recorder_{selected_prompt['id']}",
                            audio_frames_count=len(audio_frames),
                        )
                        log_recorder_snapshot("prompt_attempt", debug_snapshot)
                        render_recorder_status(
                            attempt,
                            target_duration_sec=float(selected_prompt["response_seconds"]),
                            evaluation_running=st.session_state.get("prompt_assessment_running", False),
                            title="Status deiner Antwort",
                        )
                        render_recorder_debug("prompt_attempt", debug_snapshot)
                    prompt_actions = st.columns(2)
                    if prompt_actions[0].button("Antwort neu aufnehmen", key=f"reset_record_{selected_prompt['id']}"):
                        if HAS_NATIVE_AUDIO_INPUT:
                            reset_audio_input_recorder(
                                session_key="prompt_attempt",
                                version_key="prompt_audio_input_version",
                            )
                            st.session_state["prompt_attempt"] = create_prompt_attempt(selected_prompt)
                        else:
                            st.session_state["prompt_attempt"] = create_prompt_attempt(selected_prompt)
                        dashboard_rerun()
                    prompt_run_clicked = prompt_actions[1].button(
                        "Auswertung läuft..." if st.session_state.get("prompt_assessment_running") else "Antwort auswerten",
                        key=f"finalize_record_{selected_prompt['id']}",
                        disabled=(
                            not bool(attempt.get("saved_path"))
                            or not prompt_whisper_availability.get("cached")
                            or st.session_state.get("prompt_assessment_running", False)
                            or attempt_is_expired
                        ),
                    )
                    if not prompt_whisper_availability.get("cached"):
                        st.caption("Die Auswertung bleibt gesperrt, bis ein lokales Whisper-Modell für den Prompt bereitsteht.")
                    elif attempt_is_expired:
                        st.caption("Die Auswertung ist gesperrt, weil das Zeitlimit überschritten wurde.")
                    elif not attempt.get("saved_path"):
                        st.caption("Die Auswertung wird aktiv, sobald deine Antwort beendet und gespeichert wurde.")
                    elif Path(attempt["saved_path"]).exists():
                        st.audio(Path(attempt["saved_path"]).read_bytes(), format="audio/wav")

                    if prompt_run_clicked:
                        response_path = Path(attempt["saved_path"]) if attempt.get("saved_path") else None
                        if attempt_is_expired:
                            st.warning("Zeitlimit überschritten – starte die Übung neu, bevor du eine Antwort auswertest.")
                        elif response_path is None:
                            st.warning("Bitte beende zuerst die Aufnahme. Danach wird die Antwort automatisch gespeichert.")
                        elif not prompt_whisper_availability.get("cached"):
                            st.error("Das gewählte Whisper-Modell ist für den Prompt noch nicht lokal verfügbar.")
                        else:
                            st.session_state["prompt_assessment_request"] = build_prompt_assessment_request(
                                attempt=attempt,
                                prompt=selected_prompt,
                                response_path=response_path,
                                log_dir=log_dir,
                                whisper=prompt_whisper,
                                llm=prompt_llm,
                                notes=prompt_notes,
                                provider=prompt_provider,
                                speaker_id=speaker_id,
                                ui_locale=ui_locale,
                                target_cefr=attempt.get("cefr"),
                            )
                            st.session_state["prompt_assessment_running"] = True
                            st.session_state["prompt_attempt"] = None
                            dashboard_rerun()
                    plays_left = attempt.get("plays_remaining", 0)
                    if can_play_prompt(attempt):
                        if st.button(
                            f"Prompt abspielen ({plays_left} verbleibend)", key=f"play_{selected_prompt['id']}"
                        ):
                            decrement_playback(attempt)
                            attempt["last_audio"] = True
                            st.session_state["prompt_attempt"] = attempt
                    else:
                        st.caption("Maximale Anzahl an Wiedergaben erreicht.")

                    if attempt.get("last_audio"):
                        st.audio(attempt["audio"])

                    with st.expander("Stattdessen eine fertige Antwort hochladen", expanded=False):
                        response = st.file_uploader(
                            "Antwortdatei hochladen (wav/mp3/m4a)",
                            type=["wav", "mp3", "m4a", "ogg", "flac"],
                            key=f"response_{selected_prompt['id']}",
                            disabled=attempt_is_expired,
                        )
                        if attempt_is_expired:
                            st.caption("Upload ist nach Ablauf des Zeitlimits gesperrt. Starte zuerst eine neue Übung.")
                        elif response is not None:
                            if not prompt_whisper_availability.get("cached"):
                                st.error("Das gewählte Whisper-Modell ist für den Prompt noch nicht lokal verfügbar.")
                            else:
                                response_path = store_uploaded_audio(
                                    response,
                                    response.name or "response.wav",
                                    log_dir / "prompt_responses",
                                )
                                st.session_state["prompt_assessment_request"] = build_prompt_assessment_request(
                                    attempt=attempt,
                                    prompt=selected_prompt,
                                    response_path=response_path,
                                    log_dir=log_dir,
                                    whisper=prompt_whisper,
                                    llm=prompt_llm,
                                    notes=prompt_notes,
                                    provider=prompt_provider,
                                    speaker_id=speaker_id,
                                    ui_locale=ui_locale,
                                    target_cefr=attempt["cefr"],
                                )
                                st.session_state["prompt_assessment_running"] = True
                                st.session_state["prompt_attempt"] = None
                                dashboard_rerun()
        prompt_request = st.session_state.get("prompt_assessment_request")
        if prompt_request and st.session_state.get("prompt_assessment_running"):
            with st.spinner("Auswertung läuft..."):
                result = execute_assessment_request(prompt_request)
            st.session_state["prompt_assessment_running"] = False
            st.session_state["prompt_assessment_request"] = None
            if result.returncode != 0:
                st.error("Bewertung fehlgeschlagen. Siehe Log unten.")
                st.code(result.stderr or result.stdout)
            else:
                payload = parse_cli_json(result.stdout) or load_latest_report_payload(
                    Path(prompt_request["log_dir"]),
                    label=prompt_request.get("label", ""),
                )
                st.success("Bewertung abgeschlossen.")
                if payload:
                    st.session_state["prompt_payload"] = payload
                else:
                    st.code(result.stdout.strip(), language="json")
                rerun_history(Path(prompt_request["log_dir"]))
                history_df = load_history_df(Path(prompt_request["log_dir"]))
            dashboard_rerun()

        with chart_tab:
            st.header("Mein Fortschritt")
            if history_status is not None:
                level, message = history_status
                getattr(st, level, st.info)(message)
            if history_df.empty:
                st.info("Noch keine Bewertungen verfügbar.")
            else:
                history_df = history_df.sort_values("timestamp")
                history_df["date"] = history_df["timestamp"].dt.date
                metric_cols = st.columns(4)
                summary = progress_dashboard.summarise(history_records)
                metric_cols[0].metric("Versuche", summary.get("count", 0))
                metric_cols[1].metric("∅ WPM", summary.get("avg_wpm") or "–")
                metric_cols[2].metric("∅ Gesamteindruck", summary.get("avg_overall") or "–")
                metric_cols[3].metric("Bester Gesamtwert", summary.get("best_final") or "–")
                filter_cols = st.columns(2)
                speaker_options = ["Alle"] + sorted({record.speaker_id for record in history_records if record.speaker_id})
                family_options = ["Alle"] + sorted({record.task_family for record in history_records if record.task_family})
                selected_speaker = filter_cols[0].selectbox("Speaker", options=speaker_options)
                selected_family = filter_cols[1].selectbox("Aufgabentyp", options=family_options)

                filtered_records = progress_analysis.filter_records(
                    history_records,
                    speaker_id=None if selected_speaker == "Alle" else selected_speaker,
                    task_family=None if selected_family == "Alle" else selected_family,
                )
                chart_data = build_trend_chart_df(filtered_records)
                if chart_data.empty:
                    st.info("Noch keine numerischen Werte für Chart verfügbar.")
                else:
                    st.line_chart(chart_data)
                    family_summary = progress_analysis.task_family_progress(
                        filtered_records if selected_family != "Alle" else history_records,
                        speaker_id=None if selected_speaker == "Alle" else selected_speaker,
                    )
                    if family_summary:
                        st.subheader("Vergleich nach Aufgabentyp")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "task_family": row["task_family"],
                                        "count": row["count"],
                                        "avg_final": row["avg_final"],
                                        "latest_final": row["latest_final"],
                                        "grammar": progress_analysis.format_top_counts(row["grammar_counts"]),
                                        "coherence": progress_analysis.format_top_counts(row["coherence_counts"]),
                                        "latest_priorities": " | ".join(row["latest_priorities"]),
                                    }
                                    for row in family_summary
                                ]
                            ),
                            width="stretch",
                        )

                    grammar_df = build_issue_count_df(filtered_records, "grammar_error_categories")
                    coherence_df = build_issue_count_df(filtered_records, "coherence_issue_categories")
                    issue_cols = st.columns(2)
                    with issue_cols[0]:
                        st.caption("Wiederkehrende Grammatikmuster")
                        if grammar_df.empty:
                            st.info("Keine Grammatikmuster im aktuellen Filter.")
                        else:
                            st.bar_chart(grammar_df.set_index("category"))
                    with issue_cols[1]:
                        st.caption("Wiederkehrende Strukturprobleme")
                        if coherence_df.empty:
                            st.info("Keine Strukturprobleme im aktuellen Filter.")
                        else:
                            st.bar_chart(coherence_df.set_index("category"))

                    if selected_family != "Alle":
                        priority_delta = progress_analysis.latest_priorities(filtered_records)
                        st.subheader("Vergleich der letzten Schwerpunkte")
                        st.write("Neueste Schwerpunkte:", ", ".join(priority_delta["latest"]) or "–")
                        st.write("Vorherige Schwerpunkte:", ", ".join(priority_delta["previous"]) or "–")
                        st.write("Neu hinzugekommen:", ", ".join(priority_delta["new"]) or "–")
                        st.write("Erledigt/entfallen:", ", ".join(priority_delta["resolved"]) or "–")

        with table_tab:
            if history_status is not None and history_df.empty:
                level, message = history_status
                getattr(st, level, st.info)(message)
            elif history_df.empty:
                st.info("Noch keine Bewertungen verfügbar.")
            else:
                st.dataframe(history_df.drop(columns=["report_path"]), width="stretch")

        with detail_tab:
            if history_status is not None and history_df.empty:
                level, message = history_status
                getattr(st, level, st.info)(message)
            elif history_df.empty:
                st.info("Noch keine Bewertungen verfügbar.")
            else:
                labels = history_df.apply(lambda r: f"{r['timestamp'].strftime('%Y-%m-%d %H:%M')} – {r['label'] or r['audio']}", axis=1)
                selection = st.selectbox("Bewertung auswählen", options=list(labels))
                selected_idx = labels.index[labels == selection][0]
                selected = history_df.loc[selected_idx]
                st.write("**Meta**")
                st.json({
                    "timestamp": selected["timestamp"].isoformat(),
                    "speaker_id": selected.get("speaker_id"),
                    "task_family": selected.get("task_family"),
                    "audio": selected["audio"],
                    "label": selected["label"],
                    "whisper": selected["whisper"],
                    "llm": selected["llm"],
                    "wpm": selected["wpm"],
                    "overall": selected["overall"],
                    "final_score": selected.get("final_score"),
                })
                report_path = Path(selected["report_path"])
                if report_path.exists():
                    try:
                        content = json.loads(report_path.read_text(encoding="utf-8"))
                        progress_delta = (content.get("report") or {}).get("progress_delta")
                        if isinstance(progress_delta, dict):
                            st.write("**Progress Delta**")
                            st.json(progress_delta)
                        st.write("**Speicherbericht**")
                        st.json(content)
                    except Exception as exc:  # pragma: no cover - defensive
                        st.error(f"Konnte JSON nicht laden: {exc}")
                else:
                    st.warning(f"Report-Datei nicht gefunden: {report_path}")


if __name__ == "__main__":
    main()

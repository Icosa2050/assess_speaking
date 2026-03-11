#!/usr/bin/env python3
"""Interactive Streamlit dashboard for assess_speaking results."""
from __future__ import annotations

import asyncio
import argparse
import csv
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

import progress_analysis
from settings import Settings
from scripts import progress_dashboard
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

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
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_LLM_MODEL = (
    DEFAULT_SETTINGS.openrouter_rubric_model
    if DEFAULT_PROVIDER == "openrouter"
    else DEFAULT_SETTINGS.ollama_model
)
DEFAULT_TASK_FAMILY = "travel_narrative" if DEFAULT_SETTINGS.task_family == "generic" else DEFAULT_SETTINGS.task_family
DEFAULT_THEME = "Il mio ultimo viaggio all'estero"
DEFAULT_TARGET_DURATION_SEC = 180.0
PRACTICE_TASK_FAMILIES = [
    "travel_narrative",
    "personal_experience",
    "opinion_monologue",
    "picture_description",
    "free_monologue",
]
PRACTICE_MODES = ["Direkt im Browser aufnehmen", "Audiodatei hochladen", "Lokale Datei verwenden"]
PRACTICE_FLOW_STEPS = ("Aufgabe wählen", "Aufnahme", "Auswertung")
RECORDER_TRANSLATIONS = {
    "start": "Aufnahme starten",
    "stop": "Aufnahme beenden",
    "select_device": "Mikrofon wählen",
    "media_api_not_available": "Dieser Browser unterstützt die Mikrofonaufnahme nicht.",
    "device_ask_permission": "Bitte Mikrofonzugriff erlauben.",
    "device_not_available": "Kein Mikrofon gefunden.",
    "device_access_denied": "Mikrofonzugriff wurde verweigert.",
}


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
    }


def remaining_time(attempt: dict, now: float | None = None) -> float:
    now = now or time.time()
    return attempt["deadline"] - now


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
    target_dir: Path,
    prefix: str,
    connection_timeout_sec: float = 8.0,
) -> dict:
    state = getattr(webrtc_ctx, "state", None)
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


def mark_recording_connecting(attempt: dict) -> dict:
    attempt = dict(attempt)
    attempt["connection_requested_at"] = attempt.get("connection_requested_at") or time.time()
    if not attempt.get("saved_path"):
        attempt["status"] = "connecting"
    return attempt


def flag_recording_requested(session_key: str) -> None:
    attempt = st.session_state.get(session_key) or create_recording_attempt()
    st.session_state[session_key] = mark_recording_connecting(attempt)


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


def store_uploaded_audio(uploaded_file: io.BytesIO, original_name: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=target_dir) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


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


def generate_practice_brief(task_family: str, theme: str, target_duration_sec: float, variant_index: int = 0) -> dict:
    theme = (theme or DEFAULT_THEME).strip()
    templates = {
        "travel_narrative": [
            {
                "title": "Racconta il viaggio come una storia chiara",
                "prompt": f"Parla in italiano del tema '{theme}'. Porta l'ascoltatore dall'inizio alla fine senza saltare passaggi importanti.",
                "cover_points": ["Dove e con chi eri", "Che cosa è successo prima, poi e alla fine", "Che cosa ti è rimasto del viaggio"],
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
    }
    options = templates.get(task_family) or templates["free_monologue"]
    chosen = options[variant_index % len(options)]
    target_minutes = round(float(target_duration_sec) / 60.0, 1)
    return {
        "title": chosen["title"],
        "prompt": chosen["prompt"],
        "cover_points": chosen["cover_points"],
        "starter_phrases": chosen["starter_phrases"],
        "success_focus": [
            f"Punta a parlare per circa {target_minutes} minuti." if target_minutes >= 1 else f"Punta a parlare per circa {int(target_duration_sec)} secondi.",
            "Usa connettivi per legare gli eventi o le idee.",
            "Chiudi con una riflessione personale invece di fermarti bruscamente.",
        ],
    }


def render_practice_brief(brief: dict) -> None:
    card_cols = st.columns([1.4, 1, 1])
    with card_cols[0]:
        st.markdown("### Dein Sprechauftrag")
        st.markdown(f"**{brief['title']}**")
        st.write(brief["prompt"])
        st.caption("Ziel: ruhig sprechen, einen roten Faden halten und sauber abschließen.")
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


def render_step_strip(current_step: int) -> None:
    step_cols = st.columns(len(PRACTICE_FLOW_STEPS))
    for idx, label in enumerate(PRACTICE_FLOW_STEPS, start=1):
        text = f"{idx}. {label}"
        if idx < current_step:
            step_cols[idx - 1].success(text)
        elif idx == current_step:
            step_cols[idx - 1].info(text)
        else:
            step_cols[idx - 1].caption(text)


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
    recorded_sec = display_duration_sec(attempt)
    target_sec = float(target_duration_sec)
    remaining_sec = max(0.0, target_sec - recorded_sec)

    st.markdown(f"### {title}")
    metrics = st.columns(3)
    metrics[0].metric("Gesprochen", format_duration(recorded_sec))
    metrics[1].metric("Ziel", format_duration(target_sec))
    metrics[2].metric("Rest", format_duration(remaining_sec))
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


def current_practice_step(attempt: dict, *, evaluation_running: bool = False) -> int:
    if evaluation_running:
        return 3
    status = attempt.get("status")
    if status in {"recording", "connecting", "ready", "error"}:
        return 2
    return 1


@st.fragment(run_every=1)
def render_practice_recorder_fragment(log_dir: Path, target_duration_sec: float) -> None:
    practice_attempt = st.session_state.get("practice_attempt") or create_recording_attempt()
    previous_status = practice_attempt.get("status")
    previous_saved_path = practice_attempt.get("saved_path")

    webrtc_ctx = webrtc_streamer(
        key="practice_recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},
        translations=RECORDER_TRANSLATIONS,
        on_change=lambda: flag_recording_requested("practice_attempt"),
    )
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
        target_dir=log_dir / "recordings",
        prefix="practice",
    )
    st.session_state["practice_attempt"] = practice_attempt

    render_recorder_status(
        practice_attempt,
        target_duration_sec=target_duration_sec,
        evaluation_running=st.session_state.get("manual_assessment_running", False),
        title="Status deiner Aufnahme",
    )
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

    if (
        previous_status != practice_attempt.get("status")
        or previous_saved_path != practice_attempt.get("saved_path")
    ):
        st.rerun()


def build_result_summary(payload: dict) -> dict:
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
        ("Sprache", bool(checks.get("language_pass"))),
        ("Thema", bool(checks.get("topic_pass"))),
        ("Dauer", bool(checks.get("duration_pass"))),
        ("Wortmenge", bool(checks.get("min_words_pass"))),
    ]
    failed_gates = [label for label, passed in gates if not passed]
    requires_review = bool(report.get("requires_human_review"))
    if requires_review:
        status_level = "warning"
        status_title = "Manuelle Prüfung empfohlen"
    elif failed_gates:
        status_level = "info"
        status_title = "Aufgabe noch nicht stabil erfüllt"
    else:
        status_level = "success"
        status_title = "Aufgabe erfüllt"
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


def render_assessment_feedback(payload: dict, *, key_prefix: str) -> None:
    summary = build_result_summary(payload)
    status_text = summary["status_title"]
    if summary["failed_gates"]:
        status_text += " – offen: " + ", ".join(summary["failed_gates"])
    if summary["status_level"] == "success":
        st.success(status_text)
    elif summary["status_level"] == "warning":
        st.warning(status_text)
    else:
        st.info(status_text)

    st.markdown("### Deine Rückmeldung")
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
        st.subheader("Das gelingt dir schon")
        if summary["strengths"]:
            for item in summary["strengths"]:
                st.markdown(f"- {item}")
        else:
            st.caption("Hier erscheinen die Dinge, die bereits stabil wirken.")
    with right:
        st.subheader("Darauf solltest du als Nächstes achten")
        if summary["priorities"]:
            for idx, item in enumerate(summary["priorities"], start=1):
                st.markdown(f"{idx}. {item}")
        else:
            st.caption("Noch keine Prioritäten vorhanden.")
        if summary["next_focus"]:
            st.markdown(f"**Konkreter Fokus für den nächsten Versuch:** {summary['next_focus']}")

    if summary["next_exercise"]:
        st.info(f"**Nächste Übung für dich:** {summary['next_exercise']}")

    if summary["progress_lines"]:
        st.subheader("Vergleich zum letzten Versuch")
        for line in summary["progress_lines"]:
            st.markdown(f"- {line}")

    issue_cols = st.columns(2)
    with issue_cols[0]:
        st.caption("Wiederkehrende Grammatikmuster")
        if summary["recurring_grammar"]:
            st.write(", ".join(summary["recurring_grammar"]))
        else:
            st.write("–")
    with issue_cols[1]:
        st.caption("Wiederkehrende Strukturprobleme")
        if summary["recurring_coherence"]:
            st.write(", ".join(summary["recurring_coherence"]))
        else:
            st.write("–")

    with st.expander("So wurde bewertet", expanded=False):
        detail_cols = st.columns(3)
        detail_cols[0].metric("Bewertungsart", summary["mode_label"])
        detail_cols[1].metric("Sprachurteil", summary["llm_score"] if summary["llm_score"] is not None else "–")
        detail_cols[2].metric(
            "Messwerte",
            summary["deterministic_score"] if summary["deterministic_score"] is not None else "–",
        )

    baseline = summary["baseline"]
    if isinstance(baseline, dict):
        st.subheader("CEFR-Baseline")
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if summary["warnings"]:
        st.caption("Warnungen: " + ", ".join(summary["warnings"]))

    with st.expander("Rohdaten und JSON"):
        st.json(payload)
    if st.button("Gleiche Aufgabe erneut versuchen", key=f"{key_prefix}_retry"):
        st.session_state[f"{key_prefix}_payload"] = None
        dashboard_rerun()


def main() -> None:
    st.set_page_config(page_title="Assess Speaking Dashboard", layout="wide")
    st.title("Assess Speaking – Interactive Dashboard")
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

    log_dir_str = st.sidebar.text_input("Log-Verzeichnis", str(DEFAULT_LOG_DIR))
    log_dir = Path(log_dir_str).expanduser().resolve()
    prompts = load_prompts(PROMPTS_FILE)
    if "prompt_attempt" not in st.session_state:
        st.session_state["prompt_attempt"] = None
    st.session_state.setdefault("practice_attempt", create_recording_attempt())
    st.session_state.setdefault("manual_payload", None)
    st.session_state.setdefault("manual_assessment_running", False)
    st.session_state.setdefault("prompt_payload", None)
    st.session_state.setdefault("prompt_assessment_running", False)
    st.session_state.setdefault("practice_prompt_variant", 0)
    st.session_state.setdefault("practice_mode", "Direkt im Browser aufnehmen")
    st.sidebar.markdown("""
    **Workflow**
    1. Aufgabe definieren
    2. Direkt im Browser sprechen
    3. Aufnahme prüfen und auswerten
    4. Coaching lesen und direkt erneut versuchen
    """)

    history_df = pd.DataFrame()
    history_records = []
    warning_container = st.empty()
    try:
        history_records = load_history_records(log_dir)
        history_df = load_history_df(log_dir)
    except FileNotFoundError:
        warning_container.info("Noch keine `history.csv` gefunden – führe zuerst eine Bewertung aus.")
    except ValueError as exc:  # pragma: no cover - defensive
        warning_container.error(f"Konnte history.csv nicht lesen: {exc}")

    st.header("Sprechtraining")
    st.markdown(
        """
        <div class="practice-hero">
          <strong>Sprich zuerst. Aufnahme und Aufgabenfokus stehen im Mittelpunkt.</strong>
          <div class="practice-subtle">
            Wähle ein Thema, hole dir einen klaren Sprechauftrag und versuche die Aufgabe direkt noch einmal nach dem Feedback.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    context_cols = st.columns([1, 1])
    with context_cols[0]:
        speaker_id = st.text_input(
            "Speaker ID",
            value=st.session_state.get("speaker_id", DEFAULT_SETTINGS.speaker_id or ""),
            help="Für Timeline und Vergleich gleicher Sprecher.",
        )
        task_family = st.selectbox(
            "Task-Family",
            options=PRACTICE_TASK_FAMILIES,
            index=PRACTICE_TASK_FAMILIES.index(DEFAULT_TASK_FAMILY)
            if DEFAULT_TASK_FAMILY in PRACTICE_TASK_FAMILIES
            else 0,
            help="Vergleiche werden innerhalb derselben Task-Family gemacht.",
        )
    with context_cols[1]:
        theme = st.text_area(
            "Thema",
            value=st.session_state.get("theme", DEFAULT_THEME),
            help="Beispiel: Il mio ultimo viaggio all'estero",
        )
        target_duration_sec = st.number_input(
            "Zielsprechdauer (Sekunden)",
            min_value=30.0,
            max_value=600.0,
            step=30.0,
            value=float(st.session_state.get("target_duration_sec", DEFAULT_TARGET_DURATION_SEC)),
        )
    st.session_state["speaker_id"] = speaker_id
    st.session_state["theme"] = theme
    st.session_state["target_duration_sec"] = target_duration_sec
    st.caption("Sprache: Italienisch. Die Gates prüfen Sprache, Thema, Dauer und Wortmenge.")
    control_cols = st.columns([1, 1])
    with control_cols[1]:
        if st.button("Neue Aufgabenfassung", key="rotate_practice_prompt"):
            st.session_state["practice_prompt_variant"] += 1
            dashboard_rerun()
    practice_mode = st.session_state.get("practice_mode", "Direkt im Browser aufnehmen")
    with control_cols[0]:
        if practice_mode == "Direkt im Browser aufnehmen":
            st.info("Primärer Weg: direkt sprechen. Upload und lokaler Dateipfad sind nur Ausweichwege.")
        else:
            st.warning("Du nutzt gerade eine vorhandene Aufnahme. Für echtes Sprechtraining ist die Browseraufnahme der bevorzugte Weg.")
            if st.button("Zur Direktaufnahme zurückkehren", key="switch_back_to_capture"):
                st.session_state["practice_mode"] = "Direkt im Browser aufnehmen"
                dashboard_rerun()

    practice_brief = generate_practice_brief(
        task_family=task_family,
        theme=theme,
        target_duration_sec=target_duration_sec,
        variant_index=st.session_state.get("practice_prompt_variant", 0),
    )
    render_practice_brief(practice_brief)
    practice_attempt = st.session_state.get("practice_attempt") or create_recording_attempt()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded = None
        existing_path = ""
        render_step_strip(
            2
            if practice_mode != "Direkt im Browser aufnehmen"
            else current_practice_step(
                practice_attempt,
                evaluation_running=st.session_state.get("manual_assessment_running", False),
            )
        )
        st.markdown("### Jetzt sprechen")
        st.caption("Die Aufnahme wird nach dem Stoppen automatisch gespeichert. Erst danach ist die Auswertung aktiv.")
        if practice_mode != "Direkt im Browser aufnehmen":
            with st.expander("Alternative: vorhandene Aufnahme nutzen", expanded=True):
                alternative_mode = st.radio(
                    "Alternative wählen",
                    options=["Audiodatei hochladen", "Lokale Datei verwenden"],
                    index=0 if practice_mode == "Audiodatei hochladen" else 1,
                    horizontal=True,
                    key="practice_mode_radio",
                )
                st.session_state["practice_mode"] = alternative_mode
                practice_mode = alternative_mode
                if practice_mode == "Audiodatei hochladen":
                    uploaded = st.file_uploader(
                        "Audio-Datei hinzufügen",
                        type=["wav", "mp3", "m4a", "flac", "ogg"],
                    )
                else:
                    existing_path = st.text_input("Lokalen Pfad verwenden", "")
        else:
            with st.expander("Stattdessen eine vorhandene Aufnahme nutzen", expanded=False):
                alternative_mode = st.radio(
                    "Alternative wählen",
                    options=["Audiodatei hochladen", "Lokale Datei verwenden"],
                    horizontal=True,
                    key="practice_mode_radio",
                )
                if st.button("Alternative aktivieren", key="activate_alternative_mode"):
                    st.session_state["practice_mode"] = alternative_mode
                    dashboard_rerun()
        if practice_mode == "Direkt im Browser aufnehmen":
            render_practice_recorder_fragment(log_dir, float(target_duration_sec))
            practice_attempt = st.session_state.get("practice_attempt") or create_recording_attempt()
        elif practice_mode == "Audiodatei hochladen":
            st.caption("Sekundärer Weg: eine bereits vorhandene Aufnahme auswerten.")
            uploaded = uploaded or st.file_uploader("Audio-Datei hinzufügen", type=["wav", "mp3", "m4a", "flac", "ogg"])
        else:
            st.caption("Sekundärer Weg: nutze einen lokalen Dateipfad nur dann, wenn die Aufnahme schon auf diesem Rechner liegt.")
            existing_path = existing_path or st.text_input("Lokalen Pfad verwenden", "")
    with col_right:
        st.markdown("### Danach")
        label = st.text_input("Label", "")
        with st.expander("Erweiterte Optionen", expanded=False):
            notes = st.text_area("Notiz", "", height=100)
            provider = st.selectbox("LLM-Anbieter", options=["openrouter", "ollama"], index=0 if DEFAULT_PROVIDER == "openrouter" else 1)
            whisper_model = st.text_input("Whisper-Modell", value=DEFAULT_WHISPER_MODEL)
            llm_default = DEFAULT_LLM_MODEL if provider == DEFAULT_PROVIDER else (
                DEFAULT_SETTINGS.openrouter_rubric_model if provider == "openrouter" else DEFAULT_SETTINGS.ollama_model
            )
            llm_model = st.text_input("LLM-Modell", value=llm_default)
            if not RTC_CONFIGURATION.get("iceServers"):
                st.caption("Recorder läuft lokal ohne externen STUN-Server. Für Fernzugriff kannst du `ASSESS_SPEAKING_STUN_URLS` setzen.")
        run_label = "Aufnahme auswerten" if practice_mode == "Direkt im Browser aufnehmen" else "Datei auswerten"
        run_disabled = practice_mode == "Direkt im Browser aufnehmen" and not bool(practice_attempt.get("saved_path"))
        run_button = st.button(run_label, type="primary", disabled=run_disabled)
        if run_disabled:
            st.caption("Die Auswertung wird erst aktiv, wenn die Aufnahme beendet und erfolgreich gespeichert wurde.")

    if run_button:
        audio_path: Path | None = None
        if practice_mode == "Direkt im Browser aufnehmen":
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
        elif existing_path:
            potential = Path(existing_path).expanduser()
            if potential.exists():
                audio_path = potential
            else:
                st.error(f"Datei nicht gefunden: {potential}")
        else:
            st.warning("Bitte nimm Audio auf, lade eine Datei hoch oder gib einen lokalen Pfad an.")

        if audio_path:
            st.session_state["manual_assessment_running"] = True
            with st.spinner("Auswertung läuft..."):
                result = run_assessment(
                    audio_path,
                    log_dir,
                    whisper_model,
                    llm_model,
                    label,
                    notes,
                    provider=provider,
                    speaker_id=speaker_id,
                    task_family=task_family,
                    theme=theme,
                    target_duration_sec=target_duration_sec,
                )
            st.session_state["manual_assessment_running"] = False
            if result.returncode != 0:
                st.error("Bewertung fehlgeschlagen. Siehe Log unten.")
                st.code(result.stderr or result.stdout)
            else:
                payload = parse_cli_json(result.stdout) or load_latest_report_payload(log_dir, label=label)
                if payload:
                    st.session_state["manual_payload"] = payload
                st.success("Bewertung abgeschlossen – Verlauf aktualisiert.")
                rerun_history(log_dir)
                history_df = load_history_df(log_dir)
                if practice_mode == "Direkt im Browser aufnehmen":
                    st.session_state["practice_attempt"] = create_recording_attempt()
        else:
            st.session_state["manual_assessment_running"] = False
            st.session_state["manual_payload"] = None

    if st.session_state.get("manual_payload"):
        render_assessment_feedback(st.session_state["manual_payload"], key_prefix="manual")

    trainer_tab, chart_tab, table_tab, detail_tab = st.tabs(["Prompt-Trainer", "Trend", "Tabelle", "Details"])

    with trainer_tab:
        existing_prompt_payload = st.session_state.get("prompt_payload")
        if isinstance(existing_prompt_payload, dict):
            st.subheader("Letztes Prompt-Ergebnis")
            render_assessment_feedback(existing_prompt_payload, key_prefix="prompt")
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
                    "LLM-Anbieter (Prompt)",
                    options=["openrouter", "ollama"],
                    index=0 if provider == "openrouter" else 1,
                    key="prompt_provider",
                )
                prompt_whisper = st.text_input(
                    "Whisper-Modell (Prompt)", value=whisper_model, key="prompt_whisper_model"
                )
                prompt_llm_default = (
                    DEFAULT_SETTINGS.openrouter_rubric_model
                    if prompt_provider == "openrouter"
                    else DEFAULT_SETTINGS.ollama_model
                )
                prompt_llm = st.text_input(
                    "LLM-Modell (Prompt)", value=prompt_llm_default, key="prompt_llm_model"
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
                st.info(
                    f"Verbleibende Zeit: {max(0, int(remaining))}s von {selected_prompt['response_seconds']}s"
                )
                if st.button("Übung abbrechen", key=f"cancel_{selected_prompt['id']}"):
                    st.session_state["prompt_attempt"] = None
                    dashboard_rerun()

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
                    target_dir=log_dir / "prompt_responses",
                    prefix=f"prompt_{selected_prompt['id']}",
                )
                st.session_state["prompt_attempt"] = attempt
                render_recorder_status(
                    attempt,
                    target_duration_sec=float(selected_prompt["response_seconds"]),
                    evaluation_running=st.session_state.get("prompt_assessment_running", False),
                    title="Status deiner Antwort",
                )
                prompt_actions = st.columns(2)
                if prompt_actions[0].button("Antwort neu aufnehmen", key=f"reset_record_{selected_prompt['id']}"):
                    st.session_state["prompt_attempt"] = create_prompt_attempt(selected_prompt)
                    dashboard_rerun()
                prompt_run_clicked = prompt_actions[1].button(
                    "Antwort auswerten",
                    key=f"finalize_record_{selected_prompt['id']}",
                    disabled=not bool(attempt.get("saved_path")),
                )
                if not attempt.get("saved_path"):
                    st.caption("Die Auswertung wird aktiv, sobald deine Antwort beendet und gespeichert wurde.")
                elif Path(attempt["saved_path"]).exists():
                    st.audio(Path(attempt["saved_path"]).read_bytes(), format="audio/wav")

                if prompt_run_clicked:
                    response_path = Path(attempt["saved_path"]) if attempt.get("saved_path") else None
                    if response_path is None:
                        st.warning("Bitte beende zuerst die Aufnahme. Danach wird die Antwort automatisch gespeichert.")
                    else:
                        st.session_state["prompt_assessment_running"] = True
                        with st.spinner("Auswertung läuft..."):
                            result = run_assessment(
                                response_path,
                                log_dir,
                                prompt_whisper,
                                prompt_llm,
                                attempt["label"],
                                prompt_notes,
                                target_cefr=attempt.get("cefr"),
                                provider=prompt_provider,
                                speaker_id=speaker_id,
                                task_family="prompt_trainer",
                                theme=selected_prompt["title"],
                                target_duration_sec=float(selected_prompt["response_seconds"]),
                            )
                        st.session_state["prompt_assessment_running"] = False
                        if result.returncode != 0:
                            st.error("Bewertung fehlgeschlagen. Siehe Log unten.")
                            st.code(result.stderr or result.stdout)
                        else:
                            payload = parse_cli_json(result.stdout) or load_latest_report_payload(
                                log_dir,
                                label=attempt["label"],
                            )
                            st.success("Bewertung abgeschlossen.")
                            if payload:
                                st.session_state["prompt_payload"] = payload
                                render_assessment_feedback(payload, key_prefix="prompt")
                            else:
                                st.code(result.stdout.strip(), language="json")
                            rerun_history(log_dir)
                            history_df = load_history_df(log_dir)
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
                    )
                    if response is not None:
                        if time.time() > attempt["deadline"]:
                            st.error("Zeitlimit überschritten – lade die Datei nach einem neuen Versuch erneut hoch.")
                        else:
                            response_path = store_uploaded_audio(
                                response,
                                response.name or "response.wav",
                                log_dir / "prompt_responses",
                            )
                            st.session_state["prompt_assessment_running"] = True
                            with st.spinner("Auswertung läuft..."):
                                result = run_assessment(
                                    response_path,
                                    log_dir,
                                    prompt_whisper,
                                    prompt_llm,
                                    attempt["label"],
                                    prompt_notes,
                                    target_cefr=attempt["cefr"],
                                    provider=prompt_provider,
                                    speaker_id=speaker_id,
                                    task_family="prompt_trainer",
                                    theme=selected_prompt["title"],
                                    target_duration_sec=float(selected_prompt["response_seconds"]),
                                )
                            st.session_state["prompt_assessment_running"] = False
                            if result.returncode != 0:
                                st.error("Bewertung fehlgeschlagen. Siehe Log unten.")
                                st.code(result.stderr or result.stdout)
                            else:
                                payload = parse_cli_json(result.stdout) or load_latest_report_payload(
                                    log_dir,
                                    label=attempt["label"],
                                )
                                st.success("Bewertung abgeschlossen.")
                                if payload:
                                    st.session_state["prompt_payload"] = payload
                                    render_assessment_feedback(payload, key_prefix="prompt")
                                else:
                                    st.code(result.stdout.strip(), language="json")
                                rerun_history(log_dir)
                                history_df = load_history_df(log_dir)
                                st.session_state["prompt_attempt"] = None
                                dashboard_rerun()

    with chart_tab:
        st.header("Mein Fortschritt")
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
                        use_container_width=True,
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
        if history_df.empty:
            st.info("Noch keine Bewertungen verfügbar.")
        else:
            st.dataframe(history_df.drop(columns=["report_path"]), use_container_width=True)

    with detail_tab:
        if history_df.empty:
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

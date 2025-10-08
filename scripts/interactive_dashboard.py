#!/usr/bin/env python3
"""Interactive Streamlit dashboard for assess_speaking results."""
from __future__ import annotations

import argparse
import io
import json
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

from scripts import progress_dashboard
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--log-dir")
_known_args, _ = _parser.parse_known_args()

DEFAULT_LOG_DIR = Path(_known_args.log_dir).expanduser().resolve() if _known_args.log_dir else PROJECT_ROOT / "reports"
ASSESS_SCRIPT = PROJECT_ROOT / "assess_speaking.py"
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "prompts.json"
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@st.cache_data(show_spinner=False)
def load_history_df(log_dir: Path) -> pd.DataFrame:
    records = progress_dashboard.load_history(log_dir / "history.csv")
    if not records:
        return pd.DataFrame()
    data = {
        "timestamp": [r.timestamp for r in records],
        "label": [r.label for r in records],
        "audio": [r.audio for r in records],
        "whisper": [r.whisper for r in records],
        "llm": [r.llm for r in records],
        "duration_sec": [r.duration_sec for r in records],
        "wpm": [r.wpm for r in records],
        "word_count": [r.word_count for r in records],
        "overall": [r.overall for r in records],
        "report_path": [r.report_path for r in records],
    }
    return pd.DataFrame(data)


def rerun_history(log_dir: Path):
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
        "chunks": [],
        "sample_rate": None,
        "channels": None,
        "sample_width": 2,
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


def run_assessment(audio_path: Path, log_dir: Path, whisper: str, llm: str, label: str, notes: str, target_cefr: str | None = None) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ASSESS_SCRIPT),
        str(audio_path),
        "--whisper",
        whisper,
        "--llm",
        llm,
        "--log-dir",
        str(log_dir),
    ]
    if label:
        cmd.extend(["--label", label])
    if notes:
        cmd.extend(["--notes", notes])
    if target_cefr:
        cmd.extend(["--target-cefr", target_cefr])

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


def main() -> None:
    st.set_page_config(page_title="Assess Speaking Dashboard", layout="wide")
    st.title("Assess Speaking – Interactive Dashboard")

    log_dir_str = st.sidebar.text_input("Log-Verzeichnis", str(DEFAULT_LOG_DIR))
    log_dir = Path(log_dir_str).expanduser().resolve()
    prompts = load_prompts(PROMPTS_FILE)
    if "prompt_attempt" not in st.session_state:
        st.session_state["prompt_attempt"] = None
    st.sidebar.markdown("""
    **Workflow**
    1. Audio hochladen oder vorhandene Datei wählen
    2. Label/Notiz setzen, Whisper & LLM Modell bestimmen
    3. Bewertung starten → Ergebnisse landen in `history.csv`
    4. Verlauf & Analyse unten prüfen
    """)

    history_df = pd.DataFrame()
    warning_container = st.empty()
    try:
        history_df = load_history_df(log_dir)
    except FileNotFoundError:
        warning_container.info("Noch keine `history.csv` gefunden – führe zuerst eine Bewertung aus.")
    except ValueError as exc:  # pragma: no cover - defensive
        warning_container.error(f"Konnte history.csv nicht lesen: {exc}")

    st.header("Neue Bewertung starten")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded = st.file_uploader("Audio-Datei hinzufügen", type=["wav", "mp3", "m4a", "flac", "ogg"])
        label = st.text_input("Label", "")
        notes = st.text_input("Notiz", "")
    with col_right:
        whisper_model = st.text_input("Whisper-Modell", value="large-v3")
        llm_model = st.text_input("Ollama-Modell", value="llama3.1")
        existing_path = st.text_input("Oder vorhandenen Pfad nutzen", "")
        run_button = st.button("Bewertung starten", type="primary")

    if run_button:
        audio_path: Path | None = None
        if uploaded:
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
            st.warning("Bitte eine Audio-Datei hochladen oder Pfad angeben.")

        if audio_path:
            with st.spinner("Bewertung läuft..."):
                result = run_assessment(audio_path, log_dir, whisper_model, llm_model, label, notes)
            if result.returncode != 0:
                st.error("Bewertung fehlgeschlagen. Siehe Log unten.")
                st.code(result.stderr or result.stdout)
            else:
                st.success("Bewertung abgeschlossen – Verlauf aktualisiert.")
                st.code(result.stdout.strip(), language="json")
                rerun_history(log_dir)
                history_df = load_history_df(log_dir)

    st.header("Verlauf & Analyse")
    if history_df.empty:
        st.info("Noch keine Bewertungen verfügbar.")
        return

    history_df = history_df.sort_values("timestamp")
    history_df["date"] = history_df["timestamp"].dt.date

    metric_cols = st.columns(4)
    summary = progress_dashboard.summarise(progress_dashboard.load_history(log_dir / "history.csv"))
    metric_cols[0].metric("Runs", summary.get("count", 0))
    metric_cols[1].metric("∅ WPM", summary.get("avg_wpm") or "–")
    metric_cols[2].metric("∅ Overall", summary.get("avg_overall") or "–")
    metric_cols[3].metric("Best Overall", summary.get("best_overall") or "–")

    trainer_tab, chart_tab, table_tab, detail_tab = st.tabs(["Prompt-Trainer", "Trend", "Tabelle", "Details"])

    with trainer_tab:
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

            col_models = st.columns(2)
            prompt_whisper = col_models[0].text_input(
                "Whisper-Modell (Prompt)", value=whisper_model, key="prompt_whisper_model"
            )
            prompt_llm = col_models[1].text_input(
                "Ollama-Modell (Prompt)", value=llm_model, key="prompt_llm_model"
            )
            prompt_notes = st.text_input("Notiz (optional)", key="prompt_notes")

            attempt = st.session_state.get("prompt_attempt")
            if attempt and attempt["id"] != selected_prompt["id"]:
                st.warning("Es läuft gerade ein Versuch für einen anderen Prompt.")
                if st.button("Aktuellen Versuch verwerfen"):
                    st.session_state["prompt_attempt"] = None
                st.write("Wähle den ursprünglichen Prompt oder verwerfe den Versuch.")
            elif attempt is None:
                if st.button("Versuch starten", key=f"start_{selected_prompt['id']}"):
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
                if st.button("Versuch abbrechen", key=f"cancel_{selected_prompt['id']}"):
                    st.session_state["prompt_attempt"] = None
                    st.experimental_rerun()

                webrtc_ctx = webrtc_streamer(
                    key=f"recorder_{selected_prompt['id']}",
                    mode=WebRtcMode.SENDONLY,
                    audio_receiver_size=256,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={"audio": True, "video": False},
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
                        st.session_state["prompt_attempt"] = attempt

                cols = st.columns(2)
                if cols[0].button("Aufnahme zurücksetzen", key=f"reset_record_{selected_prompt['id']}"):
                    attempt["chunks"] = []
                    st.session_state["prompt_attempt"] = attempt
                if cols[1].button("Aufnahme speichern & bewerten", key=f"finalize_record_{selected_prompt['id']}"):
                    if not attempt.get("chunks"):
                        st.warning("Keine Audio-Daten aufgenommen.")
                    else:
                        response_dir = log_dir / "prompt_responses"
                        response_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{attempt['id']}_{int(time.time())}.wav"
                        response_path = response_dir / filename
                        write_attempt_audio(attempt, response_path)
                        with st.spinner("Bewertung läuft..."):
                            result = run_assessment(
                                response_path,
                                log_dir,
                                prompt_whisper,
                                prompt_llm,
                                attempt["label"],
                                prompt_notes,
                                target_cefr=attempt.get("cefr"),
                            )
                        if result.returncode != 0:
                            st.error("Bewertung fehlgeschlagen. Siehe Log unten.")
                            st.code(result.stderr or result.stdout)
                        else:
                            payload = parse_cli_json(result.stdout)
                            st.success("Bewertung abgeschlossen.")
                            if payload:
                                st.subheader("Ergebnis")
                                st.json(payload)
                                baseline = payload.get("baseline_comparison")
                                if baseline:
                                    st.markdown(
                                        f"**Baseline {baseline['level']}** – {baseline.get('comment', '')}"
                                    )
                                    rows = [
                                        {
                                            "Metrik": metric,
                                            "Soll": entry["expected"],
                                            "Ist": entry["actual"],
                                            "OK": "✅" if entry["ok"] else "⚠️",
                                        }
                                        for metric, entry in baseline["targets"].items()
                                    ]
                                    st.dataframe(pd.DataFrame(rows))
                            else:
                                st.code(result.stdout.strip(), language="json")
                            rerun_history(log_dir)
                            history_df = load_history_df(log_dir)
                        st.session_state["prompt_attempt"] = None
                        st.experimental_rerun()
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

                response = st.file_uploader(
                    "Antwort aufnehmen und hier hochladen (wav/mp3/m4a)",
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
                        with st.spinner("Bewertung läuft..."):
                            result = run_assessment(
                                response_path,
                                log_dir,
                                prompt_whisper,
                                prompt_llm,
                                attempt["label"],
                                prompt_notes,
                                target_cefr=attempt["cefr"],
                            )
                        if result.returncode != 0:
                            st.error("Bewertung fehlgeschlagen. Siehe Log unten.")
                            st.code(result.stderr or result.stdout)
                        else:
                            payload = parse_cli_json(result.stdout)
                            st.success("Bewertung abgeschlossen.")
                            if payload:
                                st.subheader("Ergebnis")
                                st.json(payload)
                                baseline = payload.get("baseline_comparison")
                                if baseline:
                                    st.markdown(
                                        f"**Baseline {baseline['level']}** – {baseline.get('comment', '')}"
                                    )
                                    rows = [
                                        {
                                            "Metrik": metric,
                                            "Soll": entry["expected"],
                                            "Ist": entry["actual"],
                                            "OK": "✅" if entry["ok"] else "⚠️",
                                        }
                                        for metric, entry in baseline["targets"].items()
                                    ]
                                    st.dataframe(pd.DataFrame(rows))
                            else:
                                st.code(result.stdout.strip(), language="json")
                            rerun_history(log_dir)
                            history_df = load_history_df(log_dir)
                            st.session_state["prompt_attempt"] = None
                            st.experimental_rerun()

    with chart_tab:
        chart_data = history_df.set_index("timestamp")[["wpm", "overall"]].dropna(how="all")
        if chart_data.empty:
            st.info("Noch keine numerischen Werte für Chart verfügbar.")
        else:
            st.line_chart(chart_data)

    with table_tab:
        st.dataframe(history_df.drop(columns=["report_path"]), use_container_width=True)

    with detail_tab:
        labels = history_df.apply(lambda r: f"{r['timestamp'].strftime('%Y-%m-%d %H:%M')} – {r['label'] or r['audio']}", axis=1)
        selection = st.selectbox("Bewertung auswählen", options=list(labels))
        selected_idx = labels.index[labels == selection][0]
        selected = history_df.loc[selected_idx]
        st.write("**Meta**")
        st.json({
            "timestamp": selected["timestamp"].isoformat(),
            "audio": selected["audio"],
            "label": selected["label"],
            "whisper": selected["whisper"],
            "llm": selected["llm"],
            "wpm": selected["wpm"],
            "overall": selected["overall"],
        })
        report_path = Path(selected["report_path"])
        if report_path.exists():
            try:
                content = json.loads(report_path.read_text(encoding="utf-8"))
                st.write("**Speicherbericht**")
                st.json(content)
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Konnte JSON nicht laden: {exc}")
        else:
            st.warning(f"Report-Datei nicht gefunden: {report_path}")


if __name__ == "__main__":
    main()

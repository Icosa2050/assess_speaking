#!/usr/bin/env python3
import argparse, csv, json, math, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from feedback import generate_feedback
from lms import (
    build_canvas_submission_data,
    build_moodle_submission_data,
    upload_to_canvas,
    upload_to_moodle,
)

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime for CLI ergonomics
    WhisperModel = None  # type: ignore

try:
    import parselmouth  # type: ignore
    from parselmouth.praat import call  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime for CLI ergonomics
    parselmouth = None  # type: ignore
    call = None  # type: ignore

FILLERS = {"eh","ehm","mmm","cioè","allora","dunque","tipo","insomma"}
COHESION = {
    "inoltre","per quanto riguarda","tuttavia","ciò nonostante","in definitiva",
    "da un lato","dall’altro","a mio avviso","tenuto conto di","a quanto pare",
    "presumibilmente","parrebbe che","pertanto","quindi","invece","comunque"
}

# Heuristic CEFR baselines derived from the Council of Europe's global scale and
# EF SET can-do descriptions, with speaking-rate expectations anchored to the
# average conversational speed (120-150 wpm) reported by VirtualSpeech.
# Sources:
# - Council of Europe, CEFR global scale (https://www.coe.int/...global-scale)
# - EF SET CEFR guides for B1/B2/C1 (https://www.efset.org/cefr/<level>/)
# - VirtualSpeech, average speaking rate (https://virtualspeech.com/...words-per-minute)
CEFR_BASELINES = {
    "B1": {
        "wpm_min": 80,
        "wpm_max": 130,
        "fillers_max": 6,
        "cohesion_min": 0,
        "complexity_min": 0,
        "notes": "Produce testo connesso su esperienze personali; ritmo ancora in sviluppo ma comprensibile."},
    "B2": {
        "wpm_min": 100,
        "wpm_max": 150,
        "fillers_max": 4,
        "cohesion_min": 1,
        "complexity_min": 1,
        "notes": "Interazione fluida e spontanea con idee articolate su temi conosciuti."},
    "C1": {
        "wpm_min": 110,
        "wpm_max": 160,
        "fillers_max": 3,
        "cohesion_min": 2,
        "complexity_min": 2,
        "notes": "Discorso ben strutturato e preciso, con uso flessibile del linguaggio."},
}

LMS_TOKEN_ENVS = {
    "canvas": "CANVAS_TOKEN",
    "moodle": "MOODLE_TOKEN",
}

def load_audio_features(wav_path: Path):
    if parselmouth is None or call is None:
        raise RuntimeError(
            "praat-parselmouth is not available. Install dependencies via `python -m pip install -r requirements.txt`."
        )
    snd = parselmouth.Sound(str(wav_path))
    intensity = snd.to_intensity()
    thr = call(intensity, "Get mean", 0, 0) - 10
    step = 0.01; t = 0.0; pauses = []; in_pause=False; p_start=0.0
    dur_total = snd.get_total_duration()
    while t < dur_total:
        val = call(intensity, "Get value at time", t, "Cubic")
        if (math.isnan(val) or val < thr):
            if not in_pause:
                in_pause=True; p_start=t
        else:
            if in_pause:
                dur = t - p_start
                if dur >= 0.3: pauses.append((p_start, t, dur))
                in_pause=False
        t += step
    if in_pause:
        dur = dur_total-p_start
        if dur >= 0.3: pauses.append((p_start, dur_total, dur))
    return {"duration_sec": dur_total, "pauses": pauses}

def transcribe(path: Path, model_size="large-v3"):
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not available. Install dependencies via `python -m pip install -r requirements.txt`."
        )
    try:
        model = WhisperModel(model_size, compute_type="int8_float32")
    except ImportError as exc:
        if "socksio" in str(exc).lower():
            raise RuntimeError(
                "SOCKS proxy detected but 'socksio' is missing. "
                "Install dependencies via `python -m pip install -r requirements.txt` "
                "or `python -m pip install socksio`."
            ) from exc
        raise
    except Exception as exc:
        if exc.__class__.__module__.startswith(("httpx", "httpcore", "huggingface_hub")):
            raise RuntimeError(
                "Whisper model download failed while initializing faster-whisper. "
                "Check proxy/network access to Hugging Face or pre-download the model."
            ) from exc
        raise
    segments, info = model.transcribe(str(path), vad_filter=True, word_timestamps=True, language="it")
    words = []; full_text=[]
    for seg in segments:
        full_text.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                token = w.word.strip().lower()
                if token: words.append({"t0": w.start, "t1": w.end, "text": token})
    return {"text":" ".join(full_text).strip(), "words":words}

def metrics_from(words, audio_feats):
    duration = audio_feats["duration_sec"]
    pause_total = sum(p[2] for p in audio_feats["pauses"])
    speaking_time = max(0.001, duration - pause_total)
    tokens = [re.sub(r"[^a-zà-ù’']", "", w["text"]) for w in words]
    tokens = [t for t in tokens if t]
    word_count = len(tokens)
    wpm = word_count / (speaking_time/60.0)
    fillers = sum(1 for t in tokens if t in FILLERS)
    text = " " + re.sub(r"\s+", " ", " ".join(t for t in tokens)) + " "
    cohesion_hits = 0
    for m in COHESION:
        parts = [re.escape(part) for part in m.split()]
        pat = r"\b" + r"\s+".join(parts) + r"\b"
        cohesion_hits += len(re.findall(pat, text, flags=re.IGNORECASE))
    rel_markers = len(re.findall(r"\bche\b|\bcui\b|\bnella quale\b|\bnei quali\b", text))
    cond_markers = len(re.findall(r"\bse\b|\bqualora\b", text))
    complexity = rel_markers + cond_markers
    return {
        "duration_sec": round(duration,2),
        "pause_count": len(audio_feats["pauses"]),
        "pause_total_sec": round(pause_total,2),
        "speaking_time_sec": round(speaking_time,2),
        "word_count": word_count,
        "wpm": round(wpm,1),
        "fillers": fillers,
        "cohesion_markers": cohesion_hits,
        "complexity_index": complexity
    }

def rubric_prompt_it(transcript, metr):
    return f"""
Sei un esaminatore CEFR per italiano. Valuta SOLO la competenza orale in base al trascritto (potrebbero esserci errori ASR).
Assegna punteggi 1–5 per: 1) Fluidità, 2) Coerenza/Cohesione, 3) Correttezza grammaticale, 4) Ampiezza lessicale.
Dai anche 2 esempi concreti da migliorare per ciascun criterio e un voto complessivo (media).

METRICHE OGGETTIVE:
- Durata: {metr['duration_sec']} s; Tempo di parola: {metr['speaking_time_sec']} s; Pausa totale: {metr['pause_total_sec']} s; Pausenanzahl: {metr['pause_count']}
- Parole: {metr['word_count']} → WPM: {metr['wpm']}
- Filler: {metr['fillers']} ; Marcatori di coesione rilevati: {metr['cohesion_markers']}
- Indice di complessità (relativi/periodi ipotetici, euristico): {metr['complexity_index']}

TRASCRITTO:
\"\"\"{transcript.strip()}\"\"\"

RISPONDI IN JSON con le chiavi:
fluency, cohesion, accuracy, range, overall, comments_fluency, comments_cohesion, comments_accuracy, comments_range, overall_comment.
"""

def call_ollama(model, prompt):
    # Query Ollama HTTP API; also handle /api/tags for model validation
    try:
        proc = subprocess.run(
            ["curl","-s","http://localhost:11434/api/generate",
             "-d", json.dumps({"model": model, "prompt": prompt, "stream": False})],
            capture_output=True, text=True, check=True)
        resp = proc.stdout
        try:
            return json.loads(resp)["response"]
        except Exception:
            return resp
    except subprocess.CalledProcessError as e:
        return json.dumps({"error":"ollama_not_running_or_model_missing","detail":e.stderr})

def list_ollama_models():
    try:
        p = subprocess.run(["curl","-s","http://localhost:11434/api/tags"], capture_output=True, text=True, check=True)
        return p.stdout
    except subprocess.CalledProcessError as e:
        return json.dumps({"error":"ollama_tags_failed","detail":e.stderr})

def selftest(model="llama3.1"):
    prompt = "Valuta brevemente (JSON) un testo fittizio: 'Oggi parlo dell'efficienza energetica nelle case.' Dai punteggi CEFR 1–5 per fluency, cohesion, accuracy, range e un commento."
    return call_ollama(model, prompt)

def extract_rubric_json(payload: str) -> Optional[dict]:
    if not payload:
        return None
    match = re.search(r"```json\s*(\{.*?\})\s*```", payload, flags=re.DOTALL)
    candidate = match.group(1) if match else None
    if not candidate:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = payload[start:end+1]
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def build_report_path(log_dir: Path, audio: Path, label: Optional[str], when: datetime) -> Path:
    timestamp = when.strftime("%Y%m%dT%H%M%S")
    slug_parts = [audio.stem.replace(" ", "_") or "audio"]
    if label:
        slug_parts.append(re.sub(r"[^a-zA-Z0-9_-]", "_", label.strip()) or "label")
    slug = "-".join(slug_parts)
    return log_dir / f"{timestamp}_{slug}.json"


def append_history(history_path: Path, row: dict):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    exists = history_path.exists()
    fieldnames = [
        "timestamp","audio","whisper","llm","label","duration_sec",
        "wpm","word_count","overall","report_path"
    ]
    with history_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def evaluate_baseline(level: Optional[str], metrics: dict) -> Optional[dict]:
    if not level:
        return None
    cfg = CEFR_BASELINES.get(level.upper())
    if not cfg:
        return None

    def within_range(value: Optional[float], low: float, high: float) -> bool:
        if value is None:
            return False
        return low <= value <= high

    targets = {
        "wpm": {
            "expected": f"{cfg['wpm_min']}–{cfg['wpm_max']}",
            "actual": metrics.get("wpm"),
            "ok": within_range(metrics.get("wpm"), cfg['wpm_min'], cfg['wpm_max']),
        },
        "fillers": {
            "expected": f"≤{cfg['fillers_max']}",
            "actual": metrics.get("fillers"),
            "ok": metrics.get("fillers", 0) <= cfg['fillers_max'],
        },
        "cohesion_markers": {
            "expected": f"≥{cfg['cohesion_min']}",
            "actual": metrics.get("cohesion_markers"),
            "ok": metrics.get("cohesion_markers", 0) >= cfg['cohesion_min'],
        },
        "complexity_index": {
            "expected": f"≥{cfg['complexity_min']}",
            "actual": metrics.get("complexity_index"),
            "ok": metrics.get("complexity_index", 0) >= cfg['complexity_min'],
        },
    }
    passed = all(item["ok"] for item in targets.values())
    return {
        "level": level.upper(),
        "passed": passed,
        "targets": targets,
        "comment": cfg["notes"],
    }

def run_assessment(
    audio: Path,
    whisper_model: str = "large-v3",
    llm_model: str = "llama3.1",
    *,
    feedback_enabled: bool = False,
    train_dir: Path = Path("training"),
    target_cefr: Optional[str] = None,
) -> dict:
    tmp_wav = audio
    created_tmp = False
    if audio.suffix.lower() not in [".wav"]:
        tmp_wav = audio.with_suffix(".tmp.wav")
        created_tmp = True
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", str(tmp_wav)],
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is required for non-WAV input. Please install it via Homebrew: `brew install ffmpeg`."
            ) from exc

    try:
        audio_feats = load_audio_features(tmp_wav)
        asr = transcribe(tmp_wav, whisper_model)
        metr = metrics_from(asr["words"], audio_feats)
        prompt = rubric_prompt_it(asr["text"], metr)
        llm_json = call_ollama(llm_model, prompt)

        out = {
            "metrics": metr,
            "transcript_full": asr["text"],
            "transcript_preview": asr["text"][:400],
            "llm_rubric": llm_json,
        }
        baseline = evaluate_baseline(target_cefr, metr) if target_cefr else None
        if baseline:
            out["baseline_comparison"] = baseline

        if feedback_enabled:
            try:
                suggestions = generate_feedback(metr, train_dir)
                if suggestions:
                    out["suggested_training"] = suggestions
            except RuntimeError as e:
                # Feedback generation failures should not block the main assessment.
                print(f"[feedback] Warning: {e}", file=sys.stderr)

        return out
    finally:
        if created_tmp and tmp_wav.name.endswith(".tmp.wav"):
            try:
                tmp_wav.unlink()
            except Exception:
                pass


def lms_config_requested(args) -> bool:
    return any(
        [
            args.lms_type,
            args.lms_url,
            args.lms_token,
            args.lms_course_id is not None,
            args.lms_assign_id is not None,
            args.lms_score is not None,
            args.lms_dry_run,
        ]
    )


def resolve_lms_token(lms_type: Optional[str], cli_token: Optional[str]):
    if cli_token:
        return cli_token, "cli"
    if not lms_type:
        return None, None
    env_name = LMS_TOKEN_ENVS.get(lms_type)
    if env_name and os.getenv(env_name):
        return os.getenv(env_name), f"env:{env_name}"
    return None, None


def validate_lms_config(args, resolved_token: Optional[str]) -> None:
    if not lms_config_requested(args):
        return
    if not args.lms_type:
        raise RuntimeError("Incomplete LMS configuration: missing --lms-type.")

    missing = []
    if not args.lms_url:
        missing.append("--lms-url")
    if not resolved_token:
        env_name = LMS_TOKEN_ENVS.get(args.lms_type, "provider token env var")
        missing.append(f"--lms-token or {env_name}")
    if args.lms_assign_id is None:
        missing.append("--lms-assign-id")
    if args.lms_type == "canvas" and args.lms_course_id is None:
        missing.append("--lms-course-id")

    if missing:
        raise RuntimeError(f"Incomplete LMS configuration: missing {', '.join(missing)}.")


def build_lms_dry_run_preview(
    args,
    *,
    token_source: str,
    attachment_path: Path,
    attachment_size_bytes: int,
    resources: list | None,
):
    score = args.lms_score or 0
    if args.lms_type == "canvas":
        submission_data = build_canvas_submission_data(score=score, resources=resources)
    else:
        submission_data = build_moodle_submission_data(score=score, resources=resources)

    preview = {
        "dry_run": True,
        "provider": args.lms_type,
        "base_url": args.lms_url,
        "assignment_id": args.lms_assign_id,
        "token_source": token_source,
        "attachment_path": str(attachment_path.resolve()),
        "attachment_size_bytes": attachment_size_bytes,
        "submission_data": submission_data,
    }
    if args.lms_type == "canvas":
        preview["course_id"] = args.lms_course_id
    return preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", type=Path, help="Pfad zu WAV/MP3/M4A/...")
    ap.add_argument("--whisper", default="large-v3", help="faster-whisper Modell")
    ap.add_argument("--llm", default="llama3.1", help="Ollama-Modell (z. B. llama3.1, llama3.2:3b, qwen2.5:14b)")
    # New optional flags for training‑material feedback
    ap.add_argument("--feedback", action="store_true", help="Generate training‑material suggestions based on metrics")
    ap.add_argument("--train-dir", type=Path, default=Path("training"), help="Directory containing manifest.json with training resources")
    # --- LMS integration flags ---------------------------------------
    ap.add_argument("--lms-type", choices=["canvas", "moodle"], help="LMS provider to upload results to")
    ap.add_argument("--lms-url", help="Base URL of the LMS API (e.g. https://canvas.example.edu)")
    ap.add_argument("--lms-token", help="Access token for LMS authentication")
    ap.add_argument("--lms-course-id", type=int, help="Canvas course id (required when --lms-type=canvas)")
    ap.add_argument("--lms-assign-id", type=int, help="Assignment id to submit the report to")
    ap.add_argument("--lms-score", type=float, help="Optional score to include in the submission")
    ap.add_argument("--lms-dry-run", action="store_true", help="Show LMS submission details without uploading")
    ap.add_argument("--list-ollama", action="store_true", help="verfügbare Ollama-Modelle anzeigen")
    ap.add_argument("--selftest", action="store_true", help="Mini-Test gegen Ollama (ohne Audio) ausführen")
    ap.add_argument("--log-dir", default="reports", help="Pfad zum Speichern der Ergebnisse (Default: reports)")
    ap.add_argument("--no-log", action="store_true", help="Speichern der Ergebnisse deaktivieren")
    ap.add_argument("--label", help="Optionales Label für die Auswertung (z. B. Lerner, Aufgabe)")
    ap.add_argument("--notes", help="Freitextnotiz, wird nur im gespeicherten Bericht abgelegt")
    ap.add_argument("--target-cefr", choices=sorted(CEFR_BASELINES), help="Optionales CEFR-Ziel zur Baseline-Bewertung")
    args = ap.parse_args()

    if args.list_ollama:
        print(list_ollama_models()); return

    if args.selftest:
        print(selftest(args.llm)); return

    if not args.audio:
        print("Bitte Audio-Datei angeben oder --selftest bzw. --list-ollama nutzen.", file=sys.stderr)
        sys.exit(2)

    lms_token, lms_token_source = resolve_lms_token(args.lms_type, args.lms_token)
    validate_lms_config(args, lms_token)

    assessment = run_assessment(
        args.audio,
        args.whisper,
        args.llm,
        feedback_enabled=args.feedback,
        train_dir=args.train_dir,
        target_cefr=args.target_cefr,
    )
    metr = assessment["metrics"]
    llm_json = assessment["llm_rubric"]

    run_dt = datetime.now()
    meta = {
        "timestamp": run_dt.isoformat(timespec="seconds"),
        "audio_path": str(args.audio.resolve()),
        "whisper_model": args.whisper,
        "llm_model": args.llm,
    }
    if args.label:
        meta["label"] = args.label

    out = {
        "meta": meta,
        "metrics": metr,
        "transcript_preview": assessment["transcript_preview"],
        "llm_rubric": llm_json,
    }
    if "baseline_comparison" in assessment:
        out["baseline_comparison"] = assessment["baseline_comparison"]
    if "suggested_training" in assessment:
        out["suggested_training"] = assessment["suggested_training"]
    stdout_json = json.dumps(out, ensure_ascii=False, indent=2)
    print(stdout_json)

    # ------------------------------------------------------------------
    # Optional LMS upload – we create a small report file for the attachment.
    if lms_config_requested(args):
        attachment_path = Path("report.json")
        score = args.lms_score or 0
        resources = out.get("suggested_training")
        if args.lms_dry_run:
            preview = build_lms_dry_run_preview(
                args,
                token_source=lms_token_source or "unknown",
                attachment_path=attachment_path,
                attachment_size_bytes=len(stdout_json.encode("utf-8")),
                resources=resources,
            )
            print("[lms] Dry run:", file=sys.stderr)
            print(json.dumps(preview, ensure_ascii=False, indent=2), file=sys.stderr)
        else:
            attachment_path.write_text(stdout_json, encoding="utf-8")
            try:
                if args.lms_type == "canvas":
                    upload_to_canvas(
                        base_url=args.lms_url,
                        token=lms_token,
                        course_id=args.lms_course_id,
                        assignment_id=args.lms_assign_id,
                        score=score,
                        attachment_path=attachment_path,
                        resources=resources,
                    )
                elif args.lms_type == "moodle":
                    upload_to_moodle(
                        base_url=args.lms_url,
                        token=lms_token,
                        assignment_id=args.lms_assign_id,
                        score=score,
                        attachment_path=attachment_path,
                        resources=resources,
                    )
                print("[lms] Report uploaded successfully.")
            except RuntimeError as e:
                print(f"[lms] Failed to upload: {e}", file=sys.stderr)
            finally:
                try:
                    attachment_path.unlink()
                except Exception:
                    pass

    if not args.no_log:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = build_report_path(log_dir, args.audio, args.label, run_dt)
        saved_payload = {
            **out,
            "transcript_full": assessment["transcript_full"],
            "notes": args.notes or "",
            "report_path": str(report_path.resolve()),
        }
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(saved_payload, fh, ensure_ascii=False, indent=2)

        rubric_obj = extract_rubric_json(llm_json) if isinstance(llm_json, str) else None
        append_history(
            log_dir / "history.csv",
            {
                "timestamp": meta["timestamp"],
                "audio": args.audio.name,
                "whisper": args.whisper,
                "llm": args.llm,
                "label": args.label or "",
                "duration_sec": metr.get("duration_sec", ""),
                "wpm": metr.get("wpm", ""),
                "word_count": metr.get("word_count", ""),
                "overall": (rubric_obj or {}).get("overall", ""),
                "report_path": str(report_path.resolve()),
            },
        )
        print(f"Ergebnis gespeichert in {report_path}", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        sys.exit(1)

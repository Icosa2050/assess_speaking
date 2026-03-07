#!/usr/bin/env python3
"""CLI entrypoint and compatibility layer for speaking assessment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from asr import transcribe as _transcribe
from assessment_prompts import PROMPT_VERSION, rubric_prompt_it as _rubric_prompt_it, selftest_prompt_it
from audio_features import load_audio_features as _load_audio_features
from feedback import generate_feedback
from llm_client import LLMClientError, generate_rubric, list_ollama_models as _list_ollama_models
from lms import (
    build_canvas_submission_data,
    build_moodle_submission_data,
    upload_to_canvas,
    upload_to_moodle,
)
from metrics import metrics_from as _metrics_from
from schemas import AssessmentReport, RubricResult, SchemaValidationError
from scoring import compute_checks, deterministic_score, final_scores, rubric_score
from settings import Settings

# Heuristic CEFR baselines derived from the Council of Europe's global scale and
# EF SET can-do descriptions, with speaking-rate expectations anchored to the
# average conversational speed (120-150 wpm) reported by VirtualSpeech.
CEFR_BASELINES = {
    "B1": {
        "wpm_min": 80,
        "wpm_max": 130,
        "fillers_max": 6,
        "cohesion_min": 0,
        "complexity_min": 0,
        "notes": "Produce testo connesso su esperienze personali; ritmo ancora in sviluppo ma comprensibile.",
    },
    "B2": {
        "wpm_min": 100,
        "wpm_max": 150,
        "fillers_max": 4,
        "cohesion_min": 1,
        "complexity_min": 1,
        "notes": "Interazione fluida e spontanea con idee articolate su temi conosciuti.",
    },
    "C1": {
        "wpm_min": 110,
        "wpm_max": 160,
        "fillers_max": 3,
        "cohesion_min": 2,
        "complexity_min": 2,
        "notes": "Discorso ben strutturato e preciso, con uso flessibile del linguaggio.",
    },
}

LMS_TOKEN_ENVS = {
    "canvas": "CANVAS_TOKEN",
    "moodle": "MOODLE_TOKEN",
}


def load_audio_features(wav_path: Path, threshold_offset_db: float = -10.0) -> dict:
    return _load_audio_features(wav_path, threshold_offset_db=threshold_offset_db)


def transcribe(
    path: Path,
    model_size: str = "large-v3",
    language: str | None = None,
    compute_type: str = "default",
    fallback_compute_type: str | None = "int8",
) -> dict:
    return _transcribe(
        path,
        model_size=model_size,
        language=language,
        compute_type=compute_type,
        fallback_compute_type=fallback_compute_type,
    )


def metrics_from(words: list[dict], audio_feats: dict) -> dict:
    return _metrics_from(words, audio_feats)


def rubric_prompt_it(transcript: str, metrics: dict, theme: str = "tema libero") -> str:
    return _rubric_prompt_it(transcript, metrics, theme)


def call_ollama(model: str, prompt: str) -> str:
    try:
        proc = subprocess.run(
            [
                "curl",
                "-s",
                "http://localhost:11434/api/generate",
                "-d",
                json.dumps({"model": model, "prompt": prompt, "stream": False}),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = proc.stdout
        try:
            return json.loads(raw)["response"]
        except Exception:
            return raw
    except subprocess.CalledProcessError as exc:
        return json.dumps({"error": "ollama_not_running_or_model_missing", "detail": exc.stderr})


def list_ollama_models() -> str:
    try:
        proc = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout
    except subprocess.CalledProcessError as exc:
        return json.dumps({"error": "ollama_tags_failed", "detail": exc.stderr})


def _balanced_json_object(payload: str) -> Optional[dict]:
    if not payload:
        return None
    payload = payload.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        payload = "\n".join(lines).strip()
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(payload):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = payload[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = None
    return None


def extract_rubric_json(payload: str) -> Optional[dict]:
    return _balanced_json_object(payload)


def build_report_path(log_dir: Path, audio: Path, label: Optional[str], when: datetime) -> Path:
    timestamp = when.strftime("%Y%m%dT%H%M%S")
    slug_parts = [audio.stem.replace(" ", "_") or "audio"]
    if label:
        slug_parts.append(re.sub(r"[^a-zA-Z0-9_-]", "_", label.strip()) or "label")
    slug = "-".join(slug_parts)
    return log_dir / f"{timestamp}_{slug}.json"


def append_history(history_path: Path, row: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    exists = history_path.exists()
    fieldnames = [
        "timestamp",
        "audio",
        "whisper",
        "llm",
        "label",
        "duration_sec",
        "wpm",
        "word_count",
        "overall",
        "report_path",
    ]
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


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
            "ok": within_range(metrics.get("wpm"), cfg["wpm_min"], cfg["wpm_max"]),
        },
        "fillers": {
            "expected": f"≤{cfg['fillers_max']}",
            "actual": metrics.get("fillers"),
            "ok": metrics.get("fillers", 0) <= cfg["fillers_max"],
        },
        "cohesion_markers": {
            "expected": f"≥{cfg['cohesion_min']}",
            "actual": metrics.get("cohesion_markers"),
            "ok": metrics.get("cohesion_markers", 0) >= cfg["cohesion_min"],
        },
        "complexity_index": {
            "expected": f"≥{cfg['complexity_min']}",
            "actual": metrics.get("complexity_index"),
            "ok": metrics.get("complexity_index", 0) >= cfg["complexity_min"],
        },
    }
    passed = all(item["ok"] for item in targets.values())
    return {
        "level": level.upper(),
        "passed": passed,
        "targets": targets,
        "comment": cfg["notes"],
    }


def _infer_provider(
    provider: Optional[str],
    llm_model: Optional[str],
    llm_legacy: Optional[str],
    settings: Settings,
) -> str:
    if provider:
        return provider
    if llm_legacy:
        return "ollama"
    if llm_model:
        return "openrouter" if "/" in llm_model else "ollama"
    return settings.provider


def _resolve_model(provider: str, llm_model: Optional[str], llm_legacy: Optional[str], settings: Settings) -> str:
    if llm_model:
        return llm_model
    if llm_legacy:
        return llm_legacy
    if provider == "openrouter":
        return settings.openrouter_model
    return settings.ollama_model


def selftest(
    model: str | None = None,
    provider: str | None = None,
    timeout_sec: float | None = None,
) -> str:
    settings = Settings.from_env()
    chosen_provider = _infer_provider(provider, model, None, settings)
    chosen_model = model or _resolve_model(chosen_provider, None, None, settings)
    prompt = selftest_prompt_it()

    if chosen_provider == "ollama":
        return call_ollama(chosen_model, prompt)

    try:
        rubric, _raw = generate_rubric(
            provider=chosen_provider,
            model=chosen_model,
            prompt=prompt,
            timeout_sec=timeout_sec or settings.llm_timeout_sec,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            max_validation_retries=1,
        )
        return json.dumps(rubric.to_dict(), ensure_ascii=False, indent=2)
    except LLMClientError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _convert_to_wav(audio_path: Path) -> Path:
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
            prefix=f"{audio_path.stem}-",
        ) as tmp_handle:
            tmp_wav = Path(tmp_handle.name)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", str(tmp_wav)],
            check=True,
            capture_output=True,
        )
        return tmp_wav
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required for non-WAV input. Please install it via Homebrew: `brew install ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        detail = stderr or str(exc)
        raise RuntimeError(f"Audio conversion failed: {detail}") from exc


def _asr_speaking_time_from_words(words: list[dict]) -> float:
    spans = [
        (float(word["t0"]), float(word["t1"]))
        for word in words
        if "t0" in word and "t1" in word
    ]
    if not spans:
        return 0.0
    starts, ends = zip(*spans)
    return max(0.0, max(ends) - min(starts))


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


def _validate_rubric_payload(payload: Optional[dict]) -> RubricResult | None:
    if payload is None:
        return None
    try:
        return RubricResult.from_dict(payload)
    except SchemaValidationError:
        return None


def run_assessment(
    audio: Path,
    whisper_model: str = "large-v3",
    llm_model: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    feedback_enabled: bool = False,
    train_dir: Path = Path("training"),
    target_cefr: Optional[str] = None,
    theme: str = "tema libero",
    target_duration_sec: float = 120.0,
    expected_language: Optional[str] = None,
    min_word_count: Optional[int] = None,
    llm_timeout_sec: Optional[float] = None,
    asr_compute_type: Optional[str] = None,
    asr_fallback_compute_type: Optional[str] = None,
    pause_threshold_offset_db: Optional[float] = None,
) -> dict:
    settings = Settings.from_env()
    chosen_provider = _infer_provider(provider, llm_model, None, settings)
    chosen_model = _resolve_model(chosen_provider, llm_model, None, settings)
    chosen_language = expected_language or settings.expected_language
    chosen_min_words = min_word_count if min_word_count is not None else settings.min_word_count
    chosen_llm_timeout = llm_timeout_sec if llm_timeout_sec is not None else settings.llm_timeout_sec
    chosen_asr_compute_type = asr_compute_type or settings.asr_compute_type
    chosen_asr_fallback = (
        settings.asr_fallback_compute_type
        if asr_fallback_compute_type is None
        else asr_fallback_compute_type
    )
    chosen_pause_threshold = (
        settings.pause_threshold_offset_db
        if pause_threshold_offset_db is None
        else pause_threshold_offset_db
    )

    tmp_wav = audio
    created_tmp = False
    if audio.suffix.lower() != ".wav":
        tmp_wav = _convert_to_wav(audio)
        created_tmp = True

    try:
        timings_ms: dict[str, float] = {}
        warnings: list[str] = []
        errors: list[str] = []
        rubric_obj: RubricResult | None = None
        llm_raw = ""

        stage_start = time.perf_counter()
        audio_feats = load_audio_features(tmp_wav, threshold_offset_db=chosen_pause_threshold)
        timings_ms["audio_features"] = _elapsed_ms(stage_start)

        stage_start = time.perf_counter()
        asr_result = transcribe(
            tmp_wav,
            whisper_model,
            language=None,
            compute_type=chosen_asr_compute_type,
            fallback_compute_type=chosen_asr_fallback,
        )
        timings_ms["asr"] = _elapsed_ms(stage_start)

        metrics = metrics_from(asr_result["words"], audio_feats)
        baseline = evaluate_baseline(target_cefr, metrics) if target_cefr else None
        transcript = asr_result["text"]
        detected_language = str(asr_result.get("detected_language") or chosen_language)
        language_probability = asr_result.get("language_probability")
        language_pass = detected_language.lower() == chosen_language.lower()
        if not language_pass:
            warnings.extend(["language_mismatch", "llm_skipped_language_mismatch"])
            errors.append(
                f"Detected language '{detected_language}' does not match expected '{chosen_language}'."
            )
            timings_ms["llm"] = 0.0
        elif metrics["word_count"] < chosen_min_words:
            warnings.append("llm_skipped_low_word_count")
            timings_ms["llm"] = 0.0
        else:
            prompt = rubric_prompt_it(transcript, metrics, theme)
            stage_start = time.perf_counter()
            try:
                if chosen_provider == "ollama":
                    llm_raw = call_ollama(chosen_model, prompt)
                    rubric_obj = _validate_rubric_payload(extract_rubric_json(llm_raw))
                    if rubric_obj is None:
                        warnings.append("llm_invalid_schema")
                        errors.append("LLM response did not match rubric schema.")
                else:
                    rubric_obj, llm_raw = generate_rubric(
                        provider=chosen_provider,
                        model=chosen_model,
                        prompt=prompt,
                        timeout_sec=chosen_llm_timeout,
                        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                        max_validation_retries=1,
                    )
            except LLMClientError as exc:
                warnings.append("llm_unavailable")
                errors.append(str(exc))
                llm_raw = json.dumps({"error": "llm_unavailable", "detail": str(exc)})
            timings_ms["llm"] = _elapsed_ms(stage_start)

        if not llm_raw and not rubric_obj:
            llm_raw = json.dumps({"error": "llm_skipped"})

        det_score = deterministic_score(metrics)
        llm_score = rubric_score(rubric_obj)
        checks = compute_checks(
            metrics=metrics,
            rubric=rubric_obj,
            target_duration_sec=target_duration_sec,
            min_word_count=chosen_min_words,
            duration_pass_ratio=settings.duration_pass_ratio,
            language_pass=language_pass,
        )
        asr_speaking_time_sec = round(_asr_speaking_time_from_words(asr_result["words"]), 2)
        speaking_time_delta_sec = round(abs(float(metrics["speaking_time_sec"]) - asr_speaking_time_sec), 2)
        asr_pause_consistent = speaking_time_delta_sec <= max(3.0, float(metrics["duration_sec"]) * 0.25)
        if not asr_pause_consistent:
            warnings.append("asr_pause_mismatch")
        checks["asr_speaking_time_sec"] = asr_speaking_time_sec
        checks["speaking_time_delta_sec"] = speaking_time_delta_sec
        checks["asr_pause_consistent"] = asr_pause_consistent
        scores = final_scores(
            deterministic=det_score,
            llm=llm_score,
            topic_pass=checks["topic_pass"],
            topic_fail_cap_score=settings.topic_fail_cap_score,
        )
        requires_human_review = llm_score is None or not language_pass

        report = {
            "timestamp_utc": AssessmentReport.now_timestamp(),
            "input": {
                "provider": chosen_provider,
                "llm_model": chosen_model,
                "whisper_model": whisper_model,
                "expected_language": chosen_language,
                "detected_language": detected_language,
                "detected_language_probability": language_probability,
                "theme": theme,
                "target_duration_sec": target_duration_sec,
                "prompt_version": PROMPT_VERSION,
                "asr_compute_type": chosen_asr_compute_type,
                "asr_fallback_compute_type": chosen_asr_fallback,
                "asr_compute_type_used": asr_result.get("compute_type_used", chosen_asr_compute_type),
                "asr_compute_fallback_used": bool(asr_result.get("compute_fallback_used", False)),
                "pause_threshold_offset_db": chosen_pause_threshold,
            },
            "metrics": metrics,
            "checks": checks,
            "scores": scores,
            "requires_human_review": requires_human_review,
            "transcript_preview": transcript[:400],
            "warnings": warnings,
            "errors": errors,
            "rubric": rubric_obj.to_dict() if rubric_obj else None,
            "timings_ms": timings_ms,
        }

        suggestions = None
        if feedback_enabled:
            try:
                suggestions = generate_feedback(metrics, train_dir)
                if suggestions:
                    report["suggested_training"] = suggestions
            except RuntimeError as exc:
                report["warnings"].append("feedback_generation_failed")
                report["errors"].append(str(exc))

        validated_report = AssessmentReport.from_dict(report).to_dict()

        out = {
            "metrics": metrics,
            "transcript_full": transcript,
            "transcript_preview": transcript[:400],
            "llm_rubric": llm_raw,
            "report": validated_report,
        }
        if baseline:
            out["baseline_comparison"] = baseline
        if suggestions:
            out["suggested_training"] = suggestions
        return out
    finally:
        if created_tmp:
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
    if args.lms_type == "canvas":
        submission_data = build_canvas_submission_data(score=args.lms_score, resources=resources)
    else:
        submission_data = build_moodle_submission_data(score=args.lms_score, resources=resources)

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


def main() -> None:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", type=Path, help="Pfad zu WAV/MP3/M4A/...")
    ap.add_argument("--whisper", default="large-v3", help="faster-whisper Modell")
    ap.add_argument("--provider", choices=["openrouter", "ollama"], help="LLM provider")
    ap.add_argument("--llm-model", help="LLM model name for the selected provider")
    ap.add_argument("--llm", help="Legacy alias for local Ollama model selection")
    ap.add_argument("--expected-language", default=settings.expected_language, help="Erwarteter Sprachcode")
    ap.add_argument("--theme", default="tema libero", help="Thema der Sprechaufgabe")
    ap.add_argument("--target-duration-sec", type=float, default=120.0, help="Zielsprechdauer in Sekunden")
    ap.add_argument("--min-word-count", type=int, default=settings.min_word_count, help="Minimale Wortzahl für LLM-Bewertung")
    ap.add_argument("--llm-timeout", type=float, default=settings.llm_timeout_sec, help="LLM timeout in Sekunden")
    ap.add_argument("--asr-compute-type", default=settings.asr_compute_type, help="faster-whisper compute type")
    ap.add_argument("--asr-fallback-compute-type", default=settings.asr_fallback_compute_type, help="Fallback compute type")
    ap.add_argument("--pause-threshold-offset-db", type=float, default=settings.pause_threshold_offset_db, help="Pause threshold offset in dB")
    ap.add_argument("--feedback", action="store_true", help="Generate training-material suggestions based on metrics")
    ap.add_argument("--train-dir", type=Path, default=Path("training"), help="Directory containing manifest.json with training resources")
    ap.add_argument("--lms-type", choices=["canvas", "moodle"], help="LMS provider to upload results to")
    ap.add_argument("--lms-url", help="Base URL of the LMS API (e.g. https://canvas.example.edu)")
    ap.add_argument("--lms-token", help="Access token for LMS authentication")
    ap.add_argument("--lms-course-id", type=int, help="Canvas course id (required when --lms-type=canvas)")
    ap.add_argument("--lms-assign-id", type=int, help="Assignment id to submit the report to")
    ap.add_argument("--lms-score", type=float, help="Optional score to include in the submission")
    ap.add_argument("--lms-dry-run", action="store_true", help="Show LMS submission details without uploading")
    ap.add_argument("--list-ollama", action="store_true", help="verfügbare Ollama-Modelle anzeigen")
    ap.add_argument("--selftest", action="store_true", help="Mini-Test gegen das LLM (ohne Audio) ausführen")
    ap.add_argument("--log-dir", default="reports", help="Pfad zum Speichern der Ergebnisse (Default: reports)")
    ap.add_argument("--no-log", action="store_true", help="Speichern der Ergebnisse deaktivieren")
    ap.add_argument("--label", help="Optionales Label für die Auswertung (z. B. Lerner, Aufgabe)")
    ap.add_argument("--notes", help="Freitextnotiz, wird nur im gespeicherten Bericht abgelegt")
    ap.add_argument("--target-cefr", choices=sorted(CEFR_BASELINES), help="Optionales CEFR-Ziel zur Baseline-Bewertung")
    args = ap.parse_args()

    chosen_provider = _infer_provider(args.provider, args.llm_model, args.llm, settings)
    chosen_model = _resolve_model(chosen_provider, args.llm_model, args.llm, settings)

    if args.list_ollama:
        print(list_ollama_models())
        return

    if args.selftest:
        print(selftest(model=chosen_model, provider=chosen_provider, timeout_sec=args.llm_timeout))
        return

    if not args.audio:
        print("Bitte Audio-Datei angeben oder --selftest bzw. --list-ollama nutzen.", file=sys.stderr)
        sys.exit(2)

    lms_token, lms_token_source = resolve_lms_token(args.lms_type, args.lms_token)
    validate_lms_config(args, lms_token)

    assessment = run_assessment(
        args.audio,
        args.whisper,
        chosen_model,
        provider=chosen_provider,
        feedback_enabled=args.feedback,
        train_dir=args.train_dir,
        target_cefr=args.target_cefr,
        theme=args.theme,
        target_duration_sec=args.target_duration_sec,
        expected_language=args.expected_language,
        min_word_count=args.min_word_count,
        llm_timeout_sec=args.llm_timeout,
        asr_compute_type=args.asr_compute_type,
        asr_fallback_compute_type=args.asr_fallback_compute_type,
        pause_threshold_offset_db=args.pause_threshold_offset_db,
    )
    metrics = assessment["metrics"]
    llm_json = assessment["llm_rubric"]
    report = assessment["report"]

    run_dt = datetime.now()
    meta = {
        "timestamp": run_dt.isoformat(timespec="seconds"),
        "audio_path": str(args.audio.resolve()),
        "whisper_model": args.whisper,
        "llm_model": chosen_model,
        "provider": chosen_provider,
        "theme": args.theme,
        "target_duration_sec": args.target_duration_sec,
    }
    if args.label:
        meta["label"] = args.label

    out = {
        "meta": meta,
        "metrics": metrics,
        "transcript_preview": assessment["transcript_preview"],
        "llm_rubric": llm_json,
        "report": report,
    }
    if "baseline_comparison" in assessment:
        out["baseline_comparison"] = assessment["baseline_comparison"]
    if "suggested_training" in assessment:
        out["suggested_training"] = assessment["suggested_training"]

    stdout_json = json.dumps(out, ensure_ascii=False, indent=2)
    print(stdout_json)

    if lms_config_requested(args):
        attachment_path = Path("report.json")
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
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".json",
                prefix="assess-speaking-",
                delete=False,
            ) as tmp_attachment:
                tmp_attachment.write(stdout_json)
                attachment_path = Path(tmp_attachment.name)
            try:
                if args.lms_type == "canvas":
                    upload_to_canvas(
                        base_url=args.lms_url,
                        token=lms_token,
                        course_id=args.lms_course_id,
                        assignment_id=args.lms_assign_id,
                        score=args.lms_score,
                        attachment_path=attachment_path,
                        resources=resources,
                    )
                elif args.lms_type == "moodle":
                    upload_to_moodle(
                        base_url=args.lms_url,
                        token=lms_token,
                        assignment_id=args.lms_assign_id,
                        score=args.lms_score,
                        attachment_path=attachment_path,
                        resources=resources,
                    )
                print("[lms] Report uploaded successfully.", file=sys.stderr)
            except RuntimeError as exc:
                print(f"[lms] Failed to upload: {exc}", file=sys.stderr)
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
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(saved_payload, handle, ensure_ascii=False, indent=2)

        rubric_obj = report.get("rubric")
        if rubric_obj is None and isinstance(llm_json, str):
            rubric_obj = extract_rubric_json(llm_json)
        append_history(
            log_dir / "history.csv",
            {
                "timestamp": meta["timestamp"],
                "audio": args.audio.name,
                "whisper": args.whisper,
                "llm": chosen_model,
                "label": args.label or "",
                "duration_sec": metrics.get("duration_sec", ""),
                "wpm": metrics.get("wpm", ""),
                "word_count": metrics.get("word_count", ""),
                "overall": (rubric_obj or {}).get("overall", ""),
                "report_path": str(report_path.resolve()),
            },
        )
        print(f"Ergebnis gespeichert in {report_path}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, SchemaValidationError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        sys.exit(1)

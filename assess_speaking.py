#!/usr/bin/env python3
"""CLI entrypoint and compatibility layer for speaking assessment."""

from __future__ import annotations

import argparse
from collections import Counter
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
from uuid import uuid4

from assess_core.language_profiles import default_language_profile_key, resolve_language_profile
from assess_core.schemas import AssessmentReport, REPORT_SCHEMA_VERSION, RubricResult, SchemaValidationError
from assess_core.settings import Settings
from app_shell.runtime_providers import default_base_url, normalize_provider, resolved_base_url
from scripts.progress_dashboard import infer_learning_language
from assessment_runtime.asr import transcribe as _transcribe
from assessment_runtime.assessment_prompts import (
    COACHING_PROMPT_VERSION,
    PROMPT_VERSION,
    RUBRIC_PROMPT_VERSION,
    coaching_prompt,
    coaching_prompt_it,
    rubric_prompt,
    rubric_prompt_it as _rubric_prompt_it,
    selftest_prompt_it,
)
from assessment_runtime.audio_features import load_audio_features as _load_audio_features
from assessment_runtime.dimension_scoring import aggregate_dimension_scores, score_dimensions
from assessment_runtime.feedback import build_fallback_coaching, generate_feedback
from assessment_runtime.llm_client import (
    LLMClientError,
    extract_json_object as _extract_json_object,
    generate_coaching_summary,
    generate_rubric,
    list_ollama_models as _list_ollama_models,
)
from assessment_runtime.lms import (
    build_canvas_submission_data,
    build_moodle_submission_data,
    upload_to_canvas,
    upload_to_moodle,
)
from assessment_runtime.metrics import metrics_from as _metrics_from
from assessment_runtime.scoring import compute_checks, deterministic_score, final_scores, rubric_score

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

NONE_SENTINELS = {"", "none", "null"}
TRANSCRIPTION_BASIS = "automatic_asr"
TRANSCRIPTION_CAVEAT = "Assessment is based on automatic transcription and may contain ASR errors."
HISTORY_FIELDNAMES = [
    "timestamp",
    "session_id",
    "schema_version",
    "speaker_id",
    "learning_language",
    "task_family",
    "theme",
    "audio",
    "whisper",
    "llm",
    "label",
    "target_duration_sec",
    "duration_sec",
    "wpm",
    "word_count",
    "duration_pass",
    "topic_pass",
    "language_pass",
    "fluency",
    "cohesion",
    "accuracy",
    "range",
    "overall",
    "final_score",
    "band",
    "requires_human_review",
    "top_priority_1",
    "top_priority_2",
    "top_priority_3",
    "grammar_error_categories",
    "coherence_issue_categories",
    "report_path",
]


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


def metrics_from(
    words: list[dict],
    audio_feats: dict,
    *,
    language_code: str = "it",
    language_profile_key: str | None = None,
) -> dict:
    return _metrics_from(
        words,
        audio_feats,
        language_code=language_code,
        language_profile_key=language_profile_key,
    )


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


def extract_rubric_json(payload: str) -> Optional[dict]:
    try:
        return _extract_json_object(payload)
    except SchemaValidationError:
        return None


def _normalize_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value.strip().lower() in NONE_SENTINELS:
        return None
    return value


def build_report_path(log_dir: Path, audio: Path, label: Optional[str], when: datetime) -> Path:
    timestamp = when.strftime("%Y%m%dT%H%M%S")
    slug_parts = [audio.stem.replace(" ", "_") or "audio"]
    if label:
        slug_parts.append(re.sub(r"[^a-zA-Z0-9_-]", "_", label.strip()) or "label")
    slug = "-".join(slug_parts)
    return log_dir / f"{timestamp}_{slug}.json"


def append_history(history_path: Path, row: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fieldnames != HISTORY_FIELDNAMES:
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES)
                writer.writeheader()
                for existing_row in existing_rows:
                    upgraded_row = {key: existing_row.get(key, "") for key in HISTORY_FIELDNAMES}
                    if not str(upgraded_row.get("learning_language") or "").strip():
                        upgraded_row["learning_language"] = infer_learning_language(
                            str(upgraded_row.get("report_path") or "")
                        )
                    writer.writerow(upgraded_row)

    exists = history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in HISTORY_FIELDNAMES})


def append_session_jsonl(sessions_path: Path, payload: dict) -> None:
    sessions_path.parent.mkdir(parents=True, exist_ok=True)
    with sessions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _extract_issue_categories(rubric: dict | None, field: str) -> str:
    if not isinstance(rubric, dict):
        return ""
    issues = rubric.get(field)
    if not isinstance(issues, list):
        return ""
    categories = [item.get("category", "") for item in issues if isinstance(item, dict) and item.get("category")]
    return "|".join(categories)


def _parse_history_bool(value: str) -> Optional[bool]:
    lowered = str(value).strip().lower()
    if not lowered:
        return None
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _parse_history_float(value: str) -> Optional[float]:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _split_pipe_categories(value: str) -> list[str]:
    return [item.strip() for item in str(value).split("|") if item.strip()]


def build_progress_delta(history_path: Path, report: dict) -> Optional[dict]:
    speaker_id = str(report.get("input", {}).get("speaker_id") or "").strip()
    learning_language = str(report.get("input", {}).get("learning_language") or "").strip().lower()
    task_family = str(report.get("input", {}).get("task_family") or "").strip()
    if not speaker_id or not task_family or not history_path.exists():
        return None

    with history_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        prior_rows = [
            row
            for row in reader
            if row.get("speaker_id", "").strip() == speaker_id
            and row.get("task_family", "").strip() == task_family
            and (
                not learning_language
                or not row.get("learning_language", "").strip()
                or row.get("learning_language", "").strip().lower() == learning_language
            )
        ]

    if not prior_rows:
        return None

    previous = prior_rows[-1]
    previous_priorities = [
        previous.get("top_priority_1", "").strip(),
        previous.get("top_priority_2", "").strip(),
        previous.get("top_priority_3", "").strip(),
    ]
    previous_priorities = [item for item in previous_priorities if item]
    latest_priorities = [item for item in (report.get("coaching", {}) or {}).get("top_3_priorities", []) if item]

    current_grammar = _split_pipe_categories(
        _extract_issue_categories(report.get("rubric"), "recurring_grammar_errors")
    )
    current_coherence = _split_pipe_categories(
        _extract_issue_categories(report.get("rubric"), "coherence_issues")
    )

    grammar_counts = Counter()
    coherence_counts = Counter()
    for row in prior_rows:
        grammar_counts.update(_split_pipe_categories(row.get("grammar_error_categories", "")))
        coherence_counts.update(_split_pipe_categories(row.get("coherence_issue_categories", "")))

    def _gate_change(key: str) -> str:
        current_value = bool(report.get("checks", {}).get(key))
        previous_value = _parse_history_bool(previous.get(key, ""))
        if previous_value is None:
            return "unknown"
        if previous_value == current_value:
            return "unchanged"
        return "improved" if current_value else "regressed"

    current_scores = report.get("scores", {})
    current_metrics = report.get("metrics", {})
    previous_final = _parse_history_float(previous.get("final_score", ""))
    previous_overall = _parse_history_float(previous.get("overall", ""))
    previous_wpm = _parse_history_float(previous.get("wpm", ""))

    def _delta(current_value: object, previous_value: Optional[float]) -> Optional[float]:
        try:
            current_float = float(current_value)
        except (TypeError, ValueError):
            return None
        if previous_value is None:
            return None
        return round(current_float - previous_value, 2)

    return {
        "comparison_scope": {
            "speaker_id": speaker_id,
            "learning_language": learning_language,
            "task_family": task_family,
        },
        "previous_session_id": previous.get("session_id", ""),
        "previous_timestamp": previous.get("timestamp", ""),
        "same_task_family_sessions_before": len(prior_rows),
        "score_delta": {
            "final": _delta(current_scores.get("final"), previous_final),
            "overall": _delta(current_scores.get("llm"), previous_overall),
            "wpm": _delta(current_metrics.get("wpm"), previous_wpm),
        },
        "gate_delta": {
            "duration_pass": _gate_change("duration_pass"),
            "topic_pass": _gate_change("topic_pass"),
            "language_pass": _gate_change("language_pass"),
        },
        "latest_priorities": latest_priorities,
        "previous_priorities": previous_priorities,
        "new_priorities": [item for item in latest_priorities if item not in previous_priorities],
        "resolved_priorities": [item for item in previous_priorities if item not in latest_priorities],
        "repeating_grammar_categories": [item for item in current_grammar if grammar_counts[item] > 0],
        "repeating_coherence_categories": [item for item in current_coherence if coherence_counts[item] > 0],
    }


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


def _augment_scores_with_language_profile(
    scores: dict,
    *,
    metrics: dict,
    checks: dict,
    rubric: RubricResult | None,
    expected_language: str,
    language_profile_key: str | None,
    detected_language_probability: float | None,
) -> dict:
    enriched = dict(scores)
    enriched["scorer_version"] = "legacy_hybrid_v1"
    profile = resolve_language_profile(expected_language, profile_key=language_profile_key)
    if profile is None:
        return enriched
    dimensions = score_dimensions(
        metrics=metrics,
        rubric=rubric,
        checks=checks,
        profile=profile,
        detected_language_probability=detected_language_probability,
    )
    cefr_estimate = aggregate_dimension_scores(dimensions, profile=profile)
    cefr_estimate["language"] = profile.code
    enriched["language_profile_key"] = language_profile_key or default_language_profile_key(expected_language)
    enriched["language_profile_version"] = profile.scorer_version
    enriched["dimensions"] = dimensions
    enriched["cefr_estimate"] = cefr_estimate
    return enriched


def _infer_provider(
    provider: Optional[str],
    llm_model: Optional[str],
    llm_legacy: Optional[str],
    settings: Settings,
) -> str:
    if provider:
        return normalize_provider(provider)
    if llm_legacy:
        return "ollama"
    if llm_model:
        return "openrouter" if "/" in llm_model else "ollama"
    return normalize_provider(settings.provider)


def _resolve_model(provider: str, llm_model: Optional[str], llm_legacy: Optional[str], settings: Settings) -> str:
    if llm_model:
        return llm_model
    if llm_legacy:
        return llm_legacy
    if provider == "openrouter":
        return settings.openrouter_rubric_model
    return settings.ollama_model


def _resolve_llm_api_key(provider: str) -> str | None:
    if os.getenv("LLM_API_KEY"):
        return os.getenv("LLM_API_KEY")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    if provider == "ollama":
        return os.getenv("OLLAMA_API_KEY")
    return None


def _resolve_llm_base_url(provider: str, override: str | None, settings: Settings) -> str:
    return resolved_base_url(provider, override or settings.llm_base_url or default_base_url(provider))


def selftest(
    model: str | None = None,
    provider: str | None = None,
    timeout_sec: float | None = None,
    llm_base_url: str | None = None,
) -> str:
    settings = Settings.from_env()
    chosen_provider = _infer_provider(provider, model, None, settings)
    chosen_model = model or _resolve_model(chosen_provider, None, None, settings)
    chosen_base_url = _resolve_llm_base_url(chosen_provider, llm_base_url, settings)
    api_key = _resolve_llm_api_key(chosen_provider)
    prompt = selftest_prompt_it()

    if chosen_provider == "ollama" and chosen_base_url == default_base_url("ollama") and not api_key:
        return call_ollama(chosen_model, prompt)

    try:
        rubric, _raw = generate_rubric(
            provider=chosen_provider,
            model=chosen_model,
            prompt=prompt,
            timeout_sec=timeout_sec or settings.llm_timeout_sec,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=chosen_base_url,
            api_key=api_key,
            openrouter_http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
            openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE"),
            max_validation_retries=1,
        )
        return json.dumps(rubric.to_dict(), ensure_ascii=False, indent=2)
    except LLMClientError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _convert_to_wav(audio_path: Path) -> Path:
    tmp_wav: Path | None = None
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
        if tmp_wav and tmp_wav.exists():
            tmp_wav.unlink()
        raise RuntimeError(
            "ffmpeg is required for non-WAV input. Please install it via Homebrew: `brew install ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        if tmp_wav and tmp_wav.exists():
            tmp_wav.unlink()
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


def _dry_run_assessment(
    *,
    audio: Path,
    whisper_model: str,
    llm_model: str,
    provider: str,
    expected_language: str,
    language_profile_key: str | None,
    feedback_language: str,
    theme: str,
    task_family: str,
    speaker_id: Optional[str],
    target_duration_sec: float,
    target_cefr: Optional[str],
    settings: Settings,
) -> dict:
    metrics = {
        "duration_sec": 65.0,
        "pause_count": 2,
        "pause_total_sec": 5.0,
        "speaking_time_sec": 60.0,
        "word_count": 120,
        "wpm": 120.0,
        "fillers": 1,
        "cohesion_markers": 2,
        "complexity_index": 2,
    }
    transcript = "Questa e una valutazione di prova generata in modalita dry run."
    checks = compute_checks(
        metrics=metrics,
        rubric=None,
        target_duration_sec=target_duration_sec,
        min_word_count=settings.min_word_count,
        duration_pass_ratio=settings.duration_pass_ratio,
        language_pass=True,
    )
    checks["asr_speaking_time_sec"] = metrics["speaking_time_sec"]
    checks["speaking_time_delta_sec"] = 0.0
    checks["asr_pause_consistent"] = True
    scores = final_scores(
        deterministic=deterministic_score(metrics),
        llm=None,
        topic_pass=checks["topic_pass"],
        topic_fail_cap_score=settings.topic_fail_cap_score,
    )
    scores = _augment_scores_with_language_profile(
        scores,
        metrics=metrics,
        checks=checks,
        rubric=None,
        expected_language=expected_language,
        language_profile_key=language_profile_key,
        detected_language_probability=1.0,
    )
    profile = resolve_language_profile(expected_language, profile_key=language_profile_key)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "session_id": str(uuid4()),
        "timestamp_utc": AssessmentReport.now_timestamp(),
        "input": {
            "provider": provider,
            "llm_model": llm_model,
            "whisper_model": whisper_model,
            "expected_language": expected_language,
            "feedback_language": feedback_language,
            "detected_language": expected_language,
            "detected_language_probability": 1.0,
            "theme": theme,
            "task_family": task_family,
            "speaker_id": speaker_id,
            "target_duration_sec": target_duration_sec,
            "prompt_version": PROMPT_VERSION,
            "rubric_prompt_version": RUBRIC_PROMPT_VERSION,
            "coaching_prompt_version": COACHING_PROMPT_VERSION,
            "transcription_basis": TRANSCRIPTION_BASIS,
            "transcription_caveat": TRANSCRIPTION_CAVEAT,
            "dry_run": True,
            "audio_path": str(audio),
            "scoring_model_version": "legacy_hybrid_v1",
            "language_profile": profile.code if profile is not None else None,
            "language_profile_key": language_profile_key,
            "language_profile_version": profile.scorer_version if profile is not None else None,
        },
        "metrics": metrics,
        "checks": checks,
        "scores": scores,
        "requires_human_review": True,
        "transcript_preview": transcript[:400],
        "warnings": ["dry_run"],
        "errors": [],
        "rubric": None,
        "coaching": build_fallback_coaching(
            metrics=metrics,
            checks=checks,
            theme=theme,
            target_duration_sec=target_duration_sec,
            ui_locale=feedback_language,
            learning_language=expected_language,
        ),
        "timings_ms": {"audio_features": 0.0, "asr": 0.0, "llm": 0.0},
    }
    out = {
        "metrics": metrics,
        "transcript_full": transcript,
        "transcript_preview": transcript[:400],
        "llm_rubric": json.dumps({"error": "llm_skipped_dry_run"}),
        "report": AssessmentReport.from_dict(report).to_dict(),
    }
    baseline = evaluate_baseline(target_cefr, metrics) if target_cefr else None
    if baseline:
        out["baseline_comparison"] = baseline
    return out


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
    task_family: Optional[str] = None,
    speaker_id: Optional[str] = None,
    target_duration_sec: float = 120.0,
    expected_language: Optional[str] = None,
    language_profile_key: Optional[str] = None,
    feedback_language: Optional[str] = None,
    min_word_count: Optional[int] = None,
    llm_timeout_sec: Optional[float] = None,
    llm_base_url: Optional[str] = None,
    asr_compute_type: Optional[str] = None,
    asr_fallback_compute_type: Optional[str] = None,
    pause_threshold_offset_db: Optional[float] = None,
    dry_run: bool = False,
) -> dict:
    settings = Settings.from_env()
    chosen_provider = _infer_provider(provider, llm_model, None, settings)
    chosen_model = _resolve_model(chosen_provider, llm_model, None, settings)
    chosen_language = expected_language or settings.expected_language
    chosen_profile_key = (
        str(language_profile_key).strip().lower()
        if language_profile_key is not None and str(language_profile_key).strip()
        else default_language_profile_key(chosen_language)
    )
    chosen_feedback_language = feedback_language or chosen_language
    chosen_task_family = task_family or settings.task_family
    chosen_speaker_id = speaker_id or settings.speaker_id
    chosen_min_words = min_word_count if min_word_count is not None else settings.min_word_count
    chosen_llm_timeout = llm_timeout_sec if llm_timeout_sec is not None else settings.llm_timeout_sec
    chosen_llm_base_url = _resolve_llm_base_url(chosen_provider, llm_base_url, settings)
    chosen_llm_api_key = _resolve_llm_api_key(chosen_provider)
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

    if dry_run:
        return _dry_run_assessment(
            audio=audio,
            whisper_model=whisper_model,
            llm_model=chosen_model,
            provider=chosen_provider,
            expected_language=chosen_language,
            language_profile_key=chosen_profile_key,
            feedback_language=chosen_feedback_language,
            theme=theme,
            task_family=chosen_task_family,
            speaker_id=chosen_speaker_id,
            target_duration_sec=target_duration_sec,
            target_cefr=target_cefr,
            settings=settings,
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
        coaching_obj = None
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

        metrics = metrics_from(
            asr_result["words"],
            audio_feats,
            language_code=chosen_language,
            language_profile_key=chosen_profile_key,
        )
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
            prompt = rubric_prompt(
                transcript,
                metrics,
                theme,
                expected_language=chosen_language,
                feedback_language=chosen_feedback_language,
            )
            stage_start = time.perf_counter()
            try:
                if chosen_provider == "ollama" and chosen_llm_base_url == default_base_url("ollama") and not chosen_llm_api_key:
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
                        base_url=chosen_llm_base_url,
                        api_key=chosen_llm_api_key,
                        openrouter_http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
                        openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE"),
                        max_validation_retries=1,
                    )
            except LLMClientError as exc:
                warnings.append("llm_unavailable")
                errors.append(str(exc))
                llm_raw = json.dumps({"error": "llm_unavailable", "detail": str(exc)})
            timings_ms["llm"] = _elapsed_ms(stage_start)

        if not llm_raw and not rubric_obj:
            llm_raw = json.dumps({"error": "llm_skipped"})

        if rubric_obj is not None:
            coaching_prompt_text = coaching_prompt(
                metrics=metrics,
                rubric=rubric_obj.to_dict(),
                theme=theme,
                target_duration_sec=target_duration_sec,
                expected_language=chosen_language,
                feedback_language=chosen_feedback_language,
            )
            stage_start = time.perf_counter()
            try:
                coaching_obj, _coaching_raw = generate_coaching_summary(
                    provider=chosen_provider,
                    model=chosen_model,
                    prompt=coaching_prompt_text,
                    timeout_sec=chosen_llm_timeout,
                    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url=chosen_llm_base_url,
                    api_key=chosen_llm_api_key,
                    openrouter_http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
                    openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE"),
                    max_validation_retries=1,
                )
            except LLMClientError as exc:
                warnings.append("coaching_unavailable")
                errors.append(str(exc))
                coaching_obj = None
            timings_ms["coaching"] = _elapsed_ms(stage_start)
        else:
            timings_ms["coaching"] = 0.0

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
        scores = _augment_scores_with_language_profile(
            scores,
            metrics=metrics,
            checks=checks,
            rubric=rubric_obj,
            expected_language=chosen_language,
            language_profile_key=chosen_profile_key,
            detected_language_probability=language_probability if isinstance(language_probability, (int, float)) else None,
        )
        profile = resolve_language_profile(chosen_language, profile_key=chosen_profile_key)
        requires_human_review = llm_score is None or not language_pass
        if coaching_obj is None:
            coaching_obj = build_fallback_coaching(
                metrics=metrics,
                checks=checks,
                theme=theme,
                target_duration_sec=target_duration_sec,
                ui_locale=chosen_feedback_language,
                learning_language=chosen_language,
            )

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "session_id": str(uuid4()),
            "timestamp_utc": AssessmentReport.now_timestamp(),
            "input": {
                "provider": chosen_provider,
                "llm_model": chosen_model,
                "whisper_model": whisper_model,
                "expected_language": chosen_language,
                "feedback_language": chosen_feedback_language,
                "detected_language": detected_language,
                "detected_language_probability": language_probability,
                "theme": theme,
                "task_family": chosen_task_family,
                "speaker_id": chosen_speaker_id,
                "target_duration_sec": target_duration_sec,
                "prompt_version": PROMPT_VERSION,
                "rubric_prompt_version": RUBRIC_PROMPT_VERSION,
                "coaching_prompt_version": COACHING_PROMPT_VERSION,
                "transcription_basis": TRANSCRIPTION_BASIS,
                "transcription_caveat": TRANSCRIPTION_CAVEAT,
                "asr_compute_type": chosen_asr_compute_type,
                "asr_fallback_compute_type": chosen_asr_fallback,
                "asr_compute_type_used": asr_result.get("compute_type_used", chosen_asr_compute_type),
                "asr_compute_fallback_used": bool(asr_result.get("compute_fallback_used", False)),
                "pause_threshold_offset_db": chosen_pause_threshold,
                "scoring_model_version": "legacy_hybrid_v1",
                "language_profile": profile.code if profile is not None else None,
                "language_profile_key": chosen_profile_key,
                "language_profile_version": profile.scorer_version if profile is not None else None,
            },
            "metrics": metrics,
            "checks": checks,
            "scores": scores,
            "requires_human_review": requires_human_review,
            "transcript_preview": transcript[:400],
            "warnings": warnings,
            "errors": errors,
            "rubric": rubric_obj.to_dict() if rubric_obj else None,
            "coaching": coaching_obj.to_dict() if hasattr(coaching_obj, "to_dict") else coaching_obj,
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
    ap.add_argument("--provider", choices=["openrouter", "ollama", "lmstudio", "openai_compatible"], help="LLM provider")
    ap.add_argument("--llm-model", help="LLM model name for the selected provider")
    ap.add_argument("--llm-base-url", help="Override base URL for the selected provider")
    ap.add_argument("--llm", help="Legacy alias for local Ollama model selection")
    ap.add_argument("--expected-language", default=settings.expected_language, help="Erwarteter Sprachcode")
    ap.add_argument("--language-profile-key", help="Optionaler Schlüssel für ein bestimmtes Sprachprofil")
    ap.add_argument("--feedback-language", help="Sprachcode fuer Coaching- und Kommentartexte")
    ap.add_argument("--theme", default="tema libero", help="Thema der Sprechaufgabe")
    ap.add_argument("--task-family", default=settings.task_family, help="Familie der Sprechaufgabe für Verlaufsauswertungen")
    ap.add_argument("--speaker-id", default=settings.speaker_id, help="Optionale ID der sprechenden Person")
    ap.add_argument("--target-duration-sec", type=float, default=120.0, help="Zielsprechdauer in Sekunden")
    ap.add_argument("--min-word-count", type=int, default=settings.min_word_count, help="Minimale Wortzahl für LLM-Bewertung")
    ap.add_argument("--llm-timeout", type=float, default=settings.llm_timeout_sec, help="LLM timeout in Sekunden")
    ap.add_argument("--asr-compute-type", default=settings.asr_compute_type, help="faster-whisper compute type")
    ap.add_argument(
        "--asr-fallback-compute-type",
        type=_normalize_optional_string,
        default=settings.asr_fallback_compute_type,
        help="Fallback compute type",
    )
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
    ap.add_argument("--dry-run", action="store_true", help="Erzeuge einen Stub-Bericht ohne ASR/LLM (für E2E-Tests)")
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
        print(selftest(model=chosen_model, provider=chosen_provider, timeout_sec=args.llm_timeout, llm_base_url=args.llm_base_url))
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
        task_family=args.task_family,
        speaker_id=args.speaker_id,
        target_duration_sec=args.target_duration_sec,
        expected_language=args.expected_language,
        language_profile_key=args.language_profile_key,
        feedback_language=args.feedback_language,
        min_word_count=args.min_word_count,
        llm_timeout_sec=args.llm_timeout,
        llm_base_url=args.llm_base_url,
        asr_compute_type=args.asr_compute_type,
        asr_fallback_compute_type=args.asr_fallback_compute_type,
        pause_threshold_offset_db=args.pause_threshold_offset_db,
        dry_run=args.dry_run,
    )
    metrics = assessment["metrics"]
    llm_json = assessment["llm_rubric"]
    report = dict(assessment["report"])
    log_dir = Path(args.log_dir)
    progress_delta = build_progress_delta(log_dir / "history.csv", report)
    if progress_delta:
        report["progress_delta"] = progress_delta
        report = AssessmentReport.from_dict(report).to_dict()
        assessment["report"] = report

    run_dt = datetime.now()
    meta = {
        "timestamp": run_dt.isoformat(timespec="seconds"),
        "audio_path": str(args.audio.resolve()),
        "whisper_model": args.whisper,
        "llm_model": chosen_model,
        "provider": chosen_provider,
        "theme": args.theme,
        "task_family": args.task_family,
        "speaker_id": args.speaker_id or "",
        "target_duration_sec": args.target_duration_sec,
        "feedback_language": args.feedback_language or args.expected_language,
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
        coaching_obj = report.get("coaching") or {}
        if rubric_obj is None and isinstance(llm_json, str):
            rubric_obj = extract_rubric_json(llm_json)
        append_history(
            log_dir / "history.csv",
            {
                "timestamp": meta["timestamp"],
                "session_id": report.get("session_id", ""),
                "schema_version": report.get("schema_version", ""),
                "speaker_id": report.get("input", {}).get("speaker_id", args.speaker_id or ""),
                "learning_language": report.get("input", {}).get("learning_language", args.expected_language),
                "task_family": report.get("input", {}).get("task_family", args.task_family),
                "theme": report.get("input", {}).get("theme", args.theme),
                "audio": args.audio.name,
                "whisper": args.whisper,
                "llm": chosen_model,
                "label": args.label or "",
                "target_duration_sec": report.get("input", {}).get("target_duration_sec", args.target_duration_sec),
                "duration_sec": metrics.get("duration_sec", ""),
                "wpm": metrics.get("wpm", ""),
                "word_count": metrics.get("word_count", ""),
                "duration_pass": report.get("checks", {}).get("duration_pass", ""),
                "topic_pass": report.get("checks", {}).get("topic_pass", ""),
                "language_pass": report.get("checks", {}).get("language_pass", ""),
                "fluency": (rubric_obj or {}).get("fluency", ""),
                "cohesion": (rubric_obj or {}).get("cohesion", ""),
                "accuracy": (rubric_obj or {}).get("accuracy", ""),
                "range": (rubric_obj or {}).get("range", ""),
                "overall": (rubric_obj or {}).get("overall", ""),
                "final_score": report.get("scores", {}).get("final", ""),
                "band": report.get("scores", {}).get("band", ""),
                "requires_human_review": report.get("requires_human_review", ""),
                "top_priority_1": (coaching_obj.get("top_3_priorities") or ["", "", ""])[0],
                "top_priority_2": (coaching_obj.get("top_3_priorities") or ["", "", ""])[1],
                "top_priority_3": (coaching_obj.get("top_3_priorities") or ["", "", ""])[2],
                "grammar_error_categories": _extract_issue_categories(rubric_obj, "recurring_grammar_errors"),
                "coherence_issue_categories": _extract_issue_categories(rubric_obj, "coherence_issues"),
                "report_path": str(report_path.resolve()),
            },
        )
        append_session_jsonl(
            log_dir / "sessions.jsonl",
            {
                "timestamp": meta["timestamp"],
                "session_id": report.get("session_id", ""),
                "schema_version": report.get("schema_version", ""),
                "speaker_id": report.get("input", {}).get("speaker_id", args.speaker_id or ""),
                "task_family": report.get("input", {}).get("task_family", args.task_family),
                "theme": report.get("input", {}).get("theme", args.theme),
                "report_path": str(report_path.resolve()),
                "report": report,
            },
        )
        print(f"Ergebnis gespeichert in {report_path}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, SchemaValidationError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        sys.exit(1)

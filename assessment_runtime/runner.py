from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class AssessmentRunRequest:
    audio: Path
    whisper_model: str = "large-v3"
    asr_provider: Optional[str] = None
    llm_model: Optional[str] = None
    provider: Optional[str] = None
    feedback_enabled: bool = False
    train_dir: Path = Path("training")
    target_cefr: Optional[str] = None
    prompt_id: str = ""
    prompt_text: str = ""
    retry_of_session_id: str = ""
    theme: str = "tema libero"
    task_family: Optional[str] = None
    speaker_id: Optional[str] = None
    target_duration_sec: float = 120.0
    expected_language: Optional[str] = None
    language_profile_key: Optional[str] = None
    feedback_language: Optional[str] = None
    min_word_count: Optional[int] = None
    llm_timeout_sec: Optional[float] = None
    llm_base_url: Optional[str] = None
    llm_api_key: str | Callable[[], str] | None = None
    asr_compute_type: Optional[str] = None
    asr_fallback_compute_type: Optional[str] = None
    pause_threshold_offset_db: Optional[float] = None
    dry_run: bool = False
    log_dir: Path = Path("reports")
    no_log: bool = False
    label: str = ""
    notes: str = ""
    stage_cache_dir: Path | None = None
    cloud_asr: Callable[[], dict] | None = None
    llm_connection_id: str = ""


@dataclass(frozen=True)
class AssessmentRunResult:
    meta: dict[str, Any]
    output: dict[str, Any]
    stdout_json: str
    report: dict[str, Any]
    report_path: Path | None
    saved_payload: dict[str, Any] | None


def _build_meta(*, request: AssessmentRunRequest, report: dict[str, Any], timestamp: str) -> dict[str, Any]:
    report_input = report.get("input") if isinstance(report.get("input"), dict) else {}
    scores = report.get("scores") if isinstance(report.get("scores"), dict) else {}
    # A report schema version describes its shape, not how the score or measurements were produced.
    analysis_signature = json.dumps(
        {key: report_input.get(key) for key in (
            "rubric_prompt_version", "transcription_basis", "language_profile_key",
            "asr_compute_type_used", "pause_threshold_offset_db", "llm_inference_profile",
            "language_profile_version", "coaching_prompt_version", "feedback_language",
            "transcript_quality_policy", "feedback_claim_policy", "asr_model", "asr_diagnostic_policy",
        )},
        sort_keys=True,
    )
    routes = [{key: route.get(key) for key in ("provider", "model", "route")} for route in report_input.get("cloud_routes", [])]
    if routes:
        analysis_signature = json.dumps({"analysis": analysis_signature, "routes": routes}, sort_keys=True)
    return {
        "timestamp": timestamp,
        "audio_path": str(request.audio.resolve()),
        "whisper_model": request.whisper_model,
        "asr_provider": str(report_input.get("asr_provider") or request.asr_provider or ""),
        "llm_model": str(report_input.get("llm_model") or request.llm_model or ""),
        "provider": str(report_input.get("provider") or request.provider or ""),
        "theme": str(report_input.get("theme") or request.theme),
        "task_family": str(report_input.get("task_family") or request.task_family or ""),
        "speaker_id": str(report_input.get("speaker_id") or request.speaker_id or ""),
        "target_duration_sec": report_input.get("target_duration_sec", request.target_duration_sec),
        "feedback_language": str(report_input.get("feedback_language") or request.feedback_language or request.expected_language or ""),
        "practice": {
            "version": 1,
            "goal": (request.target_cefr or "").strip().upper() or None,
            "prompt_id": request.prompt_id,
            "prompt_text": request.prompt_text,
            "retry_of_session_id": request.retry_of_session_id,
            "target_duration_sec": request.target_duration_sec,
            "scoring_version": str(report_input.get("scoring_model_version") or ""),
            "scoring_mode": str(scores.get("mode") or ""),
            "analysis_signature": analysis_signature,
            "provider": str(report_input.get("provider") or request.provider or ""),
            "model": str(report_input.get("llm_model") or request.llm_model or ""),
            "asr_provider": str(report_input.get("asr_provider") or request.asr_provider or ""),
            "whisper_model": request.whisper_model,
            "dry_run": request.dry_run,
        },
        **({"label": request.label} if request.label else {}),
    }


def _build_stdout_payload(assessment: dict[str, Any], *, meta: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    output = {
        "meta": meta,
        "metrics": assessment["metrics"],
        "transcript_preview": assessment["transcript_preview"],
        "llm_rubric": assessment["llm_rubric"],
        "report": report,
    }
    if "baseline_comparison" in assessment:
        output["baseline_comparison"] = assessment["baseline_comparison"]
    if "suggested_training" in assessment:
        output["suggested_training"] = assessment["suggested_training"]
    return output


def execute_assessment_run(request: AssessmentRunRequest, *, status_callback=None) -> AssessmentRunResult:
    from app_core.app_data import resolve_app_data_root
    from app_core.journal_lock import journal_guard
    with journal_guard(resolve_app_data_root()):
        return _execute_assessment_run(request, status_callback=status_callback)


def _execute_assessment_run(
    request: AssessmentRunRequest,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> AssessmentRunResult:
    import assess_speaking as assess_cli

    assessment = assess_cli.run_assessment(
        request.audio,
        request.whisper_model,
        request.llm_model,
        provider=request.provider,
        asr_provider=request.asr_provider,
        feedback_enabled=request.feedback_enabled,
        train_dir=request.train_dir,
        target_cefr=request.target_cefr,
        theme=request.theme,
        task_family=request.task_family,
        speaker_id=request.speaker_id,
        target_duration_sec=request.target_duration_sec,
        expected_language=request.expected_language,
        language_profile_key=request.language_profile_key,
        feedback_language=request.feedback_language,
        min_word_count=request.min_word_count,
        llm_timeout_sec=request.llm_timeout_sec,
        llm_base_url=request.llm_base_url,
        llm_api_key=request.llm_api_key,
        asr_compute_type=request.asr_compute_type,
        asr_fallback_compute_type=request.asr_fallback_compute_type,
        pause_threshold_offset_db=request.pause_threshold_offset_db,
        dry_run=request.dry_run,
        status_callback=status_callback,
        **({"stage_cache_dir": request.stage_cache_dir} if request.stage_cache_dir else {}),
        **({"cloud_asr": request.cloud_asr} if request.cloud_asr else {}),
        **({"llm_connection_id": request.llm_connection_id} if request.llm_connection_id else {}),
    )

    from assessment_runtime.eligibility import annotate_report
    report = annotate_report(dict(assessment["report"]), assessment.get("metrics"))
    if request.stage_cache_dir:
        from assessment_runtime.checkpoints import StageCache
        report["session_id"] = StageCache(request.stage_cache_dir).attempt_id()
    run_dt = datetime.now()
    resolved_log_dir = Path(request.log_dir)
    timestamp = run_dt.isoformat(timespec="seconds")
    if request.stage_cache_dir:
        timestamp = StageCache(request.stage_cache_dir).created_at() or timestamp
    meta = _build_meta(request=request, report=report, timestamp=timestamp)
    meta["analysis_timestamp"] = run_dt.isoformat(timespec="seconds")
    progress_delta = assess_cli.build_progress_delta(
        resolved_log_dir / "history.csv", report, practice=meta["practice"],
    )
    if progress_delta:
        report["progress_delta"] = progress_delta
        report = assess_cli.AssessmentReport.from_dict(report).to_dict()
        assessment["report"] = report

    output = _build_stdout_payload(assessment, meta=meta, report=report)
    stdout_json = json.dumps(output, ensure_ascii=False, indent=2)

    if request.no_log:
        return AssessmentRunResult(
            meta=meta,
            output=output,
            stdout_json=stdout_json,
            report=report,
            report_path=None,
            saved_payload=None,
        )

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    report_path = assess_cli.build_report_path(resolved_log_dir, request.audio, request.label or None, run_dt)
    # Repeated CLI runs can share the same audio/label and timestamp second.
    # A session suffix prevents overwriting the parent's retained report.
    from uuid import uuid4
    report_path = report_path.with_name(f"{report_path.stem}_{report['session_id']}_{uuid4().hex[:8]}.json")
    saved_payload = {
        **output,
        "transcript_full": assessment["transcript_full"],
        "notes": request.notes,
        "report_path": str(report_path.resolve()),
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(saved_payload, handle, ensure_ascii=False, indent=2)

    rubric_obj = report.get("rubric")
    coaching_obj = report.get("coaching") or {}
    if rubric_obj is None and isinstance(assessment["llm_rubric"], str):
        rubric_obj = assess_cli.extract_rubric_json(assessment["llm_rubric"])
    priorities = [str(item) for item in (coaching_obj.get("top_3_priorities") or []) if str(item).strip()]
    padded_priorities = priorities[:3] + [""] * max(0, 3 - len(priorities[:3]))
    # CSV consumers cannot distinguish provisional numeric observations from
    # grades. Keep them in the annotated JSON, but never publish them as grades.
    assessable = report["eligibility"]["state"] == "assessable"
    assess_cli.append_history(
        resolved_log_dir / "history.csv",
        {
            "timestamp": meta["timestamp"],
            "session_id": report.get("session_id", ""),
            "schema_version": report.get("schema_version", ""),
            "speaker_id": report.get("input", {}).get("speaker_id", request.speaker_id or ""),
            "learning_language": report.get("input", {}).get("learning_language", request.expected_language),
            "task_family": report.get("input", {}).get("task_family", request.task_family),
            "theme": report.get("input", {}).get("theme", request.theme),
            "audio": request.audio.name,
            "whisper": request.whisper_model,
            "llm": meta["llm_model"],
            "label": request.label,
            "target_duration_sec": report.get("input", {}).get("target_duration_sec", request.target_duration_sec),
            "duration_sec": assessment["metrics"].get("duration_sec", ""),
            "wpm": assessment["metrics"].get("wpm", ""),
            "word_count": assessment["metrics"].get("word_count", ""),
            "duration_pass": report.get("checks", {}).get("duration_pass", ""),
            "topic_pass": report.get("checks", {}).get("topic_pass", ""),
            "language_pass": report.get("checks", {}).get("language_pass", ""),
            "fluency": (rubric_obj or {}).get("fluency", "") if assessable else "",
            "cohesion": (rubric_obj or {}).get("cohesion", "") if assessable else "",
            "accuracy": (rubric_obj or {}).get("accuracy", "") if assessable else "",
            "range": (rubric_obj or {}).get("range", "") if assessable else "",
            "overall": (rubric_obj or {}).get("overall", "") if assessable else "",
            "final_score": report.get("scores", {}).get("final", "") if assessable else "",
            "band": report.get("scores", {}).get("band", "") if assessable else "",
            "requires_human_review": report.get("requires_human_review", ""),
            "top_priority_1": padded_priorities[0],
            "top_priority_2": padded_priorities[1],
            "top_priority_3": padded_priorities[2],
            "grammar_error_categories": assess_cli._extract_issue_categories(rubric_obj, "recurring_grammar_errors"),
            "coherence_issue_categories": assess_cli._extract_issue_categories(rubric_obj, "coherence_issues"),
            "report_path": str(report_path.resolve()),
        },
    )
    assess_cli.append_session_jsonl(
        resolved_log_dir / "sessions.jsonl",
        {
            "timestamp": meta["timestamp"],
            "session_id": report.get("session_id", ""),
            "schema_version": report.get("schema_version", ""),
            "speaker_id": report.get("input", {}).get("speaker_id", request.speaker_id or ""),
            "task_family": report.get("input", {}).get("task_family", request.task_family),
            "theme": report.get("input", {}).get("theme", request.theme),
            "report_path": str(report_path.resolve()),
            "report": report,
        },
    )

    return AssessmentRunResult(
        meta=meta,
        output=output,
        stdout_json=stdout_json,
        report=report,
        report_path=report_path,
        saved_payload=saved_payload,
    )

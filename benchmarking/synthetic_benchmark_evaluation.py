"""Run rendered synthetic audio through the assessment pipeline and persist comparable results."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import traceback
from typing import Any, Callable

from assess_speaking import run_assessment
from benchmarking.synthetic_audio_contracts import RenderedAudioCase, RenderedAudioContractSuite


RESPONSE_PARSER_NAME = "extract_json_object"
RUBRIC_SCHEMA_NAME = "RubricResult"
EVALUATION_SCHEMA_VERSION = 1

CEFR_ORDER = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}


AssessmentRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class EvaluationRunConfig:
    whisper_model: str
    provider: str | None
    llm_model: str | None
    feedback_language: str | None
    target_duration_sec: float
    speaker_id: str
    dry_run: bool
    include_raw_llm: bool
    include_full_report: bool
    llm_timeout_sec: float | None = None
    max_consecutive_runner_errors: int | None = None
    response_parser: str = RESPONSE_PARSER_NAME
    rubric_schema: str = RUBRIC_SCHEMA_NAME
    language_profile_key: str | None = None


@dataclass(frozen=True)
class EvaluationLLMContract:
    provider: str | None
    llm_model: str | None
    whisper_model: str | None
    response_parser: str
    rubric_schema: str
    prompt_version: str | None
    rubric_prompt_version: str | None
    coaching_prompt_version: str | None
    scoring_model_version: str | None
    language_profile: str | None
    language_profile_key: str | None
    language_profile_version: str | None

    @classmethod
    def from_config_only(cls, config: "EvaluationRunConfig") -> "EvaluationLLMContract":
        return cls(
            provider=config.provider,
            llm_model=config.llm_model,
            whisper_model=config.whisper_model,
            response_parser=config.response_parser,
            rubric_schema=config.rubric_schema,
            prompt_version=None,
            rubric_prompt_version=None,
            coaching_prompt_version=None,
            scoring_model_version=None,
            language_profile=None,
            language_profile_key=config.language_profile_key,
            language_profile_version=None,
        )


@dataclass(frozen=True)
class EvaluatedRenderedCase:
    case_id: str
    source_seed_id: str
    status: str
    audio_path: Path
    expected_language: str
    feedback_language: str
    target_cefr: str
    benchmark_suite_id: str | None
    benchmark_case_id: str | None
    estimated_cefr: str | None
    cefr_delta: int | None
    cefr_match: bool | None
    final_score: float | None
    llm_score: float | None
    deterministic_score: float | None
    continuous_score: float | None
    band: int | None
    mode: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    error_type: str | None
    execution_error: str | None
    execution_traceback: str | None
    checks: dict[str, Any]
    dimensions: dict[str, Any]
    timings_ms: dict[str, Any]
    llm_contract: EvaluationLLMContract
    raw_llm_rubric: str | None
    report: dict[str, Any] | None


@dataclass(frozen=True)
class EvaluatedRenderedAudioSuite:
    suite_id: str
    manifest_id: str
    language_code: str
    task_family: str
    generated_at_utc: str
    run_status: str
    success_ratio: float
    config: EvaluationRunConfig
    cases: tuple[EvaluatedRenderedCase, ...]


def compare_cefr_levels(expected: str | None, actual: str | None) -> int | None:
    if not expected or not actual:
        return None
    expected_rank = CEFR_ORDER.get(str(expected).strip().upper())
    actual_rank = CEFR_ORDER.get(str(actual).strip().upper())
    if expected_rank is None or actual_rank is None:
        return None
    return actual_rank - expected_rank


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _theme_for_case(case: RenderedAudioCase) -> str:
    return case.topic_tag


def _slugify(value: str | None, *, fallback: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "")).strip("-").lower()
    return text or fallback


def _build_contract(
    report_input: dict[str, Any],
    *,
    config: EvaluationRunConfig,
) -> EvaluationLLMContract:
    return EvaluationLLMContract(
        provider=report_input.get("provider") or config.provider,
        llm_model=report_input.get("llm_model") or config.llm_model,
        whisper_model=report_input.get("whisper_model") or config.whisper_model,
        response_parser=config.response_parser,
        rubric_schema=config.rubric_schema,
        prompt_version=report_input.get("prompt_version"),
        rubric_prompt_version=report_input.get("rubric_prompt_version"),
        coaching_prompt_version=report_input.get("coaching_prompt_version"),
        scoring_model_version=report_input.get("scoring_model_version"),
        language_profile=report_input.get("language_profile"),
        language_profile_key=report_input.get("language_profile_key"),
        language_profile_version=report_input.get("language_profile_version"),
    )


def _serialize_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _serialize_raw_llm(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(_serialize_json_safe(value), ensure_ascii=False)


def _extract_cefr_estimate(scores: dict[str, Any]) -> tuple[str | None, float | None]:
    cefr_estimate = dict(scores.get("cefr_estimate") or {})
    estimated_cefr = cefr_estimate.get("cefr") or cefr_estimate.get("level")
    continuous_score = cefr_estimate.get("continuous_score")
    if continuous_score is None:
        continuous_score = cefr_estimate.get("continuous")
    return (
        str(estimated_cefr) if estimated_cefr else None,
        _safe_float(continuous_score),
    )


@contextmanager
def checkpoint_lock(checkpoint_path: str | Path):
    lock_path = Path(f"{checkpoint_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Checkpoint is already locked by another runner: {lock_path}") from exc
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def evaluate_rendered_audio_case(
    case: RenderedAudioCase,
    *,
    config: EvaluationRunConfig,
    runner: AssessmentRunner = run_assessment,
) -> EvaluatedRenderedCase:
    resolved_feedback_language = config.feedback_language or case.expected_language
    try:
        result = runner(
            audio=case.audio_path,
            whisper_model=config.whisper_model,
            llm_model=config.llm_model,
            provider=config.provider,
            feedback_enabled=False,
            target_cefr=case.target_cefr,
            theme=_theme_for_case(case),
            task_family=case.task_family,
            speaker_id=config.speaker_id,
            target_duration_sec=(
                case.target_duration_sec if case.target_duration_sec is not None else config.target_duration_sec
            ),
            expected_language=case.expected_language,
            language_profile_key=config.language_profile_key,
            feedback_language=resolved_feedback_language,
            llm_timeout_sec=config.llm_timeout_sec,
            dry_run=config.dry_run,
        )
    except Exception as exc:
        return EvaluatedRenderedCase(
            case_id=case.case_id,
            source_seed_id=case.source_seed_id,
            status="runner_error",
            audio_path=case.audio_path,
            expected_language=case.expected_language,
            feedback_language=resolved_feedback_language,
            target_cefr=case.target_cefr,
            benchmark_suite_id=case.benchmark_suite_id,
            benchmark_case_id=case.benchmark_case_id,
            estimated_cefr=None,
            cefr_delta=None,
            cefr_match=None,
            final_score=None,
            llm_score=None,
            deterministic_score=None,
            continuous_score=None,
            band=None,
            mode=None,
            warnings=(),
            errors=(str(exc),),
            error_type=type(exc).__name__,
            execution_error=str(exc),
            execution_traceback=traceback.format_exc(),
            checks={},
            dimensions={},
            timings_ms={},
            llm_contract=EvaluationLLMContract.from_config_only(config),
            raw_llm_rubric=None,
            report=None,
        )

    report = dict(result.get("report") or {})
    report_input = dict(report.get("input") or {})
    scores = dict(report.get("scores") or {})
    checks = dict(report.get("checks") or {})
    estimated_cefr, continuous_score = _extract_cefr_estimate(scores)
    cefr_delta = compare_cefr_levels(case.target_cefr, estimated_cefr)
    raw_llm = result.get("llm_rubric")
    if not config.include_raw_llm:
        raw_llm = None
    full_report = report if config.include_full_report else None
    return EvaluatedRenderedCase(
        case_id=case.case_id,
        source_seed_id=case.source_seed_id,
        status="ok",
        audio_path=case.audio_path,
        expected_language=case.expected_language,
        feedback_language=resolved_feedback_language,
        target_cefr=case.target_cefr,
        benchmark_suite_id=case.benchmark_suite_id,
        benchmark_case_id=case.benchmark_case_id,
        estimated_cefr=str(estimated_cefr) if estimated_cefr else None,
        cefr_delta=cefr_delta,
        cefr_match=cefr_delta == 0 if cefr_delta is not None else None,
        final_score=_safe_float(scores.get("final")),
        llm_score=_safe_float(scores.get("llm")),
        deterministic_score=_safe_float(scores.get("deterministic")),
        continuous_score=continuous_score,
        band=_safe_int(scores.get("band")),
        mode=str(scores.get("mode")) if scores.get("mode") is not None else None,
        warnings=tuple(str(item) for item in report.get("warnings") or ()),
        errors=tuple(str(item) for item in report.get("errors") or ()),
        error_type=None,
        execution_error=None,
        execution_traceback=None,
        checks=_serialize_json_safe(checks),
        dimensions=_serialize_json_safe(dict(scores.get("dimensions") or {})),
        timings_ms=_serialize_json_safe(dict(report.get("timings_ms") or {})),
        llm_contract=_build_contract(report_input, config=config),
        raw_llm_rubric=_serialize_raw_llm(raw_llm),
        report=_serialize_json_safe(full_report) if full_report is not None else None,
    )


def build_skipped_case(
    case: RenderedAudioCase,
    *,
    config: EvaluationRunConfig,
    reason: str,
) -> EvaluatedRenderedCase:
    resolved_feedback_language = config.feedback_language or case.expected_language
    return EvaluatedRenderedCase(
        case_id=case.case_id,
        source_seed_id=case.source_seed_id,
        status="skipped",
        audio_path=case.audio_path,
        expected_language=case.expected_language,
        feedback_language=resolved_feedback_language,
        target_cefr=case.target_cefr,
        benchmark_suite_id=case.benchmark_suite_id,
        benchmark_case_id=case.benchmark_case_id,
        estimated_cefr=None,
        cefr_delta=None,
        cefr_match=None,
        final_score=None,
        llm_score=None,
        deterministic_score=None,
        continuous_score=None,
        band=None,
        mode=None,
        warnings=(),
        errors=(reason,),
        error_type="CircuitBreaker",
        execution_error=reason,
        execution_traceback=None,
        checks={},
        dimensions={},
        timings_ms={},
        llm_contract=EvaluationLLMContract.from_config_only(config),
        raw_llm_rubric=None,
        report=None,
    )


def evaluate_rendered_audio_contract_suite(
    suite: RenderedAudioContractSuite,
    *,
    config: EvaluationRunConfig,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
    runner: AssessmentRunner = run_assessment,
) -> EvaluatedRenderedAudioSuite:
    provider_slug = _slugify(config.provider, fallback="default-provider")
    model_slug = _slugify(config.llm_model, fallback="default-model")
    suite_id = f"{suite.suite_id}_{provider_slug}_{model_slug}_evaluation_v1"
    lock_context = checkpoint_lock(checkpoint_path) if checkpoint_path else nullcontext()
    with lock_context:
        restored_cases: dict[str, EvaluatedRenderedCase] = {}
        if checkpoint_path and resume_from_checkpoint:
            restored_cases = load_evaluation_checkpoint_cases(
                checkpoint_path,
                manifest_id=suite.manifest_id,
                suite_id=suite_id,
                successful_only=True,
            )

        case_results: list[EvaluatedRenderedCase] = []
        consecutive_runner_errors = 0
        for case in suite.cases:
            restored_case = restored_cases.get(case.case_id)
            if restored_case is not None:
                case_results.append(restored_case)
                consecutive_runner_errors = 0 if restored_case.status == "ok" else consecutive_runner_errors
                continue
            evaluated_case = evaluate_rendered_audio_case(case, config=config, runner=runner)
            case_results.append(evaluated_case)
            if checkpoint_path:
                append_evaluation_checkpoint(
                    checkpoint_path,
                    manifest_id=suite.manifest_id,
                    suite_id=suite_id,
                    case=evaluated_case,
                )
            if evaluated_case.status == "runner_error":
                consecutive_runner_errors += 1
            else:
                consecutive_runner_errors = 0
            if (
                config.max_consecutive_runner_errors is not None
                and config.max_consecutive_runner_errors > 0
                and consecutive_runner_errors >= config.max_consecutive_runner_errors
            ):
                reason = (
                    "Skipped remaining cases after "
                    f"{consecutive_runner_errors} consecutive runner errors."
                )
                for remaining_case in suite.cases[len(case_results) :]:
                    skipped_case = build_skipped_case(remaining_case, config=config, reason=reason)
                    case_results.append(skipped_case)
                    if checkpoint_path:
                        append_evaluation_checkpoint(
                            checkpoint_path,
                            manifest_id=suite.manifest_id,
                            suite_id=suite_id,
                            case=skipped_case,
                        )
                break

    cases = tuple(case_results)
    total_cases = len(cases)
    ok_cases = sum(1 for case in cases if case.status == "ok")
    success_ratio = 0.0 if total_cases == 0 else round(ok_cases / total_cases, 4)
    skipped_cases = sum(1 for case in cases if case.status == "skipped")
    if total_cases == 0:
        run_status = "empty"
    elif skipped_cases:
        run_status = "aborted"
    elif ok_cases == 0:
        run_status = "failed"
    elif ok_cases < total_cases:
        run_status = "degraded"
    else:
        run_status = "ok"
    return EvaluatedRenderedAudioSuite(
        suite_id=suite_id,
        manifest_id=suite.manifest_id,
        language_code=suite.language_code,
        task_family=suite.task_family,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_status=run_status,
        success_ratio=success_ratio,
        config=config,
        cases=cases,
    )


def _case_to_dict(case: EvaluatedRenderedCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "source_seed_id": case.source_seed_id,
        "status": case.status,
        "audio_path": case.audio_path.as_posix(),
        "expected_language": case.expected_language,
        "feedback_language": case.feedback_language,
        "target_cefr": case.target_cefr,
        "benchmark_suite_id": case.benchmark_suite_id,
        "benchmark_case_id": case.benchmark_case_id,
        "estimated_cefr": case.estimated_cefr,
        "cefr_delta": case.cefr_delta,
        "cefr_match": case.cefr_match,
        "final_score": case.final_score,
        "llm_score": case.llm_score,
        "deterministic_score": case.deterministic_score,
        "continuous_score": case.continuous_score,
        "band": case.band,
        "mode": case.mode,
        "warnings": list(case.warnings),
        "errors": list(case.errors),
        "error_type": case.error_type,
        "execution_error": case.execution_error,
        "execution_traceback": case.execution_traceback,
        "checks": _serialize_json_safe(case.checks),
        "dimensions": _serialize_json_safe(case.dimensions),
        "timings_ms": _serialize_json_safe(case.timings_ms),
        "llm_contract": {
            "provider": case.llm_contract.provider,
            "llm_model": case.llm_contract.llm_model,
            "whisper_model": case.llm_contract.whisper_model,
            "response_parser": case.llm_contract.response_parser,
            "rubric_schema": case.llm_contract.rubric_schema,
            "prompt_version": case.llm_contract.prompt_version,
            "rubric_prompt_version": case.llm_contract.rubric_prompt_version,
            "coaching_prompt_version": case.llm_contract.coaching_prompt_version,
            "scoring_model_version": case.llm_contract.scoring_model_version,
            "language_profile": case.llm_contract.language_profile,
            "language_profile_key": case.llm_contract.language_profile_key,
            "language_profile_version": case.llm_contract.language_profile_version,
        },
        "raw_llm_rubric": case.raw_llm_rubric,
        "report": case.report,
    }


def _case_from_dict(raw: dict[str, Any]) -> EvaluatedRenderedCase:
    llm_contract_raw = dict(raw.get("llm_contract") or {})
    report_payload = dict(raw.get("report") or {}) if raw.get("report") is not None else None
    report_scores = dict(report_payload.get("scores") or {}) if report_payload is not None else {}
    estimated_cefr = str(raw["estimated_cefr"]) if raw.get("estimated_cefr") else None
    continuous_score = _safe_float(raw.get("continuous_score"))
    if estimated_cefr is None or continuous_score is None:
        derived_cefr, derived_continuous = _extract_cefr_estimate(report_scores)
        if estimated_cefr is None:
            estimated_cefr = derived_cefr
        if continuous_score is None:
            continuous_score = derived_continuous
    return EvaluatedRenderedCase(
        case_id=str(raw["case_id"]),
        source_seed_id=str(raw["source_seed_id"]),
        status=str(raw["status"]),
        audio_path=Path(str(raw["audio_path"])),
        expected_language=str(raw["expected_language"]),
        feedback_language=str(raw["feedback_language"]),
        target_cefr=str(raw["target_cefr"]),
        benchmark_suite_id=(
            str(raw["benchmark_suite_id"]) if raw.get("benchmark_suite_id") is not None else None
        ),
        benchmark_case_id=(
            str(raw["benchmark_case_id"]) if raw.get("benchmark_case_id") is not None else None
        ),
        estimated_cefr=estimated_cefr,
        cefr_delta=int(raw["cefr_delta"]) if raw.get("cefr_delta") is not None else None,
        cefr_match=bool(raw["cefr_match"]) if raw.get("cefr_match") is not None else None,
        final_score=_safe_float(raw.get("final_score")),
        llm_score=_safe_float(raw.get("llm_score")),
        deterministic_score=_safe_float(raw.get("deterministic_score")),
        continuous_score=continuous_score,
        band=_safe_int(raw.get("band")),
        mode=str(raw["mode"]) if raw.get("mode") is not None else None,
        warnings=tuple(str(item) for item in raw.get("warnings") or ()),
        errors=tuple(str(item) for item in raw.get("errors") or ()),
        error_type=str(raw["error_type"]) if raw.get("error_type") else None,
        execution_error=str(raw["execution_error"]) if raw.get("execution_error") else None,
        execution_traceback=(
            str(raw["execution_traceback"]) if raw.get("execution_traceback") else None
        ),
        checks=dict(raw.get("checks") or {}),
        dimensions=dict(raw.get("dimensions") or {}),
        timings_ms=dict(raw.get("timings_ms") or {}),
        llm_contract=EvaluationLLMContract(
            provider=llm_contract_raw.get("provider"),
            llm_model=llm_contract_raw.get("llm_model"),
            whisper_model=llm_contract_raw.get("whisper_model"),
            response_parser=str(llm_contract_raw.get("response_parser") or RESPONSE_PARSER_NAME),
            rubric_schema=str(llm_contract_raw.get("rubric_schema") or RUBRIC_SCHEMA_NAME),
            prompt_version=llm_contract_raw.get("prompt_version"),
            rubric_prompt_version=llm_contract_raw.get("rubric_prompt_version"),
            coaching_prompt_version=llm_contract_raw.get("coaching_prompt_version"),
            scoring_model_version=llm_contract_raw.get("scoring_model_version"),
            language_profile=llm_contract_raw.get("language_profile"),
            language_profile_key=llm_contract_raw.get("language_profile_key"),
            language_profile_version=llm_contract_raw.get("language_profile_version"),
        ),
        raw_llm_rubric=(
            str(raw["raw_llm_rubric"]) if raw.get("raw_llm_rubric") is not None else None
        ),
        report=report_payload,
    )


def append_evaluation_checkpoint(
    checkpoint_path: str | Path,
    *,
    manifest_id: str,
    suite_id: str,
    case: EvaluatedRenderedCase,
) -> Path:
    output = Path(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    created_new_file = not output.exists()
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "suite_id": suite_id,
        "case": _case_to_dict(case),
    }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if created_new_file:
        dir_fd = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    return output


def load_evaluation_checkpoint_cases(
    checkpoint_path: str | Path,
    *,
    manifest_id: str,
    suite_id: str,
    successful_only: bool = False,
) -> dict[str, EvaluatedRenderedCase]:
    path = Path(checkpoint_path)
    if not path.exists():
        return {}
    cases: dict[str, EvaluatedRenderedCase] = {}
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_number, line in enumerate(lines, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                if line_number == len(lines):
                    continue
                raise
            if not isinstance(payload, dict):
                raise ValueError(f"Checkpoint line {line_number} must be a JSON object")
            if int(payload.get("schema_version") or 0) != EVALUATION_SCHEMA_VERSION:
                raise ValueError("Checkpoint schema_version does not match the current runner")
            if str(payload.get("manifest_id")) != manifest_id:
                raise ValueError("Checkpoint manifest_id does not match the requested suite")
            if str(payload.get("suite_id")) != suite_id:
                raise ValueError("Checkpoint suite_id does not match the requested suite")
            case_payload = payload.get("case")
            if not isinstance(case_payload, dict):
                raise ValueError(f"Checkpoint line {line_number} is missing a valid case payload")
            case = _case_from_dict(case_payload)
            cases[case.case_id] = case
    if successful_only:
        return {case_id: case for case_id, case in cases.items() if case.status == "ok"}
    return cases


def write_evaluation_manifest(
    suite: EvaluatedRenderedAudioSuite,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite_id": suite.suite_id,
        "manifest_id": suite.manifest_id,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "language_code": suite.language_code,
        "task_family": suite.task_family,
        "generated_at_utc": suite.generated_at_utc,
        "run_status": suite.run_status,
        "success_ratio": suite.success_ratio,
        "config": {
            "whisper_model": suite.config.whisper_model,
            "provider": suite.config.provider,
            "llm_model": suite.config.llm_model,
            "feedback_language": suite.config.feedback_language,
            "target_duration_sec": suite.config.target_duration_sec,
            "speaker_id": suite.config.speaker_id,
            "dry_run": suite.config.dry_run,
            "include_raw_llm": suite.config.include_raw_llm,
            "include_full_report": suite.config.include_full_report,
            "llm_timeout_sec": suite.config.llm_timeout_sec,
            "max_consecutive_runner_errors": suite.config.max_consecutive_runner_errors,
            "response_parser": suite.config.response_parser,
            "rubric_schema": suite.config.rubric_schema,
            "language_profile_key": suite.config.language_profile_key,
        },
        "summary": {
            "total_cases": len(suite.cases),
            "ok_cases": sum(1 for case in suite.cases if case.status == "ok"),
            "runner_error_cases": sum(1 for case in suite.cases if case.status == "runner_error"),
            "skipped_cases": sum(1 for case in suite.cases if case.status == "skipped"),
            "cefr_matches": sum(1 for case in suite.cases if case.cefr_match is True),
            "success_ratio": suite.success_ratio,
            "run_status": suite.run_status,
        },
        "cases": [_case_to_dict(case) for case in suite.cases],
    }
    temp_output = output.with_suffix(f"{output.suffix}.tmp")
    temp_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_output.replace(output)
    return output


def load_evaluation_manifest(path: str | Path) -> EvaluatedRenderedAudioSuite:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation manifest root payload must be an object")
    if int(payload.get("schema_version") or 0) != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Evaluation manifest schema_version does not match the current runner")
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Evaluation manifest is missing a valid config object")
    cases_payload = payload.get("cases")
    if not isinstance(cases_payload, list):
        raise ValueError("Evaluation manifest is missing a valid cases list")
    return EvaluatedRenderedAudioSuite(
        suite_id=str(payload["suite_id"]),
        manifest_id=str(payload["manifest_id"]),
        language_code=str(payload["language_code"]),
        task_family=str(payload["task_family"]),
        generated_at_utc=str(payload["generated_at_utc"]),
        run_status=str(payload["run_status"]),
        success_ratio=float(payload["success_ratio"]),
        config=EvaluationRunConfig(
            whisper_model=str(config_payload["whisper_model"]),
            provider=config_payload.get("provider"),
            llm_model=config_payload.get("llm_model"),
            feedback_language=config_payload.get("feedback_language"),
            target_duration_sec=float(config_payload["target_duration_sec"]),
            speaker_id=str(config_payload["speaker_id"]),
            dry_run=bool(config_payload.get("dry_run", False)),
            include_raw_llm=bool(config_payload.get("include_raw_llm", False)),
            include_full_report=bool(config_payload.get("include_full_report", False)),
            llm_timeout_sec=(
                float(config_payload["llm_timeout_sec"])
                if config_payload.get("llm_timeout_sec") is not None
                else None
            ),
            max_consecutive_runner_errors=(
                int(config_payload["max_consecutive_runner_errors"])
                if config_payload.get("max_consecutive_runner_errors") is not None
                else None
            ),
            response_parser=str(config_payload.get("response_parser") or RESPONSE_PARSER_NAME),
            rubric_schema=str(config_payload.get("rubric_schema") or RUBRIC_SCHEMA_NAME),
            language_profile_key=(
                str(config_payload["language_profile_key"])
                if config_payload.get("language_profile_key") is not None
                else None
            ),
        ),
        cases=tuple(_case_from_dict(dict(item)) for item in cases_payload),
    )

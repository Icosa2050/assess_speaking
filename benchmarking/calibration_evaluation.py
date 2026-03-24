"""Evaluate real-audio calibration manifests through the assessment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from assess_speaking import run_assessment
from benchmarking.calibration_manifests import (
    CalibrationCase,
    CalibrationManifest,
    CalibrationPairExpectation,
)
from benchmarking.synthetic_benchmark_evaluation import (
    EvaluationLLMContract,
    RESPONSE_PARSER_NAME,
    RUBRIC_SCHEMA_NAME,
    compare_cefr_levels,
)


CALIBRATION_EVALUATION_SCHEMA_VERSION = 1
AssessmentRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class CalibrationRunConfig:
    whisper_model: str
    provider: str | None
    llm_model: str | None
    feedback_language: str | None
    dry_run: bool
    include_raw_llm: bool
    include_full_report: bool
    llm_timeout_sec: float | None = None
    response_parser: str = RESPONSE_PARSER_NAME
    rubric_schema: str = RUBRIC_SCHEMA_NAME
    language_profile_key: str | None = None


@dataclass(frozen=True)
class EvaluatedCalibrationCase:
    case_id: str
    status: str
    audio_path: Path
    expected_language: str
    feedback_language: str
    theme: str
    speaker_id: str
    expected_cefr: str | None
    estimated_cefr: str | None
    cefr_delta: int | None
    cefr_match: bool | None
    comparison_score: float | None
    comparison_metric: str | None
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
    checks: dict[str, Any]
    dimensions: dict[str, Any]
    timings_ms: dict[str, Any]
    llm_contract: EvaluationLLMContract
    raw_llm_rubric: str | None
    report: dict[str, Any] | None


@dataclass(frozen=True)
class EvaluatedPairExpectation:
    higher_case_id: str
    lower_case_id: str
    status: str
    passed: bool | None
    metric: str | None
    higher_score: float | None
    lower_score: float | None
    notes: str | None


@dataclass(frozen=True)
class EvaluatedCalibrationManifest:
    evaluation_id: str
    manifest_id: str
    language_code: str
    language_profile_key: str | None
    task_family: str
    generated_at_utc: str
    run_status: str
    success_ratio: float
    config: CalibrationRunConfig
    cases: tuple[EvaluatedCalibrationCase, ...]
    pair_expectations: tuple[EvaluatedPairExpectation, ...]


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


def _build_contract(report_input: dict[str, Any], *, config: CalibrationRunConfig) -> EvaluationLLMContract:
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
        language_profile_key=report_input.get("language_profile_key") or config.language_profile_key,
        language_profile_version=report_input.get("language_profile_version"),
    )


def _comparison_score(
    *,
    final_score: float | None,
    continuous_score: float | None,
    deterministic_score: float | None,
    llm_score: float | None,
) -> tuple[float | None, str | None]:
    if final_score is not None:
        return final_score, "final_score"
    if continuous_score is not None:
        return continuous_score, "continuous_score"
    if deterministic_score is not None:
        return deterministic_score, "deterministic_score"
    if llm_score is not None:
        return llm_score, "llm_score"
    return None, None


def evaluate_calibration_case(
    manifest: CalibrationManifest,
    case: CalibrationCase,
    *,
    config: CalibrationRunConfig,
    runner: AssessmentRunner = run_assessment,
) -> EvaluatedCalibrationCase:
    resolved_feedback_language = config.feedback_language or case.expected_language
    effective_profile_key = manifest.language_profile_key or config.language_profile_key
    try:
        result = runner(
            audio=case.audio_path,
            whisper_model=config.whisper_model,
            llm_model=config.llm_model,
            provider=config.provider,
            feedback_enabled=False,
            target_cefr=case.expected_cefr,
            theme=case.theme,
            task_family=manifest.task_family,
            speaker_id=case.speaker_id,
            target_duration_sec=case.target_duration_sec,
            expected_language=case.expected_language,
            language_profile_key=effective_profile_key,
            feedback_language=resolved_feedback_language,
            llm_timeout_sec=config.llm_timeout_sec,
            dry_run=config.dry_run,
        )
    except Exception as exc:
        return EvaluatedCalibrationCase(
            case_id=case.case_id,
            status="runner_error",
            audio_path=case.audio_path,
            expected_language=case.expected_language,
            feedback_language=resolved_feedback_language,
            theme=case.theme,
            speaker_id=case.speaker_id,
            expected_cefr=case.expected_cefr,
            estimated_cefr=None,
            cefr_delta=None,
            cefr_match=None,
            comparison_score=None,
            comparison_metric=None,
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
    final_score = _safe_float(scores.get("final"))
    llm_score = _safe_float(scores.get("llm"))
    deterministic_score = _safe_float(scores.get("deterministic"))
    comparison_score, comparison_metric = _comparison_score(
        final_score=final_score,
        continuous_score=continuous_score,
        deterministic_score=deterministic_score,
        llm_score=llm_score,
    )
    raw_llm = result.get("llm_rubric")
    if not config.include_raw_llm:
        raw_llm = None
    full_report = report if config.include_full_report else None
    cefr_delta = compare_cefr_levels(case.expected_cefr, estimated_cefr)
    return EvaluatedCalibrationCase(
        case_id=case.case_id,
        status="ok",
        audio_path=case.audio_path,
        expected_language=case.expected_language,
        feedback_language=resolved_feedback_language,
        theme=case.theme,
        speaker_id=case.speaker_id,
        expected_cefr=case.expected_cefr,
        estimated_cefr=estimated_cefr,
        cefr_delta=cefr_delta,
        cefr_match=cefr_delta == 0 if cefr_delta is not None else None,
        comparison_score=comparison_score,
        comparison_metric=comparison_metric,
        final_score=final_score,
        llm_score=llm_score,
        deterministic_score=deterministic_score,
        continuous_score=continuous_score,
        band=_safe_int(scores.get("band")),
        mode=str(scores.get("mode")) if scores.get("mode") is not None else None,
        warnings=tuple(str(item) for item in report.get("warnings") or ()),
        errors=tuple(str(item) for item in report.get("errors") or ()),
        error_type=None,
        execution_error=None,
        checks=_serialize_json_safe(checks),
        dimensions=_serialize_json_safe(dict(scores.get("dimensions") or {})),
        timings_ms=_serialize_json_safe(dict(report.get("timings_ms") or {})),
        llm_contract=_build_contract(report_input, config=config),
        raw_llm_rubric=_serialize_raw_llm(raw_llm),
        report=_serialize_json_safe(full_report) if full_report is not None else None,
    )


def evaluate_pair_expectation(
    expectation: CalibrationPairExpectation,
    *,
    cases_by_id: dict[str, EvaluatedCalibrationCase],
) -> EvaluatedPairExpectation:
    higher_case = cases_by_id.get(expectation.higher_case_id)
    lower_case = cases_by_id.get(expectation.lower_case_id)
    if higher_case is None or lower_case is None:
        return EvaluatedPairExpectation(
            higher_case_id=expectation.higher_case_id,
            lower_case_id=expectation.lower_case_id,
            status="missing_case",
            passed=None,
            metric=None,
            higher_score=None,
            lower_score=None,
            notes=expectation.notes,
        )
    if higher_case.status != "ok" or lower_case.status != "ok":
        return EvaluatedPairExpectation(
            higher_case_id=expectation.higher_case_id,
            lower_case_id=expectation.lower_case_id,
            status="missing_scores",
            passed=None,
            metric=None,
            higher_score=higher_case.comparison_score,
            lower_score=lower_case.comparison_score,
            notes=expectation.notes,
        )
    metric = higher_case.comparison_metric or lower_case.comparison_metric
    if higher_case.comparison_score is None or lower_case.comparison_score is None:
        return EvaluatedPairExpectation(
            higher_case_id=expectation.higher_case_id,
            lower_case_id=expectation.lower_case_id,
            status="missing_scores",
            passed=None,
            metric=metric,
            higher_score=higher_case.comparison_score,
            lower_score=lower_case.comparison_score,
            notes=expectation.notes,
        )
    return EvaluatedPairExpectation(
        higher_case_id=expectation.higher_case_id,
        lower_case_id=expectation.lower_case_id,
        status="ok",
        passed=higher_case.comparison_score > lower_case.comparison_score,
        metric=metric,
        higher_score=higher_case.comparison_score,
        lower_score=lower_case.comparison_score,
        notes=expectation.notes,
    )


def evaluate_calibration_manifest(
    manifest: CalibrationManifest,
    *,
    config: CalibrationRunConfig,
    runner: AssessmentRunner = run_assessment,
) -> EvaluatedCalibrationManifest:
    cases = tuple(
        evaluate_calibration_case(manifest, case, config=config, runner=runner)
        for case in manifest.active_cases
    )
    cases_by_id = {case.case_id: case for case in cases}
    pair_expectations = tuple(
        evaluate_pair_expectation(expectation, cases_by_id=cases_by_id)
        for expectation in manifest.active_pair_expectations
    )
    total_cases = len(cases)
    ok_cases = sum(1 for case in cases if case.status == "ok")
    success_ratio = 0.0 if total_cases == 0 else round(ok_cases / total_cases, 4)
    if total_cases == 0:
        run_status = "empty"
    elif ok_cases == 0:
        run_status = "failed"
    elif ok_cases < total_cases:
        run_status = "degraded"
    else:
        run_status = "ok"
    provider_slug = str(config.provider or "default-provider").replace("/", "-")
    model_slug = str(config.llm_model or "default-model").replace("/", "-")
    evaluation_id = f"{manifest.manifest_id}_{provider_slug}_{model_slug}_calibration_v1"
    return EvaluatedCalibrationManifest(
        evaluation_id=evaluation_id,
        manifest_id=manifest.manifest_id,
        language_code=manifest.language_code,
        language_profile_key=manifest.language_profile_key or config.language_profile_key,
        task_family=manifest.task_family,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_status=run_status,
        success_ratio=success_ratio,
        config=config,
        cases=cases,
        pair_expectations=pair_expectations,
    )


def _case_to_dict(case: EvaluatedCalibrationCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "status": case.status,
        "audio_path": case.audio_path.as_posix(),
        "expected_language": case.expected_language,
        "feedback_language": case.feedback_language,
        "theme": case.theme,
        "speaker_id": case.speaker_id,
        "expected_cefr": case.expected_cefr,
        "estimated_cefr": case.estimated_cefr,
        "cefr_delta": case.cefr_delta,
        "cefr_match": case.cefr_match,
        "comparison_score": case.comparison_score,
        "comparison_metric": case.comparison_metric,
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


def _case_from_dict(raw: dict[str, Any]) -> EvaluatedCalibrationCase:
    llm_contract_raw = dict(raw.get("llm_contract") or {})
    return EvaluatedCalibrationCase(
        case_id=str(raw["case_id"]),
        status=str(raw["status"]),
        audio_path=Path(str(raw["audio_path"])),
        expected_language=str(raw["expected_language"]),
        feedback_language=str(raw["feedback_language"]),
        theme=str(raw["theme"]),
        speaker_id=str(raw["speaker_id"]),
        expected_cefr=str(raw["expected_cefr"]) if raw.get("expected_cefr") else None,
        estimated_cefr=str(raw["estimated_cefr"]) if raw.get("estimated_cefr") else None,
        cefr_delta=int(raw["cefr_delta"]) if raw.get("cefr_delta") is not None else None,
        cefr_match=bool(raw["cefr_match"]) if raw.get("cefr_match") is not None else None,
        comparison_score=_safe_float(raw.get("comparison_score")),
        comparison_metric=(
            str(raw["comparison_metric"]) if raw.get("comparison_metric") is not None else None
        ),
        final_score=_safe_float(raw.get("final_score")),
        llm_score=_safe_float(raw.get("llm_score")),
        deterministic_score=_safe_float(raw.get("deterministic_score")),
        continuous_score=_safe_float(raw.get("continuous_score")),
        band=_safe_int(raw.get("band")),
        mode=str(raw["mode"]) if raw.get("mode") is not None else None,
        warnings=tuple(str(item) for item in raw.get("warnings") or ()),
        errors=tuple(str(item) for item in raw.get("errors") or ()),
        error_type=str(raw["error_type"]) if raw.get("error_type") else None,
        execution_error=str(raw["execution_error"]) if raw.get("execution_error") else None,
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
        raw_llm_rubric=str(raw["raw_llm_rubric"]) if raw.get("raw_llm_rubric") is not None else None,
        report=dict(raw.get("report") or {}) if raw.get("report") is not None else None,
    )


def _pair_to_dict(pair: EvaluatedPairExpectation) -> dict[str, Any]:
    return {
        "higher_case_id": pair.higher_case_id,
        "lower_case_id": pair.lower_case_id,
        "status": pair.status,
        "passed": pair.passed,
        "metric": pair.metric,
        "higher_score": pair.higher_score,
        "lower_score": pair.lower_score,
        "notes": pair.notes,
    }


def _pair_from_dict(raw: dict[str, Any]) -> EvaluatedPairExpectation:
    return EvaluatedPairExpectation(
        higher_case_id=str(raw["higher_case_id"]),
        lower_case_id=str(raw["lower_case_id"]),
        status=str(raw["status"]),
        passed=bool(raw["passed"]) if raw.get("passed") is not None else None,
        metric=str(raw["metric"]) if raw.get("metric") is not None else None,
        higher_score=_safe_float(raw.get("higher_score")),
        lower_score=_safe_float(raw.get("lower_score")),
        notes=raw.get("notes"),
    )


def write_calibration_evaluation_manifest(
    evaluation: EvaluatedCalibrationManifest,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_id": evaluation.evaluation_id,
        "manifest_id": evaluation.manifest_id,
        "schema_version": CALIBRATION_EVALUATION_SCHEMA_VERSION,
        "language_code": evaluation.language_code,
        "language_profile_key": evaluation.language_profile_key,
        "task_family": evaluation.task_family,
        "generated_at_utc": evaluation.generated_at_utc,
        "run_status": evaluation.run_status,
        "success_ratio": evaluation.success_ratio,
        "config": {
            "whisper_model": evaluation.config.whisper_model,
            "provider": evaluation.config.provider,
            "llm_model": evaluation.config.llm_model,
            "feedback_language": evaluation.config.feedback_language,
            "dry_run": evaluation.config.dry_run,
            "include_raw_llm": evaluation.config.include_raw_llm,
            "include_full_report": evaluation.config.include_full_report,
            "llm_timeout_sec": evaluation.config.llm_timeout_sec,
            "response_parser": evaluation.config.response_parser,
            "rubric_schema": evaluation.config.rubric_schema,
            "language_profile_key": evaluation.config.language_profile_key,
        },
        "summary": {
            "total_cases": len(evaluation.cases),
            "ok_cases": sum(1 for case in evaluation.cases if case.status == "ok"),
            "runner_error_cases": sum(1 for case in evaluation.cases if case.status == "runner_error"),
            "pair_expectations_total": len(evaluation.pair_expectations),
            "pair_expectations_passed": sum(1 for pair in evaluation.pair_expectations if pair.passed is True),
            "cefr_labeled_cases": sum(1 for case in evaluation.cases if case.expected_cefr is not None),
            "cefr_matches": sum(1 for case in evaluation.cases if case.cefr_match is True),
            "success_ratio": evaluation.success_ratio,
            "run_status": evaluation.run_status,
        },
        "cases": [_case_to_dict(case) for case in evaluation.cases],
        "pair_expectations": [_pair_to_dict(pair) for pair in evaluation.pair_expectations],
    }
    temp_output = output.with_suffix(f"{output.suffix}.tmp")
    temp_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_output.replace(output)
    return output


def load_calibration_evaluation_manifest(path: str | Path) -> EvaluatedCalibrationManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration evaluation manifest root payload must be an object")
    if int(payload.get("schema_version") or 0) != CALIBRATION_EVALUATION_SCHEMA_VERSION:
        raise ValueError("Calibration evaluation manifest schema_version does not match the current runner")
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Calibration evaluation manifest is missing a valid config object")
    cases_payload = payload.get("cases")
    if not isinstance(cases_payload, list):
        raise ValueError("Calibration evaluation manifest is missing a valid cases list")
    pair_payload = payload.get("pair_expectations")
    if pair_payload is None:
        pair_payload = []
    if not isinstance(pair_payload, list):
        raise ValueError("Calibration evaluation manifest pair_expectations must be a list")
    return EvaluatedCalibrationManifest(
        evaluation_id=str(payload["evaluation_id"]),
        manifest_id=str(payload["manifest_id"]),
        language_code=str(payload["language_code"]),
        language_profile_key=(
            str(payload["language_profile_key"]) if payload.get("language_profile_key") is not None else None
        ),
        task_family=str(payload["task_family"]),
        generated_at_utc=str(payload["generated_at_utc"]),
        run_status=str(payload["run_status"]),
        success_ratio=float(payload["success_ratio"]),
        config=CalibrationRunConfig(
            whisper_model=str(config_payload["whisper_model"]),
            provider=config_payload.get("provider"),
            llm_model=config_payload.get("llm_model"),
            feedback_language=config_payload.get("feedback_language"),
            dry_run=bool(config_payload.get("dry_run", False)),
            include_raw_llm=bool(config_payload.get("include_raw_llm", False)),
            include_full_report=bool(config_payload.get("include_full_report", False)),
            llm_timeout_sec=(
                float(config_payload["llm_timeout_sec"])
                if config_payload.get("llm_timeout_sec") is not None
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
        pair_expectations=tuple(_pair_from_dict(dict(item)) for item in pair_payload),
    )

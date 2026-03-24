"""Compare evaluated rendered-audio runs against benchmark expectations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from benchmarking.benchmark_suites import BenchmarkCase, BenchmarkSuite, load_benchmark_suite
from benchmarking.synthetic_benchmark_evaluation import (
    EvaluatedRenderedAudioSuite,
    EvaluatedRenderedCase,
    load_evaluation_manifest,
)


@dataclass(frozen=True)
class RegressionDimensionResult:
    dimension: str
    actual: float | None
    expected_range: tuple[float, float]
    passed: bool


@dataclass(frozen=True)
class RegressionCaseResult:
    evaluation_case_id: str
    benchmark_case_id: str | None
    benchmark_suite_id: str | None
    status: str
    passed: bool
    cefr_expected: str | None
    cefr_actual: str | None
    cefr_match: bool | None
    continuous_actual: float | None
    continuous_expected_range: tuple[float, float] | None
    continuous_passed: bool | None
    dimension_results: tuple[RegressionDimensionResult, ...]
    contract_match: bool | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RegressionSuiteResult:
    benchmark_suite_id: str
    evaluation_suite_id: str
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    missing_benchmark_refs: int
    case_results: tuple[RegressionCaseResult, ...]


def _in_range(value: float | None, expected_range: tuple[float, float]) -> bool:
    if value is None:
        return False
    low, high = expected_range
    return low <= value <= high


def _contract_match(benchmark_suite: BenchmarkSuite, evaluation_case: EvaluatedRenderedCase) -> bool | None:
    contract = evaluation_case.llm_contract
    benchmark_checks = [
        benchmark_suite.llm_contract.response_parser,
        benchmark_suite.llm_contract.rubric_schema,
        benchmark_suite.llm_contract.rubric_prompt_version,
        benchmark_suite.llm_contract.coaching_prompt_version,
        benchmark_suite.language_profile_key,
        benchmark_suite.scorer_version,
    ]
    if all(value is None for value in benchmark_checks):
        return None
    checks = [
        benchmark_suite.llm_contract.response_parser is None
        or benchmark_suite.llm_contract.response_parser == contract.response_parser,
        benchmark_suite.llm_contract.rubric_schema is None
        or benchmark_suite.llm_contract.rubric_schema == contract.rubric_schema,
        benchmark_suite.llm_contract.rubric_prompt_version is None
        or benchmark_suite.llm_contract.rubric_prompt_version == contract.rubric_prompt_version,
        benchmark_suite.llm_contract.coaching_prompt_version is None
        or benchmark_suite.llm_contract.coaching_prompt_version == contract.coaching_prompt_version,
        benchmark_suite.language_profile_key is None
        or benchmark_suite.language_profile_key == contract.language_profile_key,
        benchmark_suite.scorer_version is None
        or benchmark_suite.scorer_version == contract.language_profile_version,
    ]
    return all(checks)


def _compare_case(
    benchmark_suite: BenchmarkSuite,
    benchmark_case: BenchmarkCase | None,
    evaluation_case: EvaluatedRenderedCase,
) -> RegressionCaseResult:
    issues: list[str] = []
    if benchmark_case is None:
        return RegressionCaseResult(
            evaluation_case_id=evaluation_case.case_id,
            benchmark_case_id=evaluation_case.benchmark_case_id,
            benchmark_suite_id=evaluation_case.benchmark_suite_id,
            status=evaluation_case.status,
            passed=False,
            cefr_expected=None,
            cefr_actual=evaluation_case.estimated_cefr,
            cefr_match=None,
            continuous_actual=evaluation_case.continuous_score,
            continuous_expected_range=None,
            continuous_passed=None,
            dimension_results=(),
            contract_match=None,
            issues=("missing_benchmark_case_reference",),
        )

    dimension_results: list[RegressionDimensionResult] = []
    for dimension, expected_range in benchmark_case.expected.dimension_ranges.items():
        actual = None
        raw_dimension = (evaluation_case.dimensions or {}).get(dimension)
        if isinstance(raw_dimension, dict):
            try:
                actual = float(raw_dimension.get("score"))
            except (TypeError, ValueError):
                actual = None
                issues.append(f"dimension_malformed:{dimension}")
        elif isinstance(raw_dimension, (int, float)):
            actual = float(raw_dimension)
        elif raw_dimension is not None:
            issues.append(f"dimension_malformed:{dimension}")
        passed = _in_range(actual, expected_range)
        if not passed:
            issues.append(f"dimension_out_of_range:{dimension}")
        dimension_results.append(
            RegressionDimensionResult(
                dimension=dimension,
                actual=actual,
                expected_range=expected_range,
                passed=passed,
            )
        )

    expected_continuous = benchmark_case.expected.continuous_range
    continuous_passed = (
        _in_range(evaluation_case.continuous_score, expected_continuous)
        if expected_continuous is not None
        else None
    )
    if continuous_passed is False:
        issues.append("continuous_out_of_range")
    expected_cefr = benchmark_case.expected.cefr_level
    cefr_match = (evaluation_case.estimated_cefr == expected_cefr) if expected_cefr else None
    if cefr_match is False:
        issues.append("cefr_mismatch")
    contract_match = _contract_match(benchmark_suite, evaluation_case)
    if contract_match is False:
        issues.append("llm_contract_mismatch")
    if evaluation_case.status != "ok":
        issues.append(f"evaluation_status:{evaluation_case.status}")

    passed = (
        evaluation_case.status == "ok"
        and cefr_match is not False
        and continuous_passed is not False
        and all(result.passed for result in dimension_results)
        and contract_match is not False
    )
    return RegressionCaseResult(
        evaluation_case_id=evaluation_case.case_id,
        benchmark_case_id=benchmark_case.case_id,
        benchmark_suite_id=benchmark_suite.suite_id,
        status=evaluation_case.status,
        passed=passed,
        cefr_expected=expected_cefr,
        cefr_actual=evaluation_case.estimated_cefr,
        cefr_match=cefr_match,
        continuous_actual=evaluation_case.continuous_score,
        continuous_expected_range=expected_continuous,
        continuous_passed=continuous_passed,
        dimension_results=tuple(dimension_results),
        contract_match=contract_match,
        issues=tuple(issues),
    )


def compare_evaluation_against_benchmark(
    benchmark_suite: BenchmarkSuite,
    evaluation_suite: EvaluatedRenderedAudioSuite,
) -> RegressionSuiteResult:
    benchmark_cases = {case.case_id: case for case in benchmark_suite.cases}
    case_results: list[RegressionCaseResult] = []
    for evaluation_case in evaluation_suite.cases:
        if evaluation_case.benchmark_suite_id not in {None, benchmark_suite.suite_id}:
            case_results.append(
                RegressionCaseResult(
                    evaluation_case_id=evaluation_case.case_id,
                    benchmark_case_id=evaluation_case.benchmark_case_id,
                    benchmark_suite_id=evaluation_case.benchmark_suite_id,
                    status=evaluation_case.status,
                    passed=False,
                    cefr_expected=None,
                    cefr_actual=evaluation_case.estimated_cefr,
                    cefr_match=None,
                    continuous_actual=evaluation_case.continuous_score,
                    continuous_expected_range=None,
                    continuous_passed=None,
                    dimension_results=(),
                    contract_match=None,
                    issues=("benchmark_suite_mismatch",),
                )
            )
            continue
        benchmark_case = benchmark_cases.get(evaluation_case.benchmark_case_id)
        case_results.append(_compare_case(benchmark_suite, benchmark_case, evaluation_case))

    passed_cases = sum(1 for result in case_results if result.passed)
    skipped_cases = sum(1 for result in case_results if result.status == "skipped")
    failed_cases = sum(1 for result in case_results if not result.passed and result.status != "skipped")
    missing_refs = sum(1 for result in case_results if "missing_benchmark_case_reference" in result.issues)
    return RegressionSuiteResult(
        benchmark_suite_id=benchmark_suite.suite_id,
        evaluation_suite_id=evaluation_suite.suite_id,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        skipped_cases=skipped_cases,
        missing_benchmark_refs=missing_refs,
        case_results=tuple(case_results),
    )


def _case_result_to_dict(case: RegressionCaseResult) -> dict[str, Any]:
    return {
        "evaluation_case_id": case.evaluation_case_id,
        "benchmark_case_id": case.benchmark_case_id,
        "benchmark_suite_id": case.benchmark_suite_id,
        "status": case.status,
        "passed": case.passed,
        "cefr_expected": case.cefr_expected,
        "cefr_actual": case.cefr_actual,
        "cefr_match": case.cefr_match,
        "continuous_actual": case.continuous_actual,
        "continuous_expected_range": (
            list(case.continuous_expected_range)
            if case.continuous_expected_range is not None
            else None
        ),
        "continuous_passed": case.continuous_passed,
        "dimension_results": [
            {
                "dimension": result.dimension,
                "actual": result.actual,
                "expected_range": list(result.expected_range),
                "passed": result.passed,
            }
            for result in case.dimension_results
        ],
        "contract_match": case.contract_match,
        "issues": list(case.issues),
    }


def write_regression_report(result: RegressionSuiteResult, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_suite_id": result.benchmark_suite_id,
        "evaluation_suite_id": result.evaluation_suite_id,
        "summary": {
            "total_cases": len(result.case_results),
            "passed_cases": result.passed_cases,
            "failed_cases": result.failed_cases,
            "skipped_cases": result.skipped_cases,
            "missing_benchmark_refs": result.missing_benchmark_refs,
        },
        "cases": [_case_result_to_dict(case) for case in result.case_results],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_benchmark_and_evaluation(
    benchmark_suite_path: str | Path,
    evaluation_manifest_path: str | Path,
) -> tuple[BenchmarkSuite, EvaluatedRenderedAudioSuite]:
    return load_benchmark_suite(benchmark_suite_path), load_evaluation_manifest(evaluation_manifest_path)

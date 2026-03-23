"""Synthetic benchmark suite loading and evaluation for scoring regression tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from assess_core.language_profiles import default_language_profile_key, require_resolved_language_profile
from assess_core.schemas import RubricResult
from assessment_runtime.dimension_scoring import aggregate_dimension_scores, score_dimensions


@dataclass(frozen=True)
class BenchmarkExpectation:
    cefr_level: str
    continuous_range: tuple[float, float]
    dimension_ranges: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class BenchmarkLLMContract:
    rubric_prompt_version: str | None
    coaching_prompt_version: str | None
    response_parser: str | None
    rubric_schema: str | None
    notes: str | None


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    target_level: str
    metrics: dict[str, Any]
    checks: dict[str, Any]
    rubric: RubricResult
    detected_language_probability: float
    expected: BenchmarkExpectation
    active: bool
    tags: tuple[str, ...]
    notes: str | None


@dataclass(frozen=True)
class BenchmarkSuite:
    suite_id: str
    language_code: str
    language_profile_key: str | None
    task_family: str
    suite_type: str
    scorer_version: str
    llm_contract: BenchmarkLLMContract
    active: bool
    tags: tuple[str, ...]
    notes: str | None
    cases: tuple[BenchmarkCase, ...]


def _as_range_tuple(value: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two numeric values")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        raise ValueError(f"{field_name} lower bound must be <= upper bound")
    return (low, high)


def _as_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("tags must be a list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required_root_keys = {
        "suite_id",
        "language_code",
        "task_family",
        "suite_type",
        "scorer_version",
        "cases",
    }
    missing = required_root_keys - set(payload)
    if missing:
        raise ValueError(f"Benchmark suite is missing required keys: {sorted(missing)}")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ValueError("Benchmark suite must include at least one case")
    llm_contract_payload = payload.get("llm_contract", {})
    if llm_contract_payload is None:
        llm_contract_payload = {}
    if not isinstance(llm_contract_payload, dict):
        raise ValueError("llm_contract must be an object if present")

    cases: list[BenchmarkCase] = []
    for raw_case in payload["cases"]:
        required_case_keys = {
            "case_id",
            "target_level",
            "metrics",
            "checks",
            "rubric",
            "detected_language_probability",
            "expected",
        }
        missing_case = required_case_keys - set(raw_case)
        if missing_case:
            raise ValueError(f"Benchmark case is missing required keys: {sorted(missing_case)}")
        expected = raw_case["expected"]
        dimension_ranges = {
            key: _as_range_tuple(value, field_name=f"{raw_case['case_id']}.{key}")
            for key, value in expected["dimension_ranges"].items()
        }
        cases.append(
            BenchmarkCase(
                case_id=str(raw_case["case_id"]),
                target_level=str(raw_case["target_level"]),
                metrics=dict(raw_case["metrics"]),
                checks=dict(raw_case["checks"]),
                rubric=RubricResult.from_dict(dict(raw_case["rubric"])),
                detected_language_probability=float(raw_case["detected_language_probability"]),
                expected=BenchmarkExpectation(
                    cefr_level=str(expected["cefr_level"]),
                    continuous_range=_as_range_tuple(
                        expected["continuous_range"],
                        field_name=f"{raw_case['case_id']}.continuous_range",
                    ),
                    dimension_ranges=dimension_ranges,
                ),
                active=bool(raw_case.get("active", True)),
                tags=_as_tags(raw_case.get("tags")),
                notes=raw_case.get("notes"),
            )
        )

    language_code = str(payload["language_code"])
    language_profile_key = payload.get("language_profile_key")
    if language_profile_key is not None:
        language_profile_key = str(language_profile_key).strip() or None
    if language_profile_key is None:
        language_profile_key = default_language_profile_key(language_code)

    return BenchmarkSuite(
        suite_id=str(payload["suite_id"]),
        language_code=language_code,
        language_profile_key=language_profile_key,
        task_family=str(payload["task_family"]),
        suite_type=str(payload["suite_type"]),
        scorer_version=str(payload["scorer_version"]),
        llm_contract=BenchmarkLLMContract(
            rubric_prompt_version=llm_contract_payload.get("rubric_prompt_version"),
            coaching_prompt_version=llm_contract_payload.get("coaching_prompt_version"),
            response_parser=llm_contract_payload.get("response_parser"),
            rubric_schema=llm_contract_payload.get("rubric_schema"),
            notes=llm_contract_payload.get("notes"),
        ),
        active=bool(payload.get("active", True)),
        tags=_as_tags(payload.get("tags")),
        notes=payload.get("notes"),
        cases=tuple(cases),
    )


def discover_benchmark_suites(
    root: str | Path,
    *,
    include_inactive: bool = False,
    language_codes: set[str] | None = None,
    suite_types: set[str] | None = None,
    tags: set[str] | None = None,
    tag_match: str = "any",
) -> tuple[BenchmarkSuite, ...]:
    root_path = Path(root)
    if tag_match not in {"any", "all"}:
        raise ValueError("tag_match must be 'any' or 'all'")
    suites: list[BenchmarkSuite] = []
    for path in sorted(root_path.glob("*.json")):
        suite = load_benchmark_suite(path)
        if not include_inactive and not suite.active:
            continue
        if language_codes and suite.language_code not in language_codes:
            continue
        if suite_types and suite.suite_type not in suite_types:
            continue
        if tags:
            suite_tags = set(suite.tags)
            if tag_match == "any" and not suite_tags.intersection(tags):
                continue
            if tag_match == "all" and not tags.issubset(suite_tags):
                continue
        suites.append(suite)
    return tuple(suites)


def evaluate_benchmark_case(
    case: BenchmarkCase,
    *,
    language_code: str,
    language_profile_key: str | None = None,
) -> dict[str, Any]:
    profile = require_resolved_language_profile(language_code, profile_key=language_profile_key)
    dimensions = score_dimensions(
        metrics=case.metrics,
        rubric=case.rubric,
        checks=case.checks,
        profile=profile,
        detected_language_probability=case.detected_language_probability,
    )
    cefr_estimate = {**aggregate_dimension_scores(dimensions, profile=profile), "language": profile.code}
    return {
        "dimensions": dimensions,
        "cefr_estimate": cefr_estimate,
    }

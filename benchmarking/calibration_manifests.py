"""Manifest loading and discovery for real-audio calibration and shadow evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from assess_core.language_profiles import default_language_profile_key


VALID_CEFR_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    audio_path: Path
    expected_language: str
    theme: str
    speaker_id: str
    target_duration_sec: float | None
    expected_cefr: str | None
    active: bool
    tags: tuple[str, ...]
    notes: str | None


@dataclass(frozen=True)
class CalibrationPairExpectation:
    higher_case_id: str
    lower_case_id: str
    active: bool
    notes: str | None


@dataclass(frozen=True)
class CalibrationManifest:
    manifest_id: str
    language_code: str
    language_profile_key: str | None
    task_family: str
    version: str
    active: bool
    tags: tuple[str, ...]
    notes: str | None
    cases: tuple[CalibrationCase, ...]
    pair_expectations: tuple[CalibrationPairExpectation, ...]

    @property
    def active_cases(self) -> tuple[CalibrationCase, ...]:
        return tuple(case for case in self.cases if case.active)

    @property
    def active_pair_expectations(self) -> tuple[CalibrationPairExpectation, ...]:
        return tuple(expectation for expectation in self.pair_expectations if expectation.active)


def _as_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("tags must be a list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _as_positive_float_or_none(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def _resolve_audio_path(path_value: Any, *, manifest_path: Path, case_id: str) -> Path:
    candidate = Path(str(path_value))
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"{case_id}.audio_path does not exist: {candidate}")
    return candidate


def load_calibration_manifest(path: str | Path) -> CalibrationManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration manifest root payload must be an object")

    required_root_keys = {
        "manifest_id",
        "language_code",
        "task_family",
        "version",
        "cases",
    }
    missing = required_root_keys - set(payload)
    if missing:
        raise ValueError(f"Calibration manifest is missing required keys: {sorted(missing)}")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ValueError("Calibration manifest must include at least one case")

    language_code = str(payload["language_code"]).strip().lower()
    language_profile_key = payload.get("language_profile_key")
    if language_profile_key is not None:
        language_profile_key = str(language_profile_key).strip() or None
    if language_profile_key is None:
        language_profile_key = default_language_profile_key(language_code)

    cases: list[CalibrationCase] = []
    seen_case_ids: set[str] = set()
    for raw_case in payload["cases"]:
        required_case_keys = {
            "case_id",
            "audio_path",
            "theme",
            "speaker_id",
        }
        missing_case = required_case_keys - set(raw_case)
        if missing_case:
            raise ValueError(f"Calibration case is missing required keys: {sorted(missing_case)}")
        case_id = str(raw_case["case_id"])
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate calibration case_id: {case_id}")
        seen_case_ids.add(case_id)

        expected_language = str(raw_case.get("expected_language") or language_code).strip().lower()
        expected_cefr = raw_case.get("expected_cefr")
        if expected_cefr is not None:
            expected_cefr = str(expected_cefr).strip().upper() or None
        if expected_cefr and expected_cefr not in VALID_CEFR_LEVELS:
            raise ValueError(f"{case_id}.expected_cefr must be one of {sorted(VALID_CEFR_LEVELS)}")

        cases.append(
            CalibrationCase(
                case_id=case_id,
                audio_path=_resolve_audio_path(
                    raw_case["audio_path"],
                    manifest_path=manifest_path,
                    case_id=case_id,
                ),
                expected_language=expected_language,
                theme=str(raw_case["theme"]),
                speaker_id=str(raw_case["speaker_id"]),
                target_duration_sec=_as_positive_float_or_none(
                    raw_case.get("target_duration_sec"),
                    field_name=f"{case_id}.target_duration_sec",
                ),
                expected_cefr=expected_cefr,
                active=bool(raw_case.get("active", True)),
                tags=_as_tags(raw_case.get("tags")),
                notes=raw_case.get("notes"),
            )
        )

    case_ids = {case.case_id for case in cases}
    pair_expectations: list[CalibrationPairExpectation] = []
    for raw_expectation in payload.get("pair_expectations") or []:
        if not isinstance(raw_expectation, dict):
            raise ValueError("pair_expectations entries must be objects")
        higher_case_id = str(raw_expectation.get("higher_case_id") or "").strip()
        lower_case_id = str(raw_expectation.get("lower_case_id") or "").strip()
        if not higher_case_id or not lower_case_id:
            raise ValueError("pair_expectations entries must include higher_case_id and lower_case_id")
        if higher_case_id not in case_ids:
            raise ValueError(f"pair_expectations references unknown higher_case_id: {higher_case_id}")
        if lower_case_id not in case_ids:
            raise ValueError(f"pair_expectations references unknown lower_case_id: {lower_case_id}")
        if higher_case_id == lower_case_id:
            raise ValueError("pair_expectations cannot compare a case against itself")
        pair_expectations.append(
            CalibrationPairExpectation(
                higher_case_id=higher_case_id,
                lower_case_id=lower_case_id,
                active=bool(raw_expectation.get("active", True)),
                notes=raw_expectation.get("notes"),
            )
        )

    return CalibrationManifest(
        manifest_id=str(payload["manifest_id"]),
        language_code=language_code,
        language_profile_key=language_profile_key,
        task_family=str(payload["task_family"]),
        version=str(payload["version"]),
        active=bool(payload.get("active", True)),
        tags=_as_tags(payload.get("tags")),
        notes=payload.get("notes"),
        cases=tuple(cases),
        pair_expectations=tuple(pair_expectations),
    )


def discover_calibration_manifests(
    root: str | Path,
    *,
    include_inactive: bool = False,
    language_codes: set[str] | None = None,
    tags: set[str] | None = None,
    tag_match: str = "any",
) -> tuple[CalibrationManifest, ...]:
    root_path = Path(root)
    if tag_match not in {"any", "all"}:
        raise ValueError("tag_match must be 'any' or 'all'")
    manifests: list[CalibrationManifest] = []
    for path in sorted(root_path.glob("*.json")):
        manifest = load_calibration_manifest(path)
        if not include_inactive and not manifest.active:
            continue
        if language_codes and manifest.language_code not in language_codes:
            continue
        if tags:
            manifest_tags = set(manifest.tags)
            if tag_match == "any" and not manifest_tags.intersection(tags):
                continue
            if tag_match == "all" and not tags.issubset(manifest_tags):
                continue
        manifests.append(manifest)
    return tuple(manifests)

"""Strict schema validation for rubric and final reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


class SchemaValidationError(ValueError):
    pass


def _require_keys(data: dict, required: set[str], scope: str) -> None:
    missing = sorted(required - set(data.keys()))
    if missing:
        raise SchemaValidationError(f"{scope}: missing keys: {', '.join(missing)}")


def _score_value(raw: Any, key: str) -> int:
    if isinstance(raw, bool):
        raise SchemaValidationError(f"{key}: must be integer 1..5")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float) and raw.is_integer():
        value = int(raw)
    else:
        raise SchemaValidationError(f"{key}: must be integer 1..5")
    if value < 1 or value > 5:
        raise SchemaValidationError(f"{key}: out of range 1..5")
    return value


def _bool_value(raw: Any, key: str) -> bool:
    if isinstance(raw, bool):
        return raw
    raise SchemaValidationError(f"{key}: must be boolean")


def _str_value(raw: Any, key: str) -> str:
    if isinstance(raw, str):
        return raw
    raise SchemaValidationError(f"{key}: must be string")


@dataclass(frozen=True)
class RubricResult:
    fluency: int
    cohesion: int
    accuracy: int
    range: int
    overall: int
    comments_fluency: str
    comments_cohesion: str
    comments_accuracy: str
    comments_range: str
    overall_comment: str
    on_topic: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RubricResult":
        required = {
            "fluency",
            "cohesion",
            "accuracy",
            "range",
            "overall",
            "comments_fluency",
            "comments_cohesion",
            "comments_accuracy",
            "comments_range",
            "overall_comment",
            "on_topic",
        }
        _require_keys(data, required, "RubricResult")
        return cls(
            fluency=_score_value(data["fluency"], "fluency"),
            cohesion=_score_value(data["cohesion"], "cohesion"),
            accuracy=_score_value(data["accuracy"], "accuracy"),
            range=_score_value(data["range"], "range"),
            overall=_score_value(data["overall"], "overall"),
            comments_fluency=_str_value(data["comments_fluency"], "comments_fluency"),
            comments_cohesion=_str_value(data["comments_cohesion"], "comments_cohesion"),
            comments_accuracy=_str_value(data["comments_accuracy"], "comments_accuracy"),
            comments_range=_str_value(data["comments_range"], "comments_range"),
            overall_comment=_str_value(data["overall_comment"], "overall_comment"),
            on_topic=_bool_value(data["on_topic"], "on_topic"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssessmentReport:
    timestamp_utc: str
    input: dict[str, Any]
    metrics: dict[str, Any]
    checks: dict[str, Any]
    scores: dict[str, Any]
    requires_human_review: bool
    transcript_preview: str
    warnings: list[str]
    errors: list[str]
    rubric: dict[str, Any] | None = None
    suggested_training: list[dict[str, Any]] | None = None
    timings_ms: dict[str, Any] | None = None

    @classmethod
    def now_timestamp(cls) -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssessmentReport":
        required = {
            "timestamp_utc",
            "input",
            "metrics",
            "checks",
            "scores",
            "requires_human_review",
            "transcript_preview",
            "warnings",
            "errors",
        }
        _require_keys(data, required, "AssessmentReport")
        if not isinstance(data["input"], dict):
            raise SchemaValidationError("input: must be object")
        if not isinstance(data["metrics"], dict):
            raise SchemaValidationError("metrics: must be object")
        if not isinstance(data["checks"], dict):
            raise SchemaValidationError("checks: must be object")
        if not isinstance(data["scores"], dict):
            raise SchemaValidationError("scores: must be object")
        if not isinstance(data["requires_human_review"], bool):
            raise SchemaValidationError("requires_human_review: must be boolean")
        if not isinstance(data["warnings"], list):
            raise SchemaValidationError("warnings: must be list")
        if not isinstance(data["errors"], list):
            raise SchemaValidationError("errors: must be list")
        if not isinstance(data["transcript_preview"], str):
            raise SchemaValidationError("transcript_preview: must be string")

        metric_required = {
            "duration_sec",
            "pause_count",
            "pause_total_sec",
            "speaking_time_sec",
            "word_count",
            "wpm",
            "fillers",
            "cohesion_markers",
            "complexity_index",
        }
        _require_keys(data["metrics"], metric_required, "metrics")
        _require_keys(data["checks"], {"duration_pass", "topic_pass", "min_words_pass", "language_pass"}, "checks")
        _require_keys(data["scores"], {"deterministic", "llm", "final", "band", "mode"}, "scores")

        rubric_dict = data.get("rubric")
        if rubric_dict is not None:
            if not isinstance(rubric_dict, dict):
                raise SchemaValidationError("rubric: must be object when present")
            RubricResult.from_dict(rubric_dict)

        suggestions = data.get("suggested_training")
        if suggestions is not None and not isinstance(suggestions, list):
            raise SchemaValidationError("suggested_training: must be list when present")
        timings_ms = data.get("timings_ms")
        if timings_ms is not None and not isinstance(timings_ms, dict):
            raise SchemaValidationError("timings_ms: must be object when present")

        return cls(
            timestamp_utc=data["timestamp_utc"],
            input=data["input"],
            metrics=data["metrics"],
            checks=data["checks"],
            scores=data["scores"],
            requires_human_review=data["requires_human_review"],
            transcript_preview=data["transcript_preview"],
            warnings=[str(value) for value in data["warnings"]],
            errors=[str(value) for value in data["errors"]],
            rubric=rubric_dict,
            suggested_training=suggestions,
            timings_ms=timings_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

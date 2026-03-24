"""Strict schema validation for rubric and final reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from assess_core.coaching_taxonomy import (
    COACHING_CONFIDENCE_LEVELS,
    COHERENCE_ISSUE_CATEGORIES,
    GRAMMAR_ERROR_CATEGORIES,
    LEXICAL_GAP_CATEGORIES,
)

REPORT_SCHEMA_VERSION = 2


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


def _int_value(raw: Any, key: str, *, minimum: int | None = None) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SchemaValidationError(f"{key}: must be integer")
    if minimum is not None and raw < minimum:
        raise SchemaValidationError(f"{key}: must be >= {minimum}")
    return raw


def _str_list(raw: Any, key: str) -> list[str]:
    if not isinstance(raw, list):
        raise SchemaValidationError(f"{key}: must be list")
    values: list[str] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, str):
            raise SchemaValidationError(f"{key}[{idx}]: must be string")
        values.append(item)
    return values


def _enum_str(raw: Any, key: str, allowed: tuple[str, ...]) -> str:
    value = _str_value(raw, key)
    if value not in allowed:
        raise SchemaValidationError(f"{key}: invalid value '{value}'")
    return value


@dataclass(frozen=True)
class CategorizedIssue:
    category: str
    explanation: str
    examples: list[str]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        key: str,
        allowed_categories: tuple[str, ...],
    ) -> "CategorizedIssue":
        _require_keys(data, {"category", "explanation", "examples"}, key)
        category = _enum_str(data["category"], f"{key}.category", allowed_categories)
        explanation = _str_value(data["explanation"], f"{key}.explanation")
        examples = _str_list(data["examples"], f"{key}.examples")
        return cls(category=category, explanation=explanation, examples=examples)


def _issue_list(raw: Any, key: str, allowed_categories: tuple[str, ...]) -> list[CategorizedIssue]:
    if not isinstance(raw, list):
        raise SchemaValidationError(f"{key}: must be list")
    issues: list[CategorizedIssue] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SchemaValidationError(f"{key}[{idx}]: must be object")
        issues.append(
            CategorizedIssue.from_dict(
                item,
                key=f"{key}[{idx}]",
                allowed_categories=allowed_categories,
            )
        )
    return issues


@dataclass(frozen=True)
class CoachingSummary:
    strengths: list[str]
    top_3_priorities: list[str]
    next_focus: str
    next_exercise: str
    coach_summary: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoachingSummary":
        _require_keys(
            data,
            {"strengths", "top_3_priorities", "next_focus", "next_exercise", "coach_summary"},
            "CoachingSummary",
        )
        strengths = _str_list(data["strengths"], "strengths")
        priorities = _str_list(data["top_3_priorities"], "top_3_priorities")
        if len(priorities) != 3:
            raise SchemaValidationError("top_3_priorities: must contain exactly 3 items")
        return cls(
            strengths=strengths,
            top_3_priorities=priorities,
            next_focus=_str_value(data["next_focus"], "next_focus"),
            next_exercise=_str_value(data["next_exercise"], "next_exercise"),
            coach_summary=_str_value(data["coach_summary"], "coach_summary"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    topic_relevance_score: int = 3
    language_ok: bool = True
    recurring_grammar_errors: list[CategorizedIssue] = field(default_factory=list)
    coherence_issues: list[CategorizedIssue] = field(default_factory=list)
    lexical_gaps: list[CategorizedIssue] = field(default_factory=list)
    evidence_quotes: list[str] = field(default_factory=list)
    confidence: str = "medium"

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
            "topic_relevance_score",
            "language_ok",
            "recurring_grammar_errors",
            "coherence_issues",
            "lexical_gaps",
            "evidence_quotes",
            "confidence",
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
            topic_relevance_score=_score_value(data["topic_relevance_score"], "topic_relevance_score"),
            language_ok=_bool_value(data["language_ok"], "language_ok"),
            recurring_grammar_errors=_issue_list(
                data["recurring_grammar_errors"],
                "recurring_grammar_errors",
                GRAMMAR_ERROR_CATEGORIES,
            ),
            coherence_issues=_issue_list(
                data["coherence_issues"],
                "coherence_issues",
                COHERENCE_ISSUE_CATEGORIES,
            ),
            lexical_gaps=_issue_list(
                data["lexical_gaps"],
                "lexical_gaps",
                LEXICAL_GAP_CATEGORIES,
            ),
            evidence_quotes=_str_list(data["evidence_quotes"], "evidence_quotes"),
            confidence=_enum_str(data["confidence"], "confidence", COACHING_CONFIDENCE_LEVELS),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssessmentReport:
    schema_version: int
    session_id: str
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
    coaching: dict[str, Any] | None = None
    progress_delta: dict[str, Any] | None = None
    suggested_training: list[dict[str, Any]] | None = None
    timings_ms: dict[str, Any] | None = None

    @classmethod
    def now_timestamp(cls) -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssessmentReport":
        required = {
            "schema_version",
            "session_id",
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
        schema_version = _int_value(data["schema_version"], "schema_version", minimum=1)
        session_id = _str_value(data["session_id"], "session_id")
        if not session_id.strip():
            raise SchemaValidationError("session_id: must be non-empty string")
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
        coaching = data.get("coaching")
        if coaching is not None:
            if not isinstance(coaching, dict):
                raise SchemaValidationError("coaching: must be object when present")
            CoachingSummary.from_dict(coaching)
        progress_delta = data.get("progress_delta")
        if progress_delta is not None and not isinstance(progress_delta, dict):
            raise SchemaValidationError("progress_delta: must be object when present")
        timings_ms = data.get("timings_ms")
        if timings_ms is not None and not isinstance(timings_ms, dict):
            raise SchemaValidationError("timings_ms: must be object when present")

        return cls(
            schema_version=schema_version,
            session_id=session_id,
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
            coaching=coaching,
            progress_delta=progress_delta,
            suggested_training=suggestions,
            timings_ms=timings_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

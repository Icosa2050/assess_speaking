"""Match saved assessment provenance using the same contract as History's comparisonKey."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def comparison_key(report: dict, practice: dict | None) -> tuple | None:
    if not isinstance(report, dict) or not isinstance(practice, dict):
        return None
    context = practice
    inputs = report.get("input") or {}
    if not isinstance(inputs, dict):
        return None
    identity = [inputs.get("speaker_id"), inputs.get("learning_language") or inputs.get("expected_language"), inputs.get("task_family")]
    required = ("goal", "scoring_version", "analysis_signature", "provider", "model", "whisper_model")
    if (context.get("version") != 1 or context.get("dry_run") or not all(identity)
            or not all(context.get(key) for key in required)
            or context.get("scoring_mode") not in {"hybrid", "deterministic_only"}
            or not report.get("session_id")):
        return None
    duration = context.get("target_duration_sec")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not 0 < duration < float("inf"):
        return None
    return tuple(identity + [context["goal"], duration, context["version"],
        context["scoring_version"], context["scoring_mode"], context["analysis_signature"],
        context["provider"], context["model"], context.get("asr_provider"), context["whisper_model"]])


def comparable_history_rows(history_path: Path, report: dict, practice: dict | None,
                            rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key = comparison_key(report, practice)
    if key is None:
        return []
    matching = []
    retry_parent = (practice or {}).get("retry_of_session_id")
    for row in rows:
        try:
            if not isinstance(row, dict):
                continue
            path = Path(row.get("report_path") or "").resolve()
            # Only saved reports from this isolated history directory are eligible.
            if not path.is_relative_to(history_path.parent.resolve()) or not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("report"), dict) or not isinstance(payload.get("meta"), dict):
                continue
            prior_report = payload["report"]
            prior_practice = payload["meta"]["practice"]
            if prior_report.get("session_id") != row.get("session_id"):
                continue
            if comparison_key(prior_report, prior_practice) != key:
                continue
            if retry_parent and (row.get("session_id") != retry_parent or
                                 prior_practice.get("prompt_text") != (practice or {}).get("prompt_text")):
                continue
            matching.append(row)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(matching, key=lambda row: str(row.get("timestamp", "")))

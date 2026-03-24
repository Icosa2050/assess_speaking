"""Cross-session progress analysis helpers."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, Optional


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_records(
    records: Iterable[object],
    *,
    speaker_id: Optional[str] = None,
    task_family: Optional[str] = None,
) -> list[object]:
    filtered = []
    for record in records:
        if speaker_id and getattr(record, "speaker_id", "") != speaker_id:
            continue
        if task_family and getattr(record, "task_family", "") != task_family:
            continue
        filtered.append(record)
    return filtered


def recurring_issue_counts(records: Iterable[object], attribute: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for category in getattr(record, attribute, ()):
            if category:
                counts[category] += 1
    return counts


def latest_priorities(records: Iterable[object]) -> dict:
    rows = list(records)
    if not rows:
        return {
            "latest": [],
            "previous": [],
            "new": [],
            "resolved": [],
        }

    latest = list(getattr(rows[-1], "top_priorities", ()))
    previous = list(getattr(rows[-2], "top_priorities", ())) if len(rows) > 1 else []
    return {
        "latest": latest,
        "previous": previous,
        "new": [item for item in latest if item not in previous],
        "resolved": [item for item in previous if item not in latest],
    }


def group_by_task_family(records: Iterable[object], *, speaker_id: Optional[str] = None) -> dict[str, list[object]]:
    groups: dict[str, list[object]] = {}
    for record in filter_records(records, speaker_id=speaker_id):
        family = getattr(record, "task_family", "") or "unclassified"
        groups.setdefault(family, []).append(record)
    return groups


def task_family_progress(records: Iterable[object], *, speaker_id: Optional[str] = None) -> list[dict]:
    rows = []
    for family, family_records in sorted(group_by_task_family(records, speaker_id=speaker_id).items()):
        final_scores = [
            value
            for record in family_records
            if (value := _safe_float(getattr(record, "final_score", None))) is not None
        ]
        latest = family_records[-1]
        grammar = recurring_issue_counts(family_records, "grammar_error_categories")
        coherence = recurring_issue_counts(family_records, "coherence_issue_categories")
        rows.append(
            {
                "task_family": family,
                "count": len(family_records),
                "avg_final": round(mean(final_scores), 2) if final_scores else None,
                "latest_final": _safe_float(getattr(latest, "final_score", None)),
                "latest_overall": _safe_float(getattr(latest, "overall", None)),
                "latest_priorities": list(getattr(latest, "top_priorities", ())),
                "grammar_counts": grammar,
                "coherence_counts": coherence,
            }
        )
    return rows


def format_top_counts(counter: Counter[str] | dict[str, int], limit: int = 3) -> str:
    if not counter:
        return "–"
    normalized = counter if isinstance(counter, Counter) else Counter(counter)
    parts = [f"{name} ({count})" for name, count in normalized.most_common(limit)]
    return ", ".join(parts)

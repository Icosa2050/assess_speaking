"""Feedback generation based on assessment metrics.

The module loads a manifest file (JSON) that describes training resources and
maps metric gaps to those resources. It provides a single public function
``generate_feedback`` that returns a list of suggestions suitable for inclusion
in the final JSON report.

The manifest format (example)::

    {
        "schema_version": "1.0",
        "resources": [
            {
                "id": "filler-drill",
                "title": "Filler reduction exercises",
                "url": "https://example.com/filler.pdf",
                "metrics": ["fillers"]
            },
            {
                "id": "pace-drill",
                "title": "Pacing practice",
                "url": "https://example.com/pace.html",
                "metrics": ["wpm"]
            }
        ]
    }

Each resource lists the metric keys it addresses. ``generate_feedback`` checks a
hand‑crafted set of simple thresholds and, for each failing metric, returns all
resources that target that metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------

def load_manifest(train_dir: Path) -> Dict[str, Any]:
    """Load ``manifest.json`` from ``train_dir``.

    The function validates that the file exists and that the top‑level keys
    ``schema_version`` and ``resources`` are present. A ``RuntimeError`` is raised
    with a helpful message if validation fails.
    """
    manifest_path = train_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Training manifest not found at {manifest_path}. "
            "Create a manifest.json describing your resources."
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in manifest: {exc}") from exc

    # Minimal schema validation
    if not isinstance(raw, dict) or "resources" not in raw:
        raise RuntimeError("Manifest must be a JSON object with a 'resources' list.")
    if not isinstance(raw["resources"], list):
        raise RuntimeError("'resources' in manifest must be a list.")
    return raw


# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------

# Simple thresholds – these can be tweaked later or made configurable via the
# manifest if needed.
TARGET_WPM = 120.0
MAX_FILLER_RATIO = 0.05  # 5 % of total words
MIN_COHESION_MARKERS = 1
MAX_COMPLEXITY_INDEX = 5


def _metric_failures(metrics: Dict[str, Any]) -> List[str]:
    """Return a list of metric keys that do not meet the thresholds.

    The keys correspond to the ``metrics`` entries used in the manifest.
    """
    failures: List[str] = []
    # words per minute
    if metrics.get("wpm", 0) < TARGET_WPM:
        failures.append("wpm")
    # filler ratio
    word_cnt = metrics.get("word_count", 0) or 1
    if metrics.get("fillers", 0) / word_cnt > MAX_FILLER_RATIO:
        failures.append("fillers")
    # cohesion markers
    if metrics.get("cohesion_markers", 0) < MIN_COHESION_MARKERS:
        failures.append("cohesion")
    # complexity index (higher values indicate more complex constructions; we
    # treat very high values as a potential issue for learners at lower levels)
    if metrics.get("complexity_index", 0) > MAX_COMPLEXITY_INDEX:
        failures.append("complexity")
    return failures


def generate_feedback(metrics: Dict[str, Any], train_dir: Path) -> List[Dict[str, str]]:
    """Generate a list of resource suggestions based on ``metrics``.

    ``train_dir`` points to the directory that contains ``manifest.json``.
    The return value is a list of dictionaries with ``id``, ``title`` and ``url``
    fields, plus a human‑readable ``reason`` explaining why the resource was
    chosen.
    """
    manifest = load_manifest(train_dir)
    failures = _metric_failures(metrics)
    if not failures:
        return []

    suggestions: List[Dict[str, str]] = []
    for res in manifest.get("resources", []):
        # each resource declares which metric keys it addresses
        res_metrics = res.get("metrics", [])
        intersect = set(failures).intersection(set(res_metrics))
        if intersect:
            suggestions.append(
                {
                    "id": res.get("id", ""),
                    "title": res.get("title", ""),
                    "url": res.get("url", ""),
                    "reason": f"Addresses metric(s): {', '.join(sorted(intersect))}",
                }
            )
    return suggestions


def build_fallback_coaching(
    *,
    metrics: Dict[str, Any],
    checks: Dict[str, Any],
    theme: str,
    target_duration_sec: float,
) -> Dict[str, Any]:
    strengths: List[str] = []
    priorities: List[str] = []

    if checks.get("language_pass"):
        strengths.append("Hai parlato nella lingua richiesta.")
    if checks.get("topic_pass"):
        strengths.append("Sei rimasto sul tema richiesto.")
    if checks.get("duration_pass"):
        strengths.append("Hai sostenuto una durata vicina all'obiettivo.")
    if metrics.get("fillers", 0) <= 2:
        strengths.append("L'uso di filler è contenuto.")
    if metrics.get("cohesion_markers", 0) >= 2:
        strengths.append("Ci sono segnali di collegamento tra le idee.")
    if not strengths:
        strengths.append("Hai completato una registrazione utile per il confronto nel tempo.")

    if not checks.get("duration_pass"):
        priorities.append(f"Parla in modo più continuo fino a circa {int(target_duration_sec)} secondi.")
    if not checks.get("topic_pass"):
        priorities.append(f"Resta più vicino al tema '{theme}'.")
    word_count = max(1, int(metrics.get("word_count", 0)))
    if metrics.get("fillers", 0) / word_count > MAX_FILLER_RATIO:
        priorities.append("Riduci filler come 'eh' o 'allora' e sostituiscili con pause silenziose.")
    if metrics.get("cohesion_markers", 0) < MIN_COHESION_MARKERS:
        priorities.append("Usa più connettivi per ordinare meglio il racconto.")
    if metrics.get("wpm", 0) < TARGET_WPM:
        priorities.append("Aumenta leggermente il ritmo senza sacrificare la chiarezza.")
    if metrics.get("complexity_index", 0) < 1:
        priorities.append("Aggiungi almeno una frase più articolata per collegare causa o tempo.")

    if not priorities:
        priorities.append("Mantieni la stessa struttura ma aggiungi più dettagli concreti.")
    while len(priorities) < 3:
        priorities.append("Ripeti lo stesso compito puntando a una narrazione più chiara e lineare.")
    priorities = priorities[:3]

    primary = priorities[0]
    next_exercise = f"Registra di nuovo il tema '{theme}' e concentrati su: {primary.lower()}"
    return {
        "strengths": strengths[:3],
        "top_3_priorities": priorities,
        "next_focus": primary,
        "next_exercise": next_exercise,
        "coach_summary": (
            f"Punto di partenza utile. Nel prossimo tentativo concentrati soprattutto su: "
            f"{priorities[0]} Poi lavora su: {priorities[1]} e {priorities[2]}"
        ),
    }

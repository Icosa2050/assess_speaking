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
import re
from typing import List, Dict, Any

from .assessment_prompts import localized_language_name, normalize_language_code

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
    ui_locale: str = "it",
    learning_language: str = "it",
    transcript: str = "",
    detected_language: str | None = None,
) -> Dict[str, Any]:
    locale = normalize_language_code(ui_locale, fallback="en")
    learning_language_name = localized_language_name(learning_language, locale=locale)
    safe_detected_language = str(detected_language or "").strip() or learning_language
    detected_language_name = localized_language_name(safe_detected_language, locale=locale)
    strengths: List[str] = []
    priorities: List[str] = []
    word_count = max(1, int(metrics.get("word_count", 0)))
    topic_gate = checks.get("topic_pass")
    transcript_excerpt = _transcript_excerpt(transcript)
    very_short_response = word_count <= 5

    text = {
        "de": {
            "strength_language": "Du hast in der geforderten Sprache gesprochen.",
            "strength_theme": "Du bist beim vorgegebenen Thema geblieben.",
            "strength_duration": "Du warst nahe an der Zielzeit.",
            "strength_fillers": "Du hast nur wenige Fuellwoerter benutzt.",
            "strength_cohesion": "Es gibt klare Verknuepfungen zwischen deinen Ideen.",
            "strength_default": "Diese Aufnahme ist ein nuetzlicher Ausgangspunkt fuer den Vergleich ueber die Zeit.",
            "priority_short_response": "Antworte mit mehreren zusammenhaengenden Saetzen statt nur mit wenigen einzelnen Woertern.",
            "priority_short_wrong_language": "Statt nur '{excerpt}' aufzunehmen, gib eine laengere Antwort in {language}, die das Thema wirklich bearbeitet.",
            "priority_language": f"Bearbeite die ganze Aufgabe in {learning_language_name}, auch wenn du dabei etwas langsamer sprichst.",
            "priority_duration": f"Sprich gleichmaessiger weiter, bis du etwa {int(target_duration_sec)} Sekunden erreichst.",
            "priority_theme": f"Bleib naeher am Thema '{theme}'.",
            "priority_fillers": "Reduziere Fuellwoerter wie 'aehm' und ersetze sie durch kurze stille Pausen.",
            "priority_cohesion": "Nutze mehr Verbindungswoerter, um deine Ideen klarer zu ordnen.",
            "priority_wpm": "Erhoehe das Tempo leicht, ohne an Klarheit zu verlieren.",
            "priority_complexity": "Baue mindestens einen etwas komplexeren Satz ein, um Ursache oder Zeit zu verbinden.",
            "priority_default": "Wiederhole dieselbe Aufgabe mit einer klareren und geradlinigeren Struktur.",
            "next_exercise": "Nimm das Thema '{theme}' noch einmal auf und konzentriere dich auf: {primary}",
            "coach_summary_short_wrong_language": "Die automatische Rueckmeldung bleibt bewusst vorsichtig, weil das Transkript nur aus wenigen Woertern besteht und als {detected_language} erkannt wurde: '{excerpt}'. Nimm eine laengere Antwort in {learning_language} auf, die das Thema wirklich bearbeitet.",
            "coach_summary_short_response": "Die automatische Rueckmeldung bleibt bewusst vorsichtig, weil das Transkript sehr kurz ist: '{excerpt}'. Nimm eine laengere Antwort auf, die das Thema wirklich bearbeitet, bevor du dich auf den Score verlaesst.",
            "coach_summary": "Das ist ein brauchbarer Ausgangspunkt. Konzentriere dich im naechsten Versuch zuerst auf: {first} Danach arbeite an: {second} und {third}",
        },
        "en": {
            "strength_language": "You spoke in the required language.",
            "strength_theme": "You stayed on the assigned theme.",
            "strength_duration": "You stayed close to the target duration.",
            "strength_fillers": "Filler use was limited.",
            "strength_cohesion": "You used some clear links between ideas.",
            "strength_default": "This recording is a useful baseline for future comparison.",
            "priority_short_response": "Answer with several connected sentences instead of only a few isolated words.",
            "priority_short_wrong_language": "Instead of only saying '{excerpt}', record a fuller answer in {language} that actually addresses the theme.",
            "priority_language": f"Complete the full task in {learning_language_name}, even if you need to speak a little more slowly.",
            "priority_duration": f"Keep speaking more continuously until you reach about {int(target_duration_sec)} seconds.",
            "priority_theme": f"Stay closer to the theme '{theme}'.",
            "priority_fillers": "Reduce fillers like 'um' or 'so' and replace them with brief silent pauses.",
            "priority_cohesion": "Use more connectors to organize your ideas more clearly.",
            "priority_wpm": "Increase the pace slightly without losing clarity.",
            "priority_complexity": "Add at least one more complex sentence to connect cause or time.",
            "priority_default": "Repeat the same task with a clearer, more linear structure.",
            "next_exercise": "Record the theme '{theme}' again and focus on: {primary}",
            "coach_summary_short_wrong_language": "Automatic feedback stayed conservative because the transcript is only a few words and was detected as {detected_language}: '{excerpt}'. Record a fuller answer in {learning_language} that actually addresses the theme.",
            "coach_summary_short_response": "Automatic feedback stayed conservative because the transcript is very short: '{excerpt}'. Record a fuller answer that actually addresses the theme before relying on the score.",
            "coach_summary": "This is a useful starting point. In your next attempt, focus first on: {first} Then work on: {second} and {third}",
        },
        "it": {
            "strength_language": "Hai parlato nella lingua richiesta.",
            "strength_theme": "Sei rimasto sul tema richiesto.",
            "strength_duration": "Hai sostenuto una durata vicina all'obiettivo.",
            "strength_fillers": "L'uso di filler e' contenuto.",
            "strength_cohesion": "Ci sono segnali di collegamento tra le idee.",
            "strength_default": "Hai completato una registrazione utile per il confronto nel tempo.",
            "priority_short_response": "Rispondi con piu' frasi collegate invece di dire solo poche parole isolate.",
            "priority_short_wrong_language": "Invece di dire solo '{excerpt}', registra una risposta piu' completa in {language} che sviluppi davvero il tema.",
            "priority_language": f"Svolgi l'intero compito in {learning_language_name}, anche se all'inizio dovrai parlare un po' piu' lentamente.",
            "priority_duration": f"Parla in modo piu' continuo fino a circa {int(target_duration_sec)} secondi.",
            "priority_theme": f"Resta piu' vicino al tema '{theme}'.",
            "priority_fillers": "Riduci filler come 'eh' o 'allora' e sostituiscili con pause silenziose.",
            "priority_cohesion": "Usa piu' connettivi per ordinare meglio il racconto.",
            "priority_wpm": "Aumenta leggermente il ritmo senza sacrificare la chiarezza.",
            "priority_complexity": "Aggiungi almeno una frase piu' articolata per collegare causa o tempo.",
            "priority_default": "Ripeti lo stesso compito puntando a una narrazione piu' chiara e lineare.",
            "next_exercise": "Registra di nuovo il tema '{theme}' e concentrati su: {primary}",
            "coach_summary_short_wrong_language": "Il feedback automatico resta prudente perche' il trascritto contiene solo poche parole ed e' stato rilevato come {detected_language}: '{excerpt}'. Registra una risposta piu' completa in {learning_language} che sviluppi davvero il tema.",
            "coach_summary_short_response": "Il feedback automatico resta prudente perche' il trascritto e' molto breve: '{excerpt}'. Registra una risposta piu' completa che sviluppi davvero il tema prima di affidarti al punteggio.",
            "coach_summary": "Punto di partenza utile. Nel prossimo tentativo concentrati soprattutto su: {first} Poi lavora su: {second} e {third}",
        },
    }[locale]

    if checks.get("language_pass"):
        strengths.append(text["strength_language"])
    if topic_gate is True:
        strengths.append(text["strength_theme"])
    if checks.get("duration_pass"):
        strengths.append(text["strength_duration"])
    if word_count >= 20 and metrics.get("fillers", 0) <= 2:
        strengths.append(text["strength_fillers"])
    if word_count >= 20 and metrics.get("cohesion_markers", 0) >= 2:
        strengths.append(text["strength_cohesion"])
    if not strengths:
        strengths.append(text["strength_default"])

    if very_short_response and checks.get("language_pass") is False:
        priorities.append(
            text["priority_short_wrong_language"].format(
                excerpt=transcript_excerpt,
                language=learning_language_name,
            )
        )
    elif very_short_response:
        priorities.append(text["priority_short_response"])
    if not checks.get("language_pass"):
        priorities.append(text["priority_language"])
    if not checks.get("duration_pass"):
        priorities.append(text["priority_duration"])
    if topic_gate is False:
        priorities.append(text["priority_theme"])
    if not very_short_response and metrics.get("fillers", 0) / word_count > MAX_FILLER_RATIO:
        priorities.append(text["priority_fillers"])
    if not very_short_response and metrics.get("cohesion_markers", 0) < MIN_COHESION_MARKERS:
        priorities.append(text["priority_cohesion"])
    if not very_short_response and metrics.get("wpm", 0) < TARGET_WPM:
        priorities.append(text["priority_wpm"])
    if not very_short_response and metrics.get("complexity_index", 0) < 1:
        priorities.append(text["priority_complexity"])

    if not priorities:
        priorities.append(text["priority_default"])
    while len(priorities) < 3:
        priorities.append(text["priority_default"])
    priorities = priorities[:3]

    primary = priorities[0]
    next_exercise = text["next_exercise"].format(theme=theme, primary=primary)
    return {
        "strengths": strengths[:3],
        "top_3_priorities": priorities,
        "next_focus": primary,
        "next_exercise": next_exercise,
        "coach_summary": _coach_summary_text(
            text=text,
            priorities=priorities,
            transcript_excerpt=transcript_excerpt,
            very_short_response=very_short_response,
            language_pass=checks.get("language_pass") is True,
            learning_language_name=learning_language_name,
            detected_language_name=detected_language_name,
        ),
    }


def _transcript_excerpt(transcript: str, *, limit: int = 60) -> str:
    cleaned = re.sub(r"\s+", " ", str(transcript or "").strip())
    if not cleaned:
        return "..."
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}…"


def _coach_summary_text(
    *,
    text: Dict[str, str],
    priorities: List[str],
    transcript_excerpt: str,
    very_short_response: bool,
    language_pass: bool,
    learning_language_name: str,
    detected_language_name: str,
) -> str:
    if very_short_response and not language_pass:
        return text["coach_summary_short_wrong_language"].format(
            excerpt=transcript_excerpt,
            learning_language=learning_language_name,
            detected_language=detected_language_name,
        )
    if very_short_response:
        return text["coach_summary_short_response"].format(
            excerpt=transcript_excerpt,
        )
    return text["coach_summary"].format(
        first=priorities[0],
        second=priorities[1],
        third=priorities[2],
    )

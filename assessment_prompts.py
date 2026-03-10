"""Prompt templates for rubric generation."""

from __future__ import annotations

import json

from coaching_taxonomy import (
    COACHING_CONFIDENCE_LEVELS,
    COHERENCE_ISSUE_CATEGORIES,
    GRAMMAR_ERROR_CATEGORIES,
    LEXICAL_GAP_CATEGORIES,
)

RUBRIC_PROMPT_VERSION = "rubric_it_v2"
COACHING_PROMPT_VERSION = "coaching_it_v1"
PROMPT_VERSION = RUBRIC_PROMPT_VERSION


def rubric_prompt_it(transcript: str, metrics: dict, theme: str = "tema libero") -> str:
    safe_transcript = transcript.replace('"""', "'''").strip()
    grammar_categories = ", ".join(GRAMMAR_ERROR_CATEGORIES)
    coherence_categories = ", ".join(COHERENCE_ISSUE_CATEGORIES)
    lexical_categories = ", ".join(LEXICAL_GAP_CATEGORIES)
    confidence_levels = ", ".join(COACHING_CONFIDENCE_LEVELS)
    return f"""
Sei un esaminatore CEFR di italiano L2. Valuta solo la produzione orale basata su trascrizione e metriche.
Il tema richiesto è: "{theme}".

Regole:
- Rispondi SOLO con JSON valido (nessun testo prima/dopo).
- I punteggi sono interi da 1 a 5.
- `on_topic` deve essere true solo se la risposta è chiaramente sul tema.
- `topic_relevance_score` deve essere un intero da 1 a 5.
- `language_ok` deve essere true solo se il parlato è chiaramente in italiano.
- Per gli errori ricorrenti usa SOLO le categorie ammesse.

METRICHE OGGETTIVE:
- Durata: {metrics['duration_sec']} s
- Tempo di parola: {metrics['speaking_time_sec']} s
- Pausa totale: {metrics['pause_total_sec']} s
- Numero pause: {metrics['pause_count']}
- Parole: {metrics['word_count']}
- WPM: {metrics['wpm']}
- Filler: {metrics['fillers']}
- Marcatori di coesione: {metrics['cohesion_markers']}
- Indice complessità: {metrics['complexity_index']}

TRASCRITTO:
\"\"\"{safe_transcript}\"\"\"

Nota:
- Il trascritto proviene da ASR automatico e può contenere errori di trascrizione.

Schema JSON richiesto:
{{
  "fluency": 1-5,
  "cohesion": 1-5,
  "accuracy": 1-5,
  "range": 1-5,
  "overall": 1-5,
  "comments_fluency": "stringa",
  "comments_cohesion": "stringa",
  "comments_accuracy": "stringa",
  "comments_range": "stringa",
  "overall_comment": "stringa",
  "on_topic": true/false,
  "topic_relevance_score": 1-5,
  "language_ok": true/false,
  "recurring_grammar_errors": [
    {{
      "category": "una di: {grammar_categories}",
      "explanation": "stringa",
      "examples": ["stringa"]
    }}
  ],
  "coherence_issues": [
    {{
      "category": "una di: {coherence_categories}",
      "explanation": "stringa",
      "examples": ["stringa"]
    }}
  ],
  "lexical_gaps": [
    {{
      "category": "una di: {lexical_categories}",
      "explanation": "stringa",
      "examples": ["stringa"]
    }}
  ],
  "evidence_quotes": ["stringa"],
  "confidence": "una di: {confidence_levels}"
}}
"""


def selftest_prompt_it() -> str:
    fake_metrics = {
        "duration_sec": 75.0,
        "speaking_time_sec": 63.0,
        "pause_total_sec": 12.0,
        "pause_count": 8,
        "word_count": 140,
        "wpm": 133.3,
        "fillers": 5,
        "cohesion_markers": 4,
        "complexity_index": 3,
    }
    transcript = (
        "Oggi parlo della mia città. Negli ultimi anni il trasporto pubblico è migliorato, "
        "tuttavia i costi sono ancora alti e molte persone preferiscono l'auto."
    )
    return rubric_prompt_it(transcript, fake_metrics, "la mia città")


def coaching_prompt_it(
    metrics: dict,
    rubric: dict,
    theme: str,
    target_duration_sec: float,
) -> str:
    rubric_json = json.dumps(rubric, ensure_ascii=False, indent=2)
    return f"""
Sei un coach di italiano L2. Usa SOLO le metriche e il rubric già validato per dare consigli pratici.
Il compito era parlare in italiano per {target_duration_sec:.0f} secondi sul tema "{theme}".

Regole:
- Rispondi SOLO con JSON valido.
- `top_3_priorities` deve contenere ESATTAMENTE 3 elementi.
- `next_exercise` deve essere un'attività pratica concreta, non un id interno o un link inventato.
- Non cambiare i fatti del rubric già validato.

METRICHE:
- Tempo di parola: {metrics['speaking_time_sec']} s
- Pausa totale: {metrics['pause_total_sec']} s
- Numero pause: {metrics['pause_count']}
- Parole: {metrics['word_count']}
- WPM: {metrics['wpm']}
- Filler: {metrics['fillers']}
- Marcatori di coesione: {metrics['cohesion_markers']}
- Indice complessità: {metrics['complexity_index']}

RUBRIC VALIDATO:
{rubric_json}

Schema JSON richiesto:
{{
  "strengths": ["stringa"],
  "top_3_priorities": ["stringa", "stringa", "stringa"],
  "next_focus": "stringa",
  "next_exercise": "stringa",
  "coach_summary": "stringa"
}}
"""

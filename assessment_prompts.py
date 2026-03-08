"""Prompt templates for rubric generation."""

from __future__ import annotations

PROMPT_VERSION = "rubric_it_v1"


def rubric_prompt_it(transcript: str, metrics: dict, theme: str = "tema libero") -> str:
    safe_transcript = transcript.replace('"""', "'''").strip()
    return f"""
Sei un esaminatore CEFR di italiano L2. Valuta solo la produzione orale basata su trascrizione e metriche.
Il tema richiesto è: "{theme}".

Regole:
- Rispondi SOLO con JSON valido (nessun testo prima/dopo).
- I punteggi sono interi da 1 a 5.
- `on_topic` deve essere true solo se la risposta è chiaramente sul tema.

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
  "on_topic": true/false
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

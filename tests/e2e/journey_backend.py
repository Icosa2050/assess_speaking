"""Real API/jobs/metrics/storage with fixture ASR and AI, for isolated journeys only.

Spawned workers re-import this test launcher. Production never imports it.
"""
from pathlib import Path
import json
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import assess_speaking
from assess_core.schemas import CoachingSummary, RubricResult

_original_assessment = assess_speaking.run_assessment


def fixture_assessment(audio, whisper_model, llm_model, **kwargs):
    theme = kwargs.get("theme", "")
    if theme == "journey-fixture-failure":
        raise RuntimeError("Journey fixture: assessment failed; retry is safe.")
    if theme == "journey-fixture-cancel":
        kwargs["status_callback"]("scoring_rubric")
        time.sleep(6)
    language = kwargs.get("expected_language", "it")
    transcript = {
        "en": "I visited a library yesterday. I enjoyed reading there and would like to return with a friend.",
        "it": "Ieri ho visitato una biblioteca. Mi è piaciuto leggere e vorrei tornarci con un amico.",
    }[language]

    def transcribe_fixture(wav, *args, **options):
        duration = assess_speaking.load_audio_features(wav)["duration_sec"]
        tokens = transcript.split()
        step = duration / len(tokens)
        return {
            "text": transcript, "detected_language": language, "language_probability": 1.0,
            "compute_type_used": "fixture", "compute_fallback_used": False,
            "words": [{"text": token, "t0": i * step, "t1": (i + .8) * step} for i, token in enumerate(tokens)],
        }

    rubric = RubricResult.from_dict({
        **{key: 4 for key in ("fluency", "cohesion", "accuracy", "range", "overall", "topic_relevance_score")},
        **{key: "Fixture feedback" for key in ("comments_fluency", "comments_cohesion", "comments_accuracy", "comments_range", "overall_comment")},
        "on_topic": True, "language_ok": True,
        "recurring_grammar_errors": [], "coherence_issues": [], "lexical_gaps": [],
        "evidence_quotes": [transcript], "confidence": "medium",
        "style_suggestions": [{
            "original": "I visited a library" if language == "en" else "vorrei tornarci con un amico",
            "suggestion": "I went to a library" if language == "en" else "mi piacerebbe tornarci con un amico",
            "explanation": "Optional alternative; the original is grammatical." if language == "en" else "Alternativa facoltativa; la frase originale è grammaticalmente corretta.",
        }],
    })
    coaching = CoachingSummary.from_dict({
        "strengths": ["Fixture strength"], "top_3_priorities": ["Fixture detail", "Fixture ending", "Fixture structure"],
        "next_focus": "Fixture next focus", "next_exercise": "Fixture next exercise", "coach_summary": "Fixture coaching",
    })
    with patch.object(assess_speaking, "transcribe", transcribe_fixture), \
         patch.object(assess_speaking, "generate_rubric", return_value=(rubric, json.dumps(rubric.to_dict()))), \
         patch.object(assess_speaking, "generate_coaching_summary", return_value=(coaching, "fixture")):
        result = _original_assessment(audio, whisper_model, llm_model, **kwargs)
    report = result["report"]
    report["input"]["scoring_model_version"] = "journey-fixture-v1"
    report["input"]["fixture_inference"] = True
    report["warnings"].append("journey_fixture_inference")
    return result


assess_speaking.run_assessment = fixture_assessment

if __name__ == "__main__":
    from scripts.run_backend import main
    raise SystemExit(main())

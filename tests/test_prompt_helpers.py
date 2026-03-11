import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import unittest
from unittest import mock

from scripts import interactive_dashboard as dashboard

RICH_HISTORY_CSV = """timestamp,session_id,schema_version,speaker_id,task_family,theme,audio,whisper,llm,label,target_duration_sec,duration_sec,wpm,word_count,duration_pass,topic_pass,language_pass,fluency,cohesion,accuracy,range,overall,final_score,band,requires_human_review,top_priority_1,top_priority_2,top_priority_3,grammar_error_categories,coherence_issue_categories,report_path
2025-10-06T14:58:01,s1,2,bern,travel_narrative,trip,demo.m4a,large-v3,llama3.1,baseline,180,43.09,95.9,54,true,true,true,3,3,3,3,3.5,3.6,4,false,Più connettivi,Meno filler,Più dettagli,preposition_choice,missing_sequence_markers,/path/to/1.json
2025-10-07T09:12:33,s2,2,bern,travel_narrative,trip,week2.m4a,medium,llama3.2:3b,week2,180,41.00,110.2,60,true,true,true,4,4,4,4,3.8,4.1,4,false,Più dettagli,Più precisione,Meno pause,preposition_choice,missing_sequence_markers|underdeveloped_detail,/path/to/2.json
"""


class PromptHelperTests(unittest.TestCase):
    def setUp(self):
        self.prompt = {
            "id": "b2_remote_work",
            "audio_path": "/tmp/audio.wav",
            "response_seconds": 90,
            "max_playbacks": 1,
            "cefr_target": "B2",
        }

    def test_create_prompt_attempt_initialises_fields(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=100.0)
        self.assertEqual(attempt["id"], self.prompt["id"])
        self.assertEqual(attempt["start"], 100.0)
        self.assertEqual(attempt["deadline"], 190.0)
        self.assertEqual(attempt["plays_remaining"], 1)
        self.assertEqual(attempt["audio"], self.prompt["audio_path"])
        self.assertEqual(attempt["cefr"], "B2")
        self.assertEqual(attempt["chunks"], [])

    def test_remaining_time_and_decrement(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=10.0)
        remaining = dashboard.remaining_time(attempt, now=55.0)
        self.assertAlmostEqual(remaining, 45.0)
        self.assertTrue(dashboard.can_play_prompt(attempt))
        dashboard.decrement_playback(attempt)
        self.assertEqual(attempt["plays_remaining"], 0)
        self.assertFalse(dashboard.can_play_prompt(attempt))
        with self.assertRaises(ValueError):
            dashboard.decrement_playback(attempt)

    def test_append_and_write_audio(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=0.0)
        # Generate a short sine wave chunk
        sample_rate = 16000
        t = np.linspace(0, 0.01, int(sample_rate * 0.01), endpoint=False)
        data = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        dashboard.append_audio_bytes(attempt, pcm, sample_rate, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.wav")
            dashboard.write_attempt_audio(attempt, Path(out_path))
            with wave.open(out_path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getframerate(), sample_rate)
                frames = wf.readframes(wf.getnframes())
                self.assertEqual(frames, pcm)

    def test_write_attempt_audio_no_chunks_raises(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "empty.wav"
            with self.assertRaises(ValueError):
                dashboard.write_attempt_audio(attempt, out_path)

    def test_parse_cli_json_variants(self):
        payload = {"metrics": {"wpm": 100}}
        json_body = json.dumps(payload)
        direct = dashboard.parse_cli_json(json_body)
        self.assertEqual(direct, payload)
        wrapped = dashboard.parse_cli_json("INFO\n" + json_body)
        self.assertEqual(wrapped, payload)

    def test_load_prompts_resolves_relative_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.json"
            prompts_path.write_text('[{"id":"p","audio":"relative.wav","response_seconds":30,"max_playbacks":1,"cefr_target":"B1"}]')
            loaded = dashboard.load_prompts(prompts_path)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(loaded[0]["audio_path"].endswith("relative.wav"))

    @mock.patch("scripts.interactive_dashboard.subprocess.run")
    def test_run_assessment_appends_dry_run_flag_when_env_set(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="{}", stderr="")
        with mock.patch.dict(os.environ, {"ASSESS_SPEAKING_DRY_RUN": "1"}, clear=False):
            dashboard.run_assessment(
                Path("sample.wav"),
                Path("reports"),
                "large-v3",
                "llama3.1",
                "label",
                "notes",
            )
        command = mock_run.call_args.args[0]
        self.assertIn("--provider", command)
        self.assertIn("--llm-model", command)
        self.assertIn("--dry-run", command)

    @mock.patch("scripts.interactive_dashboard.subprocess.run")
    def test_run_assessment_passes_learning_context_flags(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="{}", stderr="")
        dashboard.run_assessment(
            Path("sample.wav"),
            Path("reports"),
            "large-v3",
            "google/gemini-3.1-pro-preview",
            "trip-attempt",
            "notes",
            provider="openrouter",
            speaker_id="bern",
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
        )
        command = mock_run.call_args.args[0]
        self.assertIn("--speaker-id", command)
        self.assertIn("bern", command)
        self.assertIn("--task-family", command)
        self.assertIn("travel_narrative", command)
        self.assertIn("--theme", command)
        self.assertIn("Il mio ultimo viaggio all'estero", command)
        self.assertIn("--target-duration-sec", command)
        self.assertIn("180.0", command)

    def test_load_history_df_exposes_extended_columns(self):
        dashboard.load_history_records.clear()
        dashboard.load_history_df.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.csv"
            history.write_text(RICH_HISTORY_CSV, encoding="utf-8")
            frame = dashboard.load_history_df(Path(tmpdir))
            self.assertIn("speaker_id", frame.columns)
            self.assertIn("task_family", frame.columns)
            self.assertIn("final_score", frame.columns)
            self.assertEqual(frame.iloc[-1]["task_family"], "travel_narrative")
            self.assertIn("preposition_choice", frame.iloc[-1]["grammar_error_categories"])

    def test_build_issue_count_df_counts_categories(self):
        dashboard.load_history_records.clear()
        dashboard.load_history_df.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.csv"
            history.write_text(RICH_HISTORY_CSV, encoding="utf-8")
            records = dashboard.load_history_records(Path(tmpdir))
            frame = dashboard.build_issue_count_df(records, "coherence_issue_categories")
            self.assertEqual(frame.iloc[0]["category"], "missing_sequence_markers")
            self.assertEqual(int(frame.iloc[0]["count"]), 2)

    def test_build_result_summary_prefers_learner_fields(self):
        payload = {
            "report": {
                "checks": {
                    "language_pass": True,
                    "topic_pass": False,
                    "duration_pass": True,
                    "min_words_pass": True,
                },
                "scores": {
                    "final": 3.8,
                    "band": 4,
                    "llm": 4.0,
                    "deterministic": 3.2,
                    "mode": "hybrid",
                },
                "requires_human_review": False,
                "warnings": ["coaching_unavailable"],
                "rubric": {
                    "recurring_grammar_errors": [{"type": "preposition_choice"}],
                    "coherence_issues": [{"type": "missing_sequence_markers"}],
                },
                "coaching": {
                    "strengths": ["Resti sul tema."],
                    "top_3_priorities": ["Più dettagli", "Meno filler", "Più connettivi"],
                    "next_focus": "Ordina meglio gli eventi",
                    "next_exercise": "Racconta di nuovo il viaggio.",
                    "coach_summary": "Buona base.",
                },
                "progress_delta": {
                    "previous_session_id": "sess-1",
                    "score_delta": {"final": 0.4, "overall": 0.2, "wpm": 5.0},
                    "new_priorities": ["Più dettagli"],
                    "repeating_grammar_categories": ["preposition_choice"],
                    "repeating_coherence_categories": ["missing_sequence_markers"],
                },
            }
        }
        summary = dashboard.build_result_summary(payload)
        self.assertEqual(summary["status_level"], "info")
        self.assertEqual(summary["failed_gates"], ["Thema"])
        self.assertEqual(summary["priorities"][0], "Più dettagli")
        self.assertEqual(summary["recurring_grammar"], ["preposition_choice"])
        self.assertTrue(summary["progress_lines"])

    def test_build_progress_delta_lines_handles_empty_input(self):
        self.assertEqual(dashboard.build_progress_delta_lines(None), [])
        self.assertEqual(dashboard.build_progress_delta_lines({}), [])

    def test_generate_practice_brief_uses_theme_and_variant(self):
        first = dashboard.generate_practice_brief(
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
            variant_index=0,
        )
        second = dashboard.generate_practice_brief(
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
            variant_index=1,
        )
        self.assertIn("Il mio ultimo viaggio all'estero", first["prompt"])
        self.assertNotEqual(first["title"], second["title"])
        self.assertEqual(len(first["cover_points"]), 3)
        self.assertTrue(first["starter_phrases"])

    def test_create_recording_attempt_initialises_empty_audio_state(self):
        attempt = dashboard.create_recording_attempt()
        self.assertEqual(attempt["chunks"], [])
        self.assertIsNone(attempt["sample_rate"])
        self.assertEqual(attempt["sample_width"], 2)


if __name__ == "__main__":
    unittest.main()

import json
import os
import subprocess
import tempfile
import time
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
            "learning_language": "it",
        }

    def test_legacy_dashboard_notice_points_to_app_shell(self):
        self.assertIn("Legacy compatibility surface", dashboard.LEGACY_DASHBOARD_NOTICE)
        self.assertIn("streamlit_app.py", dashboard.LEGACY_DASHBOARD_NOTICE)

    def test_create_prompt_attempt_initialises_fields(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=100.0)
        self.assertEqual(attempt["id"], self.prompt["id"])
        self.assertEqual(attempt["start"], 100.0)
        self.assertEqual(attempt["deadline"], 190.0)
        self.assertEqual(attempt["plays_remaining"], 1)
        self.assertEqual(attempt["audio"], self.prompt["audio_path"])
        self.assertEqual(attempt["cefr"], "B2")
        self.assertEqual(attempt["learning_language"], "it")
        self.assertEqual(attempt["chunks"], [])

    def test_remaining_time_and_decrement(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=10.0)
        remaining = dashboard.remaining_time(attempt, now=55.0)
        self.assertAlmostEqual(remaining, 45.0)
        self.assertFalse(dashboard.attempt_expired(attempt, now=55.0))
        self.assertTrue(dashboard.attempt_expired(attempt, now=101.0))
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

    def test_persist_audio_input_stores_new_recording(self):
        class UploadedAudio:
            name = "browser.wav"

            def __init__(self, payload: bytes):
                self._payload = payload

            def getvalue(self):
                return self._payload

            def getbuffer(self):
                return memoryview(self._payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            session_state = {"practice_attempt": dashboard.create_recording_attempt()}
            with mock.patch.object(dashboard.st, "session_state", session_state):
                attempt = dashboard.persist_audio_input(
                    UploadedAudio(b"RIFF....WAVE"),
                    session_key="practice_attempt",
                    target_dir=Path(tmpdir),
                    prefix="practice",
                )
                self.assertEqual(attempt["status"], "ready")
                self.assertTrue(Path(attempt["saved_path"]).exists())
                self.assertIsNotNone(attempt["input_digest"])

    def test_reset_audio_input_recorder_resets_attempt_and_bumps_version(self):
        session_state = {
            "practice_attempt": {"status": "ready", "saved_path": "/tmp/demo.wav"},
            "practice_audio_input_version": 2,
        }
        with mock.patch.object(dashboard.st, "session_state", session_state):
            dashboard.reset_audio_input_recorder(
                session_key="practice_attempt",
                version_key="practice_audio_input_version",
            )
        self.assertEqual(session_state["practice_attempt"]["status"], "idle")
        self.assertEqual(session_state["practice_audio_input_version"], 3)

    def test_normalize_practice_mode_accepts_legacy_labels(self):
        self.assertEqual(dashboard.normalize_practice_mode("Audiodatei hochladen"), dashboard.PRACTICE_MODE_UPLOAD)
        self.assertEqual(dashboard.normalize_practice_mode("record"), dashboard.PRACTICE_MODE_RECORD)

    def test_validate_theme_library_submission_requires_theme_title(self):
        errors = dashboard.validate_theme_library_submission(
            manage_mode="it",
            language_code="it",
            language_label="Italiano",
            theme_title="",
        )
        self.assertEqual(errors["theme_title"], "Bitte gib ein Thema ein.")

    def test_validate_theme_library_submission_requires_new_language_fields(self):
        errors = dashboard.validate_theme_library_submission(
            manage_mode=dashboard.NEW_LANGUAGE_OPTION,
            language_code="",
            language_label="",
            theme_title="Nuovo tema",
        )
        self.assertEqual(errors["language_code"], "Bitte gib einen Sprachcode ein.")
        self.assertEqual(errors["language_label"], "Bitte gib einen Sprachname ein.")

    def test_parse_cli_json_variants(self):
        payload = {"metrics": {"wpm": 100}}
        json_body = json.dumps(payload)
        direct = dashboard.parse_cli_json(json_body)
        self.assertEqual(direct, payload)
        wrapped = dashboard.parse_cli_json("INFO\n" + json_body)
        self.assertEqual(wrapped, payload)

    def test_load_latest_report_payload_uses_history_report_path(self):
        payload = {"report": {"scores": {"final": 4.2}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "latest.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            history_path = tmp / "history.csv"
            history_path.write_text(
                "timestamp,label,report_path\n2026-03-11T12:00:00,prompt:test,%s\n" % report_path,
                encoding="utf-8",
            )
            loaded = dashboard.load_latest_report_payload(tmp, label="prompt:test")
        self.assertEqual(loaded, payload)

    def test_load_prompts_resolves_relative_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.json"
            prompts_path.write_text('[{"id":"p","audio":"relative.wav","response_seconds":30,"max_playbacks":1,"cefr_target":"B1"}]')
            loaded = dashboard.load_prompts(prompts_path)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(loaded[0]["audio_path"].endswith("relative.wav"))
            self.assertEqual(loaded[0]["learning_language"], "it")

    def test_build_prompt_assessment_request_uses_prompt_learning_language(self):
        request = dashboard.build_prompt_assessment_request(
            attempt={
                "label": "prompt:b2_remote_work",
                "cefr": "B2",
                "learning_language": "it",
            },
            prompt={
                "id": "b2_remote_work",
                "title": "B2 – Lavoro da casa",
                "response_seconds": 90,
                "learning_language": "it",
            },
            response_path=Path("response.wav"),
            log_dir=Path("reports"),
            whisper="large-v3",
            llm="google/gemini-3.1-pro-preview",
            notes="prompt notes",
            provider="openrouter",
            speaker_id="bern",
            ui_locale="en",
        )
        self.assertEqual(request["expected_language"], "it")
        self.assertEqual(request["feedback_language"], "en")
        self.assertEqual(request["task_family"], "prompt_trainer")
        self.assertEqual(request["theme"], "B2 – Lavoro da casa")

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
            expected_language="en",
            feedback_language="de",
            speaker_id="bern",
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
        )
        command = mock_run.call_args.args[0]
        self.assertIn("--expected-language", command)
        self.assertIn("en", command)
        self.assertIn("--feedback-language", command)
        self.assertIn("de", command)
        self.assertIn("--speaker-id", command)
        self.assertIn("bern", command)
        self.assertIn("--task-family", command)
        self.assertIn("travel_narrative", command)
        self.assertIn("--theme", command)
        self.assertIn("Il mio ultimo viaggio all'estero", command)
        self.assertIn("--target-duration-sec", command)
        self.assertIn("180.0", command)

    def test_create_assessment_request_serializes_runtime_inputs(self):
        request = dashboard.create_assessment_request(
            audio_path=Path("sample.wav"),
            log_dir=Path("reports"),
            whisper="large-v3",
            llm="google/gemini-3.1-pro-preview",
            label="trip-attempt",
            notes="notes",
            provider="openrouter",
            expected_language="it",
            feedback_language="en",
            speaker_id="bern",
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
            target_cefr="B1",
        )
        self.assertEqual(request["audio_path"], "sample.wav")
        self.assertEqual(request["provider"], "openrouter")
        self.assertEqual(request["expected_language"], "it")
        self.assertEqual(request["feedback_language"], "en")
        self.assertEqual(request["target_cefr"], "B1")

    @mock.patch("scripts.interactive_dashboard.subprocess.run")
    def test_run_assessment_passes_language_profile_key_when_set(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="{}", stderr="")
        dashboard.run_assessment(
            Path("sample.wav"),
            Path("reports"),
            "large-v3",
            "google/gemini-3.1-pro-preview",
            "trip-attempt",
            "notes",
            provider="openrouter",
            expected_language="en",
            language_profile_key="en",
            feedback_language="de",
            speaker_id="bern",
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
        )
        command = mock_run.call_args.args[0]
        self.assertIn("--language-profile-key", command)
        self.assertIn("en", command)

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
            self.assertEqual(frame.iloc[0]["category"], "Fehlende Reihenfolge-Marker")
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
        summary = dashboard.build_result_summary(payload, ui_locale="de")
        self.assertEqual(summary["status_level"], "info")
        self.assertEqual(summary["failed_gates"], ["Thema"])
        self.assertEqual(summary["priorities"][0], "Più dettagli")
        self.assertEqual(summary["recurring_grammar"], ["Präpositionen"])
        self.assertEqual(summary["mode_label"], "Vollbewertung")
        self.assertTrue(summary["progress_lines"])
        self.assertIn("Gesamtwert: +0.40.", summary["progress_lines"])
        self.assertIn("Wiederkehrende Grammatik: Präpositionen.", summary["progress_lines"])

    def test_build_progress_delta_lines_handles_empty_input(self):
        self.assertEqual(dashboard.build_progress_delta_lines(None), [])
        self.assertEqual(dashboard.build_progress_delta_lines({}), [])

    def test_generate_practice_brief_uses_theme_and_variant(self):
        first = dashboard.generate_practice_brief(
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
            language_code="it",
            variant_index=0,
        )
        second = dashboard.generate_practice_brief(
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
            language_code="it",
            variant_index=1,
        )
        self.assertIn("Il mio ultimo viaggio all'estero", first["prompt"])
        self.assertNotEqual(first["title"], second["title"])
        self.assertEqual(len(first["cover_points"]), 3)
        self.assertTrue(first["starter_phrases"])

    def test_generate_practice_brief_supports_english(self):
        brief = dashboard.generate_practice_brief(
            task_family="opinion_monologue",
            theme="The pros and cons of working from home",
            target_duration_sec=180,
            language_code="en",
            variant_index=0,
        )
        self.assertIn("The pros and cons of working from home", brief["prompt"])
        self.assertIn("Aim to speak", brief["success_focus"][0])

    def test_create_recording_attempt_initialises_empty_audio_state(self):
        attempt = dashboard.create_recording_attempt()
        self.assertEqual(attempt["chunks"], [])
        self.assertIsNone(attempt["sample_rate"])
        self.assertEqual(attempt["sample_width"], 2)
        self.assertEqual(attempt["status"], "idle")
        self.assertIsNone(attempt["connection_requested_at"])

    def test_build_rtc_configuration_defaults_to_local_only(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = dashboard.build_rtc_configuration()
        self.assertEqual(config["iceServers"], [])

    def test_build_rtc_configuration_accepts_env_urls(self):
        with mock.patch.dict(
            os.environ,
            {"ASSESS_SPEAKING_STUN_URLS": "stun:stun1.example.org, stun:stun2.example.org"},
            clear=False,
        ):
            config = dashboard.build_rtc_configuration()
        self.assertEqual(
            config["iceServers"],
            [{"urls": ["stun:stun1.example.org", "stun:stun2.example.org"]}],
        )

    def test_mark_recording_connecting_sets_pending_status(self):
        attempt = dashboard.create_recording_attempt()
        updated = dashboard.mark_recording_connecting(attempt)
        self.assertEqual(updated["status"], "connecting")
        self.assertIsNotNone(updated["connection_requested_at"])

    def test_display_duration_uses_wall_clock_while_recording(self):
        attempt = dashboard.create_recording_attempt()
        attempt["status"] = "recording"
        attempt["recording_started_at"] = 100.0
        with mock.patch("scripts.interactive_dashboard.time.time", return_value=103.6):
            elapsed = dashboard.display_duration_sec(attempt)
        self.assertGreaterEqual(elapsed, 3.5)

    def test_transport_is_usable_rejects_closed_or_missing_socket(self):
        class DummyLoop:
            def __init__(self, closed: bool):
                self._closed = closed

            def is_closed(self):
                return self._closed

        class DummyTransport:
            def __init__(self, *, sock, closing=False, loop_closed=False):
                self._sock = sock
                self._closing = closing
                self._loop = DummyLoop(loop_closed)

            def is_closing(self):
                return self._closing

        self.assertFalse(dashboard._transport_is_usable(None))
        self.assertFalse(dashboard._transport_is_usable(DummyTransport(sock=None)))
        self.assertFalse(dashboard._transport_is_usable(DummyTransport(sock=object(), closing=True)))
        self.assertFalse(dashboard._transport_is_usable(DummyTransport(sock=object(), loop_closed=True)))
        self.assertTrue(dashboard._transport_is_usable(DummyTransport(sock=object())))

    def test_sync_recording_state_surfaces_connection_timeout(self):
        attempt = dashboard.create_recording_attempt()
        attempt["connection_requested_at"] = time.time() - 9.0

        class DummyContext:
            state = None

        updated = dashboard.sync_recording_state(
            attempt,
            DummyContext(),
            target_dir=Path("/tmp"),
            prefix="practice",
            connection_timeout_sec=8.0,
        )
        self.assertEqual(updated["status"], "error")
        self.assertIn("Aufnahme wurde nicht gestartet", updated["save_error"])

    def test_resolve_webrtc_state_prefers_frontend_session_value(self):
        if dashboard.compile_state is None or dashboard.generate_frontend_component_key is None:
            self.skipTest("streamlit-webrtc internal helpers unavailable")

        class DummyContext:
            def __init__(self):
                self.state = None

            def _set_state(self, state):
                self.state = state

        frontend_key = dashboard.generate_frontend_component_key("practice_recorder")
        ctx = DummyContext()
        with mock.patch.object(dashboard.st, "session_state", {frontend_key: {"playing": True, "sdpOffer": {"type": "offer"}}}):
            state = dashboard.resolve_webrtc_state("practice_recorder", ctx)
        self.assertTrue(state.playing)
        self.assertTrue(state.signalling)
        self.assertTrue(ctx.state.playing)

    def test_sync_recording_state_uses_frontend_component_state(self):
        if dashboard.compile_state is None or dashboard.generate_frontend_component_key is None:
            self.skipTest("streamlit-webrtc internal helpers unavailable")

        class DummyContext:
            def __init__(self):
                self.state = None

            def _set_state(self, state):
                self.state = state

        attempt = dashboard.create_recording_attempt()
        frontend_key = dashboard.generate_frontend_component_key("practice_recorder")
        ctx = DummyContext()
        with mock.patch.object(
            dashboard.st,
            "session_state",
            {frontend_key: {"playing": True, "sdpOffer": {"type": "offer"}}},
        ):
            updated = dashboard.sync_recording_state(
                attempt,
                ctx,
                component_key="practice_recorder",
                target_dir=Path("/tmp"),
                prefix="practice",
                connection_timeout_sec=8.0,
            )
        self.assertEqual(updated["status"], "recording")
        self.assertTrue(updated["is_recording"])
        self.assertTrue(updated["is_signalling"])

    def test_build_recorder_debug_snapshot_includes_frontend_state(self):
        if dashboard.generate_frontend_component_key is None:
            self.skipTest("streamlit-webrtc internal helpers unavailable")

        class DummyState:
            playing = False
            signalling = False

        class DummyContext:
            state = DummyState()
            audio_receiver = object()

            def _set_state(self, state):
                self.state = state

        attempt = dashboard.create_recording_attempt()
        frontend_key = dashboard.generate_frontend_component_key("practice_recorder")
        with mock.patch.object(
            dashboard.st,
            "session_state",
            {frontend_key: {"playing": True, "sdpOffer": {"type": "offer"}}},
        ):
            snapshot = dashboard.build_recorder_debug_snapshot(
                attempt,
                DummyContext(),
                component_key="practice_recorder",
                audio_frames_count=3,
            )
        self.assertTrue(snapshot["resolved_playing"])
        self.assertEqual(snapshot["audio_frames_count"], 3)
        self.assertIsInstance(snapshot["frontend_value"], dict)

    def test_log_recorder_snapshot_keeps_recent_entries(self):
        session_state = {}
        with mock.patch.object(dashboard.st, "session_state", session_state):
            for idx in range(30):
                dashboard.log_recorder_snapshot("practice_attempt", {"ts": idx, "status": "idle"})
        self.assertEqual(len(session_state["practice_attempt_debug_log"]), 25)
        self.assertEqual(session_state["practice_attempt_debug_log"][0]["ts"], 5)


if __name__ == "__main__":
    unittest.main()

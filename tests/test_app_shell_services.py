import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_shell.services import (
    NEW_LANGUAGE_OPTION,
    create_assessment_request,
    delete_provider_connection,
    execute_assessment_request,
    hydrate_state_from_storage,
    load_report_payload,
    parse_cli_json,
    review_summary,
    set_default_provider_connection,
    store_uploaded_audio,
    test_runtime_connection,
    theme_entry_id,
    whisper_model_status,
    validate_theme_submission,
)
from app_shell.state import AppPreferences, AppShellState, ProviderConnection
from app_shell.state import DraftSession


class _FakeUpload:
    def __init__(self, data: bytes, name: str) -> None:
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


class AppShellServiceTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": ""}, clear=False)
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_dashboard_prefs")
    def test_hydrate_state_from_storage_restores_openrouter_preferences(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
    ):
        mock_load_history.return_value = []
        mock_load_prefs.return_value = {
            "ui_locale": "de",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4.5",
            "whisper_model": "base",
            "openrouter_api_key": "saved-key",
            "openrouter_http_referer": "https://example.test/app",
            "openrouter_app_title": "Assess Speaking Dev",
            "log_dir": "reports",
        }
        mock_load_library.return_value = {
            "it": {
                "label": "Italiano",
                "themes": [],
            }
        }

        state = AppShellState(
            prefs=AppPreferences(
                ui_locale="en",
                provider="ollama",
                model="llama3",
                whisper_model="small",
            )
        )

        hydrated = hydrate_state_from_storage(state)

        self.assertEqual(hydrated.prefs.ui_locale, "de")
        self.assertEqual(hydrated.prefs.provider, "openrouter")
        self.assertEqual(hydrated.prefs.model, "anthropic/claude-sonnet-4.5")
        self.assertEqual(hydrated.prefs.whisper_model, "base")
        self.assertEqual(hydrated.prefs.openrouter_api_key, "saved-key")
        self.assertEqual(hydrated.prefs.openrouter_http_referer, "https://example.test/app")
        self.assertEqual(hydrated.prefs.openrouter_app_title, "Assess Speaking Dev")

    @mock.patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": ""}, clear=False)
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_dashboard_prefs")
    def test_hydrate_state_from_storage_migrates_legacy_default_app_title(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
    ):
        mock_load_history.return_value = []
        mock_load_prefs.return_value = {
            "provider": "openrouter",
            "openrouter_app_title": "Assess Speaking",
            "log_dir": "reports",
        }
        mock_load_library.return_value = {
            "it": {
                "label": "Italiano",
                "themes": [],
            }
        }

        hydrated = hydrate_state_from_storage(AppShellState(prefs=AppPreferences()))

        self.assertEqual(hydrated.prefs.openrouter_app_title, "Speaking Studio")

    @mock.patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": ""}, clear=False)
    @mock.patch("app_shell.services.set_secret")
    @mock.patch("app_shell.services.get_secret", return_value="")
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_dashboard_prefs")
    def test_hydrate_state_from_storage_migrates_legacy_plaintext_key_into_secret_store(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
        _mock_get_secret,
        mock_set_secret,
    ):
        mock_load_history.return_value = []
        mock_load_prefs.return_value = {
            "provider": "openrouter",
            "openrouter_api_key": "saved-key",
            "log_dir": "reports",
        }
        mock_load_library.return_value = {"it": {"label": "Italiano", "themes": []}}

        hydrated = hydrate_state_from_storage(AppShellState(prefs=AppPreferences()))

        self.assertEqual(hydrated.prefs.llm_api_key, "saved-key")
        self.assertEqual(hydrated.prefs.openrouter_api_key, "saved-key")
        mock_set_secret.assert_called_once()

    @mock.patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": ""}, clear=False)
    @mock.patch("app_shell.services.load_report_payload")
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_dashboard_prefs")
    def test_hydrate_state_from_storage_backfills_draft_from_latest_history_for_speaker(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
        mock_load_report_payload,
    ):
        mock_load_prefs.return_value = {
            "speaker_id": "bern",
            "learning_language": "en",
            "cefr_level": "B2",
            "theme": "The pros and cons of working from home",
            "task_family": "opinion_monologue",
            "target_duration_sec": 120,
            "log_dir": "reports",
        }
        mock_load_library.return_value = {
            "it": {
                "label": "Italiano",
                "themes": [
                    {
                        "title": "Il mio ultimo viaggio all'estero",
                        "level": "B1",
                        "task_family": "travel_narrative",
                    }
                ],
            },
            "en": {"label": "English", "themes": []},
        }
        mock_load_history.return_value = [
            mock.Mock(
                speaker_id="bern",
                learning_language="it",
                theme="Il mio ultimo viaggio all'estero",
                task_family="travel_narrative",
                target_duration_sec=90,
                report_path="reports/latest.json",
            )
        ]
        mock_load_report_payload.return_value = {
            "baseline_comparison": {"level": "B1"},
            "report": {
                "input": {
                    "speaker_id": "bern",
                    "expected_language": "it",
                    "theme": "Il mio ultimo viaggio all'estero",
                    "task_family": "travel_narrative",
                    "target_duration_sec": 90,
                }
            },
        }

        hydrated = hydrate_state_from_storage(AppShellState(prefs=AppPreferences()))

        self.assertEqual(hydrated.draft.speaker_id, "bern")
        self.assertEqual(hydrated.draft.learning_language, "it")
        self.assertEqual(hydrated.draft.learning_language_label, "Italiano")
        self.assertEqual(hydrated.draft.cefr_level, "B1")
        self.assertEqual(hydrated.draft.theme_label, "Il mio ultimo viaggio all'estero")
        self.assertEqual(hydrated.draft.task_family, "travel_narrative")
        self.assertEqual(hydrated.draft.duration_sec, 90)

    @mock.patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": ""}, clear=False)
    @mock.patch("app_shell.services.load_report_payload")
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_dashboard_prefs")
    def test_hydrate_state_from_storage_handles_mixed_naive_and_aware_updated_at(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
        mock_load_report_payload,
    ):
        mock_load_prefs.return_value = {
            "speaker_id": "bern",
            "last_setup": {
                "speaker_id": "bern",
                "learning_language": "en",
                "language": "en",
                "cefr_level": "B2",
                "theme": "The pros and cons of working from home",
                "task_family": "opinion_monologue",
                "target_duration_sec": 120,
                "updated_at": "2026-03-20T15:20:00+00:00",
            },
            "speaker_profiles": {
                "bern": {
                    "speaker_id": "bern",
                    "learning_language": "en",
                    "language": "en",
                    "cefr_level": "B2",
                    "theme": "The pros and cons of working from home",
                    "task_family": "opinion_monologue",
                    "target_duration_sec": 120,
                    "updated_at": "2026-03-20T15:20:00+00:00",
                }
            },
            "log_dir": "reports",
        }
        mock_load_library.return_value = {
            "it": {
                "label": "Italiano",
                "themes": [
                    {
                        "title": "Il mio ultimo viaggio all'estero",
                        "level": "B1",
                        "task_family": "travel_narrative",
                    }
                ],
            },
            "en": {"label": "English", "themes": []},
        }
        mock_load_history.return_value = [
            mock.Mock(
                speaker_id="bern",
                learning_language="it",
                theme="Il mio ultimo viaggio all'estero",
                task_family="travel_narrative",
                target_duration_sec=90,
                timestamp=mock.Mock(isoformat=mock.Mock(return_value="2026-03-20T15:25:00")),
                report_path="reports/latest.json",
            )
        ]
        mock_load_report_payload.return_value = {
            "baseline_comparison": {"level": "B1"},
            "report": {
                "input": {
                    "speaker_id": "bern",
                    "expected_language": "it",
                    "theme": "Il mio ultimo viaggio all'estero",
                    "task_family": "travel_narrative",
                    "target_duration_sec": 90,
                }
            },
        }

        hydrated = hydrate_state_from_storage(AppShellState(prefs=AppPreferences()))

        self.assertEqual(hydrated.draft.learning_language, "it")
        self.assertEqual(hydrated.draft.cefr_level, "B1")
        self.assertEqual(hydrated.draft.theme_label, "Il mio ultimo viaggio all'estero")

    def test_save_state_preferences_preserves_saved_setup_when_only_app_settings_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_dir.joinpath("dashboard_prefs.json").write_text(
                """
{
  "speaker_id": "bern",
  "learning_language": "it",
  "language": "it",
  "cefr_level": "B1",
  "theme": "Il mio ultimo viaggio all'estero",
  "task_family": "travel_narrative",
  "target_duration_sec": 90,
  "last_setup": {
    "speaker_id": "bern",
    "learning_language": "it",
    "language": "it",
    "cefr_level": "B1",
    "theme": "Il mio ultimo viaggio all'estero",
    "task_family": "travel_narrative",
    "target_duration_sec": 90
  },
  "speaker_profiles": {
    "bern": {
      "speaker_id": "bern",
      "learning_language": "it",
      "language": "it",
      "cefr_level": "B1",
      "theme": "Il mio ultimo viaggio all'estero",
      "task_family": "travel_narrative",
      "target_duration_sec": 90
    }
  }
}
                """.strip(),
                encoding="utf-8",
            )
            state = AppShellState(
                prefs=AppPreferences(ui_locale="de", provider="openrouter", model="test-model", log_dir=str(log_dir)),
                draft=DraftSession(
                    speaker_id="bern",
                    learning_language="en",
                    learning_language_label="English",
                    cefr_level="B2",
                    theme_label="The pros and cons of working from home",
                    task_family="opinion_monologue",
                    duration_sec=120,
                ),
            )

            from app_shell.services import save_state_preferences

            save_state_preferences(state, persist_draft=False)
            stored = load_report_payload(log_dir / "dashboard_prefs.json")

        self.assertEqual(stored["ui_locale"], "de")
        self.assertEqual(stored["model"], "test-model")
        self.assertEqual(stored["speaker_id"], "bern")
        self.assertEqual(stored["learning_language"], "it")
        self.assertEqual(stored["cefr_level"], "B1")
        self.assertEqual(stored["last_setup"]["learning_language"], "it")
        self.assertEqual(stored["speaker_profiles"]["bern"]["learning_language"], "it")

    def test_save_state_preferences_writes_speaker_profile_for_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            state = AppShellState(
                prefs=AppPreferences(log_dir=str(log_dir)),
                draft=DraftSession(
                    speaker_id="bern",
                    learning_language="it",
                    learning_language_label="Italiano",
                    cefr_level="B1",
                    theme_label="Il mio ultimo viaggio all'estero",
                    task_family="travel_narrative",
                    duration_sec=90,
                ),
            )

            from app_shell.services import save_state_preferences

            save_state_preferences(state)
            stored = load_report_payload(log_dir / "dashboard_prefs.json")

        self.assertEqual(stored["speaker_id"], "bern")
        self.assertEqual(stored["learning_language"], "it")
        self.assertEqual(stored["last_setup"]["cefr_level"], "B1")
        self.assertEqual(stored["speaker_profiles"]["bern"]["task_family"], "travel_narrative")

    @mock.patch("app_shell.services.set_secret")
    def test_save_state_preferences_moves_runtime_secret_out_of_dashboard_prefs(self, mock_set_secret):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            state = AppShellState(
                prefs=AppPreferences(
                    log_dir=str(log_dir),
                    provider="openrouter",
                    llm_api_key="key-123",
                    openrouter_api_key="key-123",
                ),
            )

            from app_shell.services import save_state_preferences

            save_state_preferences(state, persist_draft=False)
            stored = load_report_payload(log_dir / "dashboard_prefs.json")

        self.assertNotIn("openrouter_api_key", stored)
        self.assertNotIn("llm_api_key", stored)
        mock_set_secret.assert_called_once()

    def test_set_default_provider_connection_promotes_requested_connection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = AppShellState(
                prefs=AppPreferences(
                    log_dir=tmpdir,
                    connections=[
                        ProviderConnection(
                            connection_id="primary",
                            provider_kind="openrouter",
                            label="Primary",
                            base_url="https://openrouter.ai/api/v1",
                            default_model="google/gemini-3.1-pro-preview",
                            is_default=True,
                        ),
                        ProviderConnection(
                            connection_id="local",
                            provider_kind="ollama",
                            label="Local",
                            base_url="http://localhost:11434",
                            default_model="llama3",
                            is_local=True,
                            provider_metadata={"deployment": "local"},
                        ),
                    ],
                    active_connection_id="primary",
                    setup_complete=True,
                )
            )

            updated = set_default_provider_connection(state, "local", persist_draft=False)

        self.assertTrue(updated)
        self.assertEqual(state.prefs.active_connection_id, "local")
        self.assertEqual(sum(1 for item in state.prefs.connections if item.is_default), 1)
        self.assertTrue(next(item for item in state.prefs.connections if item.connection_id == "local").is_default)

    @mock.patch("app_shell.services.delete_secret")
    def test_delete_provider_connection_removes_secret_and_promotes_remaining_connection(self, mock_delete_secret):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = AppShellState(
                prefs=AppPreferences(
                    log_dir=tmpdir,
                    llm_api_key="key-123",
                    connections=[
                        ProviderConnection(
                            connection_id="primary",
                            provider_kind="openrouter",
                            label="Primary",
                            base_url="https://openrouter.ai/api/v1",
                            default_model="google/gemini-3.1-pro-preview",
                            secret_ref="connection:primary",
                            is_default=True,
                        ),
                        ProviderConnection(
                            connection_id="backup",
                            provider_kind="lmstudio",
                            label="Backup",
                            base_url="http://localhost:1234/v1",
                            default_model="qwen2.5",
                            secret_ref="connection:backup",
                            provider_metadata={"deployment": "local", "token_optional": True},
                        ),
                    ],
                    active_connection_id="primary",
                    setup_complete=True,
                )
            )

            deleted = delete_provider_connection(state, "primary", persist_draft=False)

        self.assertTrue(deleted)
        self.assertEqual(len(state.prefs.connections), 1)
        self.assertEqual(state.prefs.active_connection_id, "backup")
        self.assertTrue(state.prefs.connections[0].is_default)
        mock_delete_secret.assert_called_once()

    @mock.patch("app_shell.services.llm_health_check")
    @mock.patch("app_shell.services.test_llm_connection")
    def test_test_runtime_connection_uses_provider_specific_health_check(self, mock_test_connection, mock_health_check):
        mock_health_check.return_value = {
            "provider": "ollama",
            "endpoint": "http://localhost:11434/api/tags",
            "payload": {"models": [{"name": "llama3"}]},
        }
        mock_test_connection.return_value = {"ok": True, "content_preview": "OK"}

        result = test_runtime_connection(
            provider="ollama_local",
            provider_choice="ollama_local",
            model="llama3",
            base_url="http://localhost:11434",
        )

        self.assertEqual(result["health_endpoint"], "http://localhost:11434/api/tags")
        self.assertEqual(result["health_payload"]["models"][0]["name"], "llama3")
        self.assertEqual(result["models_payload"]["models"][0]["name"], "llama3")
        mock_health_check.assert_called_once()
        mock_test_connection.assert_called_once()

    def test_whisper_model_status_marks_uncached_model(self):
        status = whisper_model_status("definitely-not-a-real-whisper-model")
        self.assertFalse(status["cached"])
        self.assertEqual(status["model"], "definitely-not-a-real-whisper-model")

    def test_validate_theme_submission_flags_missing_fields(self):
        errors = validate_theme_submission(
            manage_mode=NEW_LANGUAGE_OPTION,
            language_code="",
            language_label_text="",
            theme_title="",
        )
        self.assertEqual(errors["language_code"], "language_code")
        self.assertEqual(errors["language_label"], "language_label")
        self.assertEqual(errors["theme_title"], "theme_title")

    def test_theme_entry_id_is_stable(self):
        self.assertEqual(
            theme_entry_id({"title": "Il mio ultimo viaggio all'estero", "level": "B1"}),
            "b1-il-mio-ultimo-viaggio-all-estero",
        )

    def test_parse_cli_json_extracts_embedded_json(self):
        payload = parse_cli_json("noise\n{\"report\": {\"scores\": {\"final\": 4.0}}}\ntrailer")
        self.assertEqual(payload["report"]["scores"]["final"], 4.0)

    def test_store_uploaded_audio_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path, digest = store_uploaded_audio(_FakeUpload(b"abc", "sample.wav"), target_dir=tmpdir)
            self.assertTrue(Path(path).exists())
            self.assertTrue(digest)

    def test_create_assessment_request_keeps_feedback_language(self):
        request = create_assessment_request(
            audio_path=Path("sample.wav"),
            log_dir="reports",
            whisper="large-v3",
            provider="openrouter",
            llm_model="google/gemini-3.1-pro-preview",
            expected_language="it",
            feedback_language="en",
            speaker_id="bern",
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
            openrouter_api_key="key-123",
            openrouter_http_referer="http://localhost:8503",
            openrouter_app_title="Speaking Studio",
        )
        self.assertEqual(request["expected_language"], "it")
        self.assertEqual(request["feedback_language"], "en")
        self.assertEqual(request["openrouter_api_key"], "key-123")

    @mock.patch("app_shell.services.subprocess.run")
    def test_execute_assessment_request_passes_language_profile_key_to_cli(self, mock_run):
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout='{"report": {"session_id": "sess-1"}, "transcript_full": "ciao", "transcript_preview": "ciao"}',
            stderr="",
        )
        payload, error = execute_assessment_request(
            {
                "audio_path": "sample.wav",
                "log_dir": "reports",
                "whisper": "large-v3",
                "provider": "openrouter",
                "llm_model": "google/gemini-3.1-pro-preview",
                "expected_language": "en",
                "language_profile_key": "en",
                "feedback_language": "en",
                "speaker_id": "bern",
                "task_family": "travel_narrative",
                "theme": "Remote work",
                "target_duration_sec": 180,
            }
        )
        self.assertIsNone(error)
        self.assertEqual(payload["report"]["session_id"], "sess-1")
        command = mock_run.call_args.args[0]
        self.assertIn("--language-profile-key", command)
        self.assertIn("en", command)

    @mock.patch("app_shell.services.subprocess.run")
    def test_execute_assessment_request_passes_feedback_language_to_cli(self, mock_run):
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout='{"report": {"session_id": "sess-1"}, "transcript_full": "ciao", "transcript_preview": "ciao"}',
            stderr="",
        )
        payload, error = execute_assessment_request(
            {
                "audio_path": "sample.wav",
                "log_dir": "reports",
                "whisper": "large-v3",
                "provider": "openrouter",
                "llm_model": "google/gemini-3.1-pro-preview",
                "expected_language": "it",
                "feedback_language": "en",
                "openrouter_api_key": "key-123",
                "openrouter_http_referer": "http://localhost:8503",
                "openrouter_app_title": "Speaking Studio",
                "speaker_id": "bern",
                "task_family": "travel_narrative",
                "theme": "Il mio ultimo viaggio all'estero",
                "target_duration_sec": 180,
            }
        )
        self.assertIsNone(error)
        self.assertEqual(payload["report"]["session_id"], "sess-1")
        command = mock_run.call_args.args[0]
        env = mock_run.call_args.kwargs["env"]
        self.assertIn("--feedback-language", command)
        self.assertIn("en", command)
        self.assertEqual(env["OPENROUTER_API_KEY"], "key-123")
        self.assertEqual(env["OPENROUTER_HTTP_REFERER"], "http://localhost:8503")
        self.assertEqual(env["OPENROUTER_APP_TITLE"], "Speaking Studio")

    @mock.patch("app_shell.services.load_latest_report_payload")
    @mock.patch("app_shell.services.subprocess.run")
    def test_execute_assessment_request_prefers_logged_payload_when_stdout_only_has_preview(
        self,
        mock_run,
        mock_load_latest,
    ):
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout='{"report": {"session_id": "sess-1"}, "transcript_preview": "Short preview"}',
            stderr="",
        )
        mock_load_latest.return_value = {
            "report": {"session_id": "sess-1"},
            "transcript_full": "Long full transcript",
            "transcript_preview": "Short preview",
        }

        payload, error = execute_assessment_request(
            {
                "audio_path": "sample.wav",
                "log_dir": "reports",
                "whisper": "large-v3",
                "provider": "openrouter",
                "llm_model": "google/gemini-3.1-pro-preview",
                "expected_language": "it",
                "feedback_language": "it",
                "speaker_id": "bern",
                "task_family": "opinion_monologue",
                "theme": "Lavoro da remoto",
                "target_duration_sec": 180,
            }
        )

        self.assertIsNone(error)
        self.assertEqual(payload["transcript_full"], "Long full transcript")
        self.assertEqual(payload["transcript_preview"], "Short preview")
        mock_load_latest.assert_called_once_with("reports", label="")

    def test_review_summary_extracts_coaching(self):
        summary = review_summary(
            {
                "transcript_full": "Hello world",
                "notes": "Remember to add examples.",
                "baseline_comparison": {
                    "level": "B2",
                    "comment": "Good pace.",
                    "targets": {
                        "wpm": {"expected": "100-150", "actual": 120, "ok": True}
                    }
                },
                "report": {
                    "session_id": "report-1",
                    "scores": {"final": 4.0, "band": "B2"},
                    "checks": {
                        "language_pass": True,
                        "topic_pass": False,
                        "duration_pass": True,
                        "min_words_pass": True,
                    },
                    "warnings": ["language_mismatch"],
                    "coaching": {
                        "coach_summary": "Solid structure.",
                        "strengths": ["Clear sequencing"],
                        "top_3_priorities": ["More detail"],
                        "next_focus": "Add examples",
                        "next_exercise": "Travel narrative",
                    },
                    "progress_delta": {
                        "previous_session_id": "sess-1",
                        "score_delta": {"final": 0.5}
                    }
                },
            }
        )
        self.assertEqual(summary["report_id"], "report-1")
        self.assertEqual(summary["band"], "B2")
        self.assertEqual(summary["notes"], "Remember to add examples.")
        self.assertEqual(summary["strengths"], ["Clear sequencing"])
        self.assertEqual(summary["priorities"], ["More detail"])
        self.assertEqual(summary["baseline"]["level"], "B2")
        self.assertEqual(summary["progress_items"][0]["kind"], "previous_session")

    def test_load_report_payload_reads_saved_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text('{"report": {"session_id": "sess-1"}, "transcript_full": "Ciao"}', encoding="utf-8")
            payload = load_report_payload(path)
        self.assertEqual(payload["report"]["session_id"], "sess-1")
        self.assertEqual(payload["transcript_full"], "Ciao")

    def test_review_summary_keeps_missing_gates_unknown(self):
        summary = review_summary(
            {
                "report": {
                    "session_id": "report-2",
                    "scores": {"final": 3.0, "band": "B1"},
                    "checks": {},
                    "coaching": {},
                }
            }
        )
        self.assertIsNone(summary["gates"]["language_pass"])
        self.assertEqual(summary["failed_gates"], [])


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_shell.services import (
    NEW_LANGUAGE_OPTION,
    build_client_snapshot,
    build_support_bundle_request,
    create_assessment_request,
    create_support_bundle_archive,
    discover_runtime_models,
    download_whisper_model,
    delete_provider_connection,
    execute_assessment_request,
    export_support_bundle_archive,
    hydrate_state_from_storage,
    list_sample_trials,
    load_history_detail_payload,
    load_report_payload,
    parse_cli_json,
    review_summary,
    sanitize_setup_base_url,
    set_default_provider_connection,
    store_uploaded_audio,
    submit_assessment_request,
    test_runtime_connection,
    theme_entry_id,
    whisper_model_status,
    validate_theme_submission,
)
from app_shell.diagnostics import StartupDiagnostic
from app_shell.state import AppPreferences, AppShellState, ProviderConnection
from app_shell.state import DraftSession


class _FakeUpload:
    def __init__(self, data: bytes, name: str) -> None:
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


class AppShellServiceTests(unittest.TestCase):
    @mock.patch.dict(
        os.environ,
        {"APP_SHELL_SKIP_BOOTSTRAP": "", "LLM_API_KEY": "", "OPENROUTER_API_KEY": "", "OLLAMA_API_KEY": ""},
        clear=False,
    )
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_workspace_prefs")
    def test_hydrate_state_from_storage_restores_openrouter_preferences_without_legacy_key(
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
        self.assertEqual(hydrated.prefs.llm_api_key, "")
        self.assertEqual(hydrated.prefs.openrouter_http_referer, "https://example.test/app")
        self.assertEqual(hydrated.prefs.openrouter_app_title, "Assess Speaking Dev")

    @mock.patch.dict(
        os.environ,
        {"APP_SHELL_SKIP_BOOTSTRAP": "", "LLM_API_KEY": "", "OPENROUTER_API_KEY": "", "OLLAMA_API_KEY": ""},
        clear=False,
    )
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_workspace_prefs")
    def test_hydrate_state_from_storage_preserves_explicit_openrouter_app_title(
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

        self.assertEqual(hydrated.prefs.openrouter_app_title, "Assess Speaking")

    @mock.patch.dict(
        os.environ,
        {"APP_SHELL_SKIP_BOOTSTRAP": "", "LLM_API_KEY": "", "OPENROUTER_API_KEY": "", "OLLAMA_API_KEY": ""},
        clear=False,
    )
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_workspace_prefs")
    def test_hydrate_state_from_storage_ignores_legacy_plaintext_key(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
    ):
        mock_load_history.return_value = []
        mock_load_prefs.return_value = {
            "provider": "openrouter",
            "openrouter_api_key": "saved-key",
            "log_dir": "reports",
        }
        mock_load_library.return_value = {"it": {"label": "Italiano", "themes": []}}

        hydrated = hydrate_state_from_storage(AppShellState(prefs=AppPreferences()))

        self.assertEqual(hydrated.prefs.llm_api_key, "")

    @mock.patch.dict(
        os.environ,
        {"APP_SHELL_SKIP_BOOTSTRAP": "", "OPENROUTER_API_KEY": "env-key", "LLM_API_KEY": ""},
        clear=False,
    )
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_workspace_prefs")
    def test_hydrate_state_from_storage_ignores_environment_api_key(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
    ):
        mock_load_history.return_value = []
        mock_load_prefs.return_value = {
            "provider": "openrouter",
            "log_dir": "reports",
        }
        mock_load_library.return_value = {"it": {"label": "Italiano", "themes": []}}

        hydrated = hydrate_state_from_storage(AppShellState(prefs=AppPreferences()))

        self.assertEqual(hydrated.prefs.llm_api_key, "")

    @mock.patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": ""}, clear=False)
    @mock.patch("app_shell.services.load_report_payload")
    @mock.patch("app_shell.services.load_history_records")
    @mock.patch("app_shell.services.load_theme_library")
    @mock.patch("app_shell.services.load_workspace_prefs")
    def test_hydrate_state_from_storage_backfills_draft_from_latest_history_for_speaker(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
        mock_load_report_payload,
    ):
        mock_load_prefs.return_value = {
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

        hydrated = hydrate_state_from_storage(
            AppShellState(
                prefs=AppPreferences(),
                draft=DraftSession(speaker_id="bern"),
            )
        )

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
    @mock.patch("app_shell.services.load_workspace_prefs")
    def test_hydrate_state_from_storage_prefills_speaker_id_from_last_setup(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
        mock_load_report_payload,
    ):
        mock_load_prefs.return_value = {
            "last_setup": {
                "speaker_id": "bern",
                "learning_language": "it",
                "cefr_level": "B1",
                "theme": "Il mio ultimo viaggio all'estero",
                "task_family": "travel_narrative",
                "target_duration_sec": 90,
                "updated_at": "2026-03-20T15:20:00+00:00",
            },
            "speaker_profiles": {
                "bern": {
                    "speaker_id": "bern",
                    "learning_language": "it",
                    "cefr_level": "B1",
                    "theme": "Il mio ultimo viaggio all'estero",
                    "task_family": "travel_narrative",
                    "target_duration_sec": 90,
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
    @mock.patch("app_shell.services.load_workspace_prefs")
    def test_hydrate_state_from_storage_handles_mixed_naive_and_aware_updated_at(
        self,
        mock_load_prefs,
        mock_load_library,
        mock_load_history,
        mock_load_report_payload,
    ):
        mock_load_prefs.return_value = {
            "last_setup": {
                "speaker_id": "bern",
                "learning_language": "en",
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
            log_dir.joinpath("workspace_prefs.json").write_text(
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
    "cefr_level": "B1",
    "theme": "Il mio ultimo viaggio all'estero",
    "task_family": "travel_narrative",
    "target_duration_sec": 90
  },
  "speaker_profiles": {
    "bern": {
      "speaker_id": "bern",
      "learning_language": "it",
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
            stored = load_report_payload(log_dir / "workspace_prefs.json")

        self.assertEqual(stored["ui_locale"], "de")
        self.assertEqual(stored["model"], "test-model")
        self.assertNotIn("speaker_id", stored)
        self.assertNotIn("learning_language", stored)
        self.assertNotIn("language", stored)
        self.assertNotIn("cefr_level", stored)
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
            stored = load_report_payload(log_dir / "workspace_prefs.json")

        self.assertNotIn("speaker_id", stored)
        self.assertNotIn("learning_language", stored)
        self.assertNotIn("language", stored)
        self.assertEqual(stored["last_setup"]["cefr_level"], "B1")
        self.assertEqual(stored["speaker_profiles"]["bern"]["task_family"], "travel_narrative")

    @mock.patch("app_shell.services.set_secret")
    def test_save_state_preferences_does_not_persist_runtime_secret_without_saved_connection(self, mock_set_secret):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            state = AppShellState(
                prefs=AppPreferences(
                    log_dir=str(log_dir),
                    provider="openrouter",
                    llm_api_key="key-123",
                ),
            )

            from app_shell.services import save_state_preferences

            save_state_preferences(state, persist_draft=False)
            stored = load_report_payload(log_dir / "workspace_prefs.json")

        self.assertNotIn("openrouter_api_key", stored)
        self.assertNotIn("llm_api_key", stored)
        mock_set_secret.assert_not_called()

    @mock.patch("app_shell.services.secret_store_status")
    @mock.patch("app_shell.services.resolve_connection_runtime")
    def test_build_client_snapshot_marks_missing_saved_credentials(
        self,
        mock_resolve_runtime,
        mock_secret_status,
    ):
        mock_resolve_runtime.return_value = mock.Mock(
            provider="openrouter",
            model="google/gemini-3.1-pro-preview",
            api_key="",
        )
        mock_secret_status.return_value = mock.Mock(persistent=True)
        state = AppShellState(
            prefs=AppPreferences(
                connections=[
                    ProviderConnection(
                        connection_id="primary",
                        provider_kind="openrouter",
                        label="Primary",
                        base_url="https://openrouter.ai/api/v1",
                        default_model="google/gemini-3.1-pro-preview",
                        secret_ref="connection:primary",
                        is_default=True,
                    )
                ],
                active_connection_id="primary",
            )
        )

        snapshot = build_client_snapshot(state)

        self.assertTrue(snapshot["has_active_connection"])
        self.assertTrue(snapshot["provider_requires_auth"])
        self.assertFalse(snapshot["has_saved_secret"])
        self.assertTrue(snapshot["credentials_missing"])
        self.assertEqual(snapshot["credential_state"], "missing")

    @mock.patch("app_shell.services.secret_store_status")
    @mock.patch("app_shell.services.resolve_connection_runtime")
    def test_build_client_snapshot_marks_saved_credentials_available(
        self,
        mock_resolve_runtime,
        mock_secret_status,
    ):
        mock_resolve_runtime.return_value = mock.Mock(
            provider="openrouter",
            model="google/gemini-3.1-pro-preview",
            api_key="saved-key",
        )
        mock_secret_status.return_value = mock.Mock(persistent=True)
        state = AppShellState(
            prefs=AppPreferences(
                connections=[
                    ProviderConnection(
                        connection_id="primary",
                        provider_kind="openrouter",
                        label="Primary",
                        base_url="https://openrouter.ai/api/v1",
                        default_model="google/gemini-3.1-pro-preview",
                        secret_ref="connection:primary",
                        is_default=True,
                    )
                ],
                active_connection_id="primary",
            )
        )

        snapshot = build_client_snapshot(state)

        self.assertTrue(snapshot["has_saved_secret"])
        self.assertFalse(snapshot["credentials_missing"])
        self.assertEqual(snapshot["credential_state"], "saved")

    @mock.patch("app_shell.diagnostics.collect_startup_diagnostics")
    @mock.patch("app_shell.services.secret_store_status")
    @mock.patch("app_shell.services.resolve_connection_runtime")
    def test_build_support_bundle_request_keeps_only_sanitized_snapshot_and_diagnostics(
        self,
        mock_resolve_runtime,
        mock_secret_status,
        mock_collect_startup_diagnostics,
    ):
        mock_resolve_runtime.return_value = mock.Mock(
            provider="openrouter",
            model="google/gemini-3.1-pro-preview",
            api_key="saved-key",
        )
        mock_secret_status.return_value = mock.Mock(persistent=True)
        mock_collect_startup_diagnostics.return_value = [
            StartupDiagnostic(
                key="runtime_api_key",
                status="error",
                title_key="diagnostics.runtime_api_key_title",
                detail_key="diagnostics.runtime_api_key_error_detail",
                detail_args={
                    "provider": "openrouter",
                    "api_key": "diag-secret",
                    "secret_ref": "connection:primary",
                    "nested": {
                        "llm_api_key": "nested-secret",
                    },
                },
            )
        ]
        state = AppShellState(
            prefs=AppPreferences(
                connections=[
                    ProviderConnection(
                        connection_id="primary",
                        provider_kind="openrouter",
                        label="Primary",
                        base_url="https://openrouter.ai/api/v1",
                        default_model="google/gemini-3.1-pro-preview",
                        secret_ref="connection:primary",
                        is_default=True,
                    )
                ],
                active_connection_id="primary",
                log_dir="/tmp/vostavo-support",
            )
        )

        request = build_support_bundle_request(state)

        self.assertFalse(request["include_reports"])
        self.assertFalse(request["include_recordings"])
        self.assertFalse(request["include_uploads"])
        self.assertTrue(request["client_snapshot"]["has_saved_secret"])
        self.assertNotIn("secret_ref", request["client_snapshot"])
        self.assertEqual(request["client_diagnostics"][0]["detail_args"]["api_key"], "[redacted]")
        self.assertNotIn("secret_ref", request["client_diagnostics"][0]["detail_args"])
        self.assertEqual(
            request["client_diagnostics"][0]["detail_args"]["nested"]["llm_api_key"],
            "[redacted]",
        )
        mock_collect_startup_diagnostics.assert_called_once_with(state, include_runtime_health=False)

    @mock.patch("app_shell.services.backend_client.create_support_bundle")
    @mock.patch("app_shell.services.build_support_bundle_request")
    def test_create_support_bundle_archive_uses_backend_client_with_log_dir(
        self,
        mock_build_request,
        mock_create_support_bundle,
    ):
        mock_build_request.return_value = {"client_snapshot": {"has_active_connection": True}}
        mock_create_support_bundle.return_value = mock.Mock(bundle_id="bundle_123")
        state = AppShellState(prefs=AppPreferences(log_dir="/tmp/vostavo-support"))

        created = create_support_bundle_archive(state, include_reports=True)

        self.assertEqual(created.bundle_id, "bundle_123")
        mock_build_request.assert_called_once_with(
            state,
            include_reports=True,
            include_recordings=False,
            include_uploads=False,
            include_runtime_health=False,
        )
        mock_create_support_bundle.assert_called_once_with(
            mock_build_request.return_value,
            log_dir="/tmp/vostavo-support",
        )

    @mock.patch("app_shell.services.backend_client.download_support_bundle")
    @mock.patch("app_shell.services.create_support_bundle_archive")
    def test_export_support_bundle_archive_downloads_created_bundle(
        self,
        mock_create_support_bundle_archive,
        mock_download_support_bundle,
    ):
        mock_create_support_bundle_archive.return_value = mock.Mock(bundle_id="bundle_456")
        mock_download_support_bundle.return_value = Path("/tmp/support-bundle.zip")
        state = AppShellState(prefs=AppPreferences(log_dir="/tmp/vostavo-support"))

        exported = export_support_bundle_archive(state, destination="/tmp/downloads")

        self.assertEqual(exported, Path("/tmp/support-bundle.zip"))
        mock_create_support_bundle_archive.assert_called_once_with(
            state,
            include_reports=False,
            include_recordings=False,
            include_uploads=False,
            include_runtime_health=False,
        )
        mock_download_support_bundle.assert_called_once_with(
            "bundle_456",
            destination="/tmp/downloads",
            log_dir="/tmp/vostavo-support",
        )

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

    def test_sanitize_setup_base_url_strips_local_ollama_suffixes(self):
        self.assertEqual(sanitize_setup_base_url("ollama_local", "http://localhost:11434/api"), "http://localhost:11434")
        self.assertEqual(sanitize_setup_base_url("ollama_local", "http://localhost:11434/v1"), "http://localhost:11434")
        self.assertEqual(sanitize_setup_base_url("ollama_local", "http://localhost:11434/api/v1"), "http://localhost:11434")

    @mock.patch("app_shell.services.llm_health_check")
    def test_discover_runtime_models_sanitizes_local_ollama_url(self, mock_health_check):
        mock_health_check.return_value = {
            "provider": "ollama",
            "endpoint": "http://localhost:11434/api/tags",
            "payload": {"models": [{"name": "llama3"}, {"name": "mistral"}]},
        }

        result = discover_runtime_models(
            provider="ollama_local",
            provider_choice="ollama_local",
            base_url="http://localhost:11434/api",
        )

        self.assertEqual(result["health_endpoint"], "http://localhost:11434/api/tags")
        self.assertEqual(result["models"], ["llama3", "mistral"])
        self.assertEqual(mock_health_check.call_args.kwargs["base_url"], "http://localhost:11434")

    @mock.patch("app_shell.services.llm_health_check")
    @mock.patch("app_shell.services.test_llm_connection")
    def test_test_runtime_connection_uses_discovered_model_when_input_is_blank(self, mock_test_connection, mock_health_check):
        mock_health_check.return_value = {
            "provider": "lmstudio",
            "endpoint": "http://localhost:1234/v1/models",
            "payload": {"data": [{"id": "qwen2.5"}]},
        }
        mock_test_connection.return_value = {"ok": True, "content_preview": "OK", "model": "qwen2.5"}

        result = test_runtime_connection(
            provider="lmstudio_local",
            provider_choice="lmstudio_local",
            model="",
            base_url="http://localhost:1234/v1",
        )

        self.assertEqual(result["health_endpoint"], "http://localhost:1234/v1/models")
        self.assertEqual(result["models_payload"]["data"][0]["id"], "qwen2.5")
        self.assertEqual(mock_test_connection.call_args.kwargs["model"], "qwen2.5")

    @mock.patch("app_shell.services.llm_health_check")
    @mock.patch("app_shell.services.test_llm_connection")
    def test_test_runtime_connection_sanitizes_local_ollama_base_url(self, mock_test_connection, mock_health_check):
        mock_health_check.return_value = {
            "provider": "ollama",
            "endpoint": "http://localhost:11434/api/tags",
            "payload": {"models": [{"name": "llama3"}]},
        }
        mock_test_connection.return_value = {"ok": True, "content_preview": "OK", "model": "llama3"}

        result = test_runtime_connection(
            provider="ollama_local",
            provider_choice="ollama_local",
            model="llama3",
            base_url="http://localhost:11434/api",
        )

        self.assertEqual(result["base_url"], "http://localhost:11434/v1")
        self.assertEqual(result["service_base_url"], "http://localhost:11434")
        self.assertEqual(mock_health_check.call_args.kwargs["base_url"], "http://localhost:11434")
        self.assertEqual(mock_test_connection.call_args.kwargs["base_url"], "http://localhost:11434/v1")
        self.assertEqual(mock_test_connection.call_args.kwargs["timeout_sec"], 30.0)

    @mock.patch("app_shell.services.llm_health_check")
    @mock.patch("app_shell.services.test_llm_connection")
    def test_test_runtime_connection_keeps_default_timeout_for_remote_provider(self, mock_test_connection, mock_health_check):
        mock_health_check.return_value = {
            "provider": "openrouter",
            "endpoint": "https://openrouter.ai/api/v1/models",
            "payload": {"data": [{"id": "google/gemini-3.1-pro-preview"}]},
        }
        mock_test_connection.return_value = {
            "ok": True,
            "content_preview": "OK",
            "model": "google/gemini-3.1-pro-preview",
        }

        test_runtime_connection(
            provider="openrouter",
            provider_choice="openrouter",
            model="google/gemini-3.1-pro-preview",
            base_url="https://openrouter.ai/api/v1",
        )

        self.assertEqual(mock_test_connection.call_args.kwargs["timeout_sec"], 10.0)

    def test_whisper_model_status_marks_uncached_model(self):
        status = whisper_model_status("definitely-not-a-real-whisper-model")
        self.assertFalse(status["cached"])
        self.assertEqual(status["model"], "definitely-not-a-real-whisper-model")

    @mock.patch("app_shell.services.whisper_model_status", return_value={"cached": True, "cached_path": "/tmp/model"})
    @mock.patch("app_shell.services.ensure_model_downloaded")
    def test_download_whisper_model_passes_progress_callback(self, mock_ensure_downloaded, mock_status):
        callback = mock.Mock()

        result = download_whisper_model("small", progress_callback=callback)

        self.assertEqual(result["cached_path"], "/tmp/model")
        mock_ensure_downloaded.assert_called_once_with("small", progress_callback=callback)
        mock_status.assert_called_once_with("small")

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
            llm_api_key="key-123",
            openrouter_http_referer="http://localhost:8503",
            openrouter_app_title="Vostavo",
        )
        self.assertEqual(request["expected_language"], "it")
        self.assertEqual(request["feedback_language"], "en")
        self.assertEqual(request["llm_api_key"], "key-123")

    @mock.patch.dict(os.environ, {"ASSESS_SPEAKING_DRY_RUN": "1"}, clear=False)
    def test_create_assessment_request_reads_dry_run_from_environment(self):
        request = create_assessment_request(
            audio_path=Path("sample.wav"),
            log_dir="reports",
            whisper="small",
            provider="ollama",
            llm_model="llama3",
            expected_language="it",
            feedback_language="en",
            speaker_id="bern",
            task_family="travel_narrative",
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=90,
        )

        self.assertTrue(request["dry_run"])

    @mock.patch("app_shell.services.time.sleep")
    @mock.patch("app_shell.services.backend_client.get_assessment_status")
    @mock.patch("app_shell.services.backend_client.create_assessment")
    @mock.patch("app_shell.services.backend_client.upload_audio_path")
    def test_execute_assessment_request_uses_backend_and_keeps_language_profile_key(
        self,
        mock_upload,
        mock_create,
        mock_status,
        _mock_sleep,
    ):
        mock_upload.return_value = mock.Mock(audio_id="aud_1")
        mock_create.return_value = mock.Mock(assessment_id="asmt_1")
        mock_status.return_value = mock.Mock(
            status=mock.Mock(value="completed"),
            payload={"report": {"session_id": "sess-1"}, "transcript_full": "ciao", "transcript_preview": "ciao"},
            report_path=None,
            error=None,
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
        request = mock_create.call_args.args[0]
        self.assertEqual(request["audio_id"], "aud_1")
        self.assertEqual(request["language_profile_key"], "en")

    @mock.patch("app_shell.services.time.sleep")
    @mock.patch("app_shell.services.backend_client.get_assessment_status")
    @mock.patch("app_shell.services.backend_client.create_assessment")
    @mock.patch("app_shell.services.backend_client.upload_audio_path")
    def test_execute_assessment_request_returns_backend_failure_detail(
        self,
        mock_upload,
        mock_create,
        mock_status,
        _mock_sleep,
    ):
        mock_upload.return_value = mock.Mock(audio_id="aud_1")
        mock_create.return_value = mock.Mock(assessment_id="asmt_1")
        mock_status.return_value = mock.Mock(
            status=mock.Mock(value="failed"),
            payload=None,
            report_path=None,
            error=mock.Mock(detail="provider offline"),
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
                "llm_api_key": "key-123",
                "openrouter_http_referer": "http://localhost:8503",
                "openrouter_app_title": "Vostavo",
                "speaker_id": "bern",
                "task_family": "travel_narrative",
                "theme": "Il mio ultimo viaggio all'estero",
                "target_duration_sec": 180,
            }
        )
        self.assertIsNone(payload)
        self.assertEqual(error, "provider offline")
        request = mock_create.call_args.args[0]
        self.assertEqual(request["feedback_language"], "en")
        self.assertEqual(request["llm_api_key"], "key-123")
        self.assertEqual(request["openrouter_http_referer"], "http://localhost:8503")
        self.assertEqual(request["openrouter_app_title"], "Vostavo")

    @mock.patch("app_shell.services.backend_client.upload_audio_path", side_effect=RuntimeError('{"code":"backend_unavailable","detail":"offline"}'))
    def test_execute_assessment_request_returns_backend_unavailable_error(
        self,
        _mock_upload,
    ):
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

        self.assertIsNone(payload)
        self.assertEqual(error, "offline")

    @mock.patch("app_shell.services.backend_client.create_assessment")
    @mock.patch("app_shell.services.backend_client.upload_audio_path")
    @mock.patch("app_shell.services.llm_health_check")
    def test_submit_assessment_request_rejects_missing_local_model_before_upload(
        self,
        mock_health_check,
        mock_upload,
        mock_create,
    ):
        mock_health_check.return_value = {
            "provider": "ollama",
            "endpoint": "http://localhost:11434/api/tags",
            "payload": {"models": [{"name": "qwen3.5:latest"}, {"name": "qwen3:latest"}]},
        }

        job, error = submit_assessment_request(
            {
                "audio_path": "sample.wav",
                "log_dir": "reports",
                "whisper": "small",
                "provider": "ollama",
                "llm_model": "llama3",
                "llm_base_url": "http://localhost:11434/v1",
                "expected_language": "it",
                "feedback_language": "it",
                "speaker_id": "bern",
                "task_family": "travel_narrative",
                "theme": "Il mio ultimo viaggio all'estero",
                "target_duration_sec": 90,
            }
        )

        self.assertIsNone(job)
        self.assertIn("Configured Ollama model 'llama3' is not currently available.", error)
        self.assertIn("qwen3.5:latest", error)
        mock_upload.assert_not_called()
        mock_create.assert_not_called()

    @mock.patch("app_shell.services.backend_client.create_assessment")
    @mock.patch("app_shell.services.backend_client.upload_audio_path")
    @mock.patch("app_shell.services.llm_health_check")
    def test_submit_assessment_request_skips_local_validation_for_dry_run(
        self,
        mock_health_check,
        mock_upload,
        mock_create,
    ):
        mock_upload.return_value = mock.Mock(audio_id="aud_1")
        mock_create.return_value = mock.Mock(
            assessment_id="asmt_1",
            status=mock.Mock(value="queued"),
        )

        job, error = submit_assessment_request(
            {
                "audio_path": "sample.wav",
                "log_dir": "reports",
                "whisper": "small",
                "provider": "ollama",
                "llm_model": "llama3",
                "llm_base_url": "http://localhost:11434/v1",
                "expected_language": "it",
                "feedback_language": "it",
                "speaker_id": "bern",
                "task_family": "travel_narrative",
                "theme": "Il mio ultimo viaggio all'estero",
                "target_duration_sec": 90,
                "dry_run": True,
            }
        )

        self.assertIsNone(error)
        self.assertIsNotNone(job)
        self.assertEqual(job.assessment_id, "asmt_1")
        mock_health_check.assert_not_called()
        self.assertTrue(mock_create.call_args.args[0]["dry_run"])

    @mock.patch("app_shell.services.backend_client.load_history_detail")
    def test_load_history_detail_payload_returns_backend_payload(self, mock_load_history_detail):
        mock_load_history_detail.return_value = {"report": {"session_id": "sess-1"}}

        payload, error = load_history_detail_payload("sess-1", log_dir="reports")

        self.assertIsNone(error)
        self.assertEqual(payload, {"report": {"session_id": "sess-1"}})
        mock_load_history_detail.assert_called_once_with("sess-1", log_dir="reports")

    @mock.patch(
        "app_shell.services.backend_client.load_history_detail",
        side_effect=RuntimeError('{"code":"validation_error","detail":"missing"}'),
    )
    def test_load_history_detail_payload_returns_error_detail(self, _mock_load_history_detail):
        payload, error = load_history_detail_payload("sess-missing", log_dir="reports")

        self.assertIsNone(payload)
        self.assertEqual(error, "missing")

    @mock.patch("app_shell.services.backend_client.load_samples")
    def test_list_sample_trials_filters_language_and_level(self, mock_load_samples):
        mock_load_samples.return_value = [
            {
                "sample_id": "en_B1_travel_story",
                "language": "en",
                "cefr": "B1",
                "title": "travel story",
                "path": "/tmp/en-b1.wav",
            },
            {
                "sample_id": "it_B2_remote_work",
                "language": "it",
                "cefr": "B2",
                "title": "remote work",
                "path": "/tmp/it-b2.wav",
            },
        ]

        with mock.patch("app_shell.services.Path.exists", return_value=False):
            trials = list_sample_trials(log_dir="reports", language_code="it", cefr_level="B2")

        self.assertEqual(
            trials,
            [
                {
                    "sample_id": "it_B2_remote_work",
                    "language": "it",
                    "cefr": "B2",
                    "title": "remote work",
                    "path": "/tmp/it-b2.wav",
                    "available": False,
                }
            ],
        )

    @mock.patch("app_shell.services.backend_client.load_samples", side_effect=RuntimeError("offline"))
    def test_list_sample_trials_falls_back_to_local_samples(self, _mock_load_samples):
        with mock.patch("app_shell.services._local_sample_trials", return_value=[{"sample_id": "en_B1_travel_story", "language": "en", "cefr": "B1", "title": "travel story", "path": "/tmp/en.wav"}]), \
                mock.patch("app_shell.services.Path.exists", return_value=True):
            trials = list_sample_trials(language_code="en", cefr_level="B1")

        self.assertEqual(
            trials,
            [
                {
                    "sample_id": "en_B1_travel_story",
                    "language": "en",
                    "cefr": "B1",
                    "title": "travel story",
                    "path": "/tmp/en.wav",
                    "available": True,
                }
            ],
        )

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
                    "input": {"expected_language": "it"},
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
        self.assertEqual(summary["learning_language"], "it")
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

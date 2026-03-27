import os
from datetime import datetime
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from assessment_runtime import theme_library
from app_shell.state import APP_SHELL_STATE_KEY, AppPreferences, AppShellState, DraftSession, ProviderConnection, RecordingState, RecordingStatus, ReviewState

ROOT = Path(__file__).resolve().parents[1]
os.environ["APP_SHELL_SKIP_BOOTSTRAP"] = "1"


def _app_test(path: str) -> AppTest:
    return AppTest.from_file(str(ROOT / path), default_timeout=10)


def _active_runtime_prefs(
    *,
    ui_locale: str = "en",
    provider: str = "ollama",
    log_dir: str = "reports",
    connection_id: str = "primary",
    secret_ref: str = "",
) -> AppPreferences:
    default_models = {
        "ollama": "llama3",
        "lmstudio": "qwen2.5",
        "openrouter": "google/gemini-3.1-pro-preview",
        "openai_compatible": "model-1",
    }
    default_base_urls = {
        "ollama": "http://localhost:11434/v1",
        "lmstudio": "http://localhost:1234/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "openai_compatible": "https://example.com/v1",
    }
    provider_metadata = {}
    if provider == "openrouter":
        provider_metadata = {
            "http_referer": "http://localhost:8503",
            "app_title": "Speaking Studio",
        }
    model = default_models[provider]
    base_url = default_base_urls[provider]
    return AppPreferences(
        ui_locale=ui_locale,
        provider=provider,
        model=model,
        llm_base_url=base_url,
        log_dir=log_dir,
        active_connection_id=connection_id,
        connections=[
            ProviderConnection(
                connection_id=connection_id,
                provider_kind=provider,
                label=f"{provider.title()} Primary",
                base_url=base_url,
                default_model=model,
                auth_mode="bearer" if provider == "openrouter" or secret_ref else "none",
                secret_ref=secret_ref,
                is_default=True,
                is_local=provider in {"ollama", "lmstudio"},
                provider_metadata=provider_metadata,
            )
        ],
        setup_complete=True,
    )


def _history_record(
    *,
    timestamp: str,
    session_id: str,
    speaker_id: str,
    learning_language: str = "it",
    theme: str,
    task_family: str,
    final_score: float,
    band: int,
    overall: float | None = None,
    wpm: float | None = None,
    top_priorities: list[str] | None = None,
    grammar_error_categories: list[str] | None = None,
    coherence_issue_categories: list[str] | None = None,
    report_path: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=datetime.fromisoformat(timestamp),
        session_id=session_id,
        speaker_id=speaker_id,
        learning_language=learning_language,
        theme=theme,
        task_family=task_family,
        final_score=final_score,
        band=band,
        overall=overall,
        wpm=wpm,
        top_priorities=top_priorities or [],
        grammar_error_categories=grammar_error_categories or [],
        coherence_issue_categories=coherence_issue_categories or [],
        report_path=report_path,
    )


class AppShellPageTests(unittest.TestCase):
    def test_home_renders(self):
        at = _app_test("streamlit_app.py")
        at.run()
        self.assertEqual(at.session_state["_page_id"], "home")
        self.assertTrue(any(button.label for button in at.button if button.key == "home_start_new"))
        self.assertTrue(any(button.key == "home_guide" for button in at.button))

    def test_home_renders_with_german_locale(self):
        at = _app_test("streamlit_app.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="de")
        )
        at.run()
        self.assertTrue(any(button.label == "Neue Session starten" for button in at.button))

    def test_home_renders_runtime_setup_card_when_needed(self):
        at = _app_test("streamlit_app.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="de")
        )
        with patch("app_shell.services.needs_runtime_setup", return_value=True):
            at.run()
        self.assertIn("Runtime-Setup", [item.value for item in at.subheader])
        self.assertTrue(any(button.label == "Runtime-Setup oeffnen" for button in at.button))

    def test_runtime_setup_page_renders(self):
        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        at.run()
        self.assertEqual(at.session_state["_page_id"], "runtime_setup")
        self.assertTrue(any(button.key == "runtime_setup_download_model" for button in at.button))
        self.assertTrue(any(button.key == "runtime_setup_save_connection" for button in at.button))

    def test_runtime_setup_download_shows_success_after_progress_callback(self):
        def _fake_download(_model_name, progress_callback=None):
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "downloading",
                        "current_file": "model.bin",
                        "downloaded_bytes": 512,
                        "total_bytes": 1024,
                        "completed_files": 1,
                        "total_files": 2,
                    }
                )
                progress_callback({"stage": "ready", "cached_path": "/tmp/medium"})
            return {"cached_path": "/tmp/medium"}

        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        with patch("app_shell.services.download_whisper_model", side_effect=_fake_download):
            at.run()
            at.button(key="runtime_setup_download_model").click()
            at.run()

        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Whisper model is ready at /tmp/medium." in item.value for item in at.success))

    def test_runtime_setup_provider_switch_updates_dependent_fields(self):
        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en", provider="openrouter")
        )
        at.run()

        self.assertEqual(at.text_input(key="runtime_setup_base_url").value, "https://openrouter.ai/api/v1")
        self.assertEqual(at.text_input(key="runtime_setup_model").value, "google/gemini-3.1-pro-preview")

        at.selectbox(key="runtime_setup_provider_choice").set_value("ollama_local")
        at.run()

        self.assertEqual(at.text_input(key="runtime_setup_label").value, "Ollama Local")
        self.assertEqual(at.text_input(key="runtime_setup_base_url").value, "http://localhost:11434")
        self.assertEqual(at.text_input(key="runtime_setup_model").value, "llama3")
        self.assertTrue(at.text_input(key="runtime_setup_openrouter_http_referer").disabled)

    def test_runtime_setup_provider_switch_supports_lmstudio_and_openai_compatible(self):
        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en", provider="openrouter")
        )
        at.run()

        at.selectbox(key="runtime_setup_provider_choice").set_value("ollama_cloud")
        at.run()
        self.assertEqual(at.text_input(key="runtime_setup_base_url").value, "https://ollama.com/api")
        self.assertEqual(at.text_input(key="runtime_setup_model").value, "llama3")
        self.assertFalse(any(button.key == "runtime_setup_detect_local_models" for button in at.button))

        at.selectbox(key="runtime_setup_provider_choice").set_value("lmstudio_local")
        at.run()
        self.assertEqual(at.text_input(key="runtime_setup_base_url").value, "http://localhost:1234/v1")
        self.assertEqual(at.text_input(key="runtime_setup_model").value, "qwen2.5")
        self.assertTrue(any(button.key == "runtime_setup_detect_local_models" for button in at.button))

        at.selectbox(key="runtime_setup_provider_choice").set_value("openai_compatible")
        at.run()
        self.assertEqual(at.text_input(key="runtime_setup_model").value, "")
        self.assertFalse(any(button.key == "runtime_setup_detect_local_models" for button in at.button))

    def test_runtime_setup_existing_connection_notice_is_not_success_banner(self):
        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(
                ui_locale="en",
                setup_complete=True,
                active_connection_id="primary",
                connections=[
                    ProviderConnection(
                        connection_id="primary",
                        provider_kind="openrouter",
                        label="Primary",
                        base_url="https://openrouter.ai/api/v1",
                        default_model="google/gemini-3.1-pro-preview",
                        is_default=True,
                    )
                ],
            )
        )
        at.run()

        self.assertFalse(any("Runtime setup is already complete." in item.value for item in at.success))
        self.assertTrue(any("An active connection is already configured." in item.value for item in at.caption))

    def test_runtime_setup_explains_that_save_activates_connection(self):
        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )

        at.run()

        self.assertTrue(any(button.key == "runtime_setup_save_connection" and button.label == "Save and use connection" for button in at.button))
        self.assertTrue(
            any(
                "Saving also makes this the active connection." in item.value
                and "recommended but never block save" in item.value
                for item in at.caption
            )
        )

    def test_runtime_setup_detects_ollama_local_models_and_autofills(self):
        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en", provider="openrouter")
        )
        with patch(
            "app_shell.services.discover_runtime_models",
            return_value={
                "provider": "ollama",
                "base_url": "http://localhost:11434/v1",
                "health_endpoint": "http://localhost:11434/api/tags",
                "health_payload": {"models": [{"name": "mistral:latest"}, {"name": "llama3"}]},
                "models": ["mistral:latest", "llama3"],
            },
        ):
            at.run()
            at.selectbox(key="runtime_setup_provider_choice").set_value("ollama_local")
            at.run()
            at.text_input(key="runtime_setup_model").set_value("")
            at.button(key="runtime_setup_detect_local_models").click()
            at.run()

        self.assertEqual(at.text_input(key="runtime_setup_model").value, "mistral:latest")
        self.assertEqual(at.selectbox(key="runtime_setup_detected_model_choice").value, "mistral:latest")
        self.assertTrue(any("Detected 2 local model(s)" in item.value for item in at.info))

    def test_runtime_setup_detects_lmstudio_local_models_and_autofills(self):
        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en", provider="openrouter")
        )
        with patch(
            "app_shell.services.discover_runtime_models",
            return_value={
                "provider": "lmstudio",
                "base_url": "http://localhost:1234/v1",
                "health_endpoint": "http://localhost:1234/v1/models",
                "health_payload": {"data": [{"id": "qwen2.5-coder"}, {"id": "deepseek-r1"}]},
                "models": ["qwen2.5-coder", "deepseek-r1"],
            },
        ):
            at.run()
            at.selectbox(key="runtime_setup_provider_choice").set_value("lmstudio_local")
            at.run()
            at.text_input(key="runtime_setup_model").set_value("")
            at.button(key="runtime_setup_detect_local_models").click()
            at.run()

        self.assertEqual(at.text_input(key="runtime_setup_model").value, "qwen2.5-coder")
        self.assertEqual(at.selectbox(key="runtime_setup_detected_model_choice").value, "qwen2.5-coder")
        self.assertTrue(any("http://localhost:1234/v1/models" in item.value for item in at.info))

    def test_runtime_setup_test_error_shows_without_setup_complete_success_banner(self):
        at = _app_test("pages/00_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(
                ui_locale="en",
                setup_complete=True,
                provider="lmstudio",
                active_connection_id="primary",
                connections=[
                    ProviderConnection(
                        connection_id="primary",
                        provider_kind="lmstudio",
                        label="LM Studio",
                        base_url="http://localhost:1234/v1",
                        default_model="qwen2.5",
                        is_default=True,
                        provider_metadata={"deployment": "local", "token_optional": True},
                    )
                ],
            )
        )
        with patch("app_shell.services.test_runtime_connection", side_effect=RuntimeError("HTTP 404: model missing")):
            at.run()
            at.button(key="runtime_setup_test_connection").click()
            at.run()

        self.assertTrue(any("Connection test failed: HTTP 404: model missing" in item.value for item in at.warning))
        self.assertFalse(any("Runtime setup is already complete." in item.value for item in at.success))

    def test_runtime_setup_save_creates_first_connection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/00_Setup.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(ui_locale="en", log_dir=tmpdir)
            )
            at.run()
            at.selectbox(key="runtime_setup_provider_choice").set_value("openrouter")
            at.text_input(key="runtime_setup_model").set_value("google/gemini-3.1-pro-preview")
            at.text_input(key="runtime_setup_api_key").set_value("key-123")
            at.button(key="runtime_setup_save_connection").click()
            at.run()

            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertTrue(state.prefs.setup_complete)
            self.assertEqual(len(state.prefs.connections), 1)
            self.assertEqual(state.prefs.connections[0].provider_kind, "openrouter")
            self.assertEqual(state.prefs.connections[0].default_model, "google/gemini-3.1-pro-preview")
            self.assertEqual(state.prefs.llm_api_key, "key-123")

    def test_speak_guard_renders_without_setup(self):
        at = _app_test("pages/02_Speak.py")
        at.run()
        self.assertEqual(at.session_state["_page_id"], "speak")
        self.assertEqual(len(at.warning), 1)

    def test_setup_tolerates_invalid_persisted_cefr(self):
        at = _app_test("pages/01_Session_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="Z9",
            ),
        )
        at.run()
        self.assertEqual(at.session_state["_page_id"], "setup")
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any(selectbox.key == "setup_cefr" for selectbox in at.selectbox))

    def test_setup_defaults_to_first_library_theme(self):
        at = _app_test("pages/01_Session_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        at.run()
        self.assertEqual(at.selectbox(key="setup_theme_select").value, "Il mio ultimo viaggio all'estero")

    def test_setup_preserves_theme_selection_across_reruns(self):
        at = _app_test("pages/01_Session_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        at.run()
        at.selectbox(key="setup_learning_language").set_value("en")
        at.selectbox(key="setup_cefr").set_value("B2")
        at.run()
        at.selectbox(key="setup_theme_select").set_value("How travel habits have changed in recent years")
        at.run()
        at.select_slider(key="setup_duration").set_value(120)
        at.run()
        self.assertEqual(at.selectbox(key="setup_theme_select").value, "How travel habits have changed in recent years")

    def test_setup_restores_custom_theme_from_draft(self):
        at = _app_test("pages/01_Session_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="B1",
                theme_id="b1-cambiamento-climatico",
                theme_label="Cambiamento climatico",
                task_family="personal_experience",
                duration_sec=90,
                prompt_text="Parla del cambiamento climatico.",
            ),
        )
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.selectbox(key="setup_theme_select").value, "Custom theme")
        self.assertEqual(at.text_input(key="setup_custom_theme").value, "Cambiamento climatico")
        self.assertIn("Cambiamento climatico", [item.value for item in at.subheader])

    def test_setup_resets_invalid_theme_widget_state_when_options_change(self):
        at = _app_test("pages/01_Session_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        at.run()
        at.selectbox(key="setup_learning_language").set_value("en")
        at.selectbox(key="setup_cefr").set_value("B2")
        at.run()
        at.selectbox(key="setup_theme_select").set_value("How travel habits have changed in recent years")
        at.run()
        at.selectbox(key="setup_learning_language").set_value("it")
        at.selectbox(key="setup_cefr").set_value("B1")
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.selectbox(key="setup_theme_select").value, "Il mio ultimo viaggio all'estero")

    def test_setup_submit_updates_state_and_routes_to_runtime_setup_without_connection(self):
        at = _app_test("pages/01_Session_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        at.run()
        at.text_input(key="setup_speaker_id").set_value("bern")
        at.selectbox(key="setup_learning_language").set_value("en")
        at.selectbox(key="setup_cefr").set_value("B2")
        at.run()
        at.selectbox(key="setup_theme_select").set_value("The pros and cons of working from home")
        at.select_slider(key="setup_duration").set_value(120)
        at.button(key="setup_continue").click()
        at.run()

        state = at.session_state[APP_SHELL_STATE_KEY]
        self.assertEqual(state.draft.speaker_id, "bern")
        self.assertEqual(state.draft.learning_language, "en")
        self.assertEqual(state.draft.cefr_level, "B2")
        self.assertEqual(state.draft.theme_label, "The pros and cons of working from home")
        self.assertEqual(state.draft.task_family, "opinion_monologue")
        self.assertEqual(state.draft.duration_sec, 120)
        self.assertEqual(at.session_state["_next_page"], "pages/00_Setup.py")
        self.assertEqual(len(at.exception), 0)

    def test_setup_submit_routes_to_speak_when_runtime_connection_exists(self):
        at = _app_test("pages/01_Session_Setup.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=_active_runtime_prefs(ui_locale="en", provider="ollama")
        )
        at.run()
        at.text_input(key="setup_speaker_id").set_value("bern")
        at.selectbox(key="setup_learning_language").set_value("en")
        at.selectbox(key="setup_cefr").set_value("B2")
        at.run()
        at.selectbox(key="setup_theme_select").set_value("The pros and cons of working from home")
        at.select_slider(key="setup_duration").set_value(120)
        at.button(key="setup_continue").click()
        at.run()

        self.assertEqual(at.session_state["_next_page"], "pages/02_Speak.py")
        self.assertEqual(len(at.exception), 0)

    def test_setup_submit_can_save_custom_theme_for_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/01_Session_Setup.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(ui_locale="en", log_dir=tmpdir)
            )
            at.run()
            at.text_input(key="setup_speaker_id").set_value("bern")
            at.selectbox(key="setup_theme_select").set_value("Custom theme")
            at.run()
            at.text_input(key="setup_custom_theme").set_value("Cambiamento climatico")
            at.checkbox(key="setup_save_custom_theme").set_value(True)
            at.button(key="setup_continue").click()
            at.run()

            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.draft.theme_label, "Cambiamento climatico")
            self.assertEqual(at.session_state["_next_page"], "pages/00_Setup.py")

            persisted = theme_library.load_theme_library(Path(tmpdir))
            self.assertIn(
                {
                    "title": "Cambiamento climatico",
                    "level": "B1",
                    "task_family": "free_monologue",
                },
                persisted["it"]["themes"],
            )

    def test_speak_renders_prompt_when_setup_exists(self):
        at = _app_test("pages/02_Speak.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en", provider="ollama"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="B1",
                theme_id="viaggio",
                theme_label="Il mio ultimo viaggio all'estero",
                task_family="travel_narrative",
                duration_sec=90,
                prompt_id="viaggio-b1",
                prompt_text="Parla del tuo ultimo viaggio.",
            )
        )
        at.run()
        self.assertEqual(
            [warning.value for warning in at.warning],
            ["Configure Whisper and at least one inference connection before using the full runtime flow."],
        )
        self.assertTrue(any(button.key == "guard::pages/00_Setup.py" for button in at.button))

    def test_speak_renders_prompt_when_runtime_connection_exists(self):
        at = _app_test("pages/02_Speak.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=_active_runtime_prefs(ui_locale="en", provider="ollama"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="B1",
                theme_id="viaggio",
                theme_label="Il mio ultimo viaggio all'estero",
                task_family="travel_narrative",
                duration_sec=90,
                prompt_id="viaggio-b1",
                prompt_text="Parla del tuo ultimo viaggio.",
            )
        )
        at.run()
        self.assertEqual(len(at.warning), 0)
        self.assertTrue(any(button.key == "speak_submit" for button in at.button))

    def test_speak_warns_when_recording_file_is_missing(self):
        at = _app_test("pages/02_Speak.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=_active_runtime_prefs(ui_locale="en", provider="ollama"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="B1",
                theme_id="viaggio",
                theme_label="Il mio ultimo viaggio all'estero",
                task_family="travel_narrative",
                duration_sec=90,
                prompt_id="viaggio-b1",
                prompt_text="Parla del tuo ultimo viaggio.",
            ),
            recording=RecordingState(audio_path="/tmp/definitely-missing-audio.wav"),
        )
        at.run()
        self.assertEqual([warning.value for warning in at.warning], ["A recording was selected earlier, but the audio file is no longer available."])
        submit_buttons = [button for button in at.button if button.key == "speak_submit"]
        self.assertEqual(len(submit_buttons), 1)
        self.assertTrue(submit_buttons[0].disabled)

    def test_speak_keeps_saved_browser_recording_across_rerun_without_widget_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "recording.wav"
            audio_path.write_bytes(b"RIFF....WAVE")

            at = _app_test("pages/02_Speak.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=_active_runtime_prefs(ui_locale="en", provider="ollama", log_dir=tmpdir),
                draft=DraftSession(
                    session_id="draft-123",
                    speaker_id="bern",
                    learning_language="it",
                    learning_language_label="Italiano",
                    cefr_level="B1",
                    theme_id="viaggio",
                    theme_label="Il mio ultimo viaggio all'estero",
                    task_family="travel_narrative",
                    duration_sec=90,
                    prompt_id="viaggio-b1",
                    prompt_text="Parla del tuo ultimo viaggio.",
                ),
                recording=RecordingState(
                    status=RecordingStatus.READY,
                    audio_path=str(audio_path),
                    input_method="record",
                    input_digest="digest-123",
                ),
            )
            at.run()

            self.assertEqual(len(at.warning), 0)
            self.assertTrue(any("A recording is attached" in item.value for item in at.success))
            submit_buttons = [button for button in at.button if button.key == "speak_submit"]
            self.assertEqual(len(submit_buttons), 1)
            self.assertFalse(submit_buttons[0].disabled)
            self.assertEqual(
                at.session_state[APP_SHELL_STATE_KEY].recording.audio_path,
                str(audio_path),
            )

    def test_speak_switching_input_method_clears_previous_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "recordings" / "recording.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"RIFF....WAVE")

            at = _app_test("pages/02_Speak.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=_active_runtime_prefs(ui_locale="en", provider="ollama", log_dir=tmpdir),
                draft=DraftSession(
                    session_id="draft-123",
                    speaker_id="bern",
                    learning_language="it",
                    learning_language_label="Italiano",
                    cefr_level="B1",
                    theme_id="viaggio",
                    theme_label="Il mio ultimo viaggio all'estero",
                    task_family="travel_narrative",
                    duration_sec=90,
                    prompt_id="viaggio-b1",
                    prompt_text="Parla del tuo ultimo viaggio.",
                ),
                recording=RecordingState(
                    status=RecordingStatus.READY,
                    audio_path=str(audio_path),
                    input_method="record",
                    input_digest="digest-123",
                ),
            )
            at.session_state["speak_input_method"] = "upload"

            at.run()

            self.assertEqual(len(at.exception), 0)
            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.recording.audio_path, "")
            self.assertEqual(state.recording.input_method, "")
            self.assertFalse(audio_path.exists())
            self.assertEqual([item.value for item in at.info], ["No recording is attached yet."])
            submit_buttons = [button for button in at.button if button.key == "speak_submit"]
            self.assertEqual(len(submit_buttons), 1)
            self.assertTrue(submit_buttons[0].disabled)

    def test_speak_remove_recording_button_clears_attached_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "uploads" / "sample.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"RIFF....WAVE")

            at = _app_test("pages/02_Speak.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=_active_runtime_prefs(ui_locale="en", provider="ollama", log_dir=tmpdir),
                draft=DraftSession(
                    session_id="draft-123",
                    speaker_id="bern",
                    learning_language="it",
                    learning_language_label="Italiano",
                    cefr_level="B1",
                    theme_id="viaggio",
                    theme_label="Il mio ultimo viaggio all'estero",
                    task_family="travel_narrative",
                    duration_sec=90,
                    prompt_id="viaggio-b1",
                    prompt_text="Parla del tuo ultimo viaggio.",
                ),
                recording=RecordingState(
                    status=RecordingStatus.READY,
                    audio_path=str(audio_path),
                    input_method="upload",
                    input_digest="digest-123",
                ),
            )
            at.session_state["speak_input_method"] = "upload"

            at.run()
            at.button(key="speak_remove_recording").click()
            at.run()

            self.assertEqual(len(at.exception), 0)
            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.recording.audio_path, "")
            self.assertEqual(state.recording.input_method, "")
            self.assertFalse(audio_path.exists())
            self.assertEqual([item.value for item in at.info], ["No recording is attached yet."])

    def test_speak_warns_when_openrouter_is_selected_without_any_key(self):
        at = _app_test("pages/02_Speak.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=_active_runtime_prefs(ui_locale="en", provider="openrouter"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="en",
                learning_language_label="English",
                cefr_level="B2",
                theme_id="work-home",
                theme_label="The pros and cons of working from home",
                task_family="opinion_monologue",
                duration_sec=120,
                prompt_id="work-home-b2",
                prompt_text="Give your opinion on working from home.",
            ),
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            at.run()
        self.assertIn(
            "OpenRouter is selected, but no API key is configured in Settings and no OPENROUTER_API_KEY is available in the environment.",
            [warning.value for warning in at.warning],
        )

    def test_speak_skips_openrouter_warning_when_saved_key_exists(self):
        at = _app_test("pages/02_Speak.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=_active_runtime_prefs(ui_locale="en", provider="openrouter", secret_ref="saved-openrouter"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="en",
                learning_language_label="English",
                cefr_level="B2",
                theme_id="work-home",
                theme_label="The pros and cons of working from home",
                task_family="opinion_monologue",
                duration_sec=120,
                prompt_id="work-home-b2",
                prompt_text="Give your opinion on working from home.",
            ),
        )
        with patch("app_shell.runtime_resolver.get_secret", return_value="saved-key"), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            at.run()
        self.assertNotIn(
            "OpenRouter is selected, but no API key is configured in Settings and no OPENROUTER_API_KEY is available in the environment.",
            [warning.value for warning in at.warning],
        )

    def test_speak_skips_openrouter_warning_when_environment_key_exists(self):
        at = _app_test("pages/02_Speak.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=_active_runtime_prefs(ui_locale="en", provider="openrouter"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="en",
                learning_language_label="English",
                cefr_level="B2",
                theme_id="work-home",
                theme_label="The pros and cons of working from home",
                task_family="opinion_monologue",
                duration_sec=120,
                prompt_id="work-home-b2",
                prompt_text="Give your opinion on working from home.",
            ),
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=False):
            at.run()
        self.assertNotIn(
            "OpenRouter is selected, but no API key is configured in Settings and no OPENROUTER_API_KEY is available in the environment.",
            [warning.value for warning in at.warning],
        )

    def test_speak_preserves_label_and_notes_when_returning_from_setup(self):
        state = AppShellState(
            prefs=_active_runtime_prefs(ui_locale="en", provider="ollama"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="en",
                learning_language_label="English",
                cefr_level="B2",
                theme_id="work-home",
                theme_label="The pros and cons of working from home",
                task_family="opinion_monologue",
                duration_sec=120,
                prompt_id="work-home-b2",
                prompt_text="Give your opinion on working from home.",
            ),
        )
        speak = _app_test("pages/02_Speak.py")
        speak.session_state[APP_SHELL_STATE_KEY] = state
        speak.run()
        speak.text_input(key="speak_label").set_value("Morning run")
        speak.text_area(key="speak_notes").set_value("Mention two concrete examples.")
        speak.run()

        setup = _app_test("pages/01_Session_Setup.py")
        setup.session_state[APP_SHELL_STATE_KEY] = speak.session_state[APP_SHELL_STATE_KEY]
        setup.run()

        speak_again = _app_test("pages/02_Speak.py")
        speak_again.session_state[APP_SHELL_STATE_KEY] = setup.session_state[APP_SHELL_STATE_KEY]
        speak_again.run()
        self.assertEqual(speak_again.text_input(key="speak_label").value, "Morning run")
        self.assertEqual(speak_again.text_area(key="speak_notes").value, "Mention two concrete examples.")

    def test_speak_assessing_success_applies_review_and_marks_next_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "attempt.wav"
            audio_path.write_bytes(b"fake-audio")
            at = _app_test("pages/02_Speak.py")
            prefs = _active_runtime_prefs(ui_locale="en", provider="ollama", log_dir=tmpdir)
            prefs.provider = "openrouter"
            prefs.model = "stale-model"
            prefs.llm_base_url = "https://stale.example/v1"
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=prefs,
                draft=DraftSession(
                    session_id="draft-123",
                    speaker_id="bern",
                    learning_language="en",
                    learning_language_label="English",
                    cefr_level="B2",
                    theme_id="work-home",
                    theme_label="The pros and cons of working from home",
                    task_family="opinion_monologue",
                    duration_sec=120,
                    prompt_id="work-home-b2",
                    prompt_text="Give your opinion on working from home.",
                ),
                recording=RecordingState(
                    status=RecordingStatus.ASSESSING,
                    audio_path=str(audio_path),
                    input_method="upload",
                    label_input="Morning run",
                    notes_input="Mention two examples.",
                ),
            )
            at.session_state["speak_input_method"] = "upload"
            payload = {
                "notes": "Mention two examples.",
                "transcript_full": "Working from home can be efficient.",
                "report": {
                    "session_id": "report-1",
                    "scores": {"final": 4.1, "band": "B2"},
                    "checks": {"language_pass": True, "topic_pass": True, "duration_pass": True, "min_words_pass": True},
                    "coaching": {"coach_summary": "Clear structure."},
                },
            }
            with patch("app_shell.services.create_assessment_request", return_value={"ok": True}) as create_request, \
                    patch("app_shell.services.execute_assessment_request", return_value=(payload, None)):
                at.run()

            kwargs = create_request.call_args.kwargs
            self.assertEqual(kwargs["provider"], "ollama")
            self.assertEqual(kwargs["llm_model"], "llama3")
            self.assertEqual(kwargs["llm_base_url"], "http://localhost:11434/v1")
            self.assertEqual(at.session_state["_next_page"], "pages/03_Review.py")
            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.review.report_id, "report-1")
            self.assertEqual(state.review.transcript, "Working from home can be efficient.")
            self.assertEqual(state.review.band, "B2")

    def test_speak_assessing_error_sets_recording_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "attempt.wav"
            audio_path.write_bytes(b"fake-audio")
            at = _app_test("pages/02_Speak.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=_active_runtime_prefs(ui_locale="en", provider="ollama", log_dir=tmpdir),
                draft=DraftSession(
                    session_id="draft-123",
                    speaker_id="bern",
                    learning_language="en",
                    learning_language_label="English",
                    cefr_level="B2",
                    theme_id="work-home",
                    theme_label="The pros and cons of working from home",
                    task_family="opinion_monologue",
                    duration_sec=120,
                    prompt_id="work-home-b2",
                    prompt_text="Give your opinion on working from home.",
                ),
                recording=RecordingState(
                    status=RecordingStatus.ASSESSING,
                    audio_path=str(audio_path),
                    input_method="upload",
                ),
            )
            at.session_state["speak_input_method"] = "upload"
            with patch("app_shell.services.create_assessment_request", return_value={"ok": True}), \
                    patch("app_shell.services.execute_assessment_request", return_value=(None, "boom")):
                at.run()

            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.recording.error, "boom")
            self.assertEqual(state.recording.status, RecordingStatus.IDLE)

    def test_review_guard_renders_without_attempt(self):
        at = _app_test("pages/03_Review.py")
        at.run()
        self.assertEqual(at.session_state["_page_id"], "review")
        self.assertEqual(len(at.warning), 1)

    def test_review_renders_metrics_with_attempt(self):
        at = _app_test("pages/03_Review.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="B1",
                theme_id="viaggio",
                theme_label="Il mio ultimo viaggio all'estero",
                task_family="travel_narrative",
                duration_sec=90,
                prompt_id="viaggio-b1",
                prompt_text="Parla del tuo ultimo viaggio.",
            ),
            review=ReviewState(
                report_id="report-1",
                transcript="Parla del tuo ultimo viaggio.",
                score_overall=3.8,
                band="B1",
                summary="Placeholder summary.",
                payload={
                    "report": {
                        "session_id": "report-1",
                        "scores": {"final": 3.8, "band": "B1"},
                        "checks": {
                            "language_pass": True,
                            "topic_pass": True,
                            "duration_pass": True,
                            "min_words_pass": True,
                        },
                        "coaching": {},
                    }
                },
            ),
        )
        at.run()
        self.assertEqual(len(at.warning), 0)
        self.assertGreaterEqual(len(at.metric), 8)
        self.assertEqual(len(at.tabs), 0)
        self.assertEqual(at.text_area(key="review_transcript_view").value, "Parla del tuo ultimo viaggio.")
        self.assertTrue(any(button.key == "review_open_scoring_guide" for button in at.button))

    def test_scoring_guide_page_renders_sections(self):
        at = _app_test("pages/07_Scoring_Guide.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        at.run()
        self.assertEqual(at.session_state["_page_id"], "guide")
        subheaders = [item.value for item in at.subheader]
        self.assertIn("How to read one result", subheaders)
        self.assertIn("Final score formula", subheaders)
        self.assertIn("Deterministic score signals", subheaders)
        self.assertIn("Rubric score dimensions", subheaders)
        self.assertIn("Validation gates", subheaders)
        self.assertIn("Provisional CEFR estimate", subheaders)

    def test_review_renders_saved_notes_and_full_transcript(self):
        at = _app_test("pages/03_Review.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="B1",
                theme_id="remote-work",
                theme_label="Il lavoro da remoto",
                task_family="opinion_monologue",
                duration_sec=180,
                prompt_id="remote-work-b2",
                prompt_text="Parla del lavoro da remoto.",
            ),
            review=ReviewState(
                report_id="report-1",
                transcript="Testo completo della trascrizione.",
                score_overall=4.2,
                band="B2",
                summary="Placeholder summary.",
                payload={
                    "notes": "Ricordati di confrontare vantaggi e svantaggi.",
                    "transcript_full": "Testo completo della trascrizione.",
                    "report": {
                        "session_id": "report-1",
                        "scores": {"final": 4.2, "band": "B2"},
                        "checks": {
                            "language_pass": True,
                            "topic_pass": True,
                            "duration_pass": True,
                            "min_words_pass": True,
                        },
                        "coaching": {},
                    },
                },
            ),
        )
        at.run()
        self.assertEqual(at.text_area(key="review_saved_notes_view").value, "Ricordati di confrontare vantaggi e svantaggi.")
        self.assertEqual(at.text_area(key="review_transcript_view").value, "Testo completo della trascrizione.")

    def test_review_uses_progress_unavailable_hint_when_items_are_not_renderable(self):
        at = _app_test("pages/03_Review.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="B1",
                theme_id="viaggio",
                theme_label="Il mio ultimo viaggio all'estero",
                task_family="travel_narrative",
                duration_sec=90,
                prompt_id="viaggio-b1",
                prompt_text="Parla del tuo ultimo viaggio.",
            ),
            review=ReviewState(
                report_id="report-1",
                transcript="Parla del tuo ultimo viaggio.",
                score_overall=3.8,
                band="B1",
                summary="Placeholder summary.",
                payload={"report": {"session_id": "report-1", "scores": {"final": 3.8, "band": "B1"}}},
            ),
        )
        with patch(
            "app_shell.services.review_summary",
            return_value={
                "report_id": "report-1",
                "score_overall": 3.8,
                "band": "B1",
                "mode": "hybrid",
                "gates": {},
                "failed_gates": [],
                "progress_items": [{"kind": "delta_final", "value": "n/a"}],
                "strengths": [],
                "priorities": [],
            },
        ):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(
            any(
                caption.value
                == "Progress appears here once there is a comparable earlier attempt for the same speaker and task family."
                for caption in at.caption
            )
        )
        self.assertFalse(any(subheader.value == "Progress delta" for subheader in at.subheader))

    def test_history_renders_empty_state_for_fresh_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/04_History.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(ui_locale="en", log_dir=tmpdir)
            )
            at.run()
            self.assertEqual(at.session_state["_page_id"], "history")
            self.assertEqual(len(at.info), 1)
            self.assertEqual(at.info[0].value, "No history rows are available yet.")

    def test_history_falls_back_for_unknown_task_family(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(speaker_id="bern"),
        )
        with patch(
            "app_shell.services.load_history_records",
            return_value=[
                _history_record(
                    timestamp="2026-03-12T20:00:00",
                    session_id="sess-1",
                    speaker_id="bern",
                    theme="Custom topic",
                    task_family="debate_round",
                    final_score=4.1,
                    band=4,
                    overall=4.0,
                    wpm=121.0,
                    top_priorities=["Use clearer signposting"],
                )
            ],
        ):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertGreaterEqual(len(at.dataframe), 2)
        self.assertTrue(any("debate round" in str(dataframe.value) for dataframe in at.dataframe))

    def test_history_handles_load_errors_gracefully(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        with patch("app_shell.services.load_history_records", side_effect=RuntimeError("boom")):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([error.value for error in at.error], ["History could not be loaded right now."])
        self.assertEqual(len(at.info), 0)

    def test_history_renders_summary_metrics_and_tables(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(speaker_id="bern"),
        )
        records = [
            _history_record(
                timestamp="2026-03-12T20:00:00",
                session_id="sess-1",
                speaker_id="bern",
                learning_language="it",
                theme="The pros and cons of working from home",
                task_family="opinion_monologue",
                final_score=3.8,
                band=3,
                overall=3.7,
                wpm=118.0,
                top_priorities=["Use stronger conclusions"],
                grammar_error_categories=["agreement"],
                coherence_issue_categories=["signposting"],
            ),
            _history_record(
                timestamp="2026-03-13T08:15:00",
                session_id="sess-2",
                speaker_id="bern",
                learning_language="it",
                theme="The pros and cons of working from home",
                task_family="opinion_monologue",
                final_score=4.2,
                band=4,
                overall=4.0,
                wpm=132.0,
                top_priorities=["Use stronger conclusions", "Vary examples"],
                grammar_error_categories=["agreement"],
                coherence_issue_categories=["signposting", "sequencing"],
            ),
        ]
        with patch("app_shell.services.load_history_records", return_value=records):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any(metric.label == "Runs" and metric.value == "2" for metric in at.metric))
        self.assertTrue(any(metric.label == "Best final" and metric.value == "4.20" for metric in at.metric))
        self.assertGreaterEqual(len(at.dataframe), 2)

    def test_history_renders_persistent_attempt_details_from_saved_report(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(speaker_id="bern"),
        )
        records = [
            _history_record(
                timestamp="2026-03-12T20:00:00",
                session_id="sess-1",
                speaker_id="bern",
                theme="The pros and cons of working from home",
                task_family="opinion_monologue",
                final_score=3.8,
                band=3,
                overall=3.7,
                wpm=118.0,
                report_path="/tmp/report-1.json",
            ),
            _history_record(
                timestamp="2026-03-13T08:15:00",
                session_id="sess-2",
                speaker_id="bern",
                theme="The pros and cons of working from home",
                task_family="opinion_monologue",
                final_score=4.2,
                band=4,
                overall=4.0,
                wpm=132.0,
                report_path="/tmp/report-2.json",
            ),
        ]
        payload = {
            "notes": "Remember to compare both sides.",
            "transcript_full": "Full saved transcript for the selected attempt.",
            "report": {
                "session_id": "sess-2",
                "scores": {"final": 4.2, "band": "B2"},
                "checks": {
                    "language_pass": True,
                    "topic_pass": True,
                    "duration_pass": True,
                    "min_words_pass": True,
                },
                "coaching": {},
            },
        }
        with patch("app_shell.services.load_history_records", return_value=records), patch(
            "app_shell.services.load_report_payload",
            return_value=payload,
        ) as mock_load_report:
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.selectbox(key="history_detail_report").value, "/tmp/report-2.json")
        self.assertEqual(at.text_area(key="history_saved_notes_view").value, "Remember to compare both sides.")
        self.assertEqual(at.text_area(key="history_transcript_view").value, "Full saved transcript for the selected attempt.")
        mock_load_report.assert_called_with("/tmp/report-2.json")

    def test_history_recent_jump_buttons_render_newest_first(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(speaker_id="bern", learning_language="fr", learning_language_label="French"),
        )
        records = [
            _history_record(
                timestamp="2026-03-12T20:00:00",
                session_id="sess-1",
                speaker_id="bern",
                learning_language="it",
                theme="Theme one",
                task_family="personal_experience",
                final_score=3.8,
                band=3,
                overall=3.7,
                wpm=118.0,
                report_path="/tmp/report-1.json",
            ),
            _history_record(
                timestamp="2026-03-13T08:15:00",
                session_id="sess-2",
                speaker_id="bern",
                learning_language="en",
                theme="Theme two",
                task_family="opinion_monologue",
                final_score=4.2,
                band=4,
                overall=4.0,
                wpm=132.0,
                report_path="/tmp/report-2.json",
            ),
            _history_record(
                timestamp="2026-03-14T09:30:00",
                session_id="sess-3",
                speaker_id="bern",
                learning_language="de",
                theme="Theme three",
                task_family="picture_description",
                final_score=4.4,
                band=4,
                overall=4.1,
                wpm=126.0,
                report_path="/tmp/report-3.json",
            ),
        ]

        def _payload_for(path: str) -> dict[str, object]:
            return {
                "notes": f"Notes for {path}",
                "transcript_full": f"Transcript for {path}",
                "report": {
                    "session_id": path,
                    "scores": {"final": 4.2, "band": "B2"},
                    "checks": {
                        "language_pass": True,
                        "topic_pass": True,
                        "duration_pass": True,
                        "min_words_pass": True,
                    },
                    "coaching": {},
                },
            }

        with patch("app_shell.services.load_history_records", return_value=records), patch(
            "app_shell.services.load_report_payload",
            side_effect=_payload_for,
        ):
            at.run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.button(key="history_jump_0").label, "DE · 03-14 09:30 · Picture description")
        self.assertEqual(at.button(key="history_jump_1").label, "EN · 03-13 08:15 · Opinion monologue")
        self.assertEqual(at.button(key="history_jump_2").label, "IT · 03-12 20:00 · Personal experience")

    def test_history_attempts_table_shows_latest_attempt_first(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(speaker_id="bern"),
        )
        records = [
            _history_record(
                timestamp="2026-03-12T20:00:00",
                session_id="sess-1",
                speaker_id="bern",
                learning_language="it",
                theme="Theme one",
                task_family="personal_experience",
                final_score=3.8,
                band=3,
                overall=3.7,
                wpm=118.0,
                report_path="/tmp/report-1.json",
            ),
            _history_record(
                timestamp="2026-03-13T08:15:00",
                session_id="sess-2",
                speaker_id="bern",
                learning_language="it",
                theme="Theme two",
                task_family="opinion_monologue",
                final_score=4.2,
                band=4,
                overall=4.0,
                wpm=132.0,
                report_path="/tmp/report-2.json",
            ),
        ]
        payload = {
            "notes": "Notes",
            "transcript_full": "Transcript",
            "report": {
                "session_id": "sess-2",
                "scores": {"final": 4.2, "band": "B2"},
                "checks": {
                    "language_pass": True,
                    "topic_pass": True,
                    "duration_pass": True,
                    "min_words_pass": True,
                },
                "coaching": {},
            },
        }
        with patch("app_shell.services.load_history_records", return_value=records), patch(
            "app_shell.services.load_report_payload",
            return_value=payload,
        ):
            at.run()

        self.assertEqual(len(at.exception), 0)
        attempts_frame = next(
            dataframe.value for dataframe in at.dataframe if "Session" in getattr(dataframe.value, "columns", [])
        )
        self.assertEqual(list(attempts_frame["Session"]), ["sess-2", "sess-1"])
        self.assertEqual(list(attempts_frame["Language"]), ["IT", "IT"])

    def test_history_defaults_to_current_learning_language_filter(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(speaker_id="bern", learning_language="en", learning_language_label="English"),
        )
        records = [
            _history_record(
                timestamp="2026-03-12T20:00:00",
                session_id="sess-it",
                speaker_id="bern",
                learning_language="it",
                theme="Tema uno",
                task_family="personal_experience",
                final_score=3.8,
                band=3,
                overall=3.7,
                wpm=118.0,
                report_path="/tmp/report-it.json",
            ),
            _history_record(
                timestamp="2026-03-13T08:15:00",
                session_id="sess-en",
                speaker_id="bern",
                learning_language="en",
                theme="Theme two",
                task_family="opinion_monologue",
                final_score=4.2,
                band=4,
                overall=4.0,
                wpm=132.0,
                report_path="/tmp/report-en.json",
            ),
        ]
        payload = {
            "notes": "Notes",
            "transcript_full": "Transcript",
            "report": {
                "session_id": "sess-en",
                "scores": {"final": 4.2, "band": "B2"},
                "checks": {
                    "language_pass": True,
                    "topic_pass": True,
                    "duration_pass": True,
                    "min_words_pass": True,
                },
                "coaching": {},
            },
        }
        with patch("app_shell.services.load_history_records", return_value=records), patch(
            "app_shell.services.load_report_payload",
            return_value=payload,
        ) as mock_load_report:
            at.run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.selectbox(key="history_learning_language").value, "en")
        attempts_frame = next(
            dataframe.value for dataframe in at.dataframe if "Session" in getattr(dataframe.value, "columns", [])
        )
        self.assertEqual(list(attempts_frame["Session"]), ["sess-en"])
        self.assertEqual(at.selectbox(key="history_detail_report").value, "/tmp/report-en.json")
        mock_load_report.assert_called_with("/tmp/report-en.json")

    def test_history_handles_missing_saved_report_payload(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(speaker_id="bern"),
        )
        records = [
            _history_record(
                timestamp="2026-03-13T08:15:00",
                session_id="sess-2",
                speaker_id="bern",
                learning_language="it",
                theme="The pros and cons of working from home",
                task_family="opinion_monologue",
                final_score=4.2,
                band=4,
                overall=4.0,
                wpm=132.0,
                report_path="/tmp/report-2.json",
            ),
        ]
        with patch("app_shell.services.load_history_records", return_value=records), patch(
            "app_shell.services.load_report_payload",
            return_value=None,
        ):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([error.value for error in at.error], ["The saved report for this attempt could not be loaded."])

    def test_history_tolerates_non_numeric_log_values(self):
        at = _app_test("pages/04_History.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
            draft=DraftSession(speaker_id="bern"),
        )
        records = [
            SimpleNamespace(
                timestamp=datetime.fromisoformat("2026-03-12T20:00:00"),
                session_id="sess-1",
                speaker_id="bern",
                learning_language="",
                theme="Custom topic",
                task_family="opinion_monologue",
                final_score="",
                band="",
                overall="n/a",
                wpm="",
                top_priorities=[],
                grammar_error_categories=[],
                coherence_issue_categories=[],
            ),
            SimpleNamespace(
                timestamp=datetime.fromisoformat("2026-03-13T08:15:00"),
                session_id="sess-2",
                speaker_id="bern",
                learning_language="",
                theme="Custom topic",
                task_family="opinion_monologue",
                final_score="4.2",
                band="4",
                overall="4.0",
                wpm="132.0",
                top_priorities=["Vary examples"],
                grammar_error_categories=[],
                coherence_issue_categories=[],
            ),
        ]
        with patch("app_shell.services.load_history_records", return_value=records):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any(metric.label == "Best final" and metric.value == "4.20" for metric in at.metric))

    def test_library_reveals_new_language_inputs_reactively(self):
        at = _app_test("pages/05_Library.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        at.run()
        at.selectbox(key="library_manage_language").set_value("__new_language__")
        at.run()
        self.assertTrue(any(widget.key == "library_language_code" for widget in at.text_input))
        self.assertTrue(any(widget.key == "library_language_label" for widget in at.text_input))

    def test_library_allows_bootstrap_when_catalog_is_empty(self):
        at = _app_test("pages/05_Library.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en")
        )
        with patch("app_shell.services.load_theme_library", return_value={}):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([info.value for info in at.info], ["No language catalog is available yet."])
        self.assertEqual(at.selectbox(key="library_manage_language").options, ["Create a new language"])

    def test_library_save_adds_new_language_theme_and_selects_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/05_Library.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(ui_locale="en", log_dir=tmpdir)
            )
            at.run()
            at.selectbox(key="library_manage_language").set_value("__new_language__")
            at.run()
            at.text_input(key="library_language_code").set_value(" DE ")
            at.text_input(key="library_language_label").set_value(" Deutsch ")
            at.text_input(key="library_theme_title").set_value(" Ein wichtiges Gespraech ")
            at.selectbox(key="library_theme_level").set_value("B2")
            at.selectbox(key="library_theme_family").set_value("personal_experience")
            at.button(key="library_save_theme").click()
            at.run()

            persisted = theme_library.load_theme_library(Path(tmpdir))
            self.assertIn("de", persisted)
            self.assertEqual(persisted["de"]["label"], "Deutsch")
            self.assertEqual(persisted["de"]["themes"][0]["title"], "Ein wichtiges Gespraech")
            self.assertEqual(at.session_state["library_filter_language"], "de")
            self.assertEqual(len(at.success), 1)
            at.selectbox(key="library_manage_language").set_value("__new_language__")
            at.run()
            self.assertEqual(at.text_input(key="library_language_code").value, "")
            self.assertEqual(at.text_input(key="library_language_label").value, "")

    def test_library_can_save_two_new_languages_without_session_state_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/05_Library.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(ui_locale="en", log_dir=tmpdir)
            )
            at.run()
            at.selectbox(key="library_manage_language").set_value("__new_language__")
            at.run()
            at.text_input(key="library_language_code").set_value("en")
            at.text_input(key="library_language_label").set_value("English")
            at.text_input(key="library_theme_title").set_value("Working from home")
            at.selectbox(key="library_theme_level").set_value("B2")
            at.selectbox(key="library_theme_family").set_value("opinion_monologue")
            at.button(key="library_save_theme").click()
            at.run()

            at.selectbox(key="library_manage_language").set_value("__new_language__")
            at.run()
            at.text_input(key="library_language_code").set_value("it")
            at.text_input(key="library_language_label").set_value("Italiano")
            at.text_input(key="library_theme_title").set_value("Lavorare da casa")
            at.selectbox(key="library_theme_level").set_value("B2")
            at.selectbox(key="library_theme_family").set_value("opinion_monologue")
            at.button(key="library_save_theme").click()
            at.run()

            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.session_state["library_filter_language"], "it")
            self.assertEqual(at.selectbox(key="library_filter_language").value, "it")

    def test_settings_tolerates_invalid_saved_options(self):
        at = _app_test("pages/06_Settings.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="fr", whisper_model="giant")
        )
        at.run()
        self.assertEqual(at.session_state["_page_id"], "settings")
        self.assertEqual(len(at.exception), 0)

    def test_settings_enables_openrouter_fields_immediately_when_provider_changes(self):
        at = _app_test("pages/06_Settings.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en", provider="ollama")
        )
        at.run()
        self.assertFalse(at.text_input(key="settings_api_key").disabled)
        self.assertTrue(at.text_input(key="settings_openrouter_http_referer").disabled)
        self.assertTrue(at.text_input(key="settings_openrouter_app_title").disabled)
        at.selectbox(key="settings_provider").set_value("openrouter")
        at.run()
        self.assertFalse(at.text_input(key="settings_openrouter_http_referer").disabled)
        self.assertFalse(at.text_input(key="settings_openrouter_app_title").disabled)

    def test_settings_can_make_another_connection_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/06_Settings.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(
                    ui_locale="en",
                    log_dir=tmpdir,
                    setup_complete=True,
                    active_connection_id="primary",
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
                            connection_id="backup",
                            provider_kind="ollama",
                            label="Backup",
                            base_url="http://localhost:11434",
                            default_model="llama3",
                            is_local=True,
                            provider_metadata={"deployment": "local"},
                        ),
                    ],
                )
            )
            at.run()
            at.button(key="settings_default_backup").click()
            at.run()

            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.prefs.active_connection_id, "backup")
            self.assertTrue(next(item for item in state.prefs.connections if item.connection_id == "backup").is_default)
            self.assertEqual(at.selectbox(key="settings_connection_id").value, "backup")

    def test_settings_can_delete_a_saved_connection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/06_Settings.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(
                    ui_locale="en",
                    log_dir=tmpdir,
                    setup_complete=True,
                    active_connection_id="primary",
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
                            connection_id="backup",
                            provider_kind="lmstudio",
                            label="Backup",
                            base_url="http://localhost:1234/v1",
                            default_model="qwen2.5",
                            provider_metadata={"deployment": "local", "token_optional": True},
                        ),
                    ],
                )
            )
            at.run()
            at.button(key="settings_delete_backup").click()
            at.run()

            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(len(state.prefs.connections), 1)
            self.assertEqual(state.prefs.connections[0].connection_id, "primary")
            self.assertEqual(at.selectbox(key="settings_connection_id").value, "primary")

    def test_settings_save_updates_prefs_and_back_uses_return_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/06_Settings.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(ui_locale="en", log_dir=tmpdir),
            )
            at.session_state[APP_SHELL_STATE_KEY].nav.return_to = "library"
            at.run()
            at.selectbox(key="settings_provider").set_value("ollama")
            at.text_input(key="settings_model").set_value("llama3")
            at.selectbox(key="settings_whisper_model").set_value("base")
            at.button(key="settings_save").click()
            at.run()

            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.prefs.ui_locale, "en")
            self.assertEqual(state.prefs.provider, "ollama")
            self.assertEqual(state.prefs.model, "llama3")
            self.assertEqual(state.prefs.whisper_model, "base")
            self.assertEqual(len(at.success), 1)

            at.button(key="settings_back").click()
            at.run()
            self.assertEqual(at.session_state["_next_page"], "pages/05_Library.py")

    def test_settings_save_supports_lmstudio_runtime_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/06_Settings.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(ui_locale="en", log_dir=tmpdir),
            )
            at.run()
            at.selectbox(key="settings_provider").set_value("lmstudio")
            at.text_input(key="settings_model").set_value("qwen2.5")
            at.text_input(key="settings_base_url").set_value("http://localhost:1234/v1")
            at.text_input(key="settings_api_key").set_value("token-123")
            at.button(key="settings_save").click()
            at.run()

            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.prefs.provider, "lmstudio")
            self.assertEqual(state.prefs.model, "qwen2.5")
            self.assertEqual(state.prefs.llm_base_url, "http://localhost:1234/v1")
            self.assertEqual(state.prefs.llm_api_key, "token-123")

    def test_settings_back_falls_back_to_home(self):
        at = _app_test("pages/06_Settings.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
        )
        at.run()
        at.button(key="settings_back").click()
        at.run()
        self.assertEqual(at.session_state["_next_page"], "streamlit_app.py")

    def test_settings_download_shows_success_after_progress_callback(self):
        def _fake_download(_model_name, progress_callback=None):
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "downloading",
                        "current_file": "model.bin",
                        "downloaded_bytes": 1024,
                        "total_bytes": 2048,
                        "completed_files": 1,
                        "total_files": 3,
                    }
                )
                progress_callback({"stage": "ready", "cached_path": "/tmp/small"})
            return {"cached_path": "/tmp/small"}

        at = _app_test("pages/06_Settings.py")
        at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
            prefs=AppPreferences(ui_locale="en"),
        )
        with patch("app_shell.services.download_whisper_model", side_effect=_fake_download):
            at.run()
            at.button(key="settings_whisper_download").click()
            at.run()

        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("/tmp/small" in item.value for item in at.success))

    def test_settings_save_persists_openrouter_credentials_and_identifier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            at = _app_test("pages/06_Settings.py")
            at.session_state[APP_SHELL_STATE_KEY] = AppShellState(
                prefs=AppPreferences(ui_locale="en", log_dir=tmpdir),
            )
            at.run()
            at.selectbox(key="settings_provider").set_value("openrouter")
            at.text_input(key="settings_model").set_value("google/gemini-3.1-pro-preview")
            at.text_input(key="settings_api_key").set_value("key-123")
            at.text_input(key="settings_openrouter_http_referer").set_value("http://localhost:8503")
            at.text_input(key="settings_openrouter_app_title").set_value("Speaking Studio")
            at.button(key="settings_save").click()
            at.run()

            state = at.session_state[APP_SHELL_STATE_KEY]
            self.assertEqual(state.prefs.provider, "openrouter")
            self.assertEqual(state.prefs.llm_api_key, "key-123")
            self.assertEqual(state.prefs.openrouter_http_referer, "http://localhost:8503")
            self.assertEqual(state.prefs.openrouter_app_title, "Speaking Studio")


if __name__ == "__main__":
    unittest.main()

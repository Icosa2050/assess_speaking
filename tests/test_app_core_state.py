import unittest
from pathlib import Path

from unittest import mock

from app_core.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR
from app_core.state import (
    AppPreferences,
    AppState,
    DraftSession,
    RecordingState,
    ReviewState,
    build_default_state,
    has_recording,
    has_review,
    has_setup,
)


class AppStateTests(unittest.TestCase):
    def test_default_state_has_session_id(self):
        state = build_default_state()
        self.assertTrue(state.draft.session_id.startswith("draft-"))
        self.assertEqual(state.prefs.ui_locale, "en")

    def test_app_preferences_default_paths_follow_app_data_environment(self):
        with mock.patch.dict(
            "os.environ",
            {
                APP_DATA_HOME_ENV_VAR: "/tmp/speaking-studio-home",
                APP_CACHE_HOME_ENV_VAR: "/tmp/speaking-studio-cache",
            },
            clear=False,
        ):
            prefs = AppPreferences()
        self.assertEqual(prefs.log_dir, str((Path("/tmp/speaking-studio-home") / "reports").resolve()))
        self.assertEqual(prefs.whisper_cache_dir, str((Path("/tmp/speaking-studio-cache") / "whisper").resolve()))

    def test_locale_and_learning_language_are_independent_fields(self):
        state = AppState(
            prefs=AppPreferences(ui_locale="de"),
            draft=DraftSession(learning_language="it", learning_language_label="Italiano"),
        )
        self.assertEqual(state.prefs.ui_locale, "de")
        self.assertEqual(state.draft.learning_language, "it")

    def test_has_setup_is_false_for_fresh_state(self):
        state = build_default_state()
        self.assertFalse(has_setup(state))

    def test_has_setup_is_true_when_prompt_exists(self):
        state = AppState(
            draft=DraftSession(
                session_id="draft-123",
                speaker_id="bern",
                learning_language="it",
                learning_language_label="Italiano",
                cefr_level="B1",
                theme_id="viaggio",
                theme_label="Il mio ultimo viaggio all'estero",
                duration_sec=90,
                prompt_id="viaggio-b1",
                prompt_text="Parla del tuo ultimo viaggio.",
            )
        )
        self.assertTrue(has_setup(state))

    def test_has_recording_and_review_reflect_payloads(self):
        state = AppState(
            recording=RecordingState(audio_path="demo://viaggio", duration_sec=45),
            review=ReviewState(report_id="report-1", transcript="Ciao"),
        )
        self.assertTrue(has_recording(state))
        self.assertTrue(has_review(state))

if __name__ == "__main__":
    unittest.main()

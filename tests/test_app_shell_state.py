import unittest

from unittest import mock

from app_shell.state import (
    AppPreferences,
    AppShellState,
    DraftSession,
    RecordingState,
    RecordingStatus,
    ReviewState,
    APP_SHELL_STATE_KEY,
    build_default_state,
    has_recording,
    has_review,
    has_setup,
    set_recording_assessing,
)


class AppShellStateTests(unittest.TestCase):
    def test_default_state_has_session_id(self):
        state = build_default_state()
        self.assertTrue(state.draft.session_id.startswith("draft-"))
        self.assertEqual(state.prefs.ui_locale, "en")

    def test_locale_and_learning_language_are_independent_fields(self):
        state = AppShellState(
            prefs=AppPreferences(ui_locale="de"),
            draft=DraftSession(learning_language="it", learning_language_label="Italiano"),
        )
        self.assertEqual(state.prefs.ui_locale, "de")
        self.assertEqual(state.draft.learning_language, "it")

    def test_has_setup_is_false_for_fresh_state(self):
        state = build_default_state()
        self.assertFalse(has_setup(state))

    def test_has_setup_is_true_when_prompt_exists(self):
        state = AppShellState(
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
        state = AppShellState(
            recording=RecordingState(audio_path="demo://viaggio", duration_sec=45),
            review=ReviewState(report_id="report-1", transcript="Ciao"),
        )
        self.assertTrue(has_recording(state))
        self.assertTrue(has_review(state))

    def test_set_recording_assessing_marks_attempt_inflight(self):
        state = AppShellState(
            recording=RecordingState(
                status=RecordingStatus.READY,
                audio_path="demo://viaggio",
                error="old error",
            )
        )
        with mock.patch("streamlit.session_state", {APP_SHELL_STATE_KEY: state}):
            updated = set_recording_assessing()
        self.assertIs(updated, state)
        self.assertEqual(updated.recording.status, RecordingStatus.ASSESSING)
        self.assertEqual(updated.recording.error, "")


if __name__ == "__main__":
    unittest.main()

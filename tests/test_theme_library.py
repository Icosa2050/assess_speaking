import tempfile
import unittest
import inspect
from pathlib import Path

from assessment_runtime import theme_library


class ThemeLibraryTests(unittest.TestCase):
    def test_load_session_setup_content_falls_back_when_file_is_missing(self):
        missing = Path("/tmp/does-not-exist/session_setup_content.json")

        payload = theme_library._load_session_setup_content(path=missing)

        self.assertIn("default_theme_library", payload)
        self.assertIn("practice_brief_templates", payload)
        self.assertIn("en", payload["practice_brief_templates"])

    def test_load_session_setup_content_fills_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session_setup_content.json"
            path.write_text('{"default_theme_library": {}}', encoding="utf-8")

            payload = theme_library._load_session_setup_content(path=path)

        self.assertEqual(payload["default_theme_library"], {})
        self.assertIn("practice_brief_templates", payload)
        self.assertIn("en", payload["practice_brief_templates"])

    def test_load_theme_library_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = theme_library.load_theme_library(Path(tmpdir))
        self.assertIn("it", library)
        self.assertIn("en", library)

    def test_default_library_covers_b1_b2_c1_for_it_and_en(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = theme_library.load_theme_library(Path(tmpdir))

        for language_code in ("it", "en", "de"):
            with self.subTest(language_code=language_code):
                levels = {
                    str(theme.get("level") or "").upper()
                    for theme in library[language_code]["themes"]
                }
                self.assertTrue({"B1", "B2", "C1"}.issubset(levels))

    def test_default_theme_library_covers_practice_brief_languages(self):
        payload = theme_library._load_session_setup_content()

        self.assertLessEqual(
            set(payload["practice_brief_templates"]),
            set(payload["default_theme_library"]),
        )

    def test_save_workspace_prefs_has_one_authoritative_definition(self):
        source = inspect.getsource(theme_library)

        self.assertEqual(source.count("def save_workspace_prefs"), 1)

    def test_add_theme_supports_new_language(self):
        library = theme_library.add_theme(
            {},
            language_code="pl",
            language_label="Polski",
            title="Rozmowa, ktora zrobila na mnie wrazenie",
            level="B2",
            task_family="personal_experience",
        )
        self.assertEqual(library["pl"]["label"], "Polski")
        self.assertEqual(library["pl"]["themes"][0]["level"], "B2")

    def test_save_and_load_workspace_prefs_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            prefs = {
                "speaker_id": "bern",
                "ui_locale": "de",
                "language": "it",
                "learning_language": "it",
                "cefr_level": "B1",
                "theme": "Il mio ultimo viaggio all'estero",
            }
            theme_library.save_workspace_prefs(log_dir, prefs)
            loaded = theme_library.load_workspace_prefs(log_dir)
        self.assertEqual(loaded["speaker_id"], "bern")
        self.assertEqual(loaded["ui_locale"], "de")
        self.assertEqual(loaded["language"], "it")

    def test_save_and_load_workspace_prefs_preserves_speaker_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            prefs = {
                "speaker_id": "bern",
                "last_setup": {
                    "speaker_id": "bern",
                    "learning_language": "it",
                    "language": "it",
                    "cefr_level": "B1",
                    "theme": "Il mio ultimo viaggio all'estero",
                    "task_family": "travel_narrative",
                    "target_duration_sec": 90,
                },
                "speaker_profiles": {
                    "bern": {
                        "speaker_id": "bern",
                        "learning_language": "it",
                        "language": "it",
                        "cefr_level": "B1",
                        "theme": "Il mio ultimo viaggio all'estero",
                        "task_family": "travel_narrative",
                        "target_duration_sec": 90,
                    }
                },
            }
            theme_library.save_workspace_prefs(log_dir, prefs)
            loaded = theme_library.load_workspace_prefs(log_dir)
        self.assertEqual(loaded["last_setup"]["cefr_level"], "B1")
        self.assertEqual(loaded["speaker_profiles"]["bern"]["learning_language"], "it")


if __name__ == "__main__":
    unittest.main()

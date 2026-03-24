import tempfile
import unittest
from pathlib import Path

from assessment_runtime import theme_library


class ThemeLibraryTests(unittest.TestCase):
    def test_load_theme_library_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library = theme_library.load_theme_library(Path(tmpdir))
        self.assertIn("it", library)
        self.assertIn("en", library)

    def test_add_theme_supports_new_language(self):
        library = theme_library.add_theme(
            {},
            language_code="de",
            language_label="Deutsch",
            title="Ein Gespräch, das mich beeindruckt hat",
            level="B2",
            task_family="personal_experience",
        )
        self.assertEqual(library["de"]["label"], "Deutsch")
        self.assertEqual(library["de"]["themes"][0]["level"], "B2")

    def test_save_and_load_dashboard_prefs_roundtrip(self):
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
            theme_library.save_dashboard_prefs(log_dir, prefs)
            loaded = theme_library.load_dashboard_prefs(log_dir)
        self.assertEqual(loaded["speaker_id"], "bern")
        self.assertEqual(loaded["ui_locale"], "de")
        self.assertEqual(loaded["language"], "it")

    def test_save_and_load_dashboard_prefs_preserves_speaker_profiles(self):
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
            theme_library.save_dashboard_prefs(log_dir, prefs)
            loaded = theme_library.load_dashboard_prefs(log_dir)
        self.assertEqual(loaded["last_setup"]["cefr_level"], "B1")
        self.assertEqual(loaded["speaker_profiles"]["bern"]["learning_language"], "it")


if __name__ == "__main__":
    unittest.main()

import unittest

from app_shell.i18n import locale_key_map, t


class AppShellI18nTests(unittest.TestCase):
    def test_every_locale_has_the_same_keys(self):
        key_map = locale_key_map()
        baseline = key_map["en"]
        for locale, keys in key_map.items():
            self.assertEqual(baseline, keys, f"Locale {locale} drifted from en")

    def test_missing_keys_are_visible(self):
        self.assertEqual(t("missing.example", locale="en"), "[missing.example]")

    def test_translation_interpolation_works(self):
        translated = t(
            "common.session_summary",
            locale="en",
            session_id="draft-1",
            speaker_id="bern",
            ui_locale="en",
            learning_language="Italiano",
            cefr="B1",
        )
        self.assertIn("draft-1", translated)
        self.assertIn("bern", translated)
        self.assertIn("Italiano", translated)


if __name__ == "__main__":
    unittest.main()

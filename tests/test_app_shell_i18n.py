import unittest

from app_shell.i18n import flatten_keys, load_locale, locale_key_map, t

PREPARED_UI_LOCALES = ("fr", "es")
KNOWN_LOCALE_CODES = ("en", "de", "it", "fr", "es")


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

    def test_prepared_locales_track_english_keys(self):
        baseline = flatten_keys(load_locale("en"))
        for locale in PREPARED_UI_LOCALES:
            self.assertEqual(baseline, flatten_keys(load_locale(locale)), f"Prepared locale {locale} drifted from en")

    def test_locale_names_cover_prepared_languages(self):
        for locale in ("en", "de", "it", *PREPARED_UI_LOCALES):
            locale_names = load_locale(locale).get("locale", {})
            self.assertIsInstance(locale_names, dict)
            for code in KNOWN_LOCALE_CODES:
                self.assertIn(code, locale_names, f"Locale {locale} is missing locale.{code}")


if __name__ == "__main__":
    unittest.main()

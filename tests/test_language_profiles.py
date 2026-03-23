import unittest

from assess_core.language_profiles import (
    default_language_profile_key,
    fallback_language_profile,
    get_language_profile,
    get_language_profile_by_key,
    require_language_profile,
    require_resolved_language_profile,
)


class LanguageProfilesTests(unittest.TestCase):
    def test_get_language_profile_returns_english_profile(self):
        profile = require_language_profile("en")
        self.assertEqual(profile.label, "English")
        self.assertIn("however", profile.discourse_markers)
        self.assertIn("uh", profile.fillers)
        self.assertEqual(profile.scorer_version, "language_profile_en_v2")

    def test_get_language_profile_returns_none_for_unknown_language(self):
        self.assertIsNone(get_language_profile("xx"))

    def test_get_language_profile_by_key_returns_profile(self):
        profile = get_language_profile_by_key("en")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.code, "en")

    def test_get_language_profile_by_key_returns_italian_benchmark_variant(self):
        profile = get_language_profile_by_key("it_benchmark")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.code, "it")
        self.assertEqual(profile.scorer_version, "language_profile_it_v1")

    def test_get_language_profile_by_key_returns_italian_live_shadow_variant(self):
        profile = get_language_profile_by_key("it_live_shadow")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.code, "it")
        self.assertEqual(profile.scorer_version, "language_profile_it_v1_live_shadow")

    def test_default_language_profile_key_tracks_language_defaults(self):
        self.assertEqual(default_language_profile_key("en"), "en")
        self.assertEqual(default_language_profile_key("it"), "it")
        self.assertIsNone(default_language_profile_key("xx"))

    def test_require_resolved_language_profile_accepts_matching_profile_key(self):
        profile = require_resolved_language_profile("en", profile_key="en")
        self.assertEqual(profile.scorer_version, "language_profile_en_v2")

    def test_require_resolved_language_profile_accepts_italian_benchmark_profile_key(self):
        profile = require_resolved_language_profile("it", profile_key="it_benchmark")
        self.assertEqual(profile.scorer_version, "language_profile_it_v1")

    def test_require_resolved_language_profile_rejects_language_profile_mismatch(self):
        with self.assertRaises(KeyError):
            require_resolved_language_profile("it", profile_key="en")

    def test_require_language_profile_uses_italian_live_shadow_as_default(self):
        profile = require_language_profile("it")
        self.assertEqual(profile.scorer_version, "language_profile_it_v1_live_shadow")

    def test_fallback_language_profile_returns_generic_profile_for_unknown_language(self):
        profile = fallback_language_profile("xx")
        self.assertEqual(profile.code, "generic")
        self.assertEqual(profile.scorer_version, "language_profile_generic_v1")


if __name__ == "__main__":
    unittest.main()

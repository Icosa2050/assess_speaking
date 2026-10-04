import unicodedata
import unittest
from dataclasses import replace
from unittest import mock

from assessment_runtime.metrics import _count_phrases, metrics_from
from assess_core.language_profiles import require_language_profile_by_key


def metrics(text, language="en", profile_key=None):
    return metrics_from(
        [{"text": word} for word in text.split()],
        {"duration_sec": 30.0, "pauses": []},
        language_code=language,
        language_profile_key=profile_key,
    )


class LiveMetricsTests(unittest.TestCase):
    def test_english_connectors_from_live_workflow(self):
        self.assertEqual(metrics("However, remote work can create problems. In my view, balance helps.")["cohesion_markers"], 2)
        self.assertEqual(metrics("Culture depends not only on better rules, but also on critical reading.")["cohesion_markers"], 2)

    def test_italian_connectors_from_live_workflow(self):
        self.assertEqual(metrics("Allo stesso tempo, però, è un modello flessibile.", "it")["cohesion_markers"], 2)
        self.assertEqual(metrics("Perché eravamo stanchi ma molto contenti.", "it")["cohesion_markers"], 1)

    def test_unicode_normalization_and_apostrophes(self):
        original = "PERÒ, ciò nonostante, dall’altro lato."
        for text in (original, original.replace("’", "'"), unicodedata.normalize("NFD", original)):
            with self.subTest(text=text):
                self.assertEqual(metrics(text, "it")["cohesion_markers"], 3)

    def test_boundary_matching_repetition_and_longest_phrase(self):
        self.assertEqual(_count_phrases("thereforex somewhereas in my viewer", ("therefore", "in my view")), 0)
        self.assertEqual(_count_phrases("however however", ("however",)), 2)
        self.assertEqual(_count_phrases("dall'altro lato dall’altro", ("dall’altro", "dall'altro lato")), 2)

    def test_punctuation_does_not_join_words(self):
        result = metrics("however—but, also in\tmy\nview.")
        self.assertEqual(result["word_count"], 6)
        self.assertEqual(result["cohesion_markers"], 3)

    def test_marker_split_across_asr_words(self):
        result = metrics_from(
            [{"text": "In my"}, {"text": "view,"}, {"text": "it helps."}],
            {"duration_sec": 30.0, "pauses": []}, language_code="en",
        )
        self.assertEqual(result["cohesion_markers"], 1)
        self.assertEqual(result["word_count"], 5)

    def test_language_specific_lexicons_and_empty_text(self):
        self.assertEqual(metrics("however", "it")["cohesion_markers"], 0)
        self.assertEqual(metrics("però", "en")["cohesion_markers"], 0)
        self.assertEqual(metrics("")["word_count"], 0)

    def test_multiword_fillers_count_phrases_not_constituent_words(self):
        profile = replace(require_language_profile_by_key("en_live"), fillers=("you know", "I mean", "um"))
        with mock.patch("assessment_runtime.metrics.fallback_language_profile", return_value=profile):
            result = metrics("You helped me. I know the answer. I mean, you know, um, I mean it.")
        self.assertEqual(result["fillers"], 4)

    def test_explicit_historical_profiles_keep_original_metrics_and_versions(self):
        en = require_language_profile_by_key("en")
        it = require_language_profile_by_key("it_benchmark")
        self.assertEqual(en.scorer_version, "language_profile_en_v2")
        self.assertEqual(it.scorer_version, "language_profile_it_v1")
        self.assertEqual(metrics("In my view it works", profile_key="en")["cohesion_markers"], 0)
        self.assertEqual(metrics("Allo stesso tempo però", "it", "it_benchmark")["cohesion_markers"], 0)
        self.assertEqual(metrics("ciò nonostante", "it", "it_benchmark")["cohesion_markers"], 1)
        self.assertEqual(metrics("dall'altro", "it", "it_benchmark")["cohesion_markers"], 0)


if __name__ == "__main__":
    unittest.main()

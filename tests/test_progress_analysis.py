import unittest
from dataclasses import dataclass

import progress_analysis


@dataclass
class DummyRecord:
    speaker_id: str
    task_family: str
    final_score: float | None
    overall: float | None
    top_priorities: tuple[str, ...]
    grammar_error_categories: tuple[str, ...]
    coherence_issue_categories: tuple[str, ...]


class ProgressAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            DummyRecord(
                speaker_id="bern",
                task_family="travel_narrative",
                final_score=3.4,
                overall=3.2,
                top_priorities=("More connectors", "Less filler"),
                grammar_error_categories=("preposition_choice", "tense_consistency"),
                coherence_issue_categories=("missing_sequence_markers",),
            ),
            DummyRecord(
                speaker_id="bern",
                task_family="travel_narrative",
                final_score=4.0,
                overall=3.8,
                top_priorities=("More detail", "More accuracy"),
                grammar_error_categories=("preposition_choice",),
                coherence_issue_categories=("missing_sequence_markers", "underdeveloped_detail"),
            ),
            DummyRecord(
                speaker_id="bern",
                task_family="opinion",
                final_score=3.0,
                overall=3.0,
                top_priorities=("Stronger examples",),
                grammar_error_categories=("word_order",),
                coherence_issue_categories=("insufficient_linking",),
            ),
        ]

    def test_filter_records_by_speaker_and_task_family(self):
        filtered = progress_analysis.filter_records(
            self.records,
            speaker_id="bern",
            task_family="travel_narrative",
        )
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(record.task_family == "travel_narrative" for record in filtered))

    def test_recurring_issue_counts_counts_categories(self):
        filtered = progress_analysis.filter_records(self.records, task_family="travel_narrative")
        counts = progress_analysis.recurring_issue_counts(filtered, "grammar_error_categories")
        self.assertEqual(counts["preposition_choice"], 2)
        self.assertEqual(counts["tense_consistency"], 1)

    def test_latest_priorities_compares_last_two_records(self):
        filtered = progress_analysis.filter_records(self.records, task_family="travel_narrative")
        summary = progress_analysis.latest_priorities(filtered)
        self.assertEqual(summary["latest"], ["More detail", "More accuracy"])
        self.assertEqual(summary["previous"], ["More connectors", "Less filler"])
        self.assertIn("More detail", summary["new"])
        self.assertIn("More connectors", summary["resolved"])

    def test_task_family_progress_groups_families(self):
        summaries = progress_analysis.task_family_progress(self.records, speaker_id="bern")
        self.assertEqual([row["task_family"] for row in summaries], ["opinion", "travel_narrative"])
        travel = summaries[1]
        self.assertEqual(travel["count"], 2)
        self.assertAlmostEqual(travel["avg_final"], 3.7, places=2)
        self.assertEqual(travel["grammar_counts"]["preposition_choice"], 2)


if __name__ == "__main__":
    unittest.main()

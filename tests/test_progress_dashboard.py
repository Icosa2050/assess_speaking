import tempfile
import json
from datetime import datetime
from pathlib import Path

import unittest

from scripts import progress_dashboard


SAMPLE_CSV = """timestamp,audio,whisper,llm,label,duration_sec,wpm,word_count,overall,report_path
2025-10-06T14:58:01,demo.m4a,large-v3,llama3.1,baseline,43.09,95.9,54,3.5,/path/to/1.json
2025-10-07T09:12:33,week2.m4a,medium,llama3.2:3b,week2,41.00,110.2,60,3.8,/path/to/2.json
"""

RICH_CSV = """timestamp,session_id,schema_version,speaker_id,task_family,theme,audio,whisper,llm,label,target_duration_sec,duration_sec,wpm,word_count,duration_pass,topic_pass,language_pass,fluency,cohesion,accuracy,range,overall,final_score,band,requires_human_review,top_priority_1,top_priority_2,top_priority_3,grammar_error_categories,coherence_issue_categories,report_path
2025-10-06T14:58:01,s1,2,bern,travel_narrative,trip,demo.m4a,large-v3,llama3.1,baseline,180,43.09,95.9,54,true,true,true,3,3,3,3,3.5,3.6,4,false,Più connettivi,Meno filler,Più dettagli,preposition_choice,missing_sequence_markers,/path/to/1.json
2025-10-07T09:12:33,s2,2,bern,travel_narrative,trip,week2.m4a,medium,llama3.2:3b,week2,180,41.00,110.2,60,true,true,true,4,4,4,4,3.8,4.1,4,false,Più dettagli,Più precisione,Meno pause,preposition_choice,missing_sequence_markers,/path/to/2.json
"""


class DashboardTests(unittest.TestCase):
    def test_load_history_sorts_and_parses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.csv"
            history.write_text(SAMPLE_CSV, encoding="utf-8")
            records = progress_dashboard.load_history(history)
            self.assertEqual(len(records), 2)
            self.assertLess(records[0].timestamp, records[1].timestamp)
            self.assertAlmostEqual(records[0].wpm, 95.9)
            self.assertEqual(records[1].label, "week2")

    def test_summarise_computes_means(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.csv"
            history.write_text(SAMPLE_CSV, encoding="utf-8")
            records = progress_dashboard.load_history(history)
            summary = progress_dashboard.summarise(records)
            self.assertEqual(summary["count"], 2)
            self.assertAlmostEqual(summary["avg_wpm"], (95.9 + 110.2) / 2, places=1)
            self.assertEqual(summary["best_overall"], 3.8)

    def test_render_html_contains_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.csv"
            history.write_text(SAMPLE_CSV, encoding="utf-8")
            records = progress_dashboard.load_history(history)
            summary = progress_dashboard.summarise(records)
            html = progress_dashboard.render_html(records, summary)
            self.assertIn("Assess Speaking", html)
            self.assertIn("week2.m4a", html)

    def test_load_history_accepts_extended_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.csv"
            history.write_text(RICH_CSV, encoding="utf-8")
            records = progress_dashboard.load_history(history)
            self.assertEqual(records[0].task_family, "travel_narrative")
            self.assertEqual(records[1].speaker_id, "bern")
            self.assertAlmostEqual(records[1].final_score, 4.1)
            self.assertEqual(records[1].top_priorities, ("Più dettagli", "Più precisione", "Meno pause"))
            self.assertEqual(records[1].grammar_error_categories, ("preposition_choice",))

    def test_summarise_uses_final_score_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.csv"
            history.write_text(RICH_CSV, encoding="utf-8")
            records = progress_dashboard.load_history(history)
            summary = progress_dashboard.summarise(records)
            self.assertAlmostEqual(summary["avg_final"], (3.6 + 4.1) / 2, places=2)
            self.assertAlmostEqual(summary["best_final"], 4.1, places=2)

    def test_render_html_contains_task_family_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "history.csv"
            history.write_text(RICH_CSV, encoding="utf-8")
            records = progress_dashboard.load_history(history)
            summary = progress_dashboard.summarise(records)
            family_rows = [
                {
                    "task_family": "travel_narrative",
                    "count": 2,
                    "avg_final": 3.85,
                    "latest_final": 4.1,
                    "grammar_counts": {"preposition_choice": 2},
                    "coherence_counts": {"missing_sequence_markers": 2},
                }
            ]
            html = progress_dashboard.render_html(records, summary, family_rows=family_rows)
            self.assertIn("Task-Family Analyse", html)
            self.assertIn("preposition_choice (2)", html)
            self.assertIn("Prioritätenvergleich", html)

    def test_render_html_contains_progress_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "report": {
                            "progress_delta": {
                                "previous_session_id": "sess-1",
                                "score_delta": {"final": 0.5, "overall": 1.0, "wpm": 4.2},
                                "new_priorities": ["Più dettagli"],
                                "resolved_priorities": ["Meno filler"],
                                "repeating_grammar_categories": ["preposition_choice"],
                                "repeating_coherence_categories": ["missing_sequence_markers"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            history = Path(tmpdir) / "history.csv"
            history.write_text(
                RICH_CSV.replace("/path/to/2.json", str(report)),
                encoding="utf-8",
            )
            records = progress_dashboard.load_history(history)
            summary = progress_dashboard.summarise(records)
            html = progress_dashboard.render_html(
                records,
                summary,
                family_rows=[],
                progress_delta=progress_dashboard.load_progress_delta(records[-1].report_path),
            )
            self.assertIn("Progress Delta", html)
            self.assertIn("sess-1", html)
            self.assertIn("preposition_choice", html)


if __name__ == "__main__":
    unittest.main()

import tempfile
from datetime import datetime
from pathlib import Path

import unittest

from scripts import progress_dashboard


SAMPLE_CSV = """timestamp,audio,whisper,llm,label,duration_sec,wpm,word_count,overall,report_path
2025-10-06T14:58:01,demo.m4a,large-v3,llama3.1,baseline,43.09,95.9,54,3.5,/path/to/1.json
2025-10-07T09:12:33,week2.m4a,medium,llama3.2:3b,week2,41.00,110.2,60,3.8,/path/to/2.json
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


if __name__ == "__main__":
    unittest.main()

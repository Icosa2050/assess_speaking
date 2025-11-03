import json
import unittest
from pathlib import Path

import feedback


class FeedbackTests(unittest.TestCase):
    def _create_manifest(self, tmp_path: Path) -> Path:
        manifest = {
            "schema_version": "1.0",
            "resources": [
                {
                    "id": "res1",
                    "title": "Resource 1",
                    "url": "https://example.com/1",
                    "metrics": ["wpm", "fillers"],
                },
                {
                    "id": "res2",
                    "title": "Resource 2",
                    "url": "https://example.com/2",
                    "metrics": ["cohesion"],
                },
            ],
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        return p

    def test_load_manifest_valid(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            self._create_manifest(tmp_path)
            data = feedback.load_manifest(tmp_path)
            self.assertEqual(data["schema_version"], "1.0")
            self.assertEqual(len(data["resources"]), 2)

    def test_generate_feedback_filters_by_metric(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            self._create_manifest(tmp_path)
            metrics = {
                "wpm": 80,  # below target
                "word_count": 100,
                "fillers": 10,  # 10% > 5%
                "cohesion_markers": 0,  # below min
                "complexity_index": 2,
            }
            suggestions = feedback.generate_feedback(metrics, tmp_path)
            ids = {s["id"] for s in suggestions}
            self.assertIn("res1", ids)
            self.assertIn("res2", ids)


if __name__ == "__main__":
    unittest.main()

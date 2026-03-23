from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from corpora.lips_dataset import (
    LipsBuildConfig,
    build_excluded_audit_sample,
    build_lips_manifest,
    read_jsonl,
    summarize_lips_review,
    write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "corpora" / "lips_samples"


class LipsReviewSupportTests(unittest.TestCase):
    def test_build_excluded_audit_sample_covers_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = build_lips_manifest(
                LipsBuildConfig(input_root=FIXTURES_DIR, output_dir=Path(tmp_dir), review_sample_size=4)
            )
            excluded_rows = read_jsonl(report.excluded_path)
            excluded_sample = build_excluded_audit_sample([], sample_size=0)
            self.assertEqual(excluded_sample, [])
            from corpora.lips_dataset import LipsSectionRecord

            excluded_records = [LipsSectionRecord(**row) for row in excluded_rows]
            excluded_sample = build_excluded_audit_sample(excluded_records, sample_size=3, seed=3)

        self.assertEqual(len(excluded_sample), 3)
        self.assertEqual(
            sorted(item.proposed_exclusion_reason for item in excluded_sample),
            ["placeholder_section", "raw_mode_dialogue", "raw_mode_dialogue"],
        )

    def test_summarize_lips_review_reports_disagreements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            included_review = Path(tmp_dir) / "included_review.jsonl"
            excluded_review = Path(tmp_dir) / "excluded_review.jsonl"
            write_jsonl(
                included_review,
                [
                    {
                        "source_file": "a.txt",
                        "section_id": "SE1",
                        "mapped_task_family": "free_monologue",
                        "reviewer_accepts_mapping": False,
                        "reviewer_task_family": "personal_experience",
                        "reviewer_notes": "better as personal",
                    },
                    {
                        "source_file": "b.txt",
                        "section_id": "SE2",
                        "mapped_task_family": "picture_description",
                        "reviewer_accepts_mapping": True,
                        "reviewer_task_family": "picture_description",
                        "reviewer_notes": None,
                    },
                ],
            )
            write_jsonl(
                excluded_review,
                [
                    {
                        "source_file": "c.txt",
                        "section_id": "SE1",
                        "proposed_exclusion_reason": "turn_structure_not_monologue_like",
                        "reviewer_accepts_exclusion": False,
                        "reviewer_suggested_task_family": "opinion_monologue",
                        "reviewer_notes": "looks usable",
                    },
                    {
                        "source_file": "d.txt",
                        "section_id": "SE2",
                        "proposed_exclusion_reason": "raw_mode_dialogue",
                        "reviewer_accepts_exclusion": True,
                        "reviewer_suggested_task_family": None,
                        "reviewer_notes": None,
                    },
                ],
            )

            report = summarize_lips_review(
                included_review_path=included_review,
                excluded_review_path=excluded_review,
                output_path=Path(tmp_dir) / "summary.json",
            )

        self.assertEqual(report.included_reviewed_count, 2)
        self.assertEqual(report.included_agreement_ratio, 0.5)
        self.assertEqual(report.included_disagreements_by_family, {"free_monologue": 1})
        self.assertEqual(report.included_suggested_families, {"personal_experience": 1})
        self.assertEqual(report.excluded_reviewed_count, 2)
        self.assertEqual(report.excluded_agreement_ratio, 0.5)
        self.assertEqual(
            report.excluded_disagreements_by_reason,
            {"turn_structure_not_monologue_like": 1},
        )
        self.assertEqual(report.excluded_suggested_families, {"opinion_monologue": 1})

    def test_review_lips_manifest_script_prepare_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = Path(tmp_dir) / "artifacts"
            build_lips_manifest(LipsBuildConfig(input_root=FIXTURES_DIR, output_dir=artifact_dir, review_sample_size=4))
            script_path = REPO_ROOT / "scripts" / "review_lips_manifest.py"

            prepare_completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "prepare",
                    str(artifact_dir),
                    "--included-sample-size",
                    "4",
                    "--excluded-sample-size",
                    "3",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if prepare_completed.returncode != 0:
                self.fail(
                    "review_lips_manifest.py prepare failed\n"
                    f"stdout:\n{prepare_completed.stdout}\n"
                    f"stderr:\n{prepare_completed.stderr}"
                )
            prepare_payload = json.loads(prepare_completed.stdout)
            included_review_rows = read_jsonl(prepare_payload["included_output"])
            excluded_review_rows = read_jsonl(prepare_payload["excluded_output"])
            self.assertEqual(len(included_review_rows), 4)
            self.assertEqual(len(excluded_review_rows), 3)

            for row in included_review_rows:
                row["reviewer_accepts_mapping"] = True
                row["reviewer_task_family"] = row["mapped_task_family"]
            included_review_path = artifact_dir / "included_completed.jsonl"
            write_jsonl(included_review_path, included_review_rows)

            for index, row in enumerate(excluded_review_rows):
                row["reviewer_accepts_exclusion"] = index != 0
                row["reviewer_suggested_task_family"] = None if index != 0 else "picture_description"
            excluded_review_path = artifact_dir / "excluded_completed.jsonl"
            write_jsonl(excluded_review_path, excluded_review_rows)

            summarize_completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "summarize",
                    "--included-review",
                    str(included_review_path),
                    "--excluded-review",
                    str(excluded_review_path),
                    "--output",
                    str(artifact_dir / "review_summary.json"),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if summarize_completed.returncode != 0:
                self.fail(
                    "review_lips_manifest.py summarize failed\n"
                    f"stdout:\n{summarize_completed.stdout}\n"
                    f"stderr:\n{summarize_completed.stderr}"
                )
            summary_payload = json.loads(summarize_completed.stdout)

        self.assertEqual(summary_payload["included_reviewed_count"], 4)
        self.assertEqual(summary_payload["included_agreement_ratio"], 1.0)
        self.assertEqual(summary_payload["excluded_reviewed_count"], 3)
        self.assertAlmostEqual(summary_payload["excluded_agreement_ratio"], 2 / 3, places=4)


if __name__ == "__main__":
    unittest.main()

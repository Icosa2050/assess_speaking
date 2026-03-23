from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from corpora.celi_harvest import (
    CeliFrequencyBreakdown,
    CeliQuerySummary,
    analyze_wordlist_bundle,
    harvest_wordlist_manifest,
)
from corpora.celi_wordlists import (
    celi_wordlist_manifest_as_dict,
    discover_celi_wordlist_manifests,
    load_celi_wordlist_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "celi_wordlists" / "italian_core_benchmark_v1.json"


class CeliWordlistManifestTests(unittest.TestCase):
    def test_load_fixture_manifest(self) -> None:
        manifest = load_celi_wordlist_manifest(FIXTURE_PATH)
        self.assertEqual(manifest.manifest_id, "italian_celi_core_benchmark_v1")
        self.assertEqual(manifest.source_id, "celi")
        self.assertEqual(manifest.language_code, "it")
        self.assertEqual(manifest.default_levels, ("B1", "B2", "C1", "C2"))
        self.assertEqual(len(manifest.active_terms), 5)
        self.assertEqual(manifest.active_terms[0].term, "casa")

    def test_discover_fixture_manifest(self) -> None:
        manifests = discover_celi_wordlist_manifests(FIXTURE_PATH.parent, language_codes={"it"})
        self.assertEqual([manifest.manifest_id for manifest in manifests], ["italian_celi_core_benchmark_v1"])

    def test_manifest_as_dict_includes_terms(self) -> None:
        manifest = load_celi_wordlist_manifest(FIXTURE_PATH)
        payload = celi_wordlist_manifest_as_dict(manifest)
        self.assertEqual(payload["manifest_id"], manifest.manifest_id)
        self.assertEqual(payload["terms"][0]["term"], "casa")

    def test_harvest_wordlist_manifest_writes_bundle(self) -> None:
        manifest = load_celi_wordlist_manifest(FIXTURE_PATH)
        fake_summaries = [
            CeliQuerySummary(
                term="casa",
                level="B1",
                url="https://example.test/casa-b1",
                snapshot_path="/tmp/casa-b1.yml",
                hits_per_page=10,
                matches=10,
                different_texts=8,
                corpus_words=1000,
                corpus_texts=20,
                frequency_per_million=10000.0,
                elapsed_seconds=0.1,
                retrieved_from_cache=True,
            ),
            CeliQuerySummary(
                term="casa",
                level="B2",
                url="https://example.test/casa-b2",
                snapshot_path="/tmp/casa-b2.yml",
                hits_per_page=10,
                matches=11,
                different_texts=9,
                corpus_words=1100,
                corpus_texts=21,
                frequency_per_million=10001.0,
                elapsed_seconds=0.2,
                retrieved_from_cache=False,
            ),
        ]
        fake_breakdown = CeliFrequencyBreakdown(
            term="casa",
            level=None,
            url="https://example.test/casa",
            snapshot_path="/tmp/casa-frequency.yml",
            query_summary=fake_summaries[0],
            different_types=1,
            tokens_at_position=10,
            elapsed_seconds=0.3,
        )

        def fake_harvest_query_matrix(terms, *, levels, hits_per_page, output_dir):
            self.assertEqual(list(terms), ["casa"])
            self.assertEqual(tuple(levels), ("B1", "B2"))
            return fake_summaries

        def fake_harvest_frequency_breakdown(term, *, hits_per_page, output_dir):
            self.assertEqual(term, "casa")
            return fake_breakdown

        term = manifest.active_terms[0]
        narrow_term = type(term)(
            term_id=term.term_id,
            term=term.term,
            levels=("B1", "B2"),
            active=term.active,
            tags=term.tags,
            notes=term.notes,
        )
        narrow_manifest = type(manifest)(
            manifest_id=manifest.manifest_id,
            source_id=manifest.source_id,
            language_code=manifest.language_code,
            version=manifest.version,
            active=manifest.active,
            tags=manifest.tags,
            notes=manifest.notes,
            default_levels=("B1", "B2"),
            terms=(narrow_term,),
            source_path=manifest.source_path,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            from unittest import mock

            with (
                mock.patch("corpora.celi_harvest.harvest_query_matrix", side_effect=fake_harvest_query_matrix),
                mock.patch(
                    "corpora.celi_harvest.harvest_frequency_breakdown",
                    side_effect=fake_harvest_frequency_breakdown,
                ),
            ):
                report = harvest_wordlist_manifest(narrow_manifest, output_dir=tmp_dir)

            bundle_dir = Path(report.bundle_dir)
            self.assertTrue((bundle_dir / "bundle.json").exists())
            self.assertTrue((bundle_dir / "query_matrix.tsv").exists())
            self.assertTrue((bundle_dir / "frequency_breakdowns.tsv").exists())

            bundle_payload = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle_payload["manifest"]["manifest_id"], narrow_manifest.manifest_id)
            self.assertEqual(len(bundle_payload["query_summaries"]), 2)
            self.assertEqual(len(bundle_payload["frequency_breakdowns"]), 1)

            query_tsv = (bundle_dir / "query_matrix.tsv").read_text(encoding="utf-8")
            self.assertIn("manifest_id\tterm_id\tterm\tlevel", query_tsv)
            self.assertIn("casa", query_tsv)

    def test_analyze_wordlist_bundle_writes_rankings(self) -> None:
        bundle_payload = {
            "manifest": {
                "manifest_id": "demo",
                "default_levels": ["B1", "B2", "C1", "C2"],
            },
            "query_summaries": [
                {
                    "term_id": "beginner_term",
                    "term": "festa",
                    "level": "B1",
                    "frequency_per_million": 300.0,
                    "matches": 30,
                    "different_texts": 20,
                    "term_tags": ["event"],
                },
                {
                    "term_id": "beginner_term",
                    "term": "festa",
                    "level": "B2",
                    "frequency_per_million": 20.0,
                    "matches": 2,
                    "different_texts": 2,
                    "term_tags": ["event"],
                },
                {
                    "term_id": "beginner_term",
                    "term": "festa",
                    "level": "C1",
                    "frequency_per_million": 10.0,
                    "matches": 1,
                    "different_texts": 1,
                    "term_tags": ["event"],
                },
                {
                    "term_id": "beginner_term",
                    "term": "festa",
                    "level": "C2",
                    "frequency_per_million": 10.0,
                    "matches": 1,
                    "different_texts": 1,
                    "term_tags": ["event"],
                },
                {
                    "term_id": "advanced_term",
                    "term": "argomentazione",
                    "level": "B1",
                    "frequency_per_million": 10.0,
                    "matches": 1,
                    "different_texts": 1,
                    "term_tags": ["abstract"],
                },
                {
                    "term_id": "advanced_term",
                    "term": "argomentazione",
                    "level": "B2",
                    "frequency_per_million": 20.0,
                    "matches": 2,
                    "different_texts": 2,
                    "term_tags": ["abstract"],
                },
                {
                    "term_id": "advanced_term",
                    "term": "argomentazione",
                    "level": "C1",
                    "frequency_per_million": 100.0,
                    "matches": 10,
                    "different_texts": 8,
                    "term_tags": ["abstract"],
                },
                {
                    "term_id": "advanced_term",
                    "term": "argomentazione",
                    "level": "C2",
                    "frequency_per_million": 200.0,
                    "matches": 20,
                    "different_texts": 15,
                    "term_tags": ["abstract"],
                },
            ],
            "frequency_breakdowns": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "bundle.json"
            bundle_path.write_text(json.dumps(bundle_payload), encoding="utf-8")
            report = analyze_wordlist_bundle(bundle_path)

            analysis_json = Path(report.analysis_json_path)
            ranking_tsv = Path(report.ranking_tsv_path)
            self.assertTrue(analysis_json.exists())
            self.assertTrue(ranking_tsv.exists())

            analysis_payload = json.loads(analysis_json.read_text(encoding="utf-8"))
            self.assertEqual(analysis_payload["rows"][0]["term_id"], "advanced_term")
            self.assertGreater(analysis_payload["rows"][0]["directional_skew"], 0)
            self.assertLess(analysis_payload["rows"][-1]["directional_skew"], 0)

            ranking_text = ranking_tsv.read_text(encoding="utf-8")
            self.assertIn("f_B1\tf_B2\tf_C1\tf_C2", ranking_text)
            self.assertIn("peak_gap", ranking_text)


if __name__ == "__main__":
    unittest.main()

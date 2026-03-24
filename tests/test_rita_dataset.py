import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from corpora.rita_dataset import load_rita_archive, rita_summary_as_dict


def _write_csv(rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue()


class RitaDatasetTests(unittest.TestCase):
    def _build_fixture_archive(self, root: Path) -> Path:
        archive_path = root / "RITA_fixture.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("README.md", "# Fixture\n")
            zf.writestr("AbstractDatasetRITA_PHRAME.txt", "Fixture abstract.\n")
            zf.writestr("RITA_corpus_XMLdataset.xml", "<rita />\n")
            zf.writestr("RITA_corpus_XMLschema.xsd", "<schema />\n")
            zf.writestr(
                "RITA_exams_corpus.csv",
                _write_csv(
                    [
                        [
                            "id",
                            "text_id",
                            "CEFR",
                            "Lemma",
                            "lemma_head",
                            "deprel",
                            "occurrences",
                            "occurrences_head",
                            "cooccurrences",
                            "Unit_lemma",
                        ],
                        ["1", "10", "B1", "parco", "andare", "obj", "10", "15", "3", "andare parco"],
                        ["2", "10", "B1", "grande", "parco", "amod", "5", "10", "1", "parco grande"],
                    ]
                ),
            )
            zf.writestr(
                "RITA_exams_text_statistics.csv",
                _write_csv(
                    [
                        [
                            "text_id",
                            "TTR_Types",
                            "TTR_Lemma",
                            "TTR_Tokens",
                            "num_sentences",
                            "avg_token_sentence_length",
                            "stddev_token_sentence_length",
                            "avg_token_chr_len",
                            "stddev_token_chr_len",
                            "obj_types",
                            "amod_types",
                            "advmod_types",
                            "obj_total",
                            "amod_total",
                            "advmod_total",
                            "CEFR",
                            "a_id",
                        ],
                        ["10", "20", "18", "40", "4", "10.0", "2.0", "4.5", "1.5", "1", "1", "0", "2", "1", "0", "B1", "7"],
                        ["11", "25", "22", "55", "5", "11.0", "2.2", "4.7", "1.6", "2", "2", "1", "3", "2", "1", "C1", "8"],
                    ]
                ),
            )
            zf.writestr(
                "RITA_exams_CEFR_statistics.csv",
                _write_csv(
                    [
                        ["CEFR", "types", "lemma", "tokens"],
                        ["B1", "100", "80", "500"],
                        ["C1", "120", "95", "650"],
                    ]
                ),
            )
        return archive_path

    def test_load_rita_archive_parses_headers_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = self._build_fixture_archive(Path(tmp_dir))
            summary = load_rita_archive(archive_path)
        self.assertEqual(summary.corpus_unit_count, 2)
        self.assertEqual(summary.text_count, 2)
        self.assertEqual(summary.cefr_levels, ("B1", "C1"))
        self.assertTrue(summary.xml_available)
        self.assertTrue(summary.schema_available)
        self.assertFalse(summary.contains_full_text_column)
        self.assertEqual(summary.text_statistics[0].text_id, 10)
        self.assertEqual(summary.cefr_statistics[1].tokens, 650)

    def test_rita_summary_as_dict_is_json_friendly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = self._build_fixture_archive(Path(tmp_dir))
            summary = load_rita_archive(archive_path)
            payload = rita_summary_as_dict(summary)
        self.assertEqual(payload["corpus_unit_count"], 2)
        self.assertEqual(payload["text_count"], 2)
        self.assertEqual(payload["cefr_levels"], ["B1", "C1"])
        self.assertEqual(payload["text_statistics_sample"]["assignment_id"], "7")


if __name__ == "__main__":
    unittest.main()

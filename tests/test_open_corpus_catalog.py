import unittest

from corpora.open_corpus_catalog import (
    downloadable_open_corpus_sources,
    list_open_corpus_sources,
    resolve_open_corpus_source,
)


class OpenCorpusCatalogTests(unittest.TestCase):
    def test_catalog_contains_expected_sources(self) -> None:
        source_ids = [source.source_id for source in list_open_corpus_sources()]
        self.assertIn("rita_phrame4", source_ids)
        self.assertIn("merlin_v1_2", source_ids)
        self.assertIn("ud_italian_valico", source_ids)

    def test_downloadable_catalog_starts_with_rita(self) -> None:
        downloadable = downloadable_open_corpus_sources()
        self.assertEqual([source.source_id for source in downloadable], ["rita_phrame4"])
        self.assertEqual(downloadable[0].archive_name, "RITA_PHRAME4.zip")
        self.assertIn("zenodo", downloadable[0].download_url or "")

    def test_resolve_unknown_source_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            resolve_open_corpus_source("missing_source")


if __name__ == "__main__":
    unittest.main()

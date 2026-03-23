from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from corpora.celi_harvest import (
    build_concordance_url,
    parse_download_page_refs,
    parse_downloaded_file,
    parse_frequency_breakdown,
    parse_query_action_refs,
    parse_query_summary,
)


QUERY_SNAPSHOT_WITH_LEVEL = """\
- generic [active] [ref=e1]:
  - table [ref=e2]:
    - rowgroup [ref=e3]:
      - 'row "Your query “casa”, restricted to texts meeting criteria “CEFR level: B1”, returned 213 matches in 174 different texts (in 156,612 words [1,212 texts]; frequency: 1,360.05 instances per million words) [0.008 seconds - retrieved from cache]" [ref=e4]':
        - 'columnheader "Your query “casa”, restricted to texts meeting criteria “CEFR level: B1”, returned 213 matches in 174 different texts (in 156,612 words [1,212 texts]; frequency: 1,360.05 instances per million words) [0.008 seconds - retrieved from cache]" [ref=e5]':
      - 'row "|< << >> >| Show Page: 1 Line View Show in random order Choose action... Go!" [ref=e8]':
        - cell "Choose action... Go!" [ref=e29]:
          - combobox [ref=e30]:
            - option "Choose action..." [selected]
            - option "Frequency breakdown"
            - option "Download..."
          - button "Go!" [ref=e31]
"""

QUERY_SNAPSHOT_NO_LEVEL = """\
- generic [active] [ref=e1]:
  - table [ref=e2]:
    - rowgroup [ref=e3]:
      - 'row "Your query “casa” returned 846 matches in 579 different texts (in 608,614 words [3,041 texts]; frequency: 1,390.04 instances per million words) [0.03 seconds - retrieved from cache]" [ref=e4]':
        - 'columnheader "Your query “casa” returned 846 matches in 579 different texts (in 608,614 words [3,041 texts]; frequency: 1,390.04 instances per million words) [ref=e5]'
"""

DOWNLOAD_PAGE_SNAPSHOT = """\
- generic [active] [ref=e1]:
  - table [ref=e14]:
    - rowgroup [ref=e15]:
      - row "Download concordance" [ref=e4]:
        - columnheader "Download concordance" [ref=e5]
      - 'row "Enter name for the downloaded file: concordance" [ref=e60]':
        - cell "concordance" [ref=e62]:
          - textbox [active] [ref=e63]: concordance
      - 'row "Select from available text metadata: CEFR level Task assignment ID Nationality Text genre Text type" [ref=e70]':
        - cell "Select from available text metadata:" [ref=e71]
        - cell "CEFR level Task assignment ID Nationality Text genre Text type" [ref=e72]:
          - checkbox "CEFR level" [checked] [ref=e76]
          - text: CEFR level
          - checkbox "Task assignment ID" [checked] [ref=e79]
          - text: Task assignment ID
          - checkbox "Nationality" [checked] [ref=e80]
          - text: Nationality
          - checkbox "Text genre" [ref=e82]
          - text: Text genre
          - checkbox "Text type" [checked] [ref=e83]
          - text: Text type
      - row "Download with settings above" [ref=e84]:
        - cell "Download with settings above" [ref=e85]:
          - button "Download with settings above" [ref=e86]
"""

FREQUENCY_BREAKDOWN_SNAPSHOT = """\
- generic [active] [ref=e1]:
  - table [ref=e2]:
    - rowgroup [ref=e3]:
      - 'row "Query “casa” returned 846 matches in 579 different texts (in 608,614 words [3,041 texts]; frequency: 1,390.04 instances per million words)" [ref=e4]':
        - 'columnheader "Query “casa” returned 846 matches in 579 different texts (in 608,614 words [3,041 texts]; frequency: 1,390.04 instances per million words)" [ref=e5]':
      - row "Showing frequency breakdown of words in this query, at the query node; there is 1 different type and 846 tokens at this concordance position. [0.898 seconds]" [ref=e6]:
        - columnheader "Showing frequency breakdown of words in this query, at the query node; there is 1 different type and 846 tokens at this concordance position. [0.898 seconds]" [ref=e7]
"""


class CeliHarvestTests(unittest.TestCase):
    def test_build_concordance_url_with_level(self) -> None:
        self.assertEqual(
            build_concordance_url("casa", level="B2", hits_per_page=10),
            "https://apps.unistrapg.it/cqpweb/celi/concordance.php?theData=casa&qmode=sq_nocase&pp=10&del=begin&t=-%7Ctext_cefr%7EB2&del=end",
        )

    def test_parse_query_summary_with_level(self) -> None:
        snapshot = self._write_temp_snapshot(QUERY_SNAPSHOT_WITH_LEVEL)
        summary = parse_query_summary(snapshot, url="https://example.test/query", hits_per_page=10)
        self.assertEqual(summary.term, "casa")
        self.assertEqual(summary.level, "B1")
        self.assertEqual(summary.matches, 213)
        self.assertEqual(summary.different_texts, 174)
        self.assertEqual(summary.corpus_words, 156612)
        self.assertEqual(summary.corpus_texts, 1212)
        self.assertAlmostEqual(summary.frequency_per_million, 1360.05)
        self.assertAlmostEqual(summary.elapsed_seconds or 0.0, 0.008)
        self.assertTrue(summary.retrieved_from_cache)

    def test_parse_query_summary_without_level(self) -> None:
        snapshot = self._write_temp_snapshot(QUERY_SNAPSHOT_NO_LEVEL)
        summary = parse_query_summary(snapshot, url="https://example.test/query", hits_per_page=10)
        self.assertEqual(summary.level, None)
        self.assertEqual(summary.matches, 846)
        self.assertEqual(summary.corpus_texts, 3041)

    def test_parse_query_action_refs(self) -> None:
        snapshot = self._write_temp_snapshot(QUERY_SNAPSHOT_WITH_LEVEL)
        refs = parse_query_action_refs(snapshot)
        self.assertEqual(refs.action_ref, "e30")
        self.assertEqual(refs.go_ref, "e31")

    def test_parse_download_page_refs(self) -> None:
        snapshot = self._write_temp_snapshot(DOWNLOAD_PAGE_SNAPSHOT)
        refs = parse_download_page_refs(snapshot)
        self.assertEqual(refs.filename_ref, "e63")
        self.assertEqual(refs.download_button_ref, "e86")
        self.assertTrue(refs.checkboxes["CEFR level"].checked)
        self.assertFalse(refs.checkboxes["Text genre"].checked)
        self.assertEqual(refs.checkboxes["Text type"].ref, "e83")

    def test_parse_downloaded_file(self) -> None:
        name, path = parse_downloaded_file(
            '### Events\n- Downloading file casac2sample.txt ...\n- Downloaded file casac2sample.txt to "output/playwright/celi/casac2sample.txt"\n'
        )
        self.assertEqual(name, "casac2sample.txt")
        self.assertTrue(path.as_posix().endswith("/output/playwright/celi/casac2sample.txt"))

    def test_parse_frequency_breakdown(self) -> None:
        snapshot = self._write_temp_snapshot(FREQUENCY_BREAKDOWN_SNAPSHOT)
        breakdown = parse_frequency_breakdown(snapshot, url="https://example.test/breakdown", hits_per_page=10)
        self.assertEqual(breakdown.term, "casa")
        self.assertEqual(breakdown.different_types, 1)
        self.assertEqual(breakdown.tokens_at_position, 846)
        self.assertAlmostEqual(breakdown.elapsed_seconds, 0.898)

    def _write_temp_snapshot(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "snapshot.yml"
        path.write_text(content, encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()

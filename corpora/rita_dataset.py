"""Inspect and summarize the open RITA learner-corpus archive."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
from pathlib import Path
from typing import Any
import zipfile


@dataclass(frozen=True)
class RitaTextStatisticsRow:
    text_id: int
    cefr: str
    assignment_id: str | None
    types: int
    lemmas: int
    tokens: int
    num_sentences: int
    avg_token_sentence_length: float
    stddev_token_sentence_length: float
    avg_token_chr_len: float
    stddev_token_chr_len: float
    obj_types: int
    amod_types: int
    advmod_types: int
    obj_total: int
    amod_total: int
    advmod_total: int


@dataclass(frozen=True)
class RitaCefrStatisticsRow:
    cefr: str
    types: int
    lemmas: int
    tokens: int


@dataclass(frozen=True)
class RitaArchiveSummary:
    archive_path: Path
    archive_entries: tuple[str, ...]
    corpus_header: tuple[str, ...]
    text_statistics_header: tuple[str, ...]
    cefr_statistics_header: tuple[str, ...]
    corpus_unit_count: int
    text_count: int
    cefr_levels: tuple[str, ...]
    text_statistics: tuple[RitaTextStatisticsRow, ...]
    cefr_statistics: tuple[RitaCefrStatisticsRow, ...]
    xml_available: bool
    schema_available: bool
    contains_full_text_column: bool


def _read_csv_rows(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    raw = zf.read(name)
    text = raw.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _read_csv_header_and_count(zf: zipfile.ZipFile, name: str) -> tuple[tuple[str, ...], int]:
    raw = zf.read(name)
    text = raw.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    row_count = sum(1 for _ in reader)
    return tuple(header), row_count


def _as_int(value: str | None) -> int:
    if value is None or value == "":
        raise ValueError("Missing integer field in RITA archive")
    return int(value)


def _as_float(value: str | None) -> float:
    if value is None or value == "":
        raise ValueError("Missing float field in RITA archive")
    return float(value)


def load_rita_archive(path: str | Path) -> RitaArchiveSummary:
    archive_path = Path(path).resolve()
    with zipfile.ZipFile(archive_path) as zf:
        entries = tuple(zf.namelist())
        corpus_header, corpus_unit_count = _read_csv_header_and_count(zf, "RITA_exams_corpus.csv")
        text_rows = _read_csv_rows(zf, "RITA_exams_text_statistics.csv")
        cefr_rows = _read_csv_rows(zf, "RITA_exams_CEFR_statistics.csv")
        text_statistics = tuple(
            RitaTextStatisticsRow(
                text_id=_as_int(row["text_id"]),
                cefr=str(row["CEFR"]).strip().upper(),
                assignment_id=(str(row["a_id"]).strip() or None),
                types=_as_int(row["TTR_Types"]),
                lemmas=_as_int(row["TTR_Lemma"]),
                tokens=_as_int(row["TTR_Tokens"]),
                num_sentences=_as_int(row["num_sentences"]),
                avg_token_sentence_length=_as_float(row["avg_token_sentence_length"]),
                stddev_token_sentence_length=_as_float(row["stddev_token_sentence_length"]),
                avg_token_chr_len=_as_float(row["avg_token_chr_len"]),
                stddev_token_chr_len=_as_float(row["stddev_token_chr_len"]),
                obj_types=_as_int(row["obj_types"]),
                amod_types=_as_int(row["amod_types"]),
                advmod_types=_as_int(row["advmod_types"]),
                obj_total=_as_int(row["obj_total"]),
                amod_total=_as_int(row["amod_total"]),
                advmod_total=_as_int(row["advmod_total"]),
            )
            for row in text_rows
        )
        cefr_statistics = tuple(
            RitaCefrStatisticsRow(
                cefr=str(row["CEFR"]).strip().upper(),
                types=_as_int(row["types"]),
                lemmas=_as_int(row["lemma"]),
                tokens=_as_int(row["tokens"]),
            )
            for row in cefr_rows
        )
        text_statistics_header = tuple(text_rows[0].keys()) if text_rows else ()
        cefr_statistics_header = tuple(cefr_rows[0].keys()) if cefr_rows else ()
    return RitaArchiveSummary(
        archive_path=archive_path,
        archive_entries=entries,
        corpus_header=corpus_header,
        text_statistics_header=text_statistics_header,
        cefr_statistics_header=cefr_statistics_header,
        corpus_unit_count=corpus_unit_count,
        text_count=len(text_statistics),
        cefr_levels=tuple(sorted({row.cefr for row in text_statistics})),
        text_statistics=text_statistics,
        cefr_statistics=cefr_statistics,
        xml_available="RITA_corpus_XMLdataset.xml" in entries,
        schema_available="RITA_corpus_XMLschema.xsd" in entries,
        contains_full_text_column=any(
            field.lower() in {"text", "raw_text", "full_text", "transcript"} for field in corpus_header
        )
        or any(
            field.lower() in {"text", "raw_text", "full_text", "transcript"} for field in text_statistics_header
        ),
    )


def rita_summary_as_dict(summary: RitaArchiveSummary) -> dict[str, Any]:
    return {
        "archive_path": summary.archive_path.as_posix(),
        "archive_entries": list(summary.archive_entries),
        "corpus_header": list(summary.corpus_header),
        "text_statistics_header": list(summary.text_statistics_header),
        "cefr_statistics_header": list(summary.cefr_statistics_header),
        "corpus_unit_count": summary.corpus_unit_count,
        "text_count": summary.text_count,
        "cefr_levels": list(summary.cefr_levels),
        "xml_available": summary.xml_available,
        "schema_available": summary.schema_available,
        "contains_full_text_column": summary.contains_full_text_column,
        "text_statistics_sample": asdict(summary.text_statistics[0]) if summary.text_statistics else None,
        "cefr_statistics": [asdict(row) for row in summary.cefr_statistics],
    }

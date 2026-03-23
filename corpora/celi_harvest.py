"""Helpers for harvesting approved CELI query artifacts via the Playwright CELI session."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import subprocess
from typing import Iterable
from urllib.parse import quote

from corpora.celi_wordlists import CeliWordlistManifest, celi_wordlist_manifest_as_dict

CEFR_LEVEL_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_LEVEL_INDEX = {level: index for index, level in enumerate(CEFR_LEVEL_ORDER)}


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYWRIGHT_CELI_SCRIPT = REPO_ROOT / "scripts" / "playwright_celi.sh"

_QUERY_SUMMARY_PATTERN = re.compile(
    r"(?:Your query|Query) “(?P<term>.+?)”"
    r"(?:, restricted to texts meeting criteria “CEFR level: (?P<level>[A-Z0-9]+)”,)?"  # optional level
    r" returned (?P<matches>[\d,]+) matches in (?P<different_texts>[\d,]+) different texts "
    r"\(in (?P<corpus_words>[\d,]+) words \[(?P<corpus_texts>[\d,]+) texts\]; "
    r"frequency: (?P<frequency>[0-9,]+(?:\.[0-9]+)?) instances per million words\)"
    r"(?: \[(?P<seconds>[0-9.]+) seconds(?P<cache> - retrieved from cache)?\])?",
)
_FREQUENCY_BREAKDOWN_PATTERN = re.compile(
    r"Showing frequency breakdown .*? there is (?P<types>[\d,]+) different type[s]?"
    r" and (?P<tokens>[\d,]+) tokens .*? \[(?P<seconds>[0-9.]+) seconds(?: - retrieved from cache)?\]",
)
_ACTION_MENU_PATTERN = re.compile(
    r'cell "Choose action\.\.\. Go!" \[ref=[^\]]+\]:(?P<body>.*?)- button "Go!" \[ref=(?P<go_ref>e\d+)\]',
    re.DOTALL,
)
_DOWNLOAD_OUTPUT_PATTERN = re.compile(r'Downloaded file (?P<name>.+?) to "(?P<path>.+?)"')
_CHECKBOX_PATTERN = re.compile(
    r'checkbox "(?P<label>[^"]+)"(?P<attrs>(?: \[[^\]]+\])*) \[ref=(?P<ref>e\d+)\]'
)


@dataclass(frozen=True)
class QueryActionRefs:
    action_ref: str
    go_ref: str


@dataclass(frozen=True)
class DownloadCheckbox:
    label: str
    ref: str
    checked: bool


@dataclass(frozen=True)
class DownloadPageRefs:
    filename_ref: str
    download_button_ref: str
    checkboxes: dict[str, DownloadCheckbox]


@dataclass(frozen=True)
class CeliQuerySummary:
    term: str
    level: str | None
    url: str
    snapshot_path: str
    hits_per_page: int
    matches: int
    different_texts: int
    corpus_words: int
    corpus_texts: int
    frequency_per_million: float
    elapsed_seconds: float | None
    retrieved_from_cache: bool


@dataclass(frozen=True)
class CeliFrequencyBreakdown:
    term: str
    level: str | None
    url: str
    snapshot_path: str
    query_summary: CeliQuerySummary
    different_types: int
    tokens_at_position: int
    elapsed_seconds: float


@dataclass(frozen=True)
class CeliDownloadResult:
    term: str
    level: str | None
    hits_per_page: int
    metadata_fields: tuple[str, ...]
    query_summary: CeliQuerySummary
    download_page_snapshot: str
    configured_page_snapshot: str
    download_stdout: str
    downloaded_name: str
    downloaded_path: str


@dataclass(frozen=True)
class CeliWordlistBundleReport:
    manifest_id: str
    bundle_dir: str
    bundle_json_path: str
    query_matrix_tsv_path: str
    frequency_breakdowns_tsv_path: str | None
    query_count: int
    frequency_count: int


@dataclass(frozen=True)
class CeliTermSkewRow:
    term_id: str
    term: str
    peak_level: str
    peak_share: float
    peak_gap: float
    cefr_center: float
    directional_skew: float
    total_frequency_per_million: float
    level_frequencies: dict[str, float]
    level_shares: dict[str, float]
    matches_by_level: dict[str, int]
    different_texts_by_level: dict[str, int]
    term_tags: tuple[str, ...]


@dataclass(frozen=True)
class CeliWordlistAnalysisReport:
    bundle_json_path: str
    analysis_json_path: str
    ranking_tsv_path: str
    term_count: int
    sort_by: str
    ascending: bool


def build_concordance_url(term: str, *, level: str | None = None, hits_per_page: int = 10) -> str:
    encoded_term = quote(term, safe="")
    parts = [
        f"https://apps.unistrapg.it/cqpweb/celi/concordance.php?theData={encoded_term}",
        f"qmode=sq_nocase",
        f"pp={hits_per_page}",
        "del=begin",
    ]
    if level:
        parts.append(f"t=-%7Ctext_cefr%7E{quote(level, safe='')}")
    parts.append("del=end")
    return "&".join(parts)


def default_harvest_output_dir() -> Path:
    return REPO_ROOT / "tmp" / "celi_harvest"


def run_playwright(*args: str, timeout: int = 120) -> str:
    completed = subprocess.run(
        [str(PLAYWRIGHT_CELI_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "playwright_celi.sh failed\n"
            f"args: {json.dumps(list(args), ensure_ascii=False)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def parse_query_summary(snapshot_path: str | Path, *, url: str, hits_per_page: int) -> CeliQuerySummary:
    snapshot_text = Path(snapshot_path).read_text(encoding="utf-8")
    match = _QUERY_SUMMARY_PATTERN.search(snapshot_text)
    if not match:
        raise ValueError(f"Could not find query summary in {snapshot_path}")
    return CeliQuerySummary(
        term=match.group("term"),
        level=match.group("level"),
        url=url,
        snapshot_path=str(Path(snapshot_path).resolve()),
        hits_per_page=hits_per_page,
        matches=_parse_int(match.group("matches")),
        different_texts=_parse_int(match.group("different_texts")),
        corpus_words=_parse_int(match.group("corpus_words")),
        corpus_texts=_parse_int(match.group("corpus_texts")),
        frequency_per_million=float(match.group("frequency").replace(",", "")),
        elapsed_seconds=float(match.group("seconds")) if match.group("seconds") else None,
        retrieved_from_cache=bool(match.group("cache")),
    )


def parse_frequency_breakdown(snapshot_path: str | Path, *, url: str, hits_per_page: int) -> CeliFrequencyBreakdown:
    snapshot_text = Path(snapshot_path).read_text(encoding="utf-8")
    query_summary = parse_query_summary(snapshot_path, url=url, hits_per_page=hits_per_page)
    match = _FREQUENCY_BREAKDOWN_PATTERN.search(snapshot_text)
    if not match:
        raise ValueError(f"Could not find frequency breakdown summary in {snapshot_path}")
    return CeliFrequencyBreakdown(
        term=query_summary.term,
        level=query_summary.level,
        url=url,
        snapshot_path=str(Path(snapshot_path).resolve()),
        query_summary=query_summary,
        different_types=_parse_int(match.group("types")),
        tokens_at_position=_parse_int(match.group("tokens")),
        elapsed_seconds=float(match.group("seconds")),
    )


def parse_query_action_refs(snapshot_path: str | Path) -> QueryActionRefs:
    snapshot_text = Path(snapshot_path).read_text(encoding="utf-8")
    action_block = _ACTION_MENU_PATTERN.search(snapshot_text)
    if not action_block:
        raise ValueError(f"Could not find result-page action refs in {snapshot_path}")
    action_match = re.search(r"- combobox \[ref=(e\d+)\]:", action_block.group("body"))
    if not action_match:
        raise ValueError(f"Could not find result-page action combobox in {snapshot_path}")
    return QueryActionRefs(action_ref=action_match.group(1), go_ref=action_block.group("go_ref"))


def parse_download_page_refs(snapshot_path: str | Path) -> DownloadPageRefs:
    snapshot_text = Path(snapshot_path).read_text(encoding="utf-8")
    filename_match = re.search(
        r'row "Enter name for the downloaded file:.*?(?:\n.*?)*?textbox(?: \[[^\]]+\])* \[ref=(e\d+)\]:',
        snapshot_text,
        re.DOTALL,
    )
    download_button_match = re.search(r'- button "Download with settings above" \[ref=(e\d+)\]', snapshot_text)
    if not filename_match or not download_button_match:
        raise ValueError(f"Could not find download-page controls in {snapshot_path}")

    checkboxes: dict[str, DownloadCheckbox] = {}
    for match in _CHECKBOX_PATTERN.finditer(snapshot_text):
        label = match.group("label")
        checkboxes[label] = DownloadCheckbox(
            label=label,
            ref=match.group("ref"),
            checked="[checked]" in match.group("attrs"),
        )
    return DownloadPageRefs(
        filename_ref=filename_match.group(1),
        download_button_ref=download_button_match.group(1),
        checkboxes=checkboxes,
    )


def parse_downloaded_file(stdout: str) -> tuple[str, Path]:
    match = _DOWNLOAD_OUTPUT_PATTERN.search(stdout)
    if not match:
        raise ValueError("Could not find downloaded file path in Playwright output")
    return match.group("name"), Path(match.group("path")).resolve()


def harvest_query(
    term: str,
    *,
    level: str | None = None,
    hits_per_page: int = 10,
    output_dir: str | Path | None = None,
) -> CeliQuerySummary:
    output_root = _prepare_output_dir(output_dir)
    url = build_concordance_url(term, level=level, hits_per_page=hits_per_page)
    snapshot_path = output_root / _snapshot_name(term, level, "query")
    run_playwright("goto", url)
    run_playwright("snapshot", "--filename", str(snapshot_path))
    return parse_query_summary(snapshot_path, url=url, hits_per_page=hits_per_page)


def harvest_query_matrix(
    terms: Iterable[str],
    *,
    levels: Iterable[str | None],
    hits_per_page: int = 10,
    output_dir: str | Path | None = None,
) -> list[CeliQuerySummary]:
    results: list[CeliQuerySummary] = []
    for term in terms:
        for level in levels:
            results.append(
                harvest_query(
                    term,
                    level=level,
                    hits_per_page=hits_per_page,
                    output_dir=output_dir,
                )
            )
    return results


def harvest_frequency_breakdown(
    term: str,
    *,
    level: str | None = None,
    hits_per_page: int = 10,
    output_dir: str | Path | None = None,
) -> CeliFrequencyBreakdown:
    output_root = _prepare_output_dir(output_dir)
    query_summary = harvest_query(term, level=level, hits_per_page=hits_per_page, output_dir=output_root)
    action_refs = parse_query_action_refs(query_summary.snapshot_path)
    run_playwright("select", action_refs.action_ref, "Frequency breakdown")
    run_playwright("click", action_refs.go_ref)
    breakdown_snapshot = output_root / _snapshot_name(term, level, "frequency")
    run_playwright("snapshot", "--filename", str(breakdown_snapshot))
    return parse_frequency_breakdown(breakdown_snapshot, url=query_summary.url, hits_per_page=hits_per_page)


def harvest_export(
    term: str,
    *,
    level: str,
    metadata_fields: Iterable[str],
    hits_per_page: int = 10,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> CeliDownloadResult:
    output_root = _prepare_output_dir(output_dir)
    query_summary = harvest_query(term, level=level, hits_per_page=hits_per_page, output_dir=output_root)
    action_refs = parse_query_action_refs(query_summary.snapshot_path)
    run_playwright("select", action_refs.action_ref, "Download...")
    run_playwright("click", action_refs.go_ref)

    download_snapshot = output_root / _snapshot_name(term, level, "download-page")
    run_playwright("snapshot", "--filename", str(download_snapshot))
    download_refs = parse_download_page_refs(download_snapshot)

    chosen_filename = filename or _export_filename(term, level)
    run_playwright("fill", download_refs.filename_ref, chosen_filename)

    requested_fields = tuple(field.strip() for field in metadata_fields if field.strip())
    for field in requested_fields:
        checkbox = download_refs.checkboxes.get(field)
        if checkbox is None:
            raise KeyError(f"Download page does not expose metadata checkbox: {field}")
        if not checkbox.checked:
            run_playwright("click", checkbox.ref)

    configured_snapshot = output_root / _snapshot_name(term, level, "download-ready")
    run_playwright("snapshot", "--filename", str(configured_snapshot))
    click_output = run_playwright("click", download_refs.download_button_ref)
    downloaded_name, downloaded_path = parse_downloaded_file(click_output)

    return CeliDownloadResult(
        term=term,
        level=level,
        hits_per_page=hits_per_page,
        metadata_fields=requested_fields,
        query_summary=query_summary,
        download_page_snapshot=str(download_snapshot.resolve()),
        configured_page_snapshot=str(configured_snapshot.resolve()),
        download_stdout=click_output,
        downloaded_name=downloaded_name,
        downloaded_path=str(downloaded_path),
    )


def query_summary_as_dict(summary: CeliQuerySummary) -> dict[str, object]:
    return asdict(summary)


def frequency_breakdown_as_dict(breakdown: CeliFrequencyBreakdown) -> dict[str, object]:
    payload = asdict(breakdown)
    payload["query_summary"] = query_summary_as_dict(breakdown.query_summary)
    return payload


def download_result_as_dict(result: CeliDownloadResult) -> dict[str, object]:
    payload = asdict(result)
    payload["query_summary"] = query_summary_as_dict(result.query_summary)
    return payload


def wordlist_bundle_report_as_dict(report: CeliWordlistBundleReport) -> dict[str, object]:
    return asdict(report)


def term_skew_row_as_dict(row: CeliTermSkewRow) -> dict[str, object]:
    return asdict(row)


def wordlist_analysis_report_as_dict(report: CeliWordlistAnalysisReport) -> dict[str, object]:
    return asdict(report)


def harvest_wordlist_manifest(
    manifest: CeliWordlistManifest,
    *,
    hits_per_page: int = 10,
    output_dir: str | Path | None = None,
    include_frequency: bool = True,
) -> CeliWordlistBundleReport:
    bundle_dir = _prepare_output_dir(output_dir) / manifest.manifest_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    query_rows: list[dict[str, object]] = []
    query_summaries: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []
    frequency_breakdowns: list[dict[str, object]] = []

    for term in manifest.active_terms:
        summaries = harvest_query_matrix(
            [term.term],
            levels=term.levels,
            hits_per_page=hits_per_page,
            output_dir=bundle_dir,
        )
        for summary in summaries:
            summary_payload = query_summary_as_dict(summary)
            summary_payload["term_id"] = term.term_id
            summary_payload["term_tags"] = list(term.tags)
            summary_payload["term_notes"] = term.notes
            query_summaries.append(summary_payload)
            query_rows.append(
                {
                    "manifest_id": manifest.manifest_id,
                    "term_id": term.term_id,
                    "term": summary.term,
                    "level": summary.level or "",
                    "matches": summary.matches,
                    "different_texts": summary.different_texts,
                    "corpus_words": summary.corpus_words,
                    "corpus_texts": summary.corpus_texts,
                    "frequency_per_million": summary.frequency_per_million,
                    "elapsed_seconds": summary.elapsed_seconds if summary.elapsed_seconds is not None else "",
                    "retrieved_from_cache": summary.retrieved_from_cache,
                    "snapshot_path": summary.snapshot_path,
                    "term_tags": ",".join(term.tags),
                }
            )

        if include_frequency:
            breakdown = harvest_frequency_breakdown(
                term.term,
                hits_per_page=hits_per_page,
                output_dir=bundle_dir,
            )
            breakdown_payload = frequency_breakdown_as_dict(breakdown)
            breakdown_payload["term_id"] = term.term_id
            breakdown_payload["term_tags"] = list(term.tags)
            breakdown_payload["term_notes"] = term.notes
            frequency_breakdowns.append(breakdown_payload)
            frequency_rows.append(
                {
                    "manifest_id": manifest.manifest_id,
                    "term_id": term.term_id,
                    "term": breakdown.term,
                    "different_types": breakdown.different_types,
                    "tokens_at_position": breakdown.tokens_at_position,
                    "elapsed_seconds": breakdown.elapsed_seconds,
                    "matches": breakdown.query_summary.matches,
                    "different_texts": breakdown.query_summary.different_texts,
                    "corpus_words": breakdown.query_summary.corpus_words,
                    "corpus_texts": breakdown.query_summary.corpus_texts,
                    "frequency_per_million": breakdown.query_summary.frequency_per_million,
                    "snapshot_path": breakdown.snapshot_path,
                    "term_tags": ",".join(term.tags),
                }
            )

    bundle_json_path = bundle_dir / "bundle.json"
    bundle_json_path.write_text(
        json.dumps(
            {
                "manifest": celi_wordlist_manifest_as_dict(manifest),
                "hits_per_page": hits_per_page,
                "query_summaries": query_summaries,
                "frequency_breakdowns": frequency_breakdowns,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    query_tsv_path = bundle_dir / "query_matrix.tsv"
    _write_tsv(
        query_tsv_path,
        query_rows,
        fieldnames=(
            "manifest_id",
            "term_id",
            "term",
            "level",
            "matches",
            "different_texts",
            "corpus_words",
            "corpus_texts",
            "frequency_per_million",
            "elapsed_seconds",
            "retrieved_from_cache",
            "snapshot_path",
            "term_tags",
        ),
    )

    frequency_tsv_path: Path | None = None
    if include_frequency:
        frequency_tsv_path = bundle_dir / "frequency_breakdowns.tsv"
        _write_tsv(
            frequency_tsv_path,
            frequency_rows,
            fieldnames=(
                "manifest_id",
                "term_id",
                "term",
                "different_types",
                "tokens_at_position",
                "elapsed_seconds",
                "matches",
                "different_texts",
                "corpus_words",
                "corpus_texts",
                "frequency_per_million",
                "snapshot_path",
                "term_tags",
            ),
        )

    return CeliWordlistBundleReport(
        manifest_id=manifest.manifest_id,
        bundle_dir=str(bundle_dir.resolve()),
        bundle_json_path=str(bundle_json_path.resolve()),
        query_matrix_tsv_path=str(query_tsv_path.resolve()),
        frequency_breakdowns_tsv_path=str(frequency_tsv_path.resolve()) if frequency_tsv_path else None,
        query_count=len(query_rows),
        frequency_count=len(frequency_rows),
    )


def analyze_wordlist_bundle(
    bundle_json_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    sort_by: str = "directional_skew",
    ascending: bool = False,
) -> CeliWordlistAnalysisReport:
    if sort_by not in {"directional_skew", "cefr_center", "peak_gap"}:
        raise ValueError("sort_by must be one of: directional_skew, cefr_center, peak_gap")

    bundle_path = Path(bundle_json_path).resolve()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest_payload = payload.get("manifest") or {}
    query_summaries = payload.get("query_summaries") or []
    if not isinstance(query_summaries, list) or not query_summaries:
        raise ValueError("bundle.json must include a non-empty query_summaries list")

    level_order = _extract_bundle_level_order(manifest_payload, query_summaries)
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in query_summaries:
        grouped.setdefault(str(item["term_id"]), []).append(item)

    rows = [
        _analyze_term_rows(term_items, level_order=level_order)
        for _, term_items in sorted(grouped.items(), key=lambda item: item[0])
    ]
    rows = _sorted_skew_rows(rows, sort_by=sort_by, ascending=ascending)

    destination = Path(output_dir).resolve() if output_dir is not None else bundle_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    analysis_json_path = destination / "skew_analysis.json"
    ranking_tsv_path = destination / "skew_ranking.tsv"

    analysis_json_path.write_text(
        json.dumps(
            {
                "bundle_json_path": bundle_path.as_posix(),
                "sort_by": sort_by,
                "ascending": ascending,
                "level_order": list(level_order),
                "rows": [term_skew_row_as_dict(row) for row in rows],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tsv(
        ranking_tsv_path,
        [_term_row_for_tsv(row, level_order=level_order) for row in rows],
        fieldnames=(
            "term_id",
            "term",
            *(f"f_{level}" for level in level_order),
            *(f"p_{level}" for level in level_order),
            "peak_level",
            "peak_share",
            "peak_gap",
            "cefr_center",
            "directional_skew",
            *(f"matches_{level}" for level in level_order),
            *(f"different_texts_{level}" for level in level_order),
            "total_frequency_per_million",
            "term_tags",
        ),
    )

    return CeliWordlistAnalysisReport(
        bundle_json_path=bundle_path.as_posix(),
        analysis_json_path=str(analysis_json_path.resolve()),
        ranking_tsv_path=str(ranking_tsv_path.resolve()),
        term_count=len(rows),
        sort_by=sort_by,
        ascending=ascending,
    )


def _prepare_output_dir(output_dir: str | Path | None) -> Path:
    destination = Path(output_dir) if output_dir is not None else default_harvest_output_dir()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _snapshot_name(term: str, level: str | None, suffix: str) -> str:
    return f"{_slugify(term)}-{(level or 'all').lower()}-{suffix}.yml"


def _export_filename(term: str, level: str) -> str:
    return f"{_slugify(term)}-{level.lower()}-export"


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "query"


def _parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def _write_tsv(path: Path, rows: list[dict[str, object]], *, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _extract_bundle_level_order(
    manifest_payload: dict[str, object],
    query_summaries: list[dict[str, object]],
) -> tuple[str, ...]:
    manifest_levels = manifest_payload.get("default_levels")
    if isinstance(manifest_levels, list) and manifest_levels:
        levels = tuple(str(level).strip().upper() for level in manifest_levels if str(level).strip())
        if levels:
            return levels
    seen_levels = sorted(
        {
            str(item.get("level")).strip().upper()
            for item in query_summaries
            if str(item.get("level")).strip()
        },
        key=lambda level: CEFR_LEVEL_INDEX.get(level, 999),
    )
    if not seen_levels:
        raise ValueError("bundle.json query_summaries does not contain any CEFR levels")
    return tuple(seen_levels)


def _analyze_term_rows(
    term_items: list[dict[str, object]],
    *,
    level_order: tuple[str, ...],
) -> CeliTermSkewRow:
    base_item = term_items[0]
    level_frequencies = {level: 0.0 for level in level_order}
    matches_by_level = {level: 0 for level in level_order}
    different_texts_by_level = {level: 0 for level in level_order}
    for item in term_items:
        level = str(item.get("level") or "").strip().upper()
        if level in level_frequencies:
            level_frequencies[level] = float(item.get("frequency_per_million") or 0.0)
            matches_by_level[level] = int(item.get("matches") or 0)
            different_texts_by_level[level] = int(item.get("different_texts") or 0)

    total = sum(level_frequencies.values())
    level_shares = {
        level: (level_frequencies[level] / total if total else 0.0)
        for level in level_order
    }
    ranked_levels = sorted(
        level_order,
        key=lambda level: (level_shares[level], level_frequencies[level], -CEFR_LEVEL_INDEX.get(level, 999)),
        reverse=True,
    )
    peak_level = ranked_levels[0]
    peak_share = level_shares[peak_level]
    second_share = level_shares[ranked_levels[1]] if len(ranked_levels) > 1 else 0.0
    local_rank = {level: index + 1 for index, level in enumerate(level_order)}
    cefr_center = (
        sum(local_rank[level] * level_shares[level] for level in level_order)
        if total
        else 0.0
    )
    midpoint = (len(level_order) + 1) / 2 if level_order else 0.0
    directional_skew = (cefr_center - midpoint) * (peak_share - second_share) if total else 0.0

    return CeliTermSkewRow(
        term_id=str(base_item["term_id"]),
        term=str(base_item["term"]),
        peak_level=peak_level,
        peak_share=round(peak_share, 4),
        peak_gap=round(peak_share - second_share, 4),
        cefr_center=round(cefr_center, 4),
        directional_skew=round(directional_skew, 4),
        total_frequency_per_million=round(total, 4),
        level_frequencies={level: round(level_frequencies[level], 4) for level in level_order},
        level_shares={level: round(level_shares[level], 4) for level in level_order},
        matches_by_level=matches_by_level,
        different_texts_by_level=different_texts_by_level,
        term_tags=tuple(str(tag) for tag in base_item.get("term_tags") or ()),
    )


def _sorted_skew_rows(rows: list[CeliTermSkewRow], *, sort_by: str, ascending: bool) -> list[CeliTermSkewRow]:
    if sort_by == "cefr_center":
        key = lambda row: (row.cefr_center, row.peak_gap, row.term)
    elif sort_by == "peak_gap":
        key = lambda row: (row.peak_gap, row.directional_skew, row.term)
    else:
        key = lambda row: (row.directional_skew, row.peak_gap, row.term)
    return sorted(rows, key=key, reverse=not ascending)


def _term_row_for_tsv(row: CeliTermSkewRow, *, level_order: tuple[str, ...]) -> dict[str, object]:
    payload: dict[str, object] = {
        "term_id": row.term_id,
        "term": row.term,
        "peak_level": row.peak_level,
        "peak_share": row.peak_share,
        "peak_gap": row.peak_gap,
        "cefr_center": row.cefr_center,
        "directional_skew": row.directional_skew,
        "total_frequency_per_million": row.total_frequency_per_million,
        "term_tags": ",".join(row.term_tags),
    }
    for level in level_order:
        payload[f"f_{level}"] = row.level_frequencies.get(level, 0.0)
        payload[f"p_{level}"] = row.level_shares.get(level, 0.0)
        payload[f"matches_{level}"] = row.matches_by_level.get(level, 0)
        payload[f"different_texts_{level}"] = row.different_texts_by_level.get(level, 0)
    return payload

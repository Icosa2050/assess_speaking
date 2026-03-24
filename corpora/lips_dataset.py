"""LIPS transcript parsing plus phase-1 manifest build and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Iterator


MANIFEST_VERSION = "lips_v1_monologue_only"
VALID_RAW_MODES = {"D", "M", "DM", "MD"}
VALID_PARSE_STATUSES = {"full", "partial_metadata", "text_only", "failed"}
VALID_TURN_STRUCTURE_FLAGS = {"monologue_like", "dialogue_like", "mixed", "unknown"}
VALID_MAPPING_CONFIDENCE = {"high", "medium", "low"}
VALID_TASK_FAMILIES = (
    "travel_narrative",
    "personal_experience",
    "opinion_monologue",
    "picture_description",
    "free_monologue",
)

_SECTION_START_RE = re.compile(r"^\s*SE\s*(?P<section_num>\d+)\b(?P<rest>.*)$", re.IGNORECASE)
_TURN_RE = re.compile(r"^\s*(?P<speaker>[A-Za-z]\d*)\s*:\s*(?P<body>.*)$", re.IGNORECASE)
_PROVA_RE = re.compile(r"\(\s*prova[^)]*\)", re.IGNORECASE)
_MODE_AT_END_RE = re.compile(r"\b(?P<mode>DM|MD|D|M)\s*$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_FILENAME_CEFR_RE = re.compile(r"(?P<cefr>A1|A2|B1|B2|C1|C2)\.txt$", re.IGNORECASE)
_HEADER_KEY_VALUE_RE = re.compile(r"^\s*(?P<key>[^:]+?)\s*:\s*(?P<value>.+?)\s*$")
_HEADER_LEVEL_RE = re.compile(r"^\s*Livello\s+(?P<value>[0-9A-Za-z]+)\s*$", re.IGNORECASE)
_STRUCTURAL_ARTIFACT_RE = re.compile(r"\[\s*[^\]]*\s*\]|<\??[^>]*>")
_MOJIBAKE_RE = re.compile(r"(?:Ã.|Â.|â.)")


@dataclass(frozen=True)
class LipsTurn:
    speaker: str
    text: str
    line_number: int


@dataclass(frozen=True)
class LipsSourceMetadata:
    exam_date: str | None
    site: str | None
    candidate_id: str | None
    raw_level: str | None
    exam_code: str | None
    cassette_number: str | None
    transcriber: str | None


@dataclass(frozen=True)
class LipsSectionRecord:
    manifest_version: str
    source_corpus: str
    source_file: str
    section_id: str
    raw_mode: str | None
    turn_structure_flag: str
    parse_status: str
    exclusion_reason: str | None
    cefr_level: str | None
    prompt_topic: str | None
    candidate_text_raw: str
    candidate_text_clean: str
    examiner_context: str
    section_token_count: int
    candidate_token_count: int
    candidate_turn_count: int
    examiner_turn_count: int
    mapped_task_family: str | None
    mapping_source: str | None
    mapping_confidence: str | None
    needs_review: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class LipsFileParseFailure:
    source_file: str
    parse_status: str
    exclusion_reason: str
    error_message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipsFileParseResult:
    source_file: str
    source_metadata: LipsSourceMetadata
    sections: tuple[LipsSectionRecord, ...]
    failures: tuple[LipsFileParseFailure, ...]


@dataclass(frozen=True)
class LipsBuildConfig:
    input_root: Path
    output_dir: Path
    review_sample_size: int = 20
    min_candidate_tokens: int = 20
    large_section_review_tokens: int = 500
    seed: int = 17


@dataclass(frozen=True)
class LipsBuildReport:
    manifest_version: str
    input_root: str
    output_dir: str
    included_path: str
    excluded_path: str
    build_report_path: str
    review_sample_path: str
    total_files: int
    parsed_files: int
    file_failures: int
    total_sections: int
    included_sections: int
    excluded_sections: int
    parse_success_ratio: float
    counts_by_cefr: dict[str, int]
    counts_by_family: dict[str, int]
    counts_by_raw_mode: dict[str, int]
    counts_by_turn_structure_flag: dict[str, int]
    exclusion_reason_counts: dict[str, int]
    parse_status_counts: dict[str, int]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipsValidationConfig:
    min_usable_sections: int = 200
    min_task_families: int = 3
    target_parse_success_ratio: float = 0.95
    min_manual_agreement: float = 0.85
    require_manual_review: bool = True


@dataclass(frozen=True)
class LipsValidationReport:
    manifest_version: str
    included_path: str
    excluded_path: str
    review_path: str | None
    output_path: str
    hard_gate_passed: bool
    parse_target_passed: bool
    overall_passed: bool
    usable_section_count: int
    task_family_coverage: int
    parse_success_ratio: float
    manual_reviewed_count: int
    manual_agreement_ratio: float | None
    counts_by_cefr: dict[str, int]
    counts_by_family: dict[str, int]
    exclusion_reason_counts: dict[str, int]
    parse_status_counts: dict[str, int]
    gate_results: dict[str, bool]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipsReviewSampleEntry:
    source_file: str
    section_id: str
    cefr_level: str | None
    prompt_topic: str | None
    candidate_text_clean: str
    mapped_task_family: str | None
    mapping_confidence: str | None
    reviewer_accepts_mapping: bool | None
    reviewer_task_family: str | None
    reviewer_notes: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipsReviewAnnotation:
    source_file: str
    section_id: str
    reviewer_accepts_mapping: bool
    reviewer_task_family: str | None
    reviewer_notes: str | None


@dataclass(frozen=True)
class LipsExcludedAuditEntry:
    source_file: str
    section_id: str
    cefr_level: str | None
    raw_mode: str | None
    turn_structure_flag: str
    prompt_topic: str | None
    candidate_text_clean: str
    proposed_exclusion_reason: str
    reviewer_accepts_exclusion: bool | None
    reviewer_suggested_task_family: str | None
    reviewer_notes: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LipsExcludedAuditAnnotation:
    source_file: str
    section_id: str
    reviewer_accepts_exclusion: bool
    reviewer_suggested_task_family: str | None
    reviewer_notes: str | None


@dataclass(frozen=True)
class LipsReviewSummaryReport:
    included_review_path: str | None
    excluded_review_path: str | None
    output_path: str
    included_reviewed_count: int
    included_agreement_ratio: float | None
    included_disagreements_by_family: dict[str, int]
    included_suggested_families: dict[str, int]
    excluded_reviewed_count: int
    excluded_agreement_ratio: float | None
    excluded_disagreements_by_reason: dict[str, int]
    excluded_suggested_families: dict[str, int]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_lips_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tmp" / "lips_manifest"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path_obj = Path(path)
    if not path_obj.exists():
        return rows
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path_obj


def parse_lips_file(path: str | Path) -> LipsFileParseResult:
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="iso-8859-1")
    except UnicodeDecodeError as exc:
        return LipsFileParseResult(
            source_file=source_path.name,
            source_metadata=LipsSourceMetadata(None, None, None, None, None, None, None),
            sections=(),
            failures=(
                LipsFileParseFailure(
                    source_file=source_path.name,
                    parse_status="failed",
                    exclusion_reason="decode_error",
                    error_message=str(exc),
                ),
            ),
        )

    lines = text.splitlines()
    metadata, section_headers = _scan_metadata_and_section_headers(lines)
    if not section_headers:
        return LipsFileParseResult(
            source_file=source_path.name,
            source_metadata=metadata,
            sections=(),
            failures=(
                LipsFileParseFailure(
                    source_file=source_path.name,
                    parse_status="failed",
                    exclusion_reason="missing_section_marker",
                    error_message="No section markers found",
                ),
            ),
        )

    sections: list[LipsSectionRecord] = []
    for index, header in enumerate(section_headers):
        start_line = header["line_index"]
        end_line = (
            section_headers[index + 1]["line_index"]
            if index + 1 < len(section_headers)
            else len(lines)
        )
        section_lines = lines[start_line:end_line]
        sections.append(_parse_section(source_path.name, metadata, header, section_lines))

    return LipsFileParseResult(
        source_file=source_path.name,
        source_metadata=metadata,
        sections=tuple(sections),
        failures=(),
    )


def build_lips_manifest(config: LipsBuildConfig) -> LipsBuildReport:
    input_root = config.input_root.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    included_path = output_dir / "lips_sections_included.jsonl"
    excluded_path = output_dir / "lips_sections_excluded.jsonl"
    build_report_path = output_dir / "lips_build_report.json"
    review_sample_path = output_dir / "lips_review_sample.jsonl"
    file_failures_path = output_dir / "lips_file_failures.jsonl"

    included: list[LipsSectionRecord] = []
    excluded: list[LipsSectionRecord] = []
    file_failures: list[LipsFileParseFailure] = []

    input_files = tuple(sorted(input_root.glob("*.txt")))
    for path in input_files:
        parsed = parse_lips_file(path)
        file_failures.extend(parsed.failures)
        for section in parsed.sections:
            processed = _apply_phase_one_rules(
                section,
                min_candidate_tokens=config.min_candidate_tokens,
                large_section_review_tokens=config.large_section_review_tokens,
            )
            if processed.exclusion_reason is None:
                included.append(processed)
            else:
                excluded.append(processed)

    write_jsonl(included_path, (item.to_dict() for item in included))
    write_jsonl(excluded_path, (item.to_dict() for item in excluded))
    write_jsonl(file_failures_path, (item.to_dict() for item in file_failures))

    review_sample = build_review_sample(included, sample_size=config.review_sample_size, seed=config.seed)
    write_jsonl(review_sample_path, (item.to_dict() for item in review_sample))

    counts_by_cefr = _count_values(item.cefr_level for item in included if item.cefr_level)
    counts_by_family = _count_values(item.mapped_task_family for item in included if item.mapped_task_family)
    counts_by_raw_mode = _count_values(item.raw_mode or "UNKNOWN" for item in (*included, *excluded))
    counts_by_turn_structure = _count_values(item.turn_structure_flag for item in (*included, *excluded))
    exclusion_reason_counts = _count_values(item.exclusion_reason for item in excluded if item.exclusion_reason)
    parse_status_counts = _count_values(item.parse_status for item in (*included, *excluded))
    successful_sections = sum(1 for item in (*included, *excluded) if item.parse_status in {"full", "partial_metadata"})
    total_sections = len(included) + len(excluded)
    parse_success_ratio = successful_sections / total_sections if total_sections else 0.0

    report = LipsBuildReport(
        manifest_version=MANIFEST_VERSION,
        input_root=input_root.as_posix(),
        output_dir=output_dir.as_posix(),
        included_path=included_path.as_posix(),
        excluded_path=excluded_path.as_posix(),
        build_report_path=build_report_path.as_posix(),
        review_sample_path=review_sample_path.as_posix(),
        total_files=len(input_files),
        parsed_files=len(input_files) - len(file_failures),
        file_failures=len(file_failures),
        total_sections=total_sections,
        included_sections=len(included),
        excluded_sections=len(excluded),
        parse_success_ratio=parse_success_ratio,
        counts_by_cefr=counts_by_cefr,
        counts_by_family=counts_by_family,
        counts_by_raw_mode=counts_by_raw_mode,
        counts_by_turn_structure_flag=counts_by_turn_structure,
        exclusion_reason_counts=exclusion_reason_counts,
        parse_status_counts=parse_status_counts,
        generated_at=_utc_now(),
    )
    build_report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_review_sample(
    included_records: Iterable[LipsSectionRecord],
    *,
    sample_size: int,
    seed: int = 17,
    prioritize_needs_review: bool = True,
) -> list[LipsReviewSampleEntry]:
    candidates = list(included_records)
    if not candidates or sample_size <= 0:
        return []

    if prioritize_needs_review:
        prioritized = [item for item in candidates if item.needs_review]
        fallback = [item for item in candidates if not item.needs_review]
        ordered_candidates = prioritized + fallback
    else:
        ordered_candidates = candidates

    buckets: dict[tuple[str, str], list[LipsSectionRecord]] = {}
    for record in ordered_candidates:
        bucket_key = (
            record.cefr_level or "UNKNOWN",
            record.mapped_task_family or "UNMAPPED",
        )
        buckets.setdefault(bucket_key, []).append(record)

    bucket_keys = sorted(buckets)
    randomizer = random.Random(seed)
    sample: list[LipsSectionRecord] = []
    while bucket_keys and len(sample) < sample_size:
        next_keys: list[tuple[str, str]] = []
        for key in bucket_keys:
            records = buckets[key]
            if not records:
                continue
            if len(records) > 1:
                randomizer.shuffle(records)
            sample.append(records.pop(0))
            if len(sample) >= sample_size:
                break
            if records:
                next_keys.append(key)
        bucket_keys = next_keys
        if not next_keys:
            break

    return [
        LipsReviewSampleEntry(
            source_file=item.source_file,
            section_id=item.section_id,
            cefr_level=item.cefr_level,
            prompt_topic=item.prompt_topic,
            candidate_text_clean=item.candidate_text_clean,
            mapped_task_family=item.mapped_task_family,
            mapping_confidence=item.mapping_confidence,
            reviewer_accepts_mapping=None,
            reviewer_task_family=None,
            reviewer_notes=None,
        )
        for item in sample
    ]


def build_excluded_audit_sample(
    excluded_records: Iterable[LipsSectionRecord],
    *,
    sample_size: int,
    seed: int = 17,
) -> list[LipsExcludedAuditEntry]:
    candidates = list(excluded_records)
    if not candidates or sample_size <= 0:
        return []

    buckets: dict[tuple[str, str], list[LipsSectionRecord]] = {}
    for record in candidates:
        bucket_key = (
            record.exclusion_reason or "UNKNOWN",
            record.cefr_level or "UNKNOWN",
        )
        buckets.setdefault(bucket_key, []).append(record)

    bucket_keys = sorted(buckets)
    randomizer = random.Random(seed)
    sample: list[LipsSectionRecord] = []
    while bucket_keys and len(sample) < sample_size:
        next_keys: list[tuple[str, str]] = []
        for key in bucket_keys:
            records = buckets[key]
            if not records:
                continue
            if len(records) > 1:
                randomizer.shuffle(records)
            sample.append(records.pop(0))
            if len(sample) >= sample_size:
                break
            if records:
                next_keys.append(key)
        bucket_keys = next_keys
        if not next_keys:
            break

    return [
        LipsExcludedAuditEntry(
            source_file=item.source_file,
            section_id=item.section_id,
            cefr_level=item.cefr_level,
            raw_mode=item.raw_mode,
            turn_structure_flag=item.turn_structure_flag,
            prompt_topic=item.prompt_topic,
            candidate_text_clean=item.candidate_text_clean,
            proposed_exclusion_reason=item.exclusion_reason or "UNKNOWN",
            reviewer_accepts_exclusion=None,
            reviewer_suggested_task_family=None,
            reviewer_notes=None,
        )
        for item in sample
    ]


def load_review_annotations(path: str | Path) -> tuple[LipsReviewAnnotation, ...]:
    annotations: list[LipsReviewAnnotation] = []
    for row in read_jsonl(path):
        accepts = row.get("reviewer_accepts_mapping")
        if accepts is None:
            continue
        annotations.append(
            LipsReviewAnnotation(
                source_file=str(row["source_file"]),
                section_id=str(row["section_id"]),
                reviewer_accepts_mapping=bool(accepts),
                reviewer_task_family=(
                    str(row["reviewer_task_family"]) if row.get("reviewer_task_family") is not None else None
                ),
                reviewer_notes=(
                    str(row["reviewer_notes"]) if row.get("reviewer_notes") is not None else None
                ),
            )
        )
    return tuple(annotations)


def load_excluded_audit_annotations(path: str | Path) -> tuple[LipsExcludedAuditAnnotation, ...]:
    annotations: list[LipsExcludedAuditAnnotation] = []
    for row in read_jsonl(path):
        accepts = row.get("reviewer_accepts_exclusion")
        if accepts is None:
            continue
        annotations.append(
            LipsExcludedAuditAnnotation(
                source_file=str(row["source_file"]),
                section_id=str(row["section_id"]),
                reviewer_accepts_exclusion=bool(accepts),
                reviewer_suggested_task_family=(
                    str(row["reviewer_suggested_task_family"])
                    if row.get("reviewer_suggested_task_family") is not None
                    else None
                ),
                reviewer_notes=(
                    str(row["reviewer_notes"]) if row.get("reviewer_notes") is not None else None
                ),
            )
        )
    return tuple(annotations)


def summarize_lips_review(
    *,
    included_review_path: str | Path | None = None,
    excluded_review_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> LipsReviewSummaryReport:
    included_rows = read_jsonl(included_review_path) if included_review_path else []
    excluded_rows = read_jsonl(excluded_review_path) if excluded_review_path else []

    included_annotations = load_review_annotations(included_review_path) if included_review_path else ()
    excluded_annotations = load_excluded_audit_annotations(excluded_review_path) if excluded_review_path else ()

    included_agreement_ratio: float | None = None
    included_disagreements_by_family: dict[str, int] = {}
    included_suggested_families: dict[str, int] = {}
    if included_annotations:
        included_agreement_ratio = sum(1 for item in included_annotations if item.reviewer_accepts_mapping) / len(
            included_annotations
        )
        lookup = {
            (str(row["source_file"]), str(row["section_id"])): row
            for row in included_rows
        }
        for annotation in included_annotations:
            if annotation.reviewer_accepts_mapping:
                continue
            original = lookup.get((annotation.source_file, annotation.section_id), {})
            family = str(original.get("mapped_task_family") or "UNMAPPED")
            included_disagreements_by_family[family] = included_disagreements_by_family.get(family, 0) + 1
            if annotation.reviewer_task_family:
                included_suggested_families[annotation.reviewer_task_family] = (
                    included_suggested_families.get(annotation.reviewer_task_family, 0) + 1
                )

    excluded_agreement_ratio: float | None = None
    excluded_disagreements_by_reason: dict[str, int] = {}
    excluded_suggested_families: dict[str, int] = {}
    if excluded_annotations:
        excluded_agreement_ratio = sum(1 for item in excluded_annotations if item.reviewer_accepts_exclusion) / len(
            excluded_annotations
        )
        lookup = {
            (str(row["source_file"]), str(row["section_id"])): row
            for row in excluded_rows
        }
        for annotation in excluded_annotations:
            if annotation.reviewer_accepts_exclusion:
                continue
            original = lookup.get((annotation.source_file, annotation.section_id), {})
            reason = str(original.get("proposed_exclusion_reason") or "UNKNOWN")
            excluded_disagreements_by_reason[reason] = excluded_disagreements_by_reason.get(reason, 0) + 1
            if annotation.reviewer_suggested_task_family:
                excluded_suggested_families[annotation.reviewer_suggested_task_family] = (
                    excluded_suggested_families.get(annotation.reviewer_suggested_task_family, 0) + 1
                )

    output_path_obj = (
        Path(output_path)
        if output_path is not None
        else default_lips_output_dir() / "lips_review_summary.json"
    )
    report = LipsReviewSummaryReport(
        included_review_path=str(Path(included_review_path).resolve()) if included_review_path else None,
        excluded_review_path=str(Path(excluded_review_path).resolve()) if excluded_review_path else None,
        output_path=str(output_path_obj.resolve()),
        included_reviewed_count=len(included_annotations),
        included_agreement_ratio=included_agreement_ratio,
        included_disagreements_by_family=dict(sorted(included_disagreements_by_family.items())),
        included_suggested_families=dict(sorted(included_suggested_families.items())),
        excluded_reviewed_count=len(excluded_annotations),
        excluded_agreement_ratio=excluded_agreement_ratio,
        excluded_disagreements_by_reason=dict(sorted(excluded_disagreements_by_reason.items())),
        excluded_suggested_families=dict(sorted(excluded_suggested_families.items())),
        generated_at=_utc_now(),
    )
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_lips_manifest(
    included_path: str | Path,
    excluded_path: str | Path,
    *,
    review_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config: LipsValidationConfig | None = None,
) -> LipsValidationReport:
    config = config or LipsValidationConfig()
    included_rows = read_jsonl(included_path)
    excluded_rows = read_jsonl(excluded_path)
    all_rows = [*included_rows, *excluded_rows]

    usable_section_count = len(included_rows)
    counts_by_family = _count_values(
        str(row["mapped_task_family"]) for row in included_rows if row.get("mapped_task_family")
    )
    task_family_coverage = len(counts_by_family)
    parse_status_counts = _count_values(
        str(row.get("parse_status") or "failed")
        for row in all_rows
    )
    successful_sections = parse_status_counts.get("full", 0) + parse_status_counts.get("partial_metadata", 0)
    parse_success_ratio = successful_sections / len(all_rows) if all_rows else 0.0
    counts_by_cefr = _count_values(str(row["cefr_level"]) for row in included_rows if row.get("cefr_level"))
    exclusion_reason_counts = _count_values(
        str(row["exclusion_reason"]) for row in excluded_rows if row.get("exclusion_reason")
    )

    manual_reviewed_count = 0
    manual_agreement_ratio: float | None = None
    if review_path:
        review_annotations = load_review_annotations(review_path)
        manual_reviewed_count = len(review_annotations)
        if manual_reviewed_count:
            agreed = sum(1 for item in review_annotations if item.reviewer_accepts_mapping)
            manual_agreement_ratio = agreed / manual_reviewed_count

    gate_results = {
        "min_usable_sections": usable_section_count >= config.min_usable_sections,
        "min_task_families": task_family_coverage >= config.min_task_families,
        "min_manual_agreement": (
            (not config.require_manual_review)
            or (manual_agreement_ratio is not None and manual_agreement_ratio >= config.min_manual_agreement)
        ),
        "target_parse_success_ratio": parse_success_ratio >= config.target_parse_success_ratio,
    }
    hard_gate_passed = (
        gate_results["min_usable_sections"]
        and gate_results["min_task_families"]
        and gate_results["min_manual_agreement"]
    )
    parse_target_passed = gate_results["target_parse_success_ratio"]
    overall_passed = hard_gate_passed and parse_target_passed

    output_path = Path(output_path) if output_path else Path(included_path).with_name("lips_validation_report.json")
    report = LipsValidationReport(
        manifest_version=MANIFEST_VERSION,
        included_path=str(Path(included_path).resolve()),
        excluded_path=str(Path(excluded_path).resolve()),
        review_path=str(Path(review_path).resolve()) if review_path else None,
        output_path=str(output_path.resolve()),
        hard_gate_passed=hard_gate_passed,
        parse_target_passed=parse_target_passed,
        overall_passed=overall_passed,
        usable_section_count=usable_section_count,
        task_family_coverage=task_family_coverage,
        parse_success_ratio=parse_success_ratio,
        manual_reviewed_count=manual_reviewed_count,
        manual_agreement_ratio=manual_agreement_ratio,
        counts_by_cefr=counts_by_cefr,
        counts_by_family=counts_by_family,
        exclusion_reason_counts=exclusion_reason_counts,
        parse_status_counts=parse_status_counts,
        gate_results=gate_results,
        generated_at=_utc_now(),
    )
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _apply_phase_one_rules(
    record: LipsSectionRecord,
    *,
    min_candidate_tokens: int,
    large_section_review_tokens: int,
) -> LipsSectionRecord:
    rescued_monologue = _has_light_examiner_scaffolding(record)
    exclusion_reason = record.exclusion_reason
    if record.raw_mode in {"D", "DM", "MD"}:
        exclusion_reason = "raw_mode_dialogue"
    elif record.exclusion_reason is not None:
        exclusion_reason = record.exclusion_reason
    elif record.turn_structure_flag != "monologue_like" and not rescued_monologue:
        exclusion_reason = "turn_structure_not_monologue_like"
    elif record.candidate_token_count == 0:
        exclusion_reason = "missing_candidate_text"
    elif record.candidate_token_count < min_candidate_tokens:
        exclusion_reason = "insufficient_candidate_text"

    mapped_task_family = record.mapped_task_family
    mapping_source = record.mapping_source
    mapping_confidence = record.mapping_confidence
    if exclusion_reason is None:
        mapped_task_family, mapping_source, mapping_confidence = _map_task_family(
            record.prompt_topic,
            candidate_text=record.candidate_text_clean,
        )
        needs_review = record.needs_review
        if mapped_task_family is None or mapping_confidence == "low":
            needs_review = True
        if record.candidate_token_count > large_section_review_tokens:
            needs_review = True
        if record.parse_status != "full":
            needs_review = True
        if rescued_monologue:
            needs_review = True
    else:
        mapped_task_family = None
        mapping_source = None
        mapping_confidence = None
        needs_review = False

    return LipsSectionRecord(
        **{
            **record.to_dict(),
            "exclusion_reason": exclusion_reason,
            "mapped_task_family": mapped_task_family,
            "mapping_source": mapping_source,
            "mapping_confidence": mapping_confidence,
            "needs_review": needs_review,
        }
    )


def _has_light_examiner_scaffolding(record: LipsSectionRecord) -> bool:
    if record.turn_structure_flag == "monologue_like":
        return False
    if record.candidate_turn_count == 0:
        return False
    if record.examiner_turn_count == 0:
        return False
    if record.candidate_token_count == 0:
        return False

    examiner_tokens = _token_count(record.examiner_context)
    if examiner_tokens == 0:
        return False
    examiner_ratio = examiner_tokens / record.candidate_token_count
    prompt_topic = (record.prompt_topic or "").casefold()
    picture_keywords = ("foto", "immagine", "figura", "descrivi", "descrizione")

    if (
        prompt_topic == "foto: descrizione piazza"
        and record.candidate_turn_count <= 4
        and record.examiner_turn_count <= 4
        and record.candidate_token_count >= 140
        and examiner_tokens <= 30
        and examiner_ratio <= 0.25
    ):
        return True

    if record.raw_mode != "M":
        return False

    if (
        record.candidate_turn_count <= 3
        and record.examiner_turn_count <= 4
        and examiner_tokens <= 50
        and examiner_ratio <= 0.2
    ):
        return True

    if (
        record.candidate_token_count >= 40
        and record.candidate_turn_count <= 5
        and record.examiner_turn_count <= 6
        and examiner_tokens <= 50
        and examiner_ratio <= 0.2
    ):
        return True

    if (
        record.candidate_turn_count == 1
        and record.examiner_turn_count <= 2
        and record.candidate_token_count >= 120
        and examiner_tokens <= 100
        and examiner_ratio <= 0.3
    ):
        return True

    if (
        record.candidate_turn_count == 1
        and record.examiner_turn_count <= 4
        and record.candidate_token_count >= 300
        and examiner_tokens <= 80
        and examiner_ratio <= 0.2
    ):
        return True

    if (
        1 <= record.candidate_turn_count <= 2
        and record.examiner_turn_count <= 2
        and record.candidate_token_count >= 300
        and examiner_tokens <= 100
        and examiner_ratio <= 0.3
    ):
        return True

    if (
        5 <= record.candidate_turn_count <= 8
        and record.examiner_turn_count <= 8
        and record.candidate_token_count >= 250
        and examiner_tokens <= 70
        and examiner_ratio <= 0.17
    ):
        return True

    if (
        prompt_topic == "vacanze di natale"
        and record.raw_mode == "M"
        and record.candidate_turn_count <= 8
        and record.examiner_turn_count <= 8
        and record.candidate_token_count >= 200
        and examiner_tokens <= 60
        and examiner_ratio <= 0.3
    ):
        return True

    if (
        any(keyword in prompt_topic for keyword in picture_keywords)
        and record.candidate_turn_count == 1
        and record.examiner_turn_count <= 2
        and record.candidate_token_count >= 60
        and examiner_tokens <= 50
        and examiner_ratio <= 0.6
    ):
        return True

    return False


def _parse_section(
    source_file: str,
    metadata: LipsSourceMetadata,
    header: dict[str, Any],
    section_lines: list[str],
) -> LipsSectionRecord:
    raw_mode = header.get("raw_mode")
    prompt_topic = header.get("prompt_topic")
    turns = _extract_turns(section_lines[1:], section_start_line=header["line_index"] + 2)

    candidate_turns = tuple(turn for turn in turns if turn.speaker.startswith("C"))
    examiner_turns = tuple(turn for turn in turns if turn.speaker.startswith("E"))
    candidate_text_raw = "\n".join(turn.text for turn in candidate_turns).strip()
    examiner_context = "\n".join(turn.text for turn in examiner_turns).strip()
    candidate_text_clean = _clean_candidate_text(candidate_text_raw)
    section_token_count = _token_count(_WHITESPACE_RE.sub(" ", " ".join(turn.text for turn in turns)).strip())
    candidate_token_count = _token_count(candidate_text_clean)

    parse_status = "full"
    exclusion_reason = None
    if header.get("placeholder"):
        parse_status = "text_only"
        exclusion_reason = "placeholder_section"
    elif not candidate_turns:
        parse_status = "text_only"
        exclusion_reason = "missing_candidate_text"
    elif raw_mode is None or prompt_topic is None:
        parse_status = "partial_metadata"
    elif header.get("encoding_anomaly"):
        parse_status = "partial_metadata"

    turn_structure_flag = _classify_turn_structure(raw_mode, turns)
    needs_review = parse_status != "full" or bool(header.get("encoding_anomaly"))

    return LipsSectionRecord(
        manifest_version=MANIFEST_VERSION,
        source_corpus="lips",
        source_file=source_file,
        section_id=header["section_id"],
        raw_mode=raw_mode,
        turn_structure_flag=turn_structure_flag,
        parse_status=parse_status,
        exclusion_reason=exclusion_reason,
        cefr_level=_infer_cefr_level(source_file, metadata.raw_level),
        prompt_topic=prompt_topic,
        candidate_text_raw=candidate_text_raw,
        candidate_text_clean=candidate_text_clean,
        examiner_context=examiner_context,
        section_token_count=section_token_count,
        candidate_token_count=candidate_token_count,
        candidate_turn_count=len(candidate_turns),
        examiner_turn_count=len(examiner_turns),
        mapped_task_family=None,
        mapping_source=None,
        mapping_confidence=None,
        needs_review=needs_review,
    )


def _scan_metadata_and_section_headers(lines: list[str]) -> tuple[LipsSourceMetadata, list[dict[str, Any]]]:
    header_values: dict[str, str] = {}
    section_headers: list[dict[str, Any]] = []

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        section_match = _SECTION_START_RE.match(line)
        if section_match:
            section_headers.append(_parse_section_header(line, line_index=index))
            continue
        if section_headers:
            continue
        if match := _HEADER_KEY_VALUE_RE.match(line):
            key = _normalize_header_key(match.group("key"))
            header_values[key] = match.group("value").strip()
            continue
        if match := _HEADER_LEVEL_RE.match(line):
            header_values["livello"] = match.group("value").strip()

    metadata = LipsSourceMetadata(
        exam_date=header_values.get("data esame"),
        site=header_values.get("sede"),
        candidate_id=header_values.get("numero di matricola"),
        raw_level=header_values.get("livello"),
        exam_code=header_values.get("codice esami"),
        cassette_number=header_values.get("numero cassetta"),
        transcriber=header_values.get("trascrittore"),
    )
    return metadata, section_headers


def _parse_section_header(line: str, *, line_index: int) -> dict[str, Any]:
    section_match = _SECTION_START_RE.match(line)
    assert section_match is not None
    section_id = f"SE{section_match.group('section_num')}"
    rest = section_match.group("rest").strip()

    raw_mode: str | None = None
    mode_match = _MODE_AT_END_RE.search(rest)
    if mode_match:
        raw_mode = mode_match.group("mode").upper()
        rest = rest[: mode_match.start()].strip()

    prompt_topic = rest.strip(" :-\t")
    prompt_topic = _PROVA_RE.sub("", prompt_topic)
    prompt_topic = re.sub(r"^\s*argomento\s*:\s*", "", prompt_topic, flags=re.IGNORECASE).strip()
    prompt_topic = _WHITESPACE_RE.sub(" ", prompt_topic).strip() or None
    lowered_topic = (prompt_topic or "").casefold()
    placeholder = (
        "[non esiste]" in lowered_topic
        or "[non è nella cassetta]" in lowered_topic
        or " = monologo " in f" {lowered_topic} "
        or lowered_topic.startswith("=")
    )

    return {
        "line_index": line_index,
        "section_id": section_id,
        "raw_mode": raw_mode if raw_mode in VALID_RAW_MODES else raw_mode,
        "prompt_topic": prompt_topic,
        "placeholder": placeholder,
        "encoding_anomaly": bool(_MOJIBAKE_RE.search(line)),
    }


def _extract_turns(lines: list[str], *, section_start_line: int) -> tuple[LipsTurn, ...]:
    turns: list[LipsTurn] = []
    current_speaker: str | None = None
    current_line_number = section_start_line
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_line_number, current_parts
        if current_speaker is None:
            return
        text = _WHITESPACE_RE.sub(" ", " ".join(part.strip() for part in current_parts if part.strip())).strip()
        if text:
            turns.append(LipsTurn(speaker=current_speaker, text=text, line_number=current_line_number))
        current_speaker = None
        current_parts = []

    for offset, raw_line in enumerate(lines):
        line_number = section_start_line + offset
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            continue
        if _SECTION_START_RE.match(line):
            flush()
            break
        if match := _TURN_RE.match(line):
            flush()
            current_speaker = match.group("speaker").upper()
            current_line_number = line_number
            current_parts = [match.group("body").strip()]
            continue
        if current_speaker is not None:
            current_parts.append(line.strip())
    flush()
    return tuple(turns)


def _infer_cefr_level(source_file: str, raw_level: str | None) -> str | None:
    if match := _FILENAME_CEFR_RE.search(source_file):
        return match.group("cefr").upper()
    if raw_level:
        normalized = raw_level.strip().upper()
        if normalized in {"A1", "A2", "B1", "B2", "C1", "C2"}:
            return normalized
        numeric_map = {"1": "B1", "2": "B2", "3": "C1", "4": "C2"}
        return numeric_map.get(normalized)
    return None


def _classify_turn_structure(raw_mode: str | None, turns: tuple[LipsTurn, ...]) -> str:
    if not turns:
        return "unknown"
    speakers = [turn.speaker for turn in turns]
    alternations = sum(1 for index in range(1, len(speakers)) if speakers[index] != speakers[index - 1])
    examiner_turns = sum(1 for speaker in speakers if speaker.startswith("E"))
    candidate_turns = sum(1 for speaker in speakers if speaker.startswith("C"))
    unknown_turns = sum(1 for speaker in speakers if not speaker.startswith(("E", "C")))

    if unknown_turns:
        return "unknown"
    if examiner_turns <= 1 and candidate_turns >= 1 and alternations <= 1:
        return "monologue_like"
    if examiner_turns >= 2 and candidate_turns >= 1 and alternations >= 2:
        return "dialogue_like"
    if raw_mode in {"DM", "MD"}:
        return "mixed"
    return "mixed"


def _clean_candidate_text(text: str) -> str:
    cleaned = _STRUCTURAL_ARTIFACT_RE.sub(" ", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _map_task_family(
    prompt_topic: str | None,
    *,
    candidate_text: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    topic = (prompt_topic or "").casefold()
    candidate = (candidate_text or "").casefold()
    if not topic and not candidate:
        return None, None, None

    personal_exact_topics = (
        "la persona più importante nella propria vita",
        "letture preferite",
    )
    opinion_exact_topics = (
        "italia",
        "reality shows",
        "turismo ecologico",
    )
    personal_topic_phrases = (
        "differenze tra la lingua del candidato e quella italiana",
        "amici",
        "aspetto fisico",
        "città o campagna",
        "cucina tipica",
        "il luogo che ami di più",
        "lingua italiana",
        "motivo per il quale è in italia",
        "paese di origine",
        "paese d'origine",
        "una persona cara",
        "feste tradizionali",
        "tradizioni popolari del paese",
        "abitudini alimentari del paese nativo",
        "università brasiliana",
        "vacanze di natale",
        "proprio padre",
        "soggiorno in italia",
        "la tua città e il tuo paese",
        "esami della vita",
        "insegnante",
        "insegnante di italiano",
        "italia e la lingua italiana",
        "amici conosciuti in italia",
        "il modo in cui il candidato ha imparato la lingua italiana",
        "giornata tipica",
        "vita ideale",
        "intervista a personaggio famoso",
        "rapporti con gli italiani",
        "descrivere se stessi",
        "descrizione personale",
    )
    opinion_topic_phrases = (
        "auto condivisa",
        "aspetti dell'italia",
        "artista e pubblico",
        "argomento le lingue e le certificazioni",
        "beni culturali del proprio paese",
        "conoscenza lingue",
        "il caldo in italia",
        "la cultura italiana",
        "la musica e lo sport",
        "le lingue",
        "le letture preferite",
        "buoni motivi per viaggiare in italia",
        "creatività nella scuola",
        "cura del corpo",
        "11 settembre",
        "fallaci",
        "gli italiani",
        "moodi per rilassarsi",
        "modi per rilassarsi",
        "passaggio millennio",
        "paranormale",
        "rapporto tra italiani e stranieri",
        "donne single",
        "possibilità di svolgere un lavoro part time",
        "personaggio pubblico",
        "integrazione interculturale nelle scuole",
    )
    picture_keywords = (
        "foto",
        "immagine",
        "figura",
        "descrivi",
        "descrizione",
    )
    picture_candidate_keywords = (
        "questa immagine",
        "in questa immagine",
        "dell'immagine",
        "della foto",
        "questa foto",
    )
    abstract_image_phrases = (
        "immagine che abbiamo",
        "immagine che hanno",
        "tipica immagine",
    )
    travel_keywords = (
        "viaggio",
        "vacanza",
        "vacanze",
        "turismo",
        "itinerario",
        "spostamento",
    )
    personal_keywords = (
        "esperienza",
        "ricordo",
        "famiglia",
        "infanzia",
        "hobby",
        "musica",
        "sport",
        "amicizia",
        "amico",
        "amica",
        "tempo libero",
        "migliore amico",
        "migliore amica",
        "progetti futuri",
        "propria città",
        "mia città",
        "libro",
        "lettura",
        "studio",
        "studi",
        "lavoro",
        "carattere",
    )
    opinion_topic_keywords = (
        "opinione",
        "vantaggi",
        "svantaggi",
        "inquinamento",
        "pubblicità",
        "televisione",
        "tv",
        "programmi televisivi",
        "computer",
        "tecnologie",
        "tecnologia",
        "nuove tecnologie",
        "genitori",
        "figli",
        "giovani",
        "comportamenti giovanili",
        "calo delle nascite",
        "calo demografico",
        "problemi mondiali",
        "educazione",
        "costo della vita",
        "tolleranza",
        "clima",
        "medicina naturale",
        "realtà degli anziani",
    )
    opinion_candidate_keywords = (
        "penso",
        "credo",
        "secondo me",
        "sono d'accordo",
        "non sono d'accordo",
        "a mio parere",
        "ritengo",
        "mi sembra giusto",
        "necessità",
        "problema",
    )

    if topic in personal_exact_topics:
        return "personal_experience", "heuristic_v2", "medium"
    if topic in opinion_exact_topics:
        return "opinion_monologue", "heuristic_v2", "medium"
    if "giorni di festa" in topic or "giorni festivi" in topic:
        if "secondo me" in candidate:
            return "opinion_monologue", "heuristic_v2", "medium"
        if any(phrase in candidate for phrase in ("per me", "mi piace", "trascorro", "quando sono libera")):
            return "personal_experience", "heuristic_v2", "medium"
    if any(phrase in topic for phrase in personal_topic_phrases):
        return "personal_experience", "heuristic_v2", "medium"
    if any(phrase in topic for phrase in opinion_topic_phrases):
        return "opinion_monologue", "heuristic_v2", "medium"
    if any(keyword in topic for keyword in picture_keywords):
        return "picture_description", "heuristic_v1", "high"
    if any(keyword in candidate for keyword in picture_candidate_keywords) and not any(
        phrase in candidate for phrase in abstract_image_phrases
    ):
        return "picture_description", "heuristic_v2", "medium"
    if not topic and "vorrei fare un corso" in candidate and "vorrei lavorare" in candidate:
        return "personal_experience", "heuristic_v2", "medium"
    if any(keyword in topic for keyword in opinion_topic_keywords):
        return "opinion_monologue", "heuristic_v1", "high"
    if any(keyword in topic for keyword in travel_keywords):
        return "travel_narrative", "heuristic_v1", "medium"
    if any(keyword in topic for keyword in personal_keywords):
        return "personal_experience", "heuristic_v1", "medium"
    if any(keyword in candidate for keyword in opinion_candidate_keywords):
        return "opinion_monologue", "heuristic_v1", "medium"
    if topic or candidate:
        return "free_monologue", "heuristic_v1", "low"
    return None, None, None


def _normalize_header_key(key: str) -> str:
    normalized = key.strip().lower()
    if normalized == "data":
        return "data esame"
    if normalized in {"ivello", "livello"}:
        return "livello"
    return normalized


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _token_count(text: str) -> int:
    if not text:
        return 0
    return len([token for token in _WHITESPACE_RE.split(text.strip()) if token])


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

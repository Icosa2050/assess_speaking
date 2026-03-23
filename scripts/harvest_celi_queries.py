#!/usr/bin/env python3
"""Harvest approved CELI query artifacts via the dedicated Playwright CELI session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corpora.celi_harvest import (
    analyze_wordlist_bundle,
    default_harvest_output_dir,
    download_result_as_dict,
    frequency_breakdown_as_dict,
    harvest_export,
    harvest_frequency_breakdown,
    harvest_query_matrix,
    harvest_wordlist_manifest,
    query_summary_as_dict,
    wordlist_analysis_report_as_dict,
    wordlist_bundle_report_as_dict,
)
from corpora.celi_wordlists import load_celi_wordlist_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Harvest concordance summaries for one term.")
    _add_term_argument(query_parser)
    _add_levels_argument(query_parser)
    _add_hits_argument(query_parser)
    _add_output_dir_argument(query_parser)

    frequency_parser = subparsers.add_parser(
        "frequency",
        help="Harvest the CQPweb frequency-breakdown page for one term.",
    )
    _add_term_argument(frequency_parser)
    frequency_parser.add_argument("--level", help="Optional CEFR level restriction.")
    _add_hits_argument(frequency_parser)
    _add_output_dir_argument(frequency_parser)

    export_parser = subparsers.add_parser("export", help="Download one concordance export with metadata.")
    _add_term_argument(export_parser)
    export_parser.add_argument("--level", required=True, help="Required CEFR level restriction for exports.")
    _add_hits_argument(export_parser)
    _add_output_dir_argument(export_parser)
    export_parser.add_argument(
        "--metadata",
        default="CEFR level,Task assignment ID,Nationality,Text genre,Text type",
        help="Comma-separated list of metadata fields to include.",
    )
    export_parser.add_argument("--filename", help="Optional export filename stem shown on the CELI download page.")

    matrix_parser = subparsers.add_parser("matrix", help="Harvest a term/level query matrix.")
    matrix_parser.add_argument("--terms", required=True, help="Comma-separated term list.")
    _add_levels_argument(matrix_parser)
    _add_hits_argument(matrix_parser)
    _add_output_dir_argument(matrix_parser)
    matrix_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file for the harvested matrix payload.",
    )

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Harvest a curated CELI wordlist manifest into a stable bundle.",
    )
    manifest_parser.add_argument("--manifest", type=Path, required=True, help="Path to the CELI wordlist manifest.")
    _add_hits_argument(manifest_parser)
    _add_output_dir_argument(manifest_parser)
    manifest_parser.add_argument(
        "--skip-frequency",
        action="store_true",
        help="Skip whole-corpus frequency-breakdown pages and write only the query matrix bundle.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Rank terms in a harvested CELI bundle by CEFR skew.",
    )
    analyze_parser.add_argument("--bundle", type=Path, required=True, help="Path to a harvested CELI bundle.json file.")
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for analysis outputs. Defaults to the bundle directory.",
    )
    analyze_parser.add_argument(
        "--sort-by",
        choices=("directional_skew", "cefr_center", "peak_gap"),
        default="directional_skew",
        help="Metric used to sort the ranking output.",
    )
    analyze_parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "query":
        results = harvest_query_matrix(
            [args.term],
            levels=_split_csv(args.levels),
            hits_per_page=args.hits,
            output_dir=args.output_dir,
        )
        print(json.dumps([query_summary_as_dict(result) for result in results], ensure_ascii=False, indent=2))
        return 0

    if args.command == "frequency":
        result = harvest_frequency_breakdown(
            args.term,
            level=args.level,
            hits_per_page=args.hits,
            output_dir=args.output_dir,
        )
        print(json.dumps(frequency_breakdown_as_dict(result), ensure_ascii=False, indent=2))
        return 0

    if args.command == "export":
        result = harvest_export(
            args.term,
            level=args.level,
            metadata_fields=_split_csv(args.metadata),
            hits_per_page=args.hits,
            output_dir=args.output_dir,
            filename=args.filename,
        )
        print(json.dumps(download_result_as_dict(result), ensure_ascii=False, indent=2))
        return 0

    if args.command == "manifest":
        manifest = load_celi_wordlist_manifest(args.manifest)
        result = harvest_wordlist_manifest(
            manifest,
            hits_per_page=args.hits,
            output_dir=args.output_dir,
            include_frequency=not args.skip_frequency,
        )
        print(json.dumps(wordlist_bundle_report_as_dict(result), ensure_ascii=False, indent=2))
        return 0

    if args.command == "analyze":
        result = analyze_wordlist_bundle(
            args.bundle,
            output_dir=args.output_dir,
            sort_by=args.sort_by,
            ascending=args.ascending,
        )
        print(json.dumps(wordlist_analysis_report_as_dict(result), ensure_ascii=False, indent=2))
        return 0

    results = harvest_query_matrix(
        _split_csv(args.terms),
        levels=_split_csv(args.levels),
        hits_per_page=args.hits,
        output_dir=args.output_dir,
    )
    payload = [query_summary_as_dict(result) for result in results]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _add_term_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--term", required=True, help="Search term to submit to CELI.")


def _add_levels_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--levels",
        default="B1,B2,C1,C2",
        help="Comma-separated CEFR levels to query.",
    )


def _add_hits_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hits", type=int, default=10, help="Hits per page for concordance runs.")


def _add_output_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_harvest_output_dir(),
        help="Directory for CELI snapshots and downloads.",
    )


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

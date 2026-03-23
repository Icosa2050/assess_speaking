#!/usr/bin/env python3
"""List and fetch known open learner corpora, with first-class support for RITA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corpora.open_corpus_catalog import (
    download_open_corpus,
    list_open_corpus_sources,
    open_corpus_catalog_as_dicts,
    resolve_open_corpus_source,
)
from corpora.rita_dataset import load_rita_archive, rita_summary_as_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Print the built-in open corpus catalog.")
    list_parser.add_argument("--downloadable-only", action="store_true")

    fetch_parser = subparsers.add_parser("fetch", help="Download one known open corpus.")
    fetch_parser.add_argument("source_id")
    fetch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "corpus_ingest",
        help="Directory for downloaded archives.",
    )
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.add_argument(
        "--inspect",
        action="store_true",
        help="Emit an archive summary after download when supported.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list":
        sources = list_open_corpus_sources()
        if args.downloadable_only:
            sources = tuple(source for source in sources if source.download_url and source.archive_name)
        print(json.dumps(open_corpus_catalog_as_dicts(sources), ensure_ascii=False, indent=2))
        return 0

    source = resolve_open_corpus_source(args.source_id)
    archive_path = download_open_corpus(source, args.output_dir, overwrite=args.overwrite)
    payload: dict[str, object] = {
        "source": open_corpus_catalog_as_dicts([source])[0],
        "archive_path": archive_path.as_posix(),
    }
    if args.inspect and source.source_id == "rita_phrame4":
        payload["rita_summary"] = rita_summary_as_dict(load_rita_archive(archive_path))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

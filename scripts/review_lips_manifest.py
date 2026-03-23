#!/usr/bin/env python3
"""Prepare low-fi LIPS review artifacts and summarize completed reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corpora.lips_dataset import (
    LipsSectionRecord,
    build_excluded_audit_sample,
    build_review_sample,
    read_jsonl,
    summarize_lips_review,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create static included/excluded review artifacts.")
    prepare.add_argument("artifact_dir", type=Path)
    prepare.add_argument("--included-sample-size", type=int, default=20)
    prepare.add_argument("--excluded-sample-size", type=int, default=20)
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--included-output", type=Path)
    prepare.add_argument("--excluded-output", type=Path)

    summarize = subparsers.add_parser("summarize", help="Summarize completed review artifacts.")
    summarize.add_argument("--included-review", type=Path)
    summarize.add_argument("--excluded-review", type=Path)
    summarize.add_argument("--output", type=Path)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        artifact_dir = args.artifact_dir
        included_path = artifact_dir / "lips_sections_included.jsonl"
        excluded_path = artifact_dir / "lips_sections_excluded.jsonl"
        included_records = [LipsSectionRecord(**row) for row in read_jsonl(included_path)]
        excluded_records = [LipsSectionRecord(**row) for row in read_jsonl(excluded_path)]

        included_output = args.included_output or artifact_dir / "lips_review_sample.jsonl"
        excluded_output = args.excluded_output or artifact_dir / "lips_excluded_audit_sample.jsonl"
        included_sample = build_review_sample(
            included_records,
            sample_size=args.included_sample_size,
            seed=args.seed,
            prioritize_needs_review=True,
        )
        excluded_sample = build_excluded_audit_sample(
            excluded_records,
            sample_size=args.excluded_sample_size,
            seed=args.seed,
        )
        write_jsonl(included_output, (item.to_dict() for item in included_sample))
        write_jsonl(excluded_output, (item.to_dict() for item in excluded_sample))
        print(
            json.dumps(
                {
                    "included_output": str(included_output.resolve()),
                    "excluded_output": str(excluded_output.resolve()),
                    "included_sample_count": len(included_sample),
                    "excluded_sample_count": len(excluded_sample),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    report = summarize_lips_review(
        included_review_path=args.included_review,
        excluded_review_path=args.excluded_review,
        output_path=args.output,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

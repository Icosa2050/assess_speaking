#!/usr/bin/env python3
"""Validate generated phase-1 LIPS manifest artifacts against QC gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corpora.lips_dataset import LipsValidationConfig, validate_lips_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path, help="Directory containing built LIPS JSONL artifacts.")
    parser.add_argument("--review-file", type=Path, help="Optional completed review JSONL file.")
    parser.add_argument("--output", type=Path, help="Destination JSON path for the validation report.")
    parser.add_argument("--min-usable-sections", type=int, default=200)
    parser.add_argument("--min-task-families", type=int, default=3)
    parser.add_argument("--target-parse-success-ratio", type=float, default=0.95)
    parser.add_argument("--min-manual-agreement", type=float, default=0.85)
    parser.add_argument(
        "--allow-missing-manual-review",
        action="store_true",
        help="Allow validation to pass without completed review annotations.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    included_path = args.artifact_dir / "lips_sections_included.jsonl"
    excluded_path = args.artifact_dir / "lips_sections_excluded.jsonl"
    report = validate_lips_manifest(
        included_path,
        excluded_path,
        review_path=args.review_file,
        output_path=args.output,
        config=LipsValidationConfig(
            min_usable_sections=args.min_usable_sections,
            min_task_families=args.min_task_families,
            target_parse_success_ratio=args.target_parse_success_ratio,
            min_manual_agreement=args.min_manual_agreement,
            require_manual_review=not args.allow_missing_manual_review,
        ),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

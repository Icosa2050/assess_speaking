#!/usr/bin/env python3
"""Build the phase-1 LIPS included/excluded manifest artifacts and review sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_repo() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    _bootstrap_repo()
    from corpora.lips_dataset import default_lips_output_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Directory containing LIPS .txt transcript files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_lips_output_dir(),
        help="Destination directory for generated artifacts.",
    )
    parser.add_argument("--review-sample-size", type=int, default=20)
    parser.add_argument("--min-candidate-tokens", type=int, default=20)
    parser.add_argument("--large-section-review-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> int:
    _bootstrap_repo()
    from corpora.lips_dataset import LipsBuildConfig, build_lips_manifest

    args = build_parser().parse_args()
    report = build_lips_manifest(
        LipsBuildConfig(
            input_root=args.input_root,
            output_dir=args.output_dir,
            review_sample_size=args.review_sample_size,
            min_candidate_tokens=args.min_candidate_tokens,
            large_section_review_tokens=args.large_section_review_tokens,
            seed=args.seed,
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

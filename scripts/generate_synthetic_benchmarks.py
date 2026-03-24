#!/usr/bin/env python3
"""Render synthetic benchmark audio from a seed manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.synthetic_benchmark_generation import render_seed_manifest
from benchmarking.synthetic_seed_manifests import load_seed_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render synthetic benchmark audio from a seed manifest.")
    parser.add_argument("manifest", type=Path, help="Path to a seed manifest JSON file.")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/synthetic_benchmarks"))
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help="Optional directory of benchmark suite JSON files used to validate seed/render alignment.",
    )
    parser.add_argument("--seed-id", action="append", default=[], help="Render only the given seed_id.")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive seeds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rendered audio.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_seed_manifest(args.manifest)
    result = render_seed_manifest(
        manifest,
        args.output_dir,
        selected_seed_ids=args.seed_id,
        include_inactive=args.include_inactive,
        overwrite=args.overwrite,
        benchmark_root=args.benchmark_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

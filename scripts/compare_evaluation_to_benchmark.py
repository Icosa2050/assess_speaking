#!/usr/bin/env python3
"""Compare an evaluation manifest against a benchmark suite and write a drift report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.synthetic_benchmark_regression import (
    compare_evaluation_against_benchmark,
    load_benchmark_and_evaluation,
    write_regression_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_suite", help="Path to the benchmark suite JSON file.")
    parser.add_argument("evaluation_manifest", help="Path to the evaluation manifest JSON file.")
    parser.add_argument(
        "--output",
        help="Destination path for the regression report JSON file. Defaults next to the evaluation manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_suite, evaluation_suite = load_benchmark_and_evaluation(
        args.benchmark_suite,
        args.evaluation_manifest,
    )
    result = compare_evaluation_against_benchmark(benchmark_suite, evaluation_suite)
    output_path = (
        Path(args.output)
        if args.output
        else Path(args.evaluation_manifest).with_name("benchmark_regression_report.json")
    )
    written = write_regression_report(result, output_path)
    print(
        json.dumps(
            {
                "benchmark_suite_id": result.benchmark_suite_id,
                "evaluation_suite_id": result.evaluation_suite_id,
                "output_path": str(written),
                "passed_cases": result.passed_cases,
                "failed_cases": result.failed_cases,
                "skipped_cases": result.skipped_cases,
                "missing_benchmark_refs": result.missing_benchmark_refs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

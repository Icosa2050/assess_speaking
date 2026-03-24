#!/usr/bin/env python3
"""Evaluate a real-audio calibration manifest and write a comparable JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.calibration_evaluation import (
    CalibrationRunConfig,
    evaluate_calibration_manifest,
    write_calibration_evaluation_manifest,
)
from benchmarking.calibration_manifests import load_calibration_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to a calibration manifest JSON file.")
    parser.add_argument("--output", type=Path, help="Destination JSON path. Defaults next to the manifest.")
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--feedback-language", default=None)
    parser.add_argument("--llm-timeout-sec", type=float, default=None)
    parser.add_argument("--language-profile-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-raw-llm", action="store_true")
    parser.add_argument("--include-full-report", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_calibration_manifest(args.manifest)
    evaluation = evaluate_calibration_manifest(
        manifest,
        config=CalibrationRunConfig(
            whisper_model=args.whisper_model,
            provider=args.provider,
            llm_model=args.llm_model,
            feedback_language=args.feedback_language,
            dry_run=args.dry_run,
            include_raw_llm=args.include_raw_llm,
            include_full_report=args.include_full_report,
            llm_timeout_sec=args.llm_timeout_sec,
            language_profile_key=args.language_profile_key,
        ),
    )
    output_path = args.output or args.manifest.with_name("calibration_evaluation_report.json")
    written = write_calibration_evaluation_manifest(evaluation, output_path)
    print(
        json.dumps(
            {
                "evaluation_id": evaluation.evaluation_id,
                "manifest_id": evaluation.manifest_id,
                "output_path": str(written),
                "run_status": evaluation.run_status,
                "success_ratio": evaluation.success_ratio,
                "pair_expectations_total": len(evaluation.pair_expectations),
                "pair_expectations_passed": sum(1 for pair in evaluation.pair_expectations if pair.passed is True),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

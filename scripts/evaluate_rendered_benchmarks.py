#!/usr/bin/env python3
"""Evaluate rendered synthetic audio bundles through the assessment pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.synthetic_audio_contracts import build_rendered_audio_contract_suite
from benchmarking.synthetic_benchmark_evaluation import (
    EvaluationRunConfig,
    evaluate_rendered_audio_contract_suite,
    write_evaluation_manifest,
)
from benchmarking.synthetic_seed_manifests import load_seed_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_manifest", help="Path to the synthetic seed manifest JSON file.")
    parser.add_argument("render_manifest", help="Path to the render_manifest.json bundle file.")
    parser.add_argument(
        "--output",
        help="Destination path for the evaluation manifest JSON file. Defaults next to the render manifest.",
    )
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--feedback-language", default=None)
    parser.add_argument("--language-profile-key", default=None)
    parser.add_argument("--target-duration-sec", type=float, default=120.0)
    parser.add_argument("--speaker-id", default="synthetic-benchmark")
    parser.add_argument("--llm-timeout-sec", type=float, default=None)
    parser.add_argument(
        "--max-consecutive-runner-errors",
        type=int,
        default=None,
        help="Abort the remaining suite after this many consecutive runner errors.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-raw-llm", action="store_true")
    parser.add_argument("--omit-full-report", action="store_true")
    parser.add_argument(
        "--checkpoint-jsonl",
        help="Optional JSONL checkpoint path written after each evaluated case.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --checkpoint-jsonl if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_manifest = load_seed_manifest(args.seed_manifest)
    contract_suite = build_rendered_audio_contract_suite(seed_manifest, args.render_manifest)
    config = EvaluationRunConfig(
        whisper_model=args.whisper_model,
        provider=args.provider,
        llm_model=args.llm_model,
        feedback_language=args.feedback_language,
        target_duration_sec=args.target_duration_sec,
        speaker_id=args.speaker_id,
        dry_run=bool(args.dry_run),
        include_raw_llm=bool(args.include_raw_llm),
        include_full_report=not bool(args.omit_full_report),
        llm_timeout_sec=args.llm_timeout_sec,
        max_consecutive_runner_errors=args.max_consecutive_runner_errors,
        language_profile_key=args.language_profile_key,
    )
    evaluated = evaluate_rendered_audio_contract_suite(
        contract_suite,
        config=config,
        checkpoint_path=args.checkpoint_jsonl,
        resume_from_checkpoint=bool(args.resume),
    )
    output_path = Path(args.output) if args.output else Path(args.render_manifest).with_name("evaluation_manifest.json")
    written = write_evaluation_manifest(evaluated, output_path)
    print(
        json.dumps(
            {
                "suite_id": evaluated.suite_id,
                "output_path": str(written),
                "cases": len(evaluated.cases),
                "ok_cases": sum(1 for case in evaluated.cases if case.status == "ok"),
                "runner_error_cases": sum(1 for case in evaluated.cases if case.status == "runner_error"),
                "skipped_cases": sum(1 for case in evaluated.cases if case.status == "skipped"),
                "run_status": evaluated.run_status,
                "success_ratio": evaluated.success_ratio,
                "checkpoint_jsonl": args.checkpoint_jsonl,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

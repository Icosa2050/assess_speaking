#!/usr/bin/env python3
"""Run the existing Playwright live journeys; retain isolated evidence without an evaluator API."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit
import wave

import requests

ROOT = Path(__file__).resolve().parents[1]


def discover(provider: str) -> dict:
    defaults = {
        "ollama": ("http://127.0.0.1:11434/v1", "qwen3.5:4b"),
        "lmstudio": ("http://127.0.0.1:1234/v1", "qwen2.5-3b-instruct"),
        "openrouter": ("https://openrouter.ai/api/v1", "mistralai/mistral-small-3.2-24b-instruct"),
    }
    endpoint, default_model = defaults[provider]
    endpoint = os.environ.get(f"{provider.upper()}_E2E_BASE_URL", endpoint).rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or (
        provider == "openrouter" and endpoint != "https://openrouter.ai/api/v1"
    ):
        return {"provider": provider, "available": False, "reason": "Unsafe provider endpoint override"}
    model = os.environ.get(f"{provider.upper()}_E2E_MODEL", default_model)
    result = {"provider": provider, "endpoint": endpoint, "model": model, "available": False}
    try:
        if provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                return {**result, "reason": "OPENROUTER_API_KEY is absent"}
            response = requests.get(endpoint + "/auth/key", headers={"Authorization": f"Bearer {key}"}, timeout=20)
            if response.status_code != 200:
                return {**result, "reason": f"OpenRouter credential check HTTP {response.status_code}"}
        response = requests.get(endpoint + "/models", timeout=20)
        response.raise_for_status()
        models = [item["id"] for item in response.json()["data"]]
        result["available_models"] = models if provider != "openrouter" else [model] if model in models else []
        result["available"] = model in models
        if not result["available"]:
            result["reason"] = "Requested assessment model is not available"
        alternate = os.environ.get(f"{provider.upper()}_E2E_ALTERNATE_MODEL", "")
        if not alternate and provider == "lmstudio" and "qwen2.5-7b-instruct" in models:
            alternate = "qwen2.5-7b-instruct"
        if alternate and alternate not in models:
            result.update(available=False, reason="Requested alternate assessment model is not available")
        result["alternate_model"] = alternate or model
    except (requests.RequestException, ValueError, KeyError) as exc:
        # No response bodies, headers, or credentials go into evidence.
        result["reason"] = f"Provider discovery failed: {type(exc).__name__}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", default="ollama,lmstudio,openrouter")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--whisper", default="large-v3")
    parser.add_argument("--alternate-whisper", default="tiny")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--allow-guarded-fallback", action="store_true",
        help="Accept intentional schema/coaching-validation fallback for workflow checks only; transport failures still fail",
    )
    parser.add_argument("--grep", help="Optional Playwright test filter (recorded in the manifest)")
    args = parser.parse_args()
    providers = args.providers.split(",")
    if not providers or len(set(providers)) != len(providers) or any(p not in ("ollama", "lmstudio", "openrouter") for p in providers):
        parser.error("providers must be a unique selection of ollama, lmstudio and openrouter")
    providers = [p for p in ("ollama", "lmstudio", "openrouter") if p in providers]
    if args.repeat < 1:
        parser.error("repeat must be positive")
    output = args.output or ROOT / "frontend/output/live-journeys" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    availability = [discover(provider) for provider in providers]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(), "providers": availability,
        "whisper": args.whisper, "alternate_whisper": args.alternate_whisper,
        "repeat": args.repeat, "grep": args.grep, "python": sys.version,
        "allow_guarded_fallback": args.allow_guarded_fallback,
        "acceptance_scope": (
            "workflow_only_with_guarded_validation_fallback"
            if args.allow_guarded_fallback else "strict_model_output_and_workflow"
        ),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": {name: version(name) for name in ("faster-whisper", "ctranslate2", "av")},
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT / "frontend/package-lock.json", ROOT / "frontend/playwright.ollama.config.ts",
            ROOT / "frontend/tests/live/ollamaBilingual.spec.ts", ROOT / "frontend/tests/live/workflowEvidence.ts",
            ROOT / "scripts/run_live_journeys.py", ROOT / "scripts/inspect_journey_evidence.py", ROOT / "scripts/journey_keyring.py",
            ROOT / "assess_speaking.py", ROOT / "assess_core/language_profiles.py", ROOT / "assess_core/schemas.py",
            ROOT / "assessment_runtime/metrics.py", ROOT / "assessment_runtime/asr.py",
            ROOT / "assessment_runtime/output_validation.py", ROOT / "assessment_runtime/style_validation.py", ROOT / "assessment_runtime/feedback_claims.py", ROOT / "assessment_runtime/transcript_quality.py",
            ROOT / "assessment_runtime/comparison.py", ROOT / "assessment_runtime/runner.py",
            ROOT / "assessment_runtime/assessment_prompts.py", ROOT / "assessment_runtime/llm_client.py",
            ROOT / "app_backend/app.py", ROOT / "frontend/src/components/review/ReviewSummary.tsx",
            ROOT / "frontend/src/components/review/WarningsPanel.tsx",
            ROOT / "benchmarking/benchmark_suites.py", ROOT / "benchmarking/synthetic_benchmark_generation.py",
            ROOT / "tests/fixtures/feedback_quality/bilingual_v1.json",
            *(ROOT / "locales" / f"{locale}.json" for locale in ("en", "it", "de", "fr", "es")),
        )},
        "quality_oracle": "Human review of retained recordings, reports, and screenshots; no API evaluator",
        "samples": [], "runs": [],
    }
    for sample in sorted((ROOT / "samples/cefr").rglob("*.wav")):
        with wave.open(str(sample)) as audio:
            manifest["samples"].append({"path": str(sample.relative_to(ROOT)),
                "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                "duration_sec": audio.getnframes() / audio.getframerate(),
                "sample_rate": audio.getframerate(), "channels": audio.getnchannels()})
    manifest_path = output / "manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2))
    save_manifest()
    print(json.dumps({"evidence": str(output), "providers": availability}, indent=2), flush=True)
    if args.probe_only:
        return 0
    unavailable = [s for s in availability if not s["available"]]
    if unavailable:
        manifest["runs"] = [{"provider": s["provider"], "status": "unavailable", "reason": s.get("reason")} for s in unavailable]
        save_manifest()
        print("Every requested provider must be available; select installed providers with --providers. See manifest.json for prerequisites.", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT))
    from assessment_runtime.asr import describe_model_availability
    for size in {args.whisper, args.alternate_whisper}:
        if not describe_model_availability(size)["cached"]:
            print(f"Whisper {size} must be downloaded before the offline-ASR journey", file=sys.stderr)
            return 1
    node = shutil.which("node")
    if not node:
        print("Node is required for the existing Playwright runner", file=sys.stderr)
        return 1
    for repetition in range(1, args.repeat + 1):
        for state in availability:
            run_root = output / f'{repetition:02d}-{state["provider"]}'
            run_root.mkdir()
            environment = os.environ.copy()
            environment.pop("OPENAI_API_KEY", None)
            if state["provider"] != "openrouter":
                environment.pop("OPENROUTER_API_KEY", None)
            environment.update(NODE_ENV="development", LOCAL_E2E_PROVIDER=state["provider"],
                LOCAL_E2E_BASE_URL=state["endpoint"], LOCAL_E2E_MODEL=state["model"],
                LOCAL_E2E_ALTERNATE_MODEL=state["alternate_model"], OLLAMA_E2E_WHISPER=args.whisper,
                LIVE_E2E_ALTERNATE_WHISPER=args.alternate_whisper,
                LIVE_E2E_ALLOW_GUARDED_FALLBACK="1" if args.allow_guarded_fallback else "0",
                VOSTAVO_OLLAMA_TEST_ROOT=str(run_root), VOSTAVO_TEST_PYTHON=sys.executable, HF_HUB_OFFLINE="1")
            command = [node, str(ROOT / "frontend/node_modules/playwright/cli.js"), "test", "--config", "frontend/playwright.ollama.config.ts"]
            if args.grep:
                command += ["--grep", f"setup discovers|{args.grep}"]
            run = {"provider": state["provider"], "repeat": repetition, "directory": str(run_root),
                "allow_guarded_fallback": args.allow_guarded_fallback,
                "acceptance_scope": manifest["acceptance_scope"],
                "command": command, "started_at_utc": datetime.now(UTC).isoformat(), "status": "running"}
            manifest["runs"].append(run)
            save_manifest()
            print(f'Running {state["provider"]}, repetition {repetition}: {run_root}', flush=True)
            with (run_root / "runner.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
            from scripts.inspect_journey_evidence import inspect
            try:
                evidence = inspect(run_root)
                (run_root / "evidence-summary.json").write_text(json.dumps(evidence, indent=2))
                assessments = evidence.get("assessments", [])
                evidence_valid = bool(assessments) and all(
                    row.get("status") == "completed" and row.get("session_id")
                    and row.get("transcript") and row.get("coaching") and row.get("recordings")
                    for row in assessments
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                evidence_valid = False
                run["evidence_error"] = type(exc).__name__
            # A successful process with all tests skipped is not live coverage.
            try:
                stats = json.loads((run_root / "results.json").read_text())["stats"]
                skipped = int(stats.get("skipped", 0))
                unstable = int(stats.get("unexpected", 0)) + int(stats.get("flaky", 0))
                executed = int(stats.get("expected", 0)) + unstable
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                executed = 0
                skipped = unstable = 0
            passed = result.returncode == 0 and executed > 0 and skipped == 0 and unstable == 0 and evidence_valid
            run.update(status="passed" if passed else "failed", exit_code=result.returncode,
                       executed_tests=executed, assessment_evidence_valid=bool(evidence_valid),
                       skipped_tests=skipped, failed_or_flaky_tests=unstable,
                       finished_at_utc=datetime.now(UTC).isoformat())
            if executed == 0:
                run["reason"] = "No executed tests in the Playwright JSON report"
            elif not evidence_valid:
                run["reason"] = "No complete assessment/recording evidence; setup alone is not a live journey"
            elif skipped or unstable:
                run["reason"] = "Selected live tests skipped, failed or required a retry"
            save_manifest()
            print(f'{state["provider"]}: {run["status"]}', flush=True)
    return int(any(run["status"] == "failed" for run in manifest["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())

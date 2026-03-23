"""Deterministic synthetic audio generation from seed manifests."""

from __future__ import annotations

from benchmarking.benchmark_suites import BenchmarkCase, load_benchmark_suite
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from assessment_runtime.metrics import metrics_from as transcript_metrics_from
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from typing import Iterable

from benchmarking.synthetic_seed_manifests import (
    RenderDefaults,
    SeedManifest,
    SyntheticSeed,
    seed_manifest_fingerprint,
    synthetic_seed_fingerprint,
)

RENDERER_VERSION = "macos_say_ffmpeg_v1"


@dataclass(frozen=True)
class ResolvedRenderConfig:
    provider: str
    voice: str
    rate_wpm: int
    output_format: str
    sample_rate_hz: int
    channels: int


@dataclass(frozen=True)
class RenderDurationEstimate:
    speech_word_count: int
    pause_count: int
    pause_total_sec: float
    estimated_speech_duration_sec: float
    estimated_total_duration_sec: float


SILENCE_MARKER_RE = re.compile(r"\[\[slnc\s+(\d+)\]\]")
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)

BENCHMARK_ALIGNMENT_TOLERANCES = {
    "duration_sec": {"minimum": 8.0, "relative": 0.12},
    "pause_total_sec": {"minimum": 1.0, "relative": 0.3},
    "word_count": {"minimum": 20.0, "relative": 0.15},
    "wpm": {"minimum": 12.0, "relative": 0.12},
    "fillers": {"minimum": 1.0, "relative": 0.25},
    "cohesion_markers": {"minimum": 1.0, "relative": 0.25},
    "complexity_index": {"minimum": 1.0, "relative": 0.25},
}


def _pick_value(seed_override: object, default_value: object, fallback: object) -> object:
    if seed_override is not None:
        return seed_override
    if default_value is not None:
        return default_value
    return fallback


def resolve_render_config(manifest: SeedManifest, seed: SyntheticSeed) -> ResolvedRenderConfig:
    overrides = seed.render_overrides or RenderDefaults(
        provider=None,
        voice=None,
        rate_wpm=None,
        output_format=None,
        sample_rate_hz=None,
        channels=None,
        notes=None,
    )
    provider = str(_pick_value(overrides.provider, manifest.render_defaults.provider, "macos_say"))
    voice = _pick_value(overrides.voice, manifest.render_defaults.voice, None)
    rate_wpm = _pick_value(overrides.rate_wpm, manifest.render_defaults.rate_wpm, None)
    output_format = str(_pick_value(overrides.output_format, manifest.render_defaults.output_format, "wav")).lower()
    sample_rate_hz = int(_pick_value(overrides.sample_rate_hz, manifest.render_defaults.sample_rate_hz, 16000))
    channels = int(_pick_value(overrides.channels, manifest.render_defaults.channels, 1))

    if provider != "macos_say":
        raise ValueError(f"Unsupported render provider: {provider}")
    if not voice:
        raise ValueError(f"Seed {seed.seed_id} does not resolve to a deterministic voice")
    if rate_wpm is None:
        raise ValueError(f"Seed {seed.seed_id} does not resolve to a deterministic rate_wpm")
    if output_format != "wav":
        raise ValueError("Only wav output is supported in the current renderer")

    return ResolvedRenderConfig(
        provider=provider,
        voice=str(voice),
        rate_wpm=int(rate_wpm),
        output_format=output_format,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )


def text_to_render(seed: SyntheticSeed) -> str:
    return seed.render_text or seed.transcript


def estimate_render_duration(text: str, rate_wpm: int) -> RenderDurationEstimate:
    if rate_wpm <= 0:
        raise ValueError("rate_wpm must be greater than 0")
    pause_values_ms = [int(value) for value in SILENCE_MARKER_RE.findall(text)]
    pause_total_sec = sum(pause_values_ms) / 1000.0
    normalized_text = SILENCE_MARKER_RE.sub(" ", text)
    speech_word_count = len(WORD_RE.findall(normalized_text))
    estimated_speech_duration_sec = (speech_word_count / float(rate_wpm)) * 60.0
    estimated_total_duration_sec = estimated_speech_duration_sec + pause_total_sec
    return RenderDurationEstimate(
        speech_word_count=speech_word_count,
        pause_count=len(pause_values_ms),
        pause_total_sec=round(pause_total_sec, 3),
        estimated_speech_duration_sec=round(estimated_speech_duration_sec, 3),
        estimated_total_duration_sec=round(estimated_total_duration_sec, 3),
    )


def _render_text_metrics(seed: SyntheticSeed, duration_estimate: RenderDurationEstimate) -> dict[str, float]:
    text = text_to_render(seed)
    normalized_text = SILENCE_MARKER_RE.sub(" ", text)
    pause_durations = [int(value) / 1000.0 for value in SILENCE_MARKER_RE.findall(text)]
    words = [{"text": token} for token in WORD_RE.findall(normalized_text)]
    audio_feats = {
        "duration_sec": duration_estimate.estimated_total_duration_sec,
        "pauses": [(0.0, 0.0, pause_duration) for pause_duration in pause_durations],
    }
    return transcript_metrics_from(words, audio_feats, language_code=seed.language_code)


def _resolve_benchmark_case(
    benchmark_root: Path,
    seed: SyntheticSeed,
    cache: dict[str, BenchmarkCase],
) -> BenchmarkCase | None:
    if not seed.benchmark_suite_id or not seed.benchmark_case_id:
        return None
    cache_key = f"{seed.benchmark_suite_id}:{seed.benchmark_case_id}"
    if cache_key in cache:
        return cache[cache_key]
    suite_path = benchmark_root / f"{seed.benchmark_suite_id}.json"
    if not suite_path.exists():
        raise FileNotFoundError(
            f"Benchmark suite for seed {seed.seed_id} was not found: {suite_path}"
        )
    suite = load_benchmark_suite(suite_path)
    cases_by_id = {case.case_id: case for case in suite.cases}
    benchmark_case = cases_by_id.get(seed.benchmark_case_id)
    if benchmark_case is None:
        raise ValueError(
            f"Benchmark case {seed.benchmark_case_id!r} was not found in suite {seed.benchmark_suite_id!r}"
        )
    cache[cache_key] = benchmark_case
    return benchmark_case


def _metric_tolerance(metric_name: str, expected: float) -> float:
    tolerance_cfg = BENCHMARK_ALIGNMENT_TOLERANCES.get(metric_name)
    if tolerance_cfg is None:
        return 0.0
    minimum = float(tolerance_cfg["minimum"])
    relative = float(tolerance_cfg["relative"])
    return max(minimum, abs(expected) * relative)


def _validate_seed_benchmark_alignment(
    seed: SyntheticSeed,
    *,
    duration_estimate: RenderDurationEstimate,
    benchmark_root: Path,
    cache: dict[str, BenchmarkCase],
) -> None:
    benchmark_case = _resolve_benchmark_case(benchmark_root, seed, cache)
    if benchmark_case is None:
        return

    computed_metrics = _render_text_metrics(seed, duration_estimate)
    issues: list[str] = []
    for metric_name, expected_value in benchmark_case.metrics.items():
        if metric_name not in BENCHMARK_ALIGNMENT_TOLERANCES:
            continue
        actual_value = computed_metrics.get(metric_name)
        if actual_value is None:
            continue
        expected_float = float(expected_value)
        tolerance = _metric_tolerance(metric_name, expected_float)
        if abs(float(actual_value) - expected_float) > tolerance:
            issues.append(
                f"{metric_name} expected {expected_float:g} but estimated {float(actual_value):g}"
            )

    if issues:
        raise ValueError(
            "Seed benchmark alignment failed for "
            f"{seed.seed_id} -> {seed.benchmark_case_id}: " + "; ".join(issues)
        )


def _run_subprocess(command: list[str], *, input_text: str | None = None) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            input=input_text,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"{command[0]!r} failed: {detail or f'exit code {exc.returncode}'}") from exc


def render_seed_manifest(
    manifest: SeedManifest,
    output_root: str | Path,
    *,
    selected_seed_ids: Iterable[str] | None = None,
    include_inactive: bool = False,
    overwrite: bool = False,
    benchmark_root: str | Path | None = None,
) -> dict:
    output_root_path = Path(output_root)
    bundle_dir = output_root_path / manifest.manifest_id
    audio_dir = bundle_dir / "audio"
    transcript_dir = bundle_dir / "transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = {seed_id.strip() for seed_id in (selected_seed_ids or []) if str(seed_id).strip()}
    available_seeds = manifest.seeds if include_inactive else manifest.active_seeds
    if selected_ids:
        known_ids = {seed.seed_id for seed in available_seeds}
        missing_ids = sorted(selected_ids - known_ids)
        if missing_ids:
            raise ValueError(f"Unknown or inactive seed_id values: {missing_ids}")
        target_seeds = tuple(seed for seed in available_seeds if seed.seed_id in selected_ids)
    else:
        target_seeds = available_seeds

    benchmark_root_path = Path(benchmark_root) if benchmark_root is not None else None
    benchmark_case_cache: dict[str, BenchmarkCase] = {}
    items: list[dict] = []
    for seed in target_seeds:
        config = resolve_render_config(manifest, seed)
        duration_estimate = estimate_render_duration(text_to_render(seed), config.rate_wpm)
        if benchmark_root_path is not None:
            _validate_seed_benchmark_alignment(
                seed,
                duration_estimate=duration_estimate,
                benchmark_root=benchmark_root_path,
                cache=benchmark_case_cache,
            )
        output_audio_path = audio_dir / f"{seed.seed_id}.{config.output_format}"
        transcript_path = transcript_dir / f"{seed.seed_id}.txt"
        if output_audio_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing audio: {output_audio_path}")

        with tempfile.NamedTemporaryFile(
            suffix=".aiff",
            prefix=f"{seed.seed_id}-",
            delete=False,
        ) as handle:
            intermediate_path = Path(handle.name)
        try:
            _run_subprocess(
                [
                    "say",
                    "-v",
                    config.voice,
                    "-r",
                    str(config.rate_wpm),
                    "-o",
                    str(intermediate_path),
                    "--file-format=AIFF",
                ],
                input_text=text_to_render(seed),
            )
            _run_subprocess(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(intermediate_path),
                    "-ac",
                    str(config.channels),
                    "-ar",
                    str(config.sample_rate_hz),
                    str(output_audio_path),
                ]
            )
        finally:
            intermediate_path.unlink(missing_ok=True)

        transcript_path.write_text(seed.transcript.rstrip("\n") + "\n", encoding="utf-8")
        items.append(
            {
                "seed_id": seed.seed_id,
                "target_cefr": seed.target_cefr,
                "topic_tag": seed.topic_tag,
                "audio_path": f"audio/{output_audio_path.name}",
                "transcript_path": f"transcripts/{transcript_path.name}",
                "source_seed_fingerprint": synthetic_seed_fingerprint(seed),
                "provider": config.provider,
                "voice": config.voice,
                "rate_wpm": config.rate_wpm,
                "output_format": config.output_format,
                "sample_rate_hz": config.sample_rate_hz,
                "channels": config.channels,
                "render_text_used": text_to_render(seed),
                "target_duration_sec": seed.target_duration_sec,
                "estimated_speech_word_count": duration_estimate.speech_word_count,
                "estimated_pause_count": duration_estimate.pause_count,
                "estimated_pause_total_sec": duration_estimate.pause_total_sec,
                "estimated_speech_duration_sec": duration_estimate.estimated_speech_duration_sec,
                "estimated_render_duration_sec": duration_estimate.estimated_total_duration_sec,
                "duration_alignment_ratio": (
                    round(duration_estimate.estimated_total_duration_sec / seed.target_duration_sec, 3)
                    if seed.target_duration_sec
                    else None
                ),
                "seed_tags": list(seed.tags),
            }
        )

    render_manifest = {
        "manifest_id": manifest.manifest_id,
        "seed_manifest_version": manifest.version,
        "seed_manifest_fingerprint": seed_manifest_fingerprint(manifest),
        "renderer_version": RENDERER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "items": items,
    }
    render_manifest_path = bundle_dir / "render_manifest.json"
    render_manifest_path.write_text(
        json.dumps(render_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return render_manifest

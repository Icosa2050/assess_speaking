"""Contracts that bridge rendered synthetic audio into evaluation-ready cases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from benchmarking.synthetic_seed_manifests import (
    SeedManifest,
    seed_manifest_fingerprint,
    synthetic_seed_fingerprint,
)


@dataclass(frozen=True)
class RenderedAudioItem:
    seed_id: str
    target_cefr: str
    target_duration_sec: float | None
    topic_tag: str
    audio_path: str
    transcript_path: str
    source_seed_fingerprint: str
    provider: str
    voice: str
    rate_wpm: int
    output_format: str
    sample_rate_hz: int
    channels: int
    render_text_used: str
    estimated_speech_word_count: int | None
    estimated_pause_count: int | None
    estimated_pause_total_sec: float | None
    estimated_speech_duration_sec: float | None
    estimated_render_duration_sec: float | None
    duration_alignment_ratio: float | None
    seed_tags: tuple[str, ...]


@dataclass(frozen=True)
class RenderedAudioCase:
    case_id: str
    source_seed_id: str
    audio_path: Path
    transcript_path: Path
    ground_truth_transcript: str
    expected_language: str
    task_family: str
    target_cefr: str
    target_duration_sec: float | None
    topic_tag: str
    benchmark_suite_id: str | None
    benchmark_case_id: str | None
    provider: str
    voice: str
    rate_wpm: int
    sample_rate_hz: int
    channels: int
    estimated_render_duration_sec: float | None
    duration_alignment_ratio: float | None
    renderer_version: str
    seed_manifest_version: str
    tags: tuple[str, ...]
    notes: str | None


@dataclass(frozen=True)
class RenderedAudioContractSuite:
    suite_id: str
    manifest_id: str
    language_code: str
    task_family: str
    renderer_version: str
    seed_manifest_version: str
    cases: tuple[RenderedAudioCase, ...]


def load_render_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Render manifest root payload must be an object")
    required = {
        "manifest_id",
        "seed_manifest_version",
        "seed_manifest_fingerprint",
        "renderer_version",
        "generated_at_utc",
        "items",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Render manifest is missing required keys: {sorted(missing)}")
    if not isinstance(payload["items"], list):
        raise ValueError("Render manifest items must be a list")
    return payload


def _parse_rendered_item(raw: dict[str, Any]) -> RenderedAudioItem:
    required = {
        "seed_id",
        "target_cefr",
        "target_duration_sec",
        "topic_tag",
        "audio_path",
        "transcript_path",
        "source_seed_fingerprint",
        "provider",
        "voice",
        "rate_wpm",
        "output_format",
        "sample_rate_hz",
        "channels",
        "render_text_used",
        "seed_tags",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Rendered audio item is missing required keys: {sorted(missing)}")
    return RenderedAudioItem(
        seed_id=str(raw["seed_id"]),
        target_cefr=str(raw["target_cefr"]),
        target_duration_sec=float(raw["target_duration_sec"]) if raw.get("target_duration_sec") is not None else None,
        topic_tag=str(raw["topic_tag"]),
        audio_path=str(raw["audio_path"]),
        transcript_path=str(raw["transcript_path"]),
        source_seed_fingerprint=str(raw["source_seed_fingerprint"]),
        provider=str(raw["provider"]),
        voice=str(raw["voice"]),
        rate_wpm=int(raw["rate_wpm"]),
        output_format=str(raw["output_format"]),
        sample_rate_hz=int(raw["sample_rate_hz"]),
        channels=int(raw["channels"]),
        render_text_used=str(raw["render_text_used"]),
        estimated_speech_word_count=(
            int(raw["estimated_speech_word_count"])
            if raw.get("estimated_speech_word_count") is not None
            else None
        ),
        estimated_pause_count=(
            int(raw["estimated_pause_count"])
            if raw.get("estimated_pause_count") is not None
            else None
        ),
        estimated_pause_total_sec=(
            float(raw["estimated_pause_total_sec"])
            if raw.get("estimated_pause_total_sec") is not None
            else None
        ),
        estimated_speech_duration_sec=(
            float(raw["estimated_speech_duration_sec"])
            if raw.get("estimated_speech_duration_sec") is not None
            else None
        ),
        estimated_render_duration_sec=(
            float(raw["estimated_render_duration_sec"])
            if raw.get("estimated_render_duration_sec") is not None
            else None
        ),
        duration_alignment_ratio=(
            float(raw["duration_alignment_ratio"])
            if raw.get("duration_alignment_ratio") is not None
            else None
        ),
        seed_tags=tuple(str(item) for item in raw["seed_tags"]),
    )


def _resolve_relative_path(bundle_dir: Path, relative_or_absolute: str) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate
    return (bundle_dir / candidate).resolve()


def build_rendered_audio_contract_suite(
    seed_manifest: SeedManifest,
    render_manifest_path: str | Path,
) -> RenderedAudioContractSuite:
    render_manifest_file = Path(render_manifest_path)
    payload = load_render_manifest(render_manifest_file)
    if str(payload["manifest_id"]) != seed_manifest.manifest_id:
        raise ValueError("Render manifest does not match the provided seed manifest")
    if str(payload["seed_manifest_version"]) != seed_manifest.version:
        raise ValueError("Render manifest version does not match the provided seed manifest")
    if str(payload["seed_manifest_fingerprint"]) != seed_manifest_fingerprint(seed_manifest):
        raise ValueError("Render manifest fingerprint does not match the provided seed manifest")

    bundle_dir = render_manifest_file.parent
    seeds_by_id = {seed.seed_id: seed for seed in seed_manifest.seeds}
    cases: list[RenderedAudioCase] = []
    for raw_item in payload["items"]:
        item = _parse_rendered_item(raw_item)
        seed = seeds_by_id.get(item.seed_id)
        if seed is None:
            raise ValueError(f"Render manifest references unknown seed_id: {item.seed_id}")
        if item.source_seed_fingerprint != synthetic_seed_fingerprint(seed):
            raise ValueError(
                f"Render manifest seed fingerprint does not match seed content for {item.seed_id}"
            )
        audio_path = _resolve_relative_path(bundle_dir, item.audio_path)
        transcript_path = _resolve_relative_path(bundle_dir, item.transcript_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Rendered audio file does not exist: {audio_path}")
        if not transcript_path.exists():
            raise FileNotFoundError(f"Rendered transcript file does not exist: {transcript_path}")

        cases.append(
            RenderedAudioCase(
                case_id=f"{seed.seed_id}_rendered",
                source_seed_id=seed.seed_id,
                audio_path=audio_path,
                transcript_path=transcript_path,
                ground_truth_transcript=seed.transcript,
                expected_language=seed.language_code,
                task_family=seed.task_family,
                target_cefr=seed.target_cefr,
                target_duration_sec=item.target_duration_sec,
                topic_tag=seed.topic_tag,
                benchmark_suite_id=seed.benchmark_suite_id,
                benchmark_case_id=seed.benchmark_case_id,
                provider=item.provider,
                voice=item.voice,
                rate_wpm=item.rate_wpm,
                sample_rate_hz=item.sample_rate_hz,
                channels=item.channels,
                estimated_render_duration_sec=item.estimated_render_duration_sec,
                duration_alignment_ratio=item.duration_alignment_ratio,
                renderer_version=str(payload["renderer_version"]),
                seed_manifest_version=str(payload["seed_manifest_version"]),
                tags=tuple(dict.fromkeys((*seed.tags, *item.seed_tags, "rendered-audio"))),
                notes=seed.notes,
            )
        )

    return RenderedAudioContractSuite(
        suite_id=f"{seed_manifest.manifest_id}_rendered_audio_v1",
        manifest_id=seed_manifest.manifest_id,
        language_code=seed_manifest.language_code,
        task_family=seed_manifest.task_family,
        renderer_version=str(payload["renderer_version"]),
        seed_manifest_version=str(payload["seed_manifest_version"]),
        cases=tuple(cases),
    )

"""Seed-manifest loading and discovery for synthetic benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RenderDefaults:
    provider: str | None
    voice: str | None
    rate_wpm: int | None
    output_format: str | None
    sample_rate_hz: int | None
    channels: int | None
    notes: str | None


@dataclass(frozen=True)
class SyntheticSeed:
    seed_id: str
    language_code: str
    task_family: str
    target_cefr: str
    target_duration_sec: float | None
    topic_tag: str
    benchmark_suite_id: str | None
    benchmark_case_id: str | None
    transcript: str
    render_text: str | None
    render_overrides: RenderDefaults | None
    source_type: str
    active: bool
    tags: tuple[str, ...]
    notes: str | None


@dataclass(frozen=True)
class SeedManifest:
    manifest_id: str
    language_code: str
    task_family: str
    version: str
    active: bool
    tags: tuple[str, ...]
    notes: str | None
    render_defaults: RenderDefaults
    seeds: tuple[SyntheticSeed, ...]

    @property
    def active_seeds(self) -> tuple[SyntheticSeed, ...]:
        return tuple(seed for seed in self.seeds if seed.active)


VALID_CEFR_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


def _parse_render_defaults(payload: dict[str, Any], *, field_prefix: str) -> RenderDefaults:
    return RenderDefaults(
        provider=payload.get("provider"),
        voice=payload.get("voice"),
        rate_wpm=_as_int_or_none(payload.get("rate_wpm"), field_name=f"{field_prefix}.rate_wpm"),
        output_format=payload.get("output_format"),
        sample_rate_hz=_as_int_or_none(
            payload.get("sample_rate_hz"),
            field_name=f"{field_prefix}.sample_rate_hz",
        ),
        channels=_as_int_or_none(payload.get("channels"), field_name=f"{field_prefix}.channels"),
        notes=payload.get("notes"),
    )


def _as_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("tags must be a list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _as_int_or_none(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _as_positive_float_or_none(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def load_seed_manifest(path: str | Path) -> SeedManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Seed manifest root payload must be an object")
    required_root_keys = {
        "manifest_id",
        "language_code",
        "task_family",
        "version",
        "render_defaults",
        "seeds",
    }
    missing = required_root_keys - set(payload)
    if missing:
        raise ValueError(f"Seed manifest is missing required keys: {sorted(missing)}")
    if not isinstance(payload["render_defaults"], dict):
        raise ValueError("render_defaults must be an object")
    if not isinstance(payload["seeds"], list) or not payload["seeds"]:
        raise ValueError("Seed manifest must include at least one seed")

    render_defaults = _parse_render_defaults(payload["render_defaults"], field_prefix="render_defaults")

    seeds: list[SyntheticSeed] = []
    seen_seed_ids: set[str] = set()
    for raw_seed in payload["seeds"]:
        required_seed_keys = {
            "seed_id",
            "language_code",
            "task_family",
            "target_cefr",
            "topic_tag",
            "transcript",
            "source_type",
        }
        missing_seed = required_seed_keys - set(raw_seed)
        if missing_seed:
            raise ValueError(f"Seed entry is missing required keys: {sorted(missing_seed)}")
        seed_id = str(raw_seed["seed_id"])
        if seed_id in seen_seed_ids:
            raise ValueError(f"Duplicate seed_id in manifest: {seed_id}")
        seen_seed_ids.add(seed_id)

        seed_language = str(raw_seed["language_code"])
        seed_task_family = str(raw_seed["task_family"])
        if seed_language != str(payload["language_code"]):
            raise ValueError(f"Seed {seed_id} language mismatch with manifest")
        if seed_task_family != str(payload["task_family"]):
            raise ValueError(f"Seed {seed_id} task_family mismatch with manifest")

        target_cefr = str(raw_seed["target_cefr"]).upper()
        if target_cefr not in VALID_CEFR_LEVELS:
            raise ValueError(f"{seed_id}.target_cefr must be one of {sorted(VALID_CEFR_LEVELS)}")

        transcript = str(raw_seed["transcript"]).strip()
        if not transcript:
            raise ValueError(f"{seed_id}.transcript must not be empty")
        render_overrides_payload = raw_seed.get("render_overrides")
        if render_overrides_payload is not None and not isinstance(render_overrides_payload, dict):
            raise ValueError(f"{seed_id}.render_overrides must be an object when present")
        seeds.append(
            SyntheticSeed(
                seed_id=seed_id,
                language_code=seed_language,
                task_family=seed_task_family,
                target_cefr=target_cefr,
                target_duration_sec=_as_positive_float_or_none(
                    raw_seed.get("target_duration_sec"),
                    field_name=f"{seed_id}.target_duration_sec",
                ),
                topic_tag=str(raw_seed["topic_tag"]),
                benchmark_suite_id=(
                    str(raw_seed["benchmark_suite_id"])
                    if raw_seed.get("benchmark_suite_id") is not None
                    else None
                ),
                benchmark_case_id=(
                    str(raw_seed["benchmark_case_id"])
                    if raw_seed.get("benchmark_case_id") is not None
                    else None
                ),
                transcript=transcript,
                render_text=str(raw_seed["render_text"]) if raw_seed.get("render_text") is not None else None,
                render_overrides=(
                    _parse_render_defaults(render_overrides_payload, field_prefix=f"{seed_id}.render_overrides")
                    if render_overrides_payload is not None
                    else None
                ),
                source_type=str(raw_seed["source_type"]),
                active=bool(raw_seed.get("active", True)),
                tags=_as_tags(raw_seed.get("tags")),
                notes=raw_seed.get("notes"),
            )
        )

    return SeedManifest(
        manifest_id=str(payload["manifest_id"]),
        language_code=str(payload["language_code"]),
        task_family=str(payload["task_family"]),
        version=str(payload["version"]),
        active=bool(payload.get("active", True)),
        tags=_as_tags(payload.get("tags")),
        notes=payload.get("notes"),
        render_defaults=render_defaults,
        seeds=tuple(seeds),
    )


def _render_defaults_to_dict(render_defaults: RenderDefaults | None) -> dict[str, Any] | None:
    if render_defaults is None:
        return None
    return {
        "provider": render_defaults.provider,
        "voice": render_defaults.voice,
        "rate_wpm": render_defaults.rate_wpm,
        "output_format": render_defaults.output_format,
        "sample_rate_hz": render_defaults.sample_rate_hz,
        "channels": render_defaults.channels,
        "notes": render_defaults.notes,
    }


def _seed_to_fingerprint_payload(seed: SyntheticSeed) -> dict[str, Any]:
    return {
        "seed_id": seed.seed_id,
        "language_code": seed.language_code,
        "task_family": seed.task_family,
        "target_cefr": seed.target_cefr,
        "target_duration_sec": seed.target_duration_sec,
        "topic_tag": seed.topic_tag,
        "benchmark_suite_id": seed.benchmark_suite_id,
        "benchmark_case_id": seed.benchmark_case_id,
        "transcript": seed.transcript,
        "render_text": seed.render_text,
        "render_overrides": _render_defaults_to_dict(seed.render_overrides),
        "source_type": seed.source_type,
        "active": seed.active,
        "tags": list(seed.tags),
        "notes": seed.notes,
    }


def synthetic_seed_fingerprint(seed: SyntheticSeed) -> str:
    payload = _seed_to_fingerprint_payload(seed)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def seed_manifest_fingerprint(manifest: SeedManifest) -> str:
    payload = {
        "manifest_id": manifest.manifest_id,
        "language_code": manifest.language_code,
        "task_family": manifest.task_family,
        "version": manifest.version,
        "active": manifest.active,
        "tags": list(manifest.tags),
        "notes": manifest.notes,
        "render_defaults": _render_defaults_to_dict(manifest.render_defaults),
        "seeds": [_seed_to_fingerprint_payload(seed) for seed in manifest.seeds],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def discover_seed_manifests(
    root: str | Path,
    *,
    include_inactive: bool = False,
    language_codes: set[str] | None = None,
    task_families: set[str] | None = None,
    tags: set[str] | None = None,
    tag_match: str = "any",
) -> tuple[SeedManifest, ...]:
    root_path = Path(root)
    if tag_match not in {"any", "all"}:
        raise ValueError("tag_match must be 'any' or 'all'")

    manifests: list[SeedManifest] = []
    for path in sorted(root_path.glob("*.json")):
        manifest = load_seed_manifest(path)
        if not include_inactive and not manifest.active:
            continue
        if language_codes and manifest.language_code not in language_codes:
            continue
        if task_families and manifest.task_family not in task_families:
            continue
        if tags:
            manifest_tags = set(manifest.tags)
            if tag_match == "any" and not manifest_tags.intersection(tags):
                continue
            if tag_match == "all" and not tags.issubset(manifest_tags):
                continue
        manifests.append(manifest)
    return tuple(manifests)

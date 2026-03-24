"""Manifest loading for curated CELI benchmark wordlists."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


VALID_CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


@dataclass(frozen=True)
class CeliWordlistTerm:
    term_id: str
    term: str
    levels: tuple[str, ...]
    active: bool
    tags: tuple[str, ...]
    notes: str | None


@dataclass(frozen=True)
class CeliWordlistManifest:
    manifest_id: str
    source_id: str
    language_code: str
    version: str
    active: bool
    tags: tuple[str, ...]
    notes: str | None
    default_levels: tuple[str, ...]
    terms: tuple[CeliWordlistTerm, ...]
    source_path: Path

    @property
    def active_terms(self) -> tuple[CeliWordlistTerm, ...]:
        return tuple(term for term in self.terms if term.active)


def _as_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("tags must be a list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _as_levels(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of CEFR levels")
    parsed = tuple(str(item).strip().upper() for item in value if str(item).strip())
    if not parsed:
        raise ValueError(f"{field_name} must be a non-empty list of CEFR levels")
    invalid = [item for item in parsed if item not in VALID_CEFR_LEVELS]
    if invalid:
        raise ValueError(f"{field_name} contains invalid CEFR levels: {sorted(set(invalid))}")
    return parsed


def load_celi_wordlist_manifest(path: str | Path) -> CeliWordlistManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CELI wordlist manifest root payload must be an object")

    required_root_keys = {
        "manifest_id",
        "source_id",
        "language_code",
        "version",
        "default_levels",
        "terms",
    }
    missing = required_root_keys - set(payload)
    if missing:
        raise ValueError(f"CELI wordlist manifest is missing required keys: {sorted(missing)}")
    if not isinstance(payload["terms"], list) or not payload["terms"]:
        raise ValueError("CELI wordlist manifest must include at least one term")

    default_levels = _as_levels(payload["default_levels"], field_name="default_levels")
    terms: list[CeliWordlistTerm] = []
    seen_term_ids: set[str] = set()
    for raw_term in payload["terms"]:
        required_term_keys = {"term_id", "term"}
        missing_term = required_term_keys - set(raw_term)
        if missing_term:
            raise ValueError(f"CELI wordlist term is missing required keys: {sorted(missing_term)}")
        term_id = str(raw_term["term_id"]).strip()
        if not term_id:
            raise ValueError("term_id must not be empty")
        if term_id in seen_term_ids:
            raise ValueError(f"Duplicate term_id in CELI wordlist manifest: {term_id}")
        seen_term_ids.add(term_id)
        term_text = str(raw_term["term"]).strip()
        if not term_text:
            raise ValueError(f"{term_id}.term must not be empty")
        levels = (
            _as_levels(raw_term["levels"], field_name=f"{term_id}.levels")
            if raw_term.get("levels") is not None
            else default_levels
        )
        terms.append(
            CeliWordlistTerm(
                term_id=term_id,
                term=term_text,
                levels=levels,
                active=bool(raw_term.get("active", True)),
                tags=_as_tags(raw_term.get("tags")),
                notes=raw_term.get("notes"),
            )
        )

    return CeliWordlistManifest(
        manifest_id=str(payload["manifest_id"]),
        source_id=str(payload["source_id"]),
        language_code=str(payload["language_code"]).strip().lower(),
        version=str(payload["version"]),
        active=bool(payload.get("active", True)),
        tags=_as_tags(payload.get("tags")),
        notes=payload.get("notes"),
        default_levels=default_levels,
        terms=tuple(terms),
        source_path=manifest_path,
    )


def discover_celi_wordlist_manifests(
    root: str | Path,
    *,
    include_inactive: bool = False,
    language_codes: set[str] | None = None,
    tags: set[str] | None = None,
    tag_match: str = "any",
) -> tuple[CeliWordlistManifest, ...]:
    root_path = Path(root)
    if tag_match not in {"any", "all"}:
        raise ValueError("tag_match must be 'any' or 'all'")
    manifests: list[CeliWordlistManifest] = []
    for path in sorted(root_path.glob("*.json")):
        manifest = load_celi_wordlist_manifest(path)
        if not include_inactive and not manifest.active:
            continue
        if language_codes and manifest.language_code not in language_codes:
            continue
        if tags:
            manifest_tags = set(manifest.tags)
            if tag_match == "any" and not manifest_tags.intersection(tags):
                continue
            if tag_match == "all" and not tags.issubset(manifest_tags):
                continue
        manifests.append(manifest)
    return tuple(manifests)


def celi_wordlist_manifest_as_dict(manifest: CeliWordlistManifest) -> dict[str, Any]:
    return {
        "manifest_id": manifest.manifest_id,
        "source_id": manifest.source_id,
        "language_code": manifest.language_code,
        "version": manifest.version,
        "active": manifest.active,
        "tags": list(manifest.tags),
        "notes": manifest.notes,
        "default_levels": list(manifest.default_levels),
        "terms": [
            {
                "term_id": term.term_id,
                "term": term.term,
                "levels": list(term.levels),
                "active": term.active,
                "tags": list(term.tags),
                "notes": term.notes,
            }
            for term in manifest.terms
        ],
        "source_path": manifest.source_path.as_posix(),
    }

"""Catalog and download helpers for external learner corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Iterable
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OpenCorpusSource:
    source_id: str
    title: str
    language_code: str
    modality: str
    license_name: str | None
    homepage_url: str
    download_url: str | None
    archive_name: str | None
    tags: tuple[str, ...]
    notes: str | None


_OPEN_CORPUS_SOURCES: tuple[OpenCorpusSource, ...] = (
    OpenCorpusSource(
        source_id="rita_phrame4",
        title="RITA PHRAME4",
        language_code="it",
        modality="writing",
        license_name="CC BY 4.0",
        homepage_url="https://zenodo.org/records/10391688",
        download_url="https://zenodo.org/records/10391688/files/RITA_PHRAME4.zip?download=1",
        archive_name="RITA_PHRAME4.zip",
        tags=("italian", "cefr", "exam", "open", "phrame", "rita"),
        notes=(
            "Italian L2 CEFR certification derivative from CELI. Archive contains token-level, "
            "text-level, and CEFR-level statistics plus XML/XSD, but not a simple raw full-text table."
        ),
    ),
    OpenCorpusSource(
        source_id="merlin_v1_2",
        title="MERLIN v1.2",
        language_code="multilingual",
        modality="writing",
        license_name="CC BY-SA 4.0",
        homepage_url="https://www.merlin-platform.eu/C_mcorpus.php",
        download_url=None,
        archive_name=None,
        tags=("cefr", "exam", "error-annotation", "merlin", "open"),
        notes=(
            "Officially downloadable via Eurac CLARIN handle, but the public handle currently resolves "
            "to a landing page rather than a stable direct file URL."
        ),
    ),
    OpenCorpusSource(
        source_id="ud_italian_valico",
        title="UD Italian VALICO",
        language_code="it",
        modality="writing",
        license_name=None,
        homepage_url="https://github.com/UniversalDependencies/UD_Italian-Valico",
        download_url=None,
        archive_name=None,
        tags=("italian", "ud", "valico", "writing"),
        notes="Downloadable as a Git repository; use for a small gold subset rather than raw full VALICO.",
    ),
    OpenCorpusSource(
        source_id="valico_ud_silver",
        title="VALICO-UD Silver",
        language_code="it",
        modality="writing",
        license_name=None,
        homepage_url="https://github.com/ElisaDiNuovo/VALICO-UD_silver",
        download_url=None,
        archive_name=None,
        tags=("italian", "ud", "valico", "writing", "silver"),
        notes="Downloadable as a Git repository; useful as the largest open VALICO-derived subset we verified.",
    ),
)


def list_open_corpus_sources() -> tuple[OpenCorpusSource, ...]:
    return _OPEN_CORPUS_SOURCES


def resolve_open_corpus_source(source_id: str) -> OpenCorpusSource:
    normalized = source_id.strip().lower()
    for source in _OPEN_CORPUS_SOURCES:
        if source.source_id == normalized:
            return source
    raise KeyError(f"Unknown open corpus source: {source_id}")


def downloadable_open_corpus_sources() -> tuple[OpenCorpusSource, ...]:
    return tuple(source for source in _OPEN_CORPUS_SOURCES if source.download_url and source.archive_name)


def download_open_corpus(
    source: OpenCorpusSource,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    user_agent: str = "assess-speaking-open-corpus-fetcher/1.0",
) -> Path:
    if not source.download_url or not source.archive_name:
        raise ValueError(f"{source.source_id} does not expose a stable direct download URL")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    destination = output_path / source.archive_name
    if destination.exists() and not overwrite:
        return destination

    request = Request(source.download_url, headers={"User-Agent": user_agent})
    with urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def open_corpus_catalog_as_dicts(sources: Iterable[OpenCorpusSource] | None = None) -> list[dict[str, object]]:
    catalog = sources if sources is not None else _OPEN_CORPUS_SOURCES
    return [
        {
            "source_id": source.source_id,
            "title": source.title,
            "language_code": source.language_code,
            "modality": source.modality,
            "license_name": source.license_name,
            "homepage_url": source.homepage_url,
            "download_url": source.download_url,
            "archive_name": source.archive_name,
            "tags": list(source.tags),
            "notes": source.notes,
        }
        for source in catalog
    ]

"""ASR transcription wrapper around faster-whisper."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime for CLI ergonomics
    WhisperModel = None  # type: ignore

KNOWN_WHISPER_MODELS = ("tiny", "small", "medium", "large-v3")


def _default_hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        roots.append(Path(xdg_cache_home) / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for root in roots:
        resolved = root.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def _model_repo_id(model_size: str) -> str:
    candidate_path = Path(model_size).expanduser()
    if candidate_path.exists():
        return str(candidate_path)
    return model_size if "/" in model_size else f"Systran/faster-whisper-{model_size}"


def _resolve_cached_model_path(model_size: str) -> str | None:
    candidate_path = Path(model_size).expanduser()
    if candidate_path.exists():
        return str(candidate_path)
    if not model_size or model_size.startswith((".", "/")):
        return None
    repo_id = _model_repo_id(model_size)
    cache_dir_name = "models--" + repo_id.replace("/", "--")
    for cache_root in _default_hf_cache_roots():
        snapshots_dir = cache_root / cache_dir_name / "snapshots"
        if not snapshots_dir.exists():
            continue
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if not snapshots:
            continue
        ref_main = cache_root / cache_dir_name / "refs" / "main"
        if ref_main.exists():
            ref = ref_main.read_text(encoding="utf-8").strip()
            preferred = snapshots_dir / ref
            if preferred.is_dir():
                return str(preferred)
        return str(snapshots[-1])
    return None


def describe_model_availability(model_size: str) -> dict:
    cached_path = _resolve_cached_model_path(model_size)
    if os.getenv("ASSESS_SPEAKING_DRY_RUN") == "1":
        # Dry-run skips real transcription, so UI flows should not depend on a pre-downloaded Whisper cache.
        return {
            "model": model_size,
            "repo_id": _model_repo_id(model_size),
            "cached": True,
            "cached_path": cached_path,
        }
    return {
        "model": model_size,
        "repo_id": _model_repo_id(model_size),
        "cached": bool(cached_path),
        "cached_path": cached_path,
    }


def recommend_model_choice() -> dict:
    if describe_model_availability("large-v3")["cached"]:
        return {
            "model": "large-v3",
            "reason": "`large-v3` ist lokal vorhanden und liefert die beste Bewertungsqualität.",
        }
    if describe_model_availability("medium")["cached"]:
        return {
            "model": "medium",
            "reason": "`medium` ist lokal vorhanden und ein guter Qualitätskompromiss.",
        }
    if describe_model_availability("small")["cached"]:
        return {
            "model": "small",
            "reason": "`small` ist lokal vorhanden und schneller als die größeren Modelle.",
        }
    if describe_model_availability("tiny")["cached"]:
        return {
            "model": "tiny",
            "reason": "`tiny` ist sofort nutzbar. Für bessere Qualität solltest du später `large-v3` laden.",
        }
    return {
        "model": "tiny",
        "reason": "Noch kein Modell im Cache. Starte mit `tiny` oder lade direkt `large-v3` für bessere Qualität.",
    }


def _initialize_whisper_model(model_ref: str, compute_type: str, fallback_compute_type: str | None) -> tuple[object, str, bool]:
    try:
        model = WhisperModel(model_ref, compute_type=compute_type)
        return model, compute_type, False
    except ImportError as exc:
        if "socksio" in str(exc).lower():
            raise RuntimeError(
                "SOCKS proxy detected but 'socksio' is missing. Install dependencies via `python -m pip install -r requirements.txt` or `python -m pip install socksio`."
            ) from exc
        raise
    except Exception as first_exc:
        module_name = getattr(first_exc.__class__, "__module__", "")
        if module_name.startswith(("httpx", "httpcore", "huggingface_hub")):
            raise RuntimeError(
                "Whisper model download failed while initializing faster-whisper. Check proxy/network access to Hugging Face or pre-download the model."
            ) from first_exc
        if fallback_compute_type and fallback_compute_type != compute_type:
            try:
                model = WhisperModel(model_ref, compute_type=fallback_compute_type)
                return model, fallback_compute_type, True
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Failed to initialize faster-whisper with compute_type "
                    f"'{compute_type}' and fallback '{fallback_compute_type}'."
                ) from fallback_exc
        raise RuntimeError(f"Failed to initialize faster-whisper with compute_type '{compute_type}'.") from first_exc


def ensure_model_downloaded(
    model_size: str,
    *,
    compute_type: str = "default",
    fallback_compute_type: str | None = "int8",
) -> dict:
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not available. Install dependencies via `python -m pip install -r requirements.txt`."
        )
    availability = describe_model_availability(model_size)
    if availability["cached"]:
        return availability
    model_ref = _model_repo_id(model_size)
    model, _, _ = _initialize_whisper_model(model_ref, compute_type, fallback_compute_type)
    del model
    availability = describe_model_availability(model_size)
    if not availability["cached"]:
        raise RuntimeError(f"Model '{model_size}' initialization finished, but no local cache snapshot was found afterwards.")
    return availability


def transcribe(
    path: Path,
    model_size: str = "large-v3",
    language: str | None = None,
    compute_type: str = "default",
    fallback_compute_type: str | None = "int8",
) -> dict:
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not available. Install dependencies via `python -m pip install -r requirements.txt`."
        )

    model_ref = _resolve_cached_model_path(model_size) or model_size

    model, compute_type_used, compute_fallback_used = _initialize_whisper_model(
        model_ref,
        compute_type,
        fallback_compute_type,
    )

    segments, info = model.transcribe(str(path), vad_filter=True, word_timestamps=True, language=language)
    words = []
    full_text = []
    for segment in segments:
        full_text.append(segment.text.strip())
        if segment.words:
            for word in segment.words:
                token = word.word.strip().lower()
                if token:
                    words.append({"t0": word.start, "t1": word.end, "text": token})

    if isinstance(info, dict):
        detected_language = info.get("language")
        language_probability = info.get("language_probability")
    else:
        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)

    return {
        "text": " ".join(full_text).strip(),
        "words": words,
        "compute_type_used": compute_type_used,
        "compute_fallback_used": compute_fallback_used,
        "detected_language": detected_language,
        "language_probability": language_probability,
    }

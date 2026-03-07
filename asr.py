"""ASR transcription wrapper around faster-whisper."""

from __future__ import annotations

from pathlib import Path

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime for CLI ergonomics
    WhisperModel = None  # type: ignore


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

    try:
        model = WhisperModel(model_size, compute_type=compute_type)
        compute_type_used = compute_type
        compute_fallback_used = False
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
                model = WhisperModel(model_size, compute_type=fallback_compute_type)
                compute_type_used = fallback_compute_type
                compute_fallback_used = True
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Failed to initialize faster-whisper with compute_type "
                    f"'{compute_type}' and fallback '{fallback_compute_type}'."
                ) from fallback_exc
        else:
            raise RuntimeError(f"Failed to initialize faster-whisper with compute_type '{compute_type}'.") from first_exc

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

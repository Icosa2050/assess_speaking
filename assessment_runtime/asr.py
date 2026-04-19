"""ASR transcription runtime with provider and file-strategy semantics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm as base_tqdm

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime for CLI ergonomics
    WhisperModel = None  # type: ignore

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:  # pragma: no cover - handled by fallback initialization path
    hf_hub_download = None  # type: ignore
    snapshot_download = None  # type: ignore

KNOWN_WHISPER_MODELS = ("tiny", "small", "medium", "large-v3")
DEFAULT_ASR_PROVIDER = "faster_whisper"
KNOWN_ASR_PROVIDERS = (DEFAULT_ASR_PROVIDER,)
KNOWN_FILE_STRATEGIES = ("auto", "native", "chunked")
DownloadProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ASRCapabilities:
    supports_native_file_transcription: bool = True
    prefers_native_file_transcription: bool = True
    supports_chunked_fallback: bool = True


@dataclass(frozen=True)
class LoadedASRModel:
    model: Any
    compute_type_used: str
    compute_fallback_used: bool


class ASRProvider:
    provider_id = "base"
    display_name = "Base ASR Provider"
    capabilities = ASRCapabilities()

    def load_model(
        self,
        *,
        model_size: str,
        compute_type: str,
        fallback_compute_type: str | None,
    ) -> LoadedASRModel:
        raise NotImplementedError

    def transcribe_file_native(
        self,
        loaded_model: LoadedASRModel,
        path: Path,
        *,
        language: str | None,
        time_offset_sec: float = 0.0,
    ) -> dict[str, Any]:
        raise NotImplementedError


class FasterWhisperASRProvider(ASRProvider):
    provider_id = DEFAULT_ASR_PROVIDER
    display_name = "faster-whisper"
    capabilities = ASRCapabilities(
        supports_native_file_transcription=True,
        prefers_native_file_transcription=True,
        supports_chunked_fallback=True,
    )

    def load_model(
        self,
        *,
        model_size: str,
        compute_type: str,
        fallback_compute_type: str | None,
    ) -> LoadedASRModel:
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
        return LoadedASRModel(
            model=model,
            compute_type_used=compute_type_used,
            compute_fallback_used=compute_fallback_used,
        )

    def transcribe_file_native(
        self,
        loaded_model: LoadedASRModel,
        path: Path,
        *,
        language: str | None,
        time_offset_sec: float = 0.0,
    ) -> dict[str, Any]:
        segments, info = loaded_model.model.transcribe(
            str(path),
            vad_filter=True,
            word_timestamps=True,
            language=language,
        )
        return _build_transcription_result(
            segments,
            info,
            compute_type_used=loaded_model.compute_type_used,
            compute_fallback_used=loaded_model.compute_fallback_used,
            time_offset_sec=time_offset_sec,
        )


class FasterWhisperChunkedASRProvider(FasterWhisperASRProvider):
    provider_id = "faster_whisper_chunked"
    display_name = "faster-whisper (chunk-preferring)"
    capabilities = ASRCapabilities(
        supports_native_file_transcription=True,
        prefers_native_file_transcription=False,
        supports_chunked_fallback=True,
    )


_ASR_PROVIDERS: dict[str, ASRProvider] = {
    DEFAULT_ASR_PROVIDER: FasterWhisperASRProvider(),
    "faster_whisper_chunked": FasterWhisperChunkedASRProvider(),
}


class _SilentTqdm(base_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


class _DownloadReporter:
    def __init__(
        self,
        *,
        model_size: str,
        total_bytes: int,
        total_files: int,
        completed_files: int,
        downloaded_bytes: int,
        callback: DownloadProgressCallback | None,
    ) -> None:
        self._model_size = model_size
        self._total_bytes = max(total_bytes, 0)
        self._total_files = max(total_files, 0)
        self._completed_files = max(completed_files, 0)
        self._downloaded_bytes = max(downloaded_bytes, 0)
        self._callback = callback
        self._active_file_name = ""
        self._active_file_total = 0
        self._active_file_start = self._downloaded_bytes
        self._last_emit_at = 0.0
        self._last_signature: tuple[Any, ...] | None = None

    @property
    def downloaded_bytes(self) -> int:
        return self._downloaded_bytes

    def emit(self, stage: str, *, force: bool = False, **payload: Any) -> None:
        if self._callback is None:
            return
        event = {
            "stage": stage,
            "model": self._model_size,
            "total_bytes": self._total_bytes,
            "downloaded_bytes": self._downloaded_bytes,
            "total_files": self._total_files,
            "completed_files": self._completed_files,
            **payload,
        }
        signature = (
            stage,
            event.get("current_file"),
            event.get("file_downloaded_bytes"),
            event.get("file_total_bytes"),
            event.get("downloaded_bytes"),
            event.get("completed_files"),
        )
        now = time.monotonic()
        if not force and self._last_signature == signature and now - self._last_emit_at < 0.15:
            return
        self._last_signature = signature
        self._last_emit_at = now
        self._callback(event)

    def start_file(self, filename: str, total_bytes: int, current_bytes: int) -> None:
        self._active_file_name = filename
        self._active_file_total = max(total_bytes, 0)
        self._active_file_start = self._downloaded_bytes - max(current_bytes, 0)
        self.emit(
            "downloading",
            force=True,
            current_file=filename,
            file_downloaded_bytes=max(current_bytes, 0),
            file_total_bytes=self._active_file_total,
        )

    def update_file(self, current_bytes: int, total_bytes: int | None = None) -> None:
        if total_bytes is not None and total_bytes >= 0:
            self._active_file_total = total_bytes
        clamped_current = max(current_bytes, 0)
        self._downloaded_bytes = min(
            max(self._active_file_start + clamped_current, 0),
            self._total_bytes or max(self._active_file_start + clamped_current, 0),
        )
        self.emit(
            "downloading",
            current_file=self._active_file_name,
            file_downloaded_bytes=clamped_current,
            file_total_bytes=self._active_file_total,
        )

    def finish_file(self, filename: str, file_size: int) -> None:
        self._active_file_name = filename
        self._active_file_total = max(file_size, 0)
        self._downloaded_bytes = min(
            max(self._active_file_start + max(file_size, 0), self._downloaded_bytes),
            self._total_bytes or max(self._active_file_start + max(file_size, 0), self._downloaded_bytes),
        )
        self._completed_files = min(self._completed_files + 1, self._total_files or self._completed_files + 1)
        self.emit(
            "downloading",
            force=True,
            current_file=filename,
            file_downloaded_bytes=max(file_size, 0),
            file_total_bytes=max(file_size, 0),
        )


def _download_progress_tqdm_class(reporter: _DownloadReporter, filename: str) -> type[base_tqdm]:
    class _ProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            initial = int(kwargs.get("initial") or 0)
            total = int(kwargs.get("total") or 0)
            super().__init__(*args, **kwargs)
            reporter.start_file(filename, total, initial)

        def update(self, n: int | float = 1) -> None:
            super().update(n)
            reporter.update_file(int(self.n), int(self.total or 0))

    return _ProgressTqdm


def available_asr_providers() -> tuple[str, ...]:
    return tuple(_ASR_PROVIDERS)


def _normalize_asr_provider_key(provider: str | None) -> str:
    candidate = (provider or DEFAULT_ASR_PROVIDER).strip().lower().replace("-", "_")
    if candidate == "whisper":
        candidate = DEFAULT_ASR_PROVIDER
    if candidate in {"whisper_chunked", "segmented_whisper"}:
        candidate = "faster_whisper_chunked"
    if candidate not in _ASR_PROVIDERS:
        raise RuntimeError(
            f"Unsupported ASR provider '{provider}'. Available providers: {', '.join(available_asr_providers())}."
        )
    return candidate


def _resolve_asr_provider(provider: str | None) -> ASRProvider:
    return _ASR_PROVIDERS[_normalize_asr_provider_key(provider)]


def _normalize_file_strategy(strategy: str) -> str:
    candidate = (strategy or "auto").strip().lower()
    if candidate not in KNOWN_FILE_STRATEGIES:
        raise RuntimeError(
            f"Unsupported ASR file strategy '{strategy}'. Expected one of: {', '.join(KNOWN_FILE_STRATEGIES)}."
        )
    return candidate


def _plan_snapshot_download(model_ref: str) -> list[Any] | None:
    if snapshot_download is None:
        return None
    try:
        planned = snapshot_download(
            model_ref,
            dry_run=True,
            max_workers=1,
            tqdm_class=_SilentTqdm,
        )
    except Exception:
        return None
    return planned if isinstance(planned, list) else None


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


def _build_transcription_result(
    segments: Any,
    info: Any,
    *,
    compute_type_used: str,
    compute_fallback_used: bool,
    time_offset_sec: float = 0.0,
) -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    full_text: list[str] = []
    for segment in segments:
        full_text.append(segment.text.strip())
        if segment.words:
            for word in segment.words:
                token = word.word.strip().lower()
                if token:
                    words.append(
                        {
                            "t0": float(word.start) + time_offset_sec,
                            "t1": float(word.end) + time_offset_sec,
                            "text": token,
                        }
                    )

    if isinstance(info, dict):
        detected_language = info.get("language")
        language_probability = info.get("language_probability")
    else:
        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)

    return {
        "text": " ".join(part for part in full_text if part).strip(),
        "words": words,
        "compute_type_used": compute_type_used,
        "compute_fallback_used": compute_fallback_used,
        "detected_language": detected_language,
        "language_probability": language_probability,
    }


def _merge_chunk_transcriptions(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    text_parts = [str(result.get("text") or "").strip() for result in chunk_results]
    merged_words: list[dict[str, Any]] = []
    detected_language = None
    language_probability = None
    compute_type_used = ""
    compute_fallback_used = False

    for result in chunk_results:
        if not compute_type_used and result.get("compute_type_used"):
            compute_type_used = str(result["compute_type_used"])
        compute_fallback_used = compute_fallback_used or bool(result.get("compute_fallback_used"))
        if detected_language is None and result.get("detected_language"):
            detected_language = result.get("detected_language")
        if language_probability is None and result.get("language_probability") is not None:
            language_probability = result.get("language_probability")
        merged_words.extend(result.get("words") or [])

    return {
        "text": " ".join(part for part in text_parts if part).strip(),
        "words": merged_words,
        "compute_type_used": compute_type_used,
        "compute_fallback_used": compute_fallback_used,
        "detected_language": detected_language,
        "language_probability": language_probability,
    }


@contextmanager
def _chunk_audio_for_asr(path: Path, *, chunk_duration_sec: float) -> Any:
    if chunk_duration_sec <= 0:
        raise RuntimeError("ASR chunk duration must be positive.")
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{path.stem}-asr-chunks-"))
    output_pattern = temp_dir / "chunk-%03d.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "segment",
                "-segment_time",
                str(chunk_duration_sec),
                "-reset_timestamps",
                "1",
                str(output_pattern),
            ],
            check=True,
            capture_output=True,
        )
        chunk_paths = sorted(temp_dir.glob("chunk-*.wav"))
        if not chunk_paths:
            raise RuntimeError("Audio chunking produced no output.")
        yield chunk_paths
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required for chunked ASR fallback. Please install it via Homebrew: `brew install ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        detail = stderr or str(exc)
        raise RuntimeError(f"Audio chunking failed: {detail}") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        if frame_rate <= 0:
            return 0.0
        return handle.getnframes() / float(frame_rate)


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
    progress_callback: DownloadProgressCallback | None = None,
) -> dict:
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not available. Install dependencies via `python -m pip install -r requirements.txt`."
        )
    if progress_callback is not None:
        progress_callback({"stage": "checking_cache", "model": model_size})
    availability = describe_model_availability(model_size)
    if availability["cached"]:
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "ready",
                    "model": model_size,
                    "cached_path": availability["cached_path"],
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "completed_files": 0,
                    "total_files": 0,
                }
            )
        return availability
    model_ref = _model_repo_id(model_size)
    planned_files = None
    planned_snapshot_path = None
    if hf_hub_download is not None and "/" in model_ref:
        planned_files = _plan_snapshot_download(model_ref)
    if planned_files:
        total_bytes = sum(max(int(getattr(item, "file_size", 0) or 0), 0) for item in planned_files)
        cached_files = [item for item in planned_files if not bool(getattr(item, "will_download", True))]
        pending_files = [item for item in planned_files if bool(getattr(item, "will_download", True))]
        planned_snapshot_path = str(Path(str(getattr(planned_files[0], "local_path"))).parent)
        reporter = _DownloadReporter(
            model_size=model_size,
            total_bytes=total_bytes,
            total_files=len(planned_files),
            completed_files=len(cached_files),
            downloaded_bytes=sum(max(int(getattr(item, "file_size", 0) or 0), 0) for item in cached_files),
            callback=progress_callback,
        )
        reporter.emit(
            "starting_download",
            force=True,
            pending_files=len(pending_files),
            pending_bytes=sum(max(int(getattr(item, "file_size", 0) or 0), 0) for item in pending_files),
        )
        for item in planned_files:
            file_name = str(getattr(item, "filename"))
            file_size = max(int(getattr(item, "file_size", 0) or 0), 0)
            progress_tqdm = _download_progress_tqdm_class(reporter, file_name) if getattr(item, "will_download", False) else None
            downloaded_path = hf_hub_download(
                model_ref,
                filename=file_name,
                revision=str(getattr(item, "commit_hash")),
                tqdm_class=progress_tqdm,
            )
            if planned_snapshot_path is None:
                planned_snapshot_path = str(Path(downloaded_path).parent)
            if getattr(item, "will_download", False):
                reporter.finish_file(file_name, file_size)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "finalizing",
                    "model": model_size,
                    "downloaded_bytes": reporter.downloaded_bytes,
                    "total_bytes": total_bytes,
                    "completed_files": len(planned_files),
                    "total_files": len(planned_files),
                }
            )
        model_ref = planned_snapshot_path or model_ref
    else:
        if progress_callback is not None:
            progress_callback({"stage": "starting_download", "model": model_size})
    model, _, _ = _initialize_whisper_model(model_ref, compute_type, fallback_compute_type)
    del model
    availability = describe_model_availability(model_size)
    if not availability["cached"] and planned_snapshot_path:
        snapshot_path = Path(planned_snapshot_path)
        if snapshot_path.exists():
            availability = {
                **availability,
                "cached": True,
                "cached_path": str(snapshot_path),
            }
    if not availability["cached"]:
        raise RuntimeError(f"Model '{model_size}' initialization finished, but no local cache snapshot was found afterwards.")
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "ready",
                "model": model_size,
                "cached_path": availability["cached_path"],
                "downloaded_bytes": availability.get("downloaded_bytes", 0),
                "total_bytes": availability.get("total_bytes", 0),
                "completed_files": availability.get("completed_files", 0),
                "total_files": availability.get("total_files", 0),
            }
        )
    return availability


def transcribe_file(
    path: Path,
    model_size: str = "large-v3",
    language: str | None = None,
    compute_type: str = "default",
    fallback_compute_type: str | None = "int8",
    *,
    asr_provider: str = DEFAULT_ASR_PROVIDER,
    file_strategy: str = "auto",
    chunk_duration_sec: float = 20 * 60,
) -> dict[str, Any]:
    provider = _resolve_asr_provider(asr_provider)
    normalized_strategy = _normalize_file_strategy(file_strategy)
    loaded_model = provider.load_model(
        model_size=model_size,
        compute_type=compute_type,
        fallback_compute_type=fallback_compute_type,
    )

    if normalized_strategy == "native":
        if not provider.capabilities.supports_native_file_transcription:
            raise RuntimeError(
                f"ASR provider '{provider.provider_id}' does not support native file transcription."
            )
        return provider.transcribe_file_native(
            loaded_model,
            path,
            language=language,
        )

    if normalized_strategy == "chunked":
        if not provider.capabilities.supports_chunked_fallback:
            raise RuntimeError(
                f"ASR provider '{provider.provider_id}' does not support chunked file transcription."
            )
        return _transcribe_file_chunked(
            provider,
            loaded_model,
            path,
            language=language,
            chunk_duration_sec=chunk_duration_sec,
        )

    if provider.capabilities.supports_native_file_transcription and provider.capabilities.prefers_native_file_transcription:
        return provider.transcribe_file_native(
            loaded_model,
            path,
            language=language,
        )
    if provider.capabilities.supports_chunked_fallback:
        return _transcribe_file_chunked(
            provider,
            loaded_model,
            path,
            language=language,
            chunk_duration_sec=chunk_duration_sec,
        )
    if provider.capabilities.supports_native_file_transcription:
        return provider.transcribe_file_native(
            loaded_model,
            path,
            language=language,
        )
    raise RuntimeError(f"ASR provider '{provider.provider_id}' cannot transcribe files with the current capabilities.")


def _transcribe_file_chunked(
    provider: ASRProvider,
    loaded_model: LoadedASRModel,
    path: Path,
    *,
    language: str | None,
    chunk_duration_sec: float,
) -> dict[str, Any]:
    chunk_results: list[dict[str, Any]] = []
    elapsed_offset_sec = 0.0
    with _chunk_audio_for_asr(path, chunk_duration_sec=chunk_duration_sec) as chunk_paths:
        for chunk_path in chunk_paths:
            chunk_result = provider.transcribe_file_native(
                loaded_model,
                chunk_path,
                language=language,
                time_offset_sec=elapsed_offset_sec,
            )
            chunk_results.append(chunk_result)
            elapsed_offset_sec += _wav_duration_sec(chunk_path)
    return _merge_chunk_transcriptions(chunk_results)


def transcribe(
    path: Path,
    model_size: str = "large-v3",
    language: str | None = None,
    compute_type: str = "default",
    fallback_compute_type: str | None = "int8",
    *,
    asr_provider: str = DEFAULT_ASR_PROVIDER,
    file_strategy: str = "auto",
    chunk_duration_sec: float = 20 * 60,
) -> dict[str, Any]:
    return transcribe_file(
        path,
        model_size=model_size,
        language=language,
        compute_type=compute_type,
        fallback_compute_type=fallback_compute_type,
        asr_provider=asr_provider,
        file_strategy=file_strategy,
        chunk_duration_sec=chunk_duration_sec,
    )

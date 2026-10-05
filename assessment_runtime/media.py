"""Streaming audio decoding through the packaged PyAV libraries."""
from __future__ import annotations

from pathlib import Path
import time
import wave

RATE = 16000


class MediaDecodeError(ValueError):
    pass


def pcm_blocks(path: Path, *, max_seconds: float = 1201, timeout: float = 90):
    import av

    deadline = time.monotonic() + timeout
    count = 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise MediaDecodeError("This file could not be decoded as audio.")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
            for frame in container.decode(audio=0):
                if time.monotonic() > deadline:
                    raise MediaDecodeError("Audio decoding took too long. Export a shorter MP3 or WAV and retry.")
                for output in resampler.resample(frame):
                    remaining = max(0, int(max_seconds * RATE) - count)
                    samples = min(output.samples, remaining)
                    if samples:
                        yield output.to_ndarray().astype("<i2", copy=False).tobytes()[:samples * 2]
                        count += samples
                    if count >= max_seconds * RATE:
                        return
            for output in resampler.resample(None):
                samples = min(output.samples, max(0, int(max_seconds * RATE) - count))
                if samples:
                    yield output.to_ndarray().astype("<i2", copy=False).tobytes()[:samples * 2]
                    count += samples
    except (av.error.FFmpegError, OSError) as exc:
        raise MediaDecodeError("This file could not be decoded as audio. Choose a playable recording and retry.") from exc
    if not count:
        raise MediaDecodeError("This file could not be decoded as audio. Choose a playable recording and retry.")


def write_wav(path: Path, destination: Path) -> None:
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        for block in pcm_blocks(path):
            output.writeframesraw(block)


def write_chunks(path: Path, directory: Path, seconds: float) -> list[Path]:
    chunk_bytes = int(seconds * RATE) * 2
    if chunk_bytes <= 0:
        raise MediaDecodeError("ASR chunk duration must be positive.")
    paths = []
    output = None
    used = 0
    try:
        for block in pcm_blocks(path):
            offset = 0
            while offset < len(block):
                if output is None:
                    target = directory / f"chunk-{len(paths):05d}.wav"
                    paths.append(target)
                    output = wave.open(str(target), "wb")
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(RATE)
                    used = 0
                size = min(chunk_bytes - used, len(block) - offset)
                output.writeframesraw(block[offset:offset + size])
                used += size
                offset += size
                if used == chunk_bytes:
                    output.close()
                    output = None
    finally:
        if output is not None:
            output.close()
    return paths


def validate_duration(path: Path, max_seconds: float) -> None:
    size = sum(len(block) for block in pcm_blocks(path, max_seconds=max_seconds + 1))
    if size > max_seconds * RATE * 2:
        raise MediaDecodeError("Recordings can be at most 20 minutes. Split this recording into oral practice parts and retry.")

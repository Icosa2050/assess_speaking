"""Streaming audio decoding through the packaged PyAV libraries."""
from __future__ import annotations

from pathlib import Path
import shutil
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


def write_wav(path: Path, destination: Path, *, reserve_bytes: int = 0) -> None:
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        for block in pcm_blocks(path):
            if reserve_bytes and shutil.disk_usage(destination.parent).free < reserve_bytes + len(block):
                raise MediaDecodeError("Not enough free disk space to prepare cloud audio. Your recording is retained.")
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


def write_flac_range(source: Path, destination: Path, start: float, end: float, *, reserve_bytes: int = 0) -> None:
    """Encode a range of our mono PCM derivative using packaged libraries."""
    import av
    import numpy as np

    with wave.open(str(source), "rb") as audio, av.open(str(destination), "w", format="flac") as output:
        stream = output.add_stream("flac", rate=RATE)
        stream.layout = "mono"
        first = int(start * RATE)
        remaining = min(audio.getnframes(), int(end * RATE)) - first
        audio.setpos(first)
        position = 0
        while remaining > 0:
            if reserve_bytes and shutil.disk_usage(destination.parent).free < reserve_bytes + RATE * 2:
                raise MediaDecodeError("Not enough free disk space to prepare cloud audio. Your recording is retained.")
            samples = min(remaining, RATE)
            block = audio.readframes(samples)
            if not block:
                raise MediaDecodeError("Audio range ended unexpectedly.")
            frame = av.AudioFrame.from_ndarray(np.frombuffer(block, dtype="<i2").reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = RATE
            frame.pts = position
            for packet in stream.encode(frame):
                output.mux(packet)
            position += frame.samples
            remaining -= frame.samples
        for packet in stream.encode(None):
            output.mux(packet)

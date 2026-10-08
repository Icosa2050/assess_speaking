"""Lossless, byte-bounded hosted Whisper transcription with overlap partitioning."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
import shutil
import tempfile
import time
import wave

import httpx
from assessment_runtime.media import RATE, write_wav, write_flac_range

MAX_BYTES = 24_000_000
DISK_RESERVE = 512 * 1024 * 1024
MODELS = {'whisper-large-v3', 'whisper-large-v3-turbo'}


class GroqASRError(RuntimeError):
    pass


def normalize(result: dict, offset: float = 0, keep_start: float = 0, keep_end: float = math.inf) -> dict:
    words = []
    timed_words = 0
    for word in result.get('words') or []:
        try:
            start, end = float(word['start']) + offset, float(word['end']) + offset
        except (KeyError, TypeError, ValueError):
            raise GroqASRError('Invalid transcription timestamps.') from None
        if not math.isfinite(start + end) or end < start:
            raise GroqASRError('Invalid transcription timestamps.')
        text = str(word.get('word') or '').strip()
        timed_words += bool(text)
        if keep_start <= (start + end) / 2 < keep_end:
            if text:
                words.append({'t0': start, 't1': end, 'text': text})
    text = str(result.get('text') or '').strip()
    if text and not timed_words:
        raise GroqASRError('Cloud transcription omitted word timestamps; timing-dependent assessment is unavailable.')
    language = str(result.get('language') or '').strip().lower()
    detected = {'english': 'en', 'italian': 'it', 'german': 'de', 'spanish': 'es', 'french': 'fr'}.get(language, language or None)
    return {'text': text, 'words': words, 'segment_diagnostics': [], 'compute_type_used': None,
            'compute_fallback_used': False, 'detected_language': detected, 'language_probability': None,
            'diagnostic_policy': 'groq_uncalibrated_v2'}


def transcribe(path: Path, *, api_key: str, model: str, language: str | None = None, timeout_sec: float = 120, cached=None, cancelled=lambda: False) -> dict:
    if model not in MODELS or not api_key:
        raise GroqASRError('Select a saved Groq key and supported speech model.')
    deadline = time.monotonic() + 600
    results = []
    with tempfile.TemporaryDirectory(prefix='vostavo-groq-') as tmp:
        if shutil.disk_usage(tmp).free < DISK_RESERVE + 80_000_000:
            raise GroqASRError('Free disk space to prepare cloud audio while keeping the app’s 512 MB reserve.')
        source = Path(tmp) / 'source.wav'
        write_wav(path, source, reserve_bytes=DISK_RESERVE)
        with wave.open(str(source), 'rb') as audio:
            total = audio.getnframes() / RATE
        if not 0 < total <= 1200:
            raise GroqASRError('Audio must be between zero and twenty minutes.')
        with path.open('rb') as handle:
            audio_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        with httpx.Client(timeout=timeout_sec, follow_redirects=False) as client:
            def part(start: float, end: float, index: str):
                if cancelled() or time.monotonic() >= deadline:
                    raise GroqASRError('Cloud transcription stopped. Your recording and completed chunks are retained.')
                lo, hi = max(0, start - 1), min(total, end + 1)
                target = Path(tmp) / (index + '.flac')
                write_flac_range(source, target, lo, hi, reserve_bytes=DISK_RESERVE)
                if target.stat().st_size > MAX_BYTES:
                    target.unlink()
                    if end - start < 2:
                        raise GroqASRError('Audio could not be prepared within the upload limit.')
                    mid = (start + end) / 2
                    part(start, mid, index + 'a')
                    part(mid, end, index + 'b')
                    return
                def upload():
                    data = [('model', model), ('response_format', 'verbose_json'),
                            ('timestamp_granularities[]', 'word'), ('timestamp_granularities[]', 'segment')]
                    # Let Whisper detect language, as the local assessment does.
                    # Forcing the learner's target language can hide wrong-language audio.
                    # All multipart fields use files tuples so duplicate keys are preserved.
                    fields = [(key, (None, value)) for key, value in data]
                    with target.open('rb') as handle:
                        response = client.post('https://api.groq.com/openai/v1/audio/transcriptions',
                                               headers={'Authorization': 'Bearer ' + api_key}, files=fields + [('file', (target.name, handle, 'audio/flac'))],
                                               timeout=min(timeout_sec, max(1, deadline - time.monotonic())))
                    if response.status_code == 429:
                        raise GroqASRError('Groq transcription quota exhausted. Your recording and completed chunks are retained.')
                    if response.status_code in {401, 403}:
                        raise GroqASRError('Reconnect the Groq transcription account.')
                    if response.status_code != 200:
                        raise GroqASRError('Groq transcription failed. Retry the retained recording.')
                    return normalize(response.json(), lo, start, end)
                value = cached('chunk-' + index, {'audio_sha256': audio_hash, 'model': model, 'language': 'auto', 'start': start, 'end': end, 'policy': 2}, upload) if cached else upload()
                results.append(value)
                target.unlink()
            part(0, total, '0')
    words = [word for result in results for word in result['words']]
    languages = {result['detected_language'] for result in results if result['words'] and result['detected_language']}
    detected = next((result['detected_language'] for result in results if result['words'] and result['detected_language']), None)
    return {**results[0], 'detected_language': detected if len(languages) <= 1 else None,
            'text': results[0]['text'] if len(results) == 1 else ' '.join(word['text'] for word in words),
            'words': words, 'asr_provider': 'groq', 'asr_model': model}

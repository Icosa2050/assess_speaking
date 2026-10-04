"""ASR diagnostics independent of provider/UI; explicit opt-in fails on prerequisites."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import unicodedata
import wave

import numpy as np
import pytest

import assess_speaking
from assessment_runtime.asr import describe_model_availability

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = json.loads((ROOT / "tests/fixtures/asr/sample_references.json").read_text())["cases"]


def tokens(text):
    # TTS source omits Italian accents; this checks content, not orthography.
    plain = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    return re.findall(r"[^\W\d_]+", plain)


def word_error_rate(reference, hypothesis):
    expected, actual = tokens(reference), tokens(hypothesis)
    row = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        next_row = [i]
        for j, candidate in enumerate(actual, 1):
            next_row.append(min(row[j] + 1, next_row[j-1] + 1, row[j-1] + (word != candidate)))
        row = next_row
    return row[-1] / max(1, len(expected))


def assert_timestamps(words, duration):
    previous = 0.0
    for word in words:
        start, end = float(word["t0"]), float(word["t1"])
        assert math.isfinite(start) and math.isfinite(end)
        assert 0 <= start <= end <= duration + .25
        assert start + .1 >= previous, "Word start timestamps must be ordered"
        previous = start


@pytest.fixture
def live_asr(monkeypatch):
    if os.getenv("RUN_AUDIO_INTEGRATION") != "1":
        pytest.skip("Set RUN_AUDIO_INTEGRATION=1 for live transcription checks")
    model = os.getenv("WHISPER_MODEL", "tiny")
    assert describe_model_availability(model)["cached"], f"Download Whisper {model} before running ASR integration tests"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return model


def test_reference_manifest_matches_tracked_audio():
    assert {(c["language"], c["goal"]) for c in REFERENCES} == {(lang, goal) for lang in ("en", "it") for goal in ("B1", "B2", "C1")}
    for case in REFERENCES:
        assert hashlib.sha256((ROOT / case["path"]).read_bytes()).hexdigest() == case["sha256"]
        assert len(tokens(case["text"])) > 10


@pytest.mark.parametrize("case", REFERENCES, ids=lambda c: f'{c["language"]}-{c["goal"]}')
def test_sample_transcription_content_metrics_and_timestamps(case, live_asr):
    path = ROOT / case["path"]
    features = assess_speaking.load_audio_features(path)
    result = assess_speaking.transcribe(path, model_size=live_asr)
    assert result["detected_language"] == case["language"]
    assert word_error_rate(case["text"], result["text"]) <= .35, result["text"]
    assert result["words"]
    assert_timestamps(result["words"], features["duration_sec"])
    metrics = assess_speaking.metrics_from(result["words"], features, language_code=case["language"])
    with wave.open(str(path)) as audio:
        assert metrics["duration_sec"] == pytest.approx(audio.getnframes()/audio.getframerate(), abs=.02)
    assert metrics["word_count"] > 10
    assert metrics["wpm"] > 10
    assert metrics["speaking_time_sec"] <= metrics["duration_sec"] + .02


@pytest.mark.parametrize("filename", ["test1.m4a", "test2.m4a"])
def test_longer_m4a_decoding_and_word_timing(filename, live_asr):
    path = ROOT / "tests/audio" / filename
    # Match the production pipeline: Praat consumes decoded WAV, not the M4A container.
    decoded = assess_speaking._convert_to_wav(path)
    try:
        features = assess_speaking.load_audio_features(decoded)
        assert features["duration_sec"] > 60
        result = assess_speaking.transcribe(decoded, model_size=live_asr)
        assert len(tokens(result["text"])) > 20
        assert result["words"]
        assert_timestamps(result["words"], features["duration_sec"])
    finally:
        decoded.unlink(missing_ok=True)


def write_wav(path, samples):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(np.clip(samples, -32768, 32767).astype("<i2").tobytes())


@pytest.mark.parametrize("language", ["en", "it"])
def test_moderate_seeded_noise_preserves_sample_content(language, live_asr, tmp_path):
    case = next(c for c in REFERENCES if c["language"] == language and c["goal"] == "B1")
    with wave.open(str(ROOT / case["path"])) as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(float)
    rms = np.sqrt(np.mean(samples ** 2))
    noise = np.random.default_rng(42).normal(0, rms / 10, len(samples))  # approximately 20 dB SNR
    path = tmp_path / "noisy.wav"
    write_wav(path, samples + noise)
    result = assess_speaking.transcribe(path, model_size=live_asr)
    assert result["detected_language"] == language
    assert word_error_rate(case["text"], result["text"]) <= .4, result["text"]
    assert result["words"]
    assert_timestamps(result["words"], len(samples) / 16000)


def test_silence_does_not_invent_speech(live_asr, tmp_path):
    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(16000 * 5))
    result = assess_speaking.transcribe(path, model_size=live_asr)
    assert not result["text"].strip()
    assert not result["words"]

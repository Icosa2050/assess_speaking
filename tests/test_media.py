from pathlib import Path
import wave
from unittest.mock import patch
import pytest
from assessment_runtime.media import write_chunks, write_wav, validate_duration, MediaDecodeError

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / 'tests/fixtures/audio_synthetic'


@pytest.mark.parametrize('filename', [
    'tone.wav', 'tone.m4a', 'tone.mp4', 'tone.aac', 'tone.mp3', 'tone.webm',
    'tone.ogg', 'tone.opus', 'tone.flac', 'tone.aiff', 'tone.aif',
    'tone-vorbis.ogg', 'tone-vorbis.webm', 'tone-alac.m4a',
])
def test_synthetic_accepted_audio_decodes_without_external_ffmpeg(tmp_path, filename):
    source = SYNTHETIC / filename
    output = tmp_path / 'decoded.wav'
    with patch.dict('os.environ', {'PATH': ''}):
        write_wav(source, output)
    with wave.open(str(output)) as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 16000
        assert 10000 <= audio.getnframes() <= 14000


def test_packaged_decoder_chunks_exact_samples_without_ffmpeg(tmp_path):
    sample = ROOT / 'samples/cefr/en/B1/travel_story.wav'
    with patch.dict('os.environ', {'PATH': ''}):
        chunks = write_chunks(sample, tmp_path, 5)
        output = tmp_path / 'complete.wav'
        write_wav(sample, output)
    sizes = []
    for path in chunks:
        with wave.open(str(path)) as chunk:
            assert (chunk.getnchannels(), chunk.getsampwidth(), chunk.getframerate()) == (1, 2, 16000)
            sizes.append(chunk.getnframes())
    assert len(sizes) > 1
    assert sizes[:-1] == [80000] * (len(sizes) - 1)
    with wave.open(str(output)) as wav:
        assert sum(sizes) == wav.getnframes()


def test_decoder_handles_m4a_and_duration_limit(tmp_path):
    sample = ROOT / 'tests/audio/test1.m4a'
    output = tmp_path / 'decoded.wav'
    write_wav(sample, output)
    with wave.open(str(output)) as wav:
        assert wav.getnframes() > 0
    with pytest.raises(MediaDecodeError, match='at most'):
        validate_duration(sample, 1)
    corrupt = tmp_path / 'bad.mp3'
    corrupt.write_bytes(b'not audio')
    with pytest.raises(MediaDecodeError, match='could not be decoded'):
        write_wav(corrupt, output)

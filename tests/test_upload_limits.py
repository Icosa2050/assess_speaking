import asyncio
import hashlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_backend.jobs import JobManager
from app_backend.uploads import DISK_RESERVE_BYTES, UploadRejected, parse_upload


@pytest.fixture
def config(tmp_path):
    return build_backend_runtime_config(app_data_dir=tmp_path, port=8870)


def test_upload_stream_copy_hash_and_limit(config):
    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            return super().read(size)
    manager = JobManager(config)
    audio = b'a' * (2 * 1024 * 1024 + 3)
    result = manager.register_upload_stream(source=BoundedReader(audio), filename='clip.wav')
    assert result.sha1 == hashlib.sha1(audio).hexdigest()
    assert len(list(config.app_data.uploads_dir.iterdir())) == 2
    with patch('app_backend.jobs.MAX_UPLOAD_BYTES', 10):
        with pytest.raises(UploadRejected):
            manager.register_upload_stream(source=BoundedReader(audio), filename='clip.wav')
    assert len(list(config.app_data.uploads_dir.iterdir())) == 2


def test_failed_metadata_write_removes_audio(config):
    manager = JobManager(config)
    with patch('app_backend.jobs._write_json', side_effect=OSError('disk full')):
        with pytest.raises(OSError):
            manager.register_upload(data=b'audio', filename='a.wav')
    assert list(config.app_data.uploads_dir.iterdir()) == []


def test_upload_low_disk_and_retry(config):
    client = TestClient(create_app(config))
    with patch('app_backend.uploads.shutil.disk_usage', return_value=SimpleNamespace(free=DISK_RESERVE_BYTES-1)):
        assert client.get('/v1/uploads/limits').json()['available_bytes'] == 0
        response = client.post('/v1/uploads', files={'file': ('clip.wav', b'audio')})
        assert response.status_code == 507
        assert response.json()['detail']['code'] == 'storage_error'
    assert list(config.app_data.uploads_dir.iterdir()) == []
    assert client.post('/v1/uploads', files={'file': ('clip.wav', b'audio')}).status_code == 200
    assert client.get('/v1/health').status_code == 200


def test_oversized_and_extra_files_are_rejected(config):
    client = TestClient(create_app(config))
    with patch('app_backend.uploads.MAX_UPLOAD_BYTES', 8):
        response = client.post('/v1/uploads', files={'file': ('clip.wav', b'123456789')})
        assert response.status_code == 413
    response = client.post('/v1/uploads', files=[('file', ('a.wav', b'1')), ('file', ('b.wav', b'2'))])
    assert response.status_code == 400
    assert list(config.app_data.uploads_dir.iterdir()) == []


def test_untrusted_or_absent_content_length_still_bounded(config):
    async def run():
        chunks = iter([b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n', b'a' * 70000])
        async def receive():
            return {'type': 'http.request', 'body': next(chunks), 'more_body': True}
        request = Request({'type':'http', 'method':'POST', 'path':'/v1/uploads', 'headers':[(b'content-type', b'multipart/form-data; boundary=x')]}, receive)
        with patch('app_backend.uploads.MAX_UPLOAD_BYTES', 10):
            with pytest.raises(UploadRejected):
                await parse_upload(request, config.app_data.uploads_dir)
    asyncio.run(run())


def test_audio_duration_is_checked_before_expensive_inference(tmp_path):
    import wave
    from app_backend.uploads import validate_audio_duration
    clip = tmp_path / 'clip.wav'
    with wave.open(str(clip), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\0' * 16000 * 2 * 2)
    validate_audio_duration(clip)
    with patch('app_backend.uploads.MAX_AUDIO_SECONDS', 1):
        with pytest.raises(ValueError, match='at most'):
            validate_audio_duration(clip)
    clip.write_bytes(b'not audio')
    with pytest.raises(ValueError, match='could not be decoded'):
        validate_audio_duration(clip)


def test_concurrent_assessment_is_refused_without_starting_worker(config):
    from app_backend.jobs import AssessmentBusyError
    from app_backend.contracts import AssessmentCreateRequest
    from unittest.mock import Mock
    manager = JobManager(config)
    process = Mock()
    process.is_alive.return_value = True
    manager._processes['existing'] = process
    with pytest.raises(AssessmentBusyError):
        manager.submit(AssessmentCreateRequest(audio_id='not-even-looked-up', whisper='small', provider='ollama',
            llm_model='test', expected_language='en', feedback_language='en', speaker_id='test', task_family='generic', theme='test', target_duration_sec=90))
    assert list(config.jobs_dir.iterdir()) == []


def test_copy_does_not_reserve_spool_space_twice(config):
    # The spool is already present: 60 MiB above reserve can fit a 40 MiB copy.
    from app_backend.uploads import ensure_copy_space
    with patch('app_backend.uploads.shutil.disk_usage', return_value=SimpleNamespace(free=DISK_RESERVE_BYTES+60*1024*1024)):
        ensure_copy_space(config.app_data.uploads_dir, 40*1024*1024)
        with pytest.raises(UploadRejected) as caught:
            ensure_copy_space(config.app_data.uploads_dir, 61*1024*1024)
        assert caught.value.status == 507


def test_missing_decoder_has_actionable_error(tmp_path):
    import signal
    from app_backend.uploads import validate_audio_duration
    previous = signal.getsignal(signal.SIGTERM)
    def fail_launch(*args, **kwargs):
        assert signal.getsignal(signal.SIGTERM) != previous
        raise FileNotFoundError
    with patch('app_backend.uploads.subprocess.Popen', side_effect=fail_launch):
        with pytest.raises(RuntimeError, match='bundled audio decoder is unavailable'):
            validate_audio_duration(tmp_path / 'clip.wav')
    assert signal.getsignal(signal.SIGTERM) == previous


def test_decoder_is_killed_when_validation_is_cancelled(tmp_path):
    import signal
    from unittest.mock import Mock
    from app_backend.uploads import validate_audio_duration
    decoder = Mock()
    decoder.wait.side_effect = [SystemExit(143), 0]
    decoder.poll.return_value = None
    previous = signal.getsignal(signal.SIGTERM)
    with patch('app_backend.uploads.subprocess.Popen', return_value=decoder):
        with pytest.raises(SystemExit):
            validate_audio_duration(tmp_path / 'clip.wav')
    decoder.kill.assert_called_once()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_disconnect_closes_partial_spool(config):
    import tempfile
    from starlette.requests import ClientDisconnect
    created = []
    real_spool = tempfile.SpooledTemporaryFile
    def tracked_spool(*args, **kwargs):
        file = real_spool(*args, **kwargs)
        created.append(file)
        return file
    async def run():
        count = 0
        async def receive():
            nonlocal count
            count += 1
            if count == 1:
                return {'type':'http.request', 'body':b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\npartial', 'more_body':True}
            return {'type':'http.disconnect'}
        request = Request({'type':'http', 'method':'POST', 'path':'/v1/uploads', 'headers':[(b'content-type', b'multipart/form-data; boundary=x')]}, receive)
        with pytest.raises(ClientDisconnect):
            await parse_upload(request, config.app_data.uploads_dir)
    with patch('starlette.formparsers.SpooledTemporaryFile', side_effect=tracked_spool):
        asyncio.run(run())
    assert created and all(file.closed for file in created)
    assert list(config.app_data.uploads_dir.iterdir()) == []

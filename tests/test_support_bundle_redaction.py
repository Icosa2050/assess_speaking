import json
import zipfile
from app_backend.config import build_backend_runtime_config
from app_backend.contracts import SupportBundleCreateRequest
from app_backend.support_bundle import create_support_bundle


def test_all_default_diagnostics_and_nested_text_are_redacted_and_media_is_opt_in(tmp_path):
    config=build_backend_runtime_config(app_data_dir=tmp_path,cache_dir=tmp_path/'cache',port=8765)
    private='sk-fixture-private1234'
    config.log_file.parent.mkdir(parents=True,exist_ok=True)
    config.log_file.write_text(f'Authorization: Bearer {private}\nfailed at /Users/fixture-owner/private/project.py\n')
    config.state_file.write_text(json.dumps({'nested':{'trace':f'/home/fixture-owner/private {private}', 'api_key':private,'secret_ref':'private-ref'}}))
    config.app_data.recordings_dir.mkdir(parents=True,exist_ok=True)
    (config.app_data.recordings_dir/'learner-private.wav').write_bytes(b'fixture-only')
    config.jobs_dir.mkdir(parents=True,exist_ok=True)
    (config.jobs_dir/'learner-private.json').write_text(json.dumps({'status':'completed','payload':{'report':{'transcript_preview':'PRIVATE LEARNER TEXT'}},'request_metadata':{'speaker_id':'PRIVATE LEARNER','prompt_text':'PRIVATE LEARNER TEXT','provider':'ollama'}}))
    created=create_support_bundle(config,SupportBundleCreateRequest(client_snapshot={'microphone_settings':{'sampleRate':48000}}))
    with zipfile.ZipFile(config.app_data.temp_dir/'support-bundles'/f'{created.bundle_id}.zip') as archive:
        assert archive.testzip() is None
        contents='\n'.join(archive.read(name).decode() for name in archive.namelist())
        for forbidden in [private,'fixture-owner','private-ref','PRIVATE LEARNER',str(tmp_path)]: assert forbidden not in contents
        assert not any(name.startswith(('reports/','recordings/','uploads/')) for name in archive.namelist())
    included=create_support_bundle(config,SupportBundleCreateRequest(include_recordings=True))
    with zipfile.ZipFile(config.app_data.temp_dir/'support-bundles'/f'{included.bundle_id}.zip') as archive:
        assert 'learner-private' not in str(archive.namelist())
        audio=next(name for name in archive.namelist() if name.startswith('recordings/'))
        assert archive.read(audio)==b'fixture-only'

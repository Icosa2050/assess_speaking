"""Destination acceptance must fail before work and remain stable after publication."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from app_backend.app import create_app
from app_backend.cloud_runtime import CloudRuntime
from app_backend.config import build_backend_runtime_config
from app_backend.contracts import AssessmentCreateResponse, JobStatus
from app_backend.sharing import assessment_route
from app_core.cloud_policy import CloudSettings
from app_core.preference_lock import PREFERENCES_LOCK
from app_core.state import AppPreferences, ProviderConnection
from app_core.runtime_connections import serialize_connections
from assessment_runtime.theme_library import save_workspace_prefs


def connection(provider='ollama', **changes):
    return ProviderConnection(**{'connection_id': 'primary', 'provider_kind': provider,
                                 'default_model': 'fixture-model', 'base_url': 'http://localhost:11434/v1', **changes})


def state(*connections):
    return SimpleNamespace(prefs=AppPreferences(connections=list(connections), active_connection_id=connections[0].connection_id if connections else ''))


@pytest.mark.parametrize('endpoint,local', [('http://localhost:11434',True),('http://127.0.0.1:11434',True),('http://[::1]:11434',True),('http://192.168.1.10:11434',False),('https://models.example.test/v1',False)])
def test_endpoint_locality_comes_from_actual_host(endpoint, local):
    route=assessment_route(state(connection(base_url=endpoint)),CloudSettings(),{})
    assert route['available'] and route['audio']['local'] is True
    assert route['analysis']['local'] is local
    assert len(route['fingerprint'])==64


def test_fingerprint_tracks_routes_but_not_authentication_or_model_catalog_refresh():
    primary=connection('chatgpt',base_url='',secret_ref='secret-private',provider_metadata={'models':[{'slug':'fixture-model'}]})
    speech=connection('groq',connection_id='speech',base_url='',secret_ref='speech-private')
    fallback=connection('openrouter',connection_id='fallback',base_url='',secret_ref='fallback-private')
    current=state(primary,speech,fallback)
    settings=CloudSettings(asr_provider='groq',asr_connection_id='speech',paid_fallback_enabled=True,fallback_connection_id='fallback',openrouter_modes={'fallback':'paid'})
    before=assessment_route(current,settings,{})
    assert before['available'] and before['audio']['host']=='api.groq.com'
    assert before['analysis']['host']=='api.openai.com' and before['fallback']['host']=='openrouter.ai'
    assert 'private' not in json.dumps(before)
    primary.provider_metadata={'models':[{'slug':'fixture-model'}, {'slug':'another'}], 'expires_at':42}
    primary.secret_ref='refreshed-private'
    assert assessment_route(current,settings,{})['fingerprint']==before['fingerprint']
    settings.paid_fallback_enabled=False
    assert assessment_route(current,settings,{})['fingerprint']!=before['fingerprint']


@pytest.mark.parametrize('changes', [{'base_url':'https://user:password@example.test/v1'}, {'base_url':'https://example.test/v1?key=private'}, {'default_model':'openrouter/auto'}, {'provider_kind':'unknown'}])
def test_unknown_or_secret_bearing_routes_are_unavailable(changes):
    route=assessment_route(state(connection(**changes)),CloudSettings(),{})
    assert not route['available'] and route['fingerprint']==''
    assert 'password' not in json.dumps(route) and 'private' not in json.dumps(route)


def test_path_changes_invalidate_acceptance_without_disclosing_paths():
    conn=connection('openai_compatible',base_url='https://example.test/private-model/v1')
    current=state(conn)
    before=assessment_route(current,CloudSettings(),{})
    assert 'private-model' not in json.dumps(before)
    conn.base_url='https://example.test/other/v1'
    assert assessment_route(current,CloudSettings(),{})['fingerprint']!=before['fingerprint']


def test_cloud_jobs_copy_settings_and_connections(tmp_path):
    primary=connection('groq',base_url='',secret_ref='private-ref')
    current=state(primary)
    settings=CloudSettings()
    runtime=CloudRuntime(tmp_path,current,lambda _: 'fixture-only',settings=settings)
    primary.default_model='changed';settings.asr_provider='groq'
    assert runtime.primary.default_model=='fixture-model'
    assert runtime.settings.asr_provider=='local'


def assessment_body():
    return {"audio_id":"fixture","provider":"ollama","llm_model":"fixture-model","whisper":"small","expected_language":"en","feedback_language":"en","speaker_id":"fixture","task_family":"free_monologue","theme":"fixture","target_duration_sec":90,"target_cefr":"B1"}

def client_with_local_route(tmp_path):
    config=build_backend_runtime_config(app_data_dir=tmp_path,cache_dir=tmp_path/'cache',port=8765)
    conn=connection()
    save_workspace_prefs(config.app_data.reports_dir, {'provider':'ollama','model':'fixture-model','active_connection_id':'primary','connections':serialize_connections([conn]),'whisper_model':'small','setup_complete':True})
    app=create_app(config)
    return app,TestClient(app)


def test_api_rejects_missing_and_changed_acceptance_before_publication(tmp_path):
    app,client=client_with_local_route(tmp_path)
    route=client.post('/v1/assessment-route',json={'provider':'ollama','llm_model':'fixture-model','whisper':'small'}).json()
    assert route['available']
    body=assessment_body()
    with patch.object(app.state.job_manager,'submit') as submit:
        for receipt in ['', 'b'*64]:
            response=client.post('/v1/assessments',json={**body,'sharing_fingerprint':receipt})
            assert response.status_code==409 and response.json()['detail']['code']=='sharing_changed'
        submit.assert_not_called()
        submit.return_value=AssessmentCreateResponse(assessment_id='fixture',status=JobStatus.QUEUED)
        assert client.post('/v1/assessments',json={**body,'sharing_fingerprint':route['fingerprint']}).status_code==200
        submit.assert_called_once()


def test_preferences_cannot_change_between_acceptance_and_job_publication(tmp_path):
    app,client=client_with_local_route(tmp_path)
    route=client.get('/v1/runtime/sharing').json()
    entered=Event();release=Event();writer_entered=Event();writer_started=Event()
    def publish(*args,**kwargs):
        entered.set()
        assert release.wait(3)
        assert not writer_entered.is_set()
        return AssessmentCreateResponse(assessment_id='fixture',status=JobStatus.QUEUED)
    def writer():
        writer_started.set()
        with PREFERENCES_LOCK: writer_entered.set()
    with patch.object(app.state.job_manager,'submit',side_effect=publish),ThreadPoolExecutor(2) as pool:
        request=pool.submit(client.post,'/v1/assessments',json={**assessment_body(),'sharing_fingerprint':route['fingerprint']})
        assert entered.wait(3)
        write=pool.submit(writer)
        assert writer_started.wait(3)
        assert not writer_entered.is_set()
        release.set()
        assert request.result(timeout=3).status_code==200
        write.result(timeout=3)
    assert writer_entered.is_set()

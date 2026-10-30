from pathlib import Path
from types import SimpleNamespace
import csv
import json
import zipfile

import pytest

from app_backend.journal import Journal, JournalError, private_json, read_history, write_history


def journal(root):
    reports = root / 'reports'
    uploads = reports / 'uploads'
    recordings = reports / 'recordings'
    jobs = root / 'jobs'
    for path in (reports, uploads, recordings, jobs):
        path.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(app_data=SimpleNamespace(root=root, reports_dir=reports, uploads_dir=uploads, recordings_dir=recordings), jobs_dir=jobs)
    return Journal(config, SimpleNamespace(_processes={}))


def seed(j):
    audio = j.config.app_data.uploads_dir / 'shared.wav'
    audio.write_bytes(b'synthetic waveform')
    rows = []
    for index in range(2):
        report = j.config.app_data.reports_dir / f'report-{index}.json'
        private_json(report, {'session_id': f'session{index}', 'meta': {'audio_path': str(audio)},
                             'input': {'api_key': 'do-not-export', 'llm_base_url': 'https://private.example'},
                             'eligibility': {'status': 'unverified'}, 'transcript': {'text': 'Synthetic fixture.'}})
        rows.append({'session_id': f'session{index}', 'report_path': str(report), 'timestamp': '2026-10-08'})
    from assess_speaking import HISTORY_FIELDNAMES
    write_history(j.config.app_data.reports_dir / 'history.csv', HISTORY_FIELDNAMES, [{field: row.get(field, "") for field in HISTORY_FIELDNAMES} for row in rows])


EMPTY = {'sessions': [], 'recordings': []}


def backup(j):
    tx = j.begin()['id']
    result = j.export(tx, EMPTY)
    j.abort(tx)
    return j.directory / f"backup_{result['id']}.zip"


def test_cross_root_roundtrip_deduplicates_audio_and_preserves_eligibility(tmp_path):
    first, second = journal(tmp_path / 'first'), journal(tmp_path / 'second')
    seed(first)
    drafts = first.config.app_data.root / 'drafts' / 'v1'
    drafts.mkdir(parents=True)
    (drafts / ('a' * 32 + '.manifest')).write_bytes(b'private-draft-manifest')
    (drafts / ('a' * 32 + '-' + 'b' * 32 + '.media')).write_bytes(b'private-unsent-take')
    key_bytes = bytes(range(32))
    key_service = b'com.vostavo.desktop.drafts.v1'
    key_account = b'content-key'
    (drafts / 'key-material').write_bytes(b'private-key-material' + key_bytes + key_service + key_account)
    path = backup(first)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert len(manifest['files']) == 1
        assert str(tmp_path) not in archive.read('manifest.json').decode()
        assert 'do-not-export' not in archive.read('manifest.json').decode()
        assert not any(name.startswith('drafts/') for name in archive.namelist())
        assert all(token not in archive.read(name) for name in archive.namelist()
                   for token in (b'private-draft-manifest', b'private-unsent-take', b'private-key-material',
                                 key_bytes, key_service, key_account))
    tx = second.begin()['id']
    staged = second.stage(tx, path)
    assert staged['attempts'] == 2
    assert staged['restored_attempt_ids'] == ['session0', 'session1']
    second.commit(tx)
    second.complete(tx)
    _, rows = read_history(second.config.app_data.reports_dir / 'history.csv')
    payloads = [json.loads(Path(row['report_path']).read_text()) for row in rows]
    assert payloads[0]['meta']['audio_path'] == payloads[1]['meta']['audio_path']
    assert Path(payloads[0]['meta']['audio_path']).read_bytes() == b'synthetic waveform'
    assert payloads[0]['eligibility']['status'] == 'unverified'
    assert payloads[0]['restored_from']['jobs_resumable'] is False
    assert isinstance(payloads[0]['restored_from']['restored_at_ms'], int)
    assert not list(second.config.jobs_dir.iterdir())


def test_staged_and_committed_rollback_restores_original_index(tmp_path):
    first, second = journal(tmp_path / 'first'), journal(tmp_path / 'second')
    seed(first)
    original = second.config.app_data.reports_dir / 'history.csv'
    from assess_speaking import HISTORY_FIELDNAMES
    write_history(original, HISTORY_FIELDNAMES, [{field: {'session_id': 'original', 'timestamp': '2025-01-01'}.get(field, '') for field in HISTORY_FIELDNAMES}])
    before = original.read_bytes()
    tx = second.begin()['id']
    second.stage(tx, backup(first))
    second.commit(tx)
    restarted = journal(tmp_path / 'second')
    assert restarted.state()['phase'] == 'backend_committed'
    with pytest.raises(JournalError):
        restarted.begin_mutation()
    restarted.abort(tx)
    assert original.read_bytes() == before
    assert not list(second.config.app_data.uploads_dir.glob('restored_*'))


def test_collision_and_missing_audio_leave_originals_untouched(tmp_path):
    j = journal(tmp_path)
    seed(j)
    path = backup(j)
    tx = j.begin()['id']
    assert j.stage(tx, path)['skipped_attempts'] == ['session0', 'session1']
    j.abort(tx)
    (j.config.app_data.uploads_dir / 'shared.wav').unlink()
    tx = j.begin()['id']
    with pytest.raises(JournalError, match='missing'):
        j.export(tx, EMPTY)
    result = j.export(tx, EMPTY, allow_missing=True)
    assert len(result['missing']) == 2
    j.abort(tx)


@pytest.mark.parametrize('damage', ['path', 'duplicate', 'checksum', 'unlisted', 'format'])
def test_rejects_invalid_archives_before_any_publication(tmp_path, damage):
    source, target = journal(tmp_path / 'source'), journal(tmp_path / 'target')
    seed(source)
    original = backup(source)
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(original) as archive:
        content = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(content['manifest.json'])
    if damage == 'checksum':
        content[next(name for name in content if name != 'manifest.json')] = b'wrong'
    elif damage == 'unlisted':
        manifest['attempts'][0]['report']['meta']['audio_path'] = {'$media': 'media/' + '0' * 64 + '.wav'}
    elif damage == 'format':
        manifest['version'] = 999
    content['manifest.json'] = json.dumps(manifest).encode()
    with zipfile.ZipFile(bad, 'w') as archive:
        for name, value in content.items():
            archive.writestr(name, value)
        if damage == 'path':
            archive.writestr('../outside', 'no')
        if damage == 'duplicate':
            archive.writestr('manifest.json', content['manifest.json'])
    tx = target.begin()['id']
    with pytest.raises((JournalError, ValueError)):
        target.stage(tx, bad)
    assert not (target.config.app_data.reports_dir / 'history.csv').exists()
    target.abort(tx)


def test_alive_workers_and_direct_mutations_exclude_maintenance(tmp_path):
    j = journal(tmp_path)
    j.jobs._processes['alive'] = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(JournalError):
        j.begin()
    j.jobs._processes.clear()
    guard = j.begin_mutation()
    with pytest.raises(JournalError):
        j.begin()
    j.end_mutation(guard)
    tx = j.begin()['id']
    with pytest.raises(JournalError):
        j.begin_mutation()
    j.abort(tx)


def test_archive_undo_retains_shared_media_and_reports(tmp_path):
    j = journal(tmp_path)
    seed(j)
    assert j.archive_attempt('session0')['retained_media'] is True
    assert len(read_history(j.config.app_data.reports_dir / 'history.csv')[1]) == 1
    assert (j.config.app_data.uploads_dir / 'shared.wav').exists()
    path = backup(j)
    with zipfile.ZipFile(path) as archive:
        assert sum(item['archived'] for item in json.loads(archive.read('manifest.json'))['attempts']) == 1
    j.archive_attempt('session0', undo=True)
    assert len(read_history(j.config.app_data.reports_dir / 'history.csv')[1]) == 2


def test_failed_csv_archive_write_rolls_back_both_indexes(tmp_path, monkeypatch):
    import app_backend.journal as module
    j = journal(tmp_path)
    seed(j)
    history = j.config.app_data.reports_dir / 'history.csv'
    before = history.read_bytes()
    original = module._write_history
    calls = 0
    def fail_once(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError('Injected disk failure')
        return original(*args)
    monkeypatch.setattr(module, '_write_history', fail_once)
    with pytest.raises(OSError):
        j.archive_attempt('session0')
    assert history.read_bytes() == before
    assert not json.loads((j.directory / 'trash.json').read_text())


def test_restart_during_file_publication_rolls_back_before_admitting_writers(tmp_path):
    source, destination = journal(tmp_path / 'source'), journal(tmp_path / 'destination')
    seed(source)
    tx = destination.begin()['id']
    destination.stage(tx, backup(source))
    destination.commit(tx)
    state = destination.state(); state['phase'] = 'backend_committing'; destination.save(state)
    restarted = journal(tmp_path / 'destination')
    assert restarted.state() is None
    assert not (restarted.config.app_data.reports_dir / 'history.csv').exists()
    guard = restarted.begin_mutation(); restarted.end_mutation(guard)


def test_restore_keeps_the_canonical_history_schema_for_future_analysis(tmp_path):
    from assess_speaking import HISTORY_FIELDNAMES, append_history
    source, target = journal(tmp_path / 'source'), journal(tmp_path / 'target')
    seed(source)
    tx = target.begin()['id']; target.stage(tx, backup(source)); target.commit(tx); target.complete(tx)
    history = target.config.app_data.reports_dir / 'history.csv'
    assert read_history(history)[0] == HISTORY_FIELDNAMES
    append_history(history, {'session_id': 'new-analysis'})
    assert len(read_history(history)[1]) == 3
    row = read_history(history)[1][0]
    assert json.loads(Path(row['report_path']).read_text())['report_path'] == row['report_path']


def test_completion_receipt_survives_a_crash_before_phase_publication(tmp_path):
    source, target = journal(tmp_path / 'source'), journal(tmp_path / 'target')
    seed(source)
    tx = target.begin()['id']; target.stage(tx, backup(source)); target.commit(tx)
    private_json(target.directory / 'completed.json', [tx])
    restarted = journal(tmp_path / 'target')
    assert restarted.state() is None
    assert len(read_history(restarted.config.app_data.reports_dir / 'history.csv')[1]) == 2
    assert (restarted.directory / 'completed.json').exists()


def test_rollback_preserves_a_foreign_history_append(tmp_path):
    from assess_speaking import append_history
    source, target = journal(tmp_path / 'source'), journal(tmp_path / 'target')
    seed(source)
    tx = target.begin()['id']; target.stage(tx, backup(source)); target.commit(tx)
    append_history(target.config.app_data.reports_dir / 'history.csv', {'session_id': 'foreign-append'})
    target.abort(tx)
    assert [row['session_id'] for row in read_history(target.config.app_data.reports_dir / 'history.csv')[1]] == ['foreign-append']


def test_corrupt_recovery_metadata_keeps_history_readable_and_blocks_writes(tmp_path):
    j = journal(tmp_path); seed(j)
    j.state_path.write_text('{broken')
    restarted = journal(tmp_path)
    assert restarted.state()['phase'] == 'recovery_error'
    assert len(read_history(restarted.config.app_data.reports_dir / 'history.csv')[1]) == 2
    with pytest.raises(JournalError, match='damaged'):
        restarted.begin_mutation()


def test_cross_process_lease_protects_cli_and_api_writers(tmp_path):
    from app_core.journal_lock import journal_guard, JournalMaintenanceError
    j = journal(tmp_path)
    with journal_guard(tmp_path):
        with pytest.raises(JournalError, match='another process'):
            j.begin()
    tx = j.begin()['id']
    with pytest.raises(JournalMaintenanceError, match='in progress'):
        with journal_guard(tmp_path):
            pass
    j.abort(tx)


def test_purge_protects_shared_media_and_existing_exports(tmp_path):
    j = journal(tmp_path); seed(j)
    saved = backup(j); before = saved.read_bytes()
    j.archive_attempt('session0')
    preview = j.purge_preview('session0')
    assert len(preview['files']) == 1 and preview['retained_files'] == 1
    assert j.purge_attempt('session0')['removed_files'] == 1
    assert (j.config.app_data.uploads_dir / 'shared.wav').exists()
    assert saved.read_bytes() == before
    assert (j.config.app_data.reports_dir / 'report-1.json').exists()
    with pytest.raises(JournalError):
        j.archive_attempt('session0', undo=True)


@pytest.mark.parametrize('failure', ['move', 'index'])
def test_failed_purge_restores_files_and_undo_index(tmp_path, monkeypatch, failure):
    import app_backend.journal as module
    j = journal(tmp_path); seed(j); j.archive_attempt('session0')
    if failure == 'move':
        original = module.os.replace
        def fail_once(source, target):
            if '.purge_' in Path(target).name:
                monkeypatch.setattr(module.os, 'replace', original)
                raise OSError('Injected move failure')
            return original(source, target)
        monkeypatch.setattr(module.os, 'replace', fail_once)
    else:
        original = module.private_json
        def fail_once(path, value):
            if Path(path).name == 'trash.json':
                monkeypatch.setattr(module, 'private_json', original)
                raise OSError('Injected index failure')
            return original(path, value)
        monkeypatch.setattr(module, 'private_json', fail_once)
    with pytest.raises(OSError):
        j.purge_attempt('session0')
    assert (j.config.app_data.reports_dir / 'report-0.json').exists()
    assert 'session0' in json.loads((j.directory / 'trash.json').read_text())
    j.archive_attempt('session0', undo=True)


def test_purge_refuses_retained_recovery_jobs(tmp_path):
    j = journal(tmp_path); seed(j); j.archive_attempt('session0')
    private_json(j.config.jobs_dir / 'asmt_fixture.json', {'assessment_id': 'asmt_fixture', 'status': 'failed', 'payload': {'report': {'session_id': 'session0'}}})
    with pytest.raises(JournalError, match='recovery'):
        j.purge_preview('session0')


def test_selective_restore_recovers_missing_attempt_without_overwriting_existing(tmp_path):
    source, target = journal(tmp_path / 'source'), journal(tmp_path / 'target')
    seed(source); seed(target)
    target.archive_attempt('session1')
    target.purge_attempt('session1')
    original = (target.config.app_data.reports_dir / 'report-0.json').read_bytes()
    tx = target.begin()['id']; preview = target.stage(tx, backup(source))
    assert preview['attempts'] == 1 and preview['skipped_attempts'] == ['session0']
    target.commit(tx); target.complete(tx)
    assert (target.config.app_data.reports_dir / 'report-0.json').read_bytes() == original
    assert {row['session_id'] for row in read_history(target.config.app_data.reports_dir / 'history.csv')[1]} == {'session0', 'session1'}


def test_purge_removes_all_report_copies_completed_jobs_and_owned_cache(tmp_path):
    j = journal(tmp_path); seed(j)
    report = read_json_fixture(j.config.app_data.reports_dir / 'report-0.json')
    private_json(j.config.app_data.reports_dir / 'older-revision.json', report)
    from assess_speaking import append_session_jsonl
    sessions = j.config.app_data.reports_dir / 'sessions.jsonl'
    append_session_jsonl(sessions, report)
    retained = read_json_fixture(j.config.app_data.reports_dir / 'report-1.json')
    append_session_jsonl(sessions, retained)
    cache = j.config.jobs_dir / 'asmt_fixture-stages'; cache.mkdir(); (cache / 'asr.json').write_text('{"fixture":true}')
    job = j.config.jobs_dir / 'asmt_fixture.json'
    private_json(job, {'status': 'completed', 'payload': {'report': {'session_id': 'session0'}}, 'stage_cache_dir': str(cache)})
    j.archive_attempt('session0'); preview = j.purge_preview('session0'); j.purge_attempt('session0', preview['fingerprint'])
    assert not job.exists() and not cache.exists()
    assert not (j.config.app_data.reports_dir / 'older-revision.json').exists()
    assert list(j._jsonl_rows(sessions)) == [(json.dumps(retained, ensure_ascii=False) + '\n', retained)]
    assert (j.config.app_data.uploads_dir / 'shared.wav').exists()


def read_json_fixture(path):
    return json.loads(path.read_text())


def test_uploaded_audio_registration_does_not_prevent_removal(tmp_path):
    j = journal(tmp_path); seed(j)
    audio = j.config.app_data.uploads_dir / ('aud_' + 'a' * 32 + '.ogg'); audio.write_bytes(b'owned synthetic audio')
    sidecar = audio.with_suffix('.json'); private_json(sidecar, {'path': str(audio)})
    report = j.config.app_data.reports_dir / 'report-0.json'
    private_json(report, {'session_id': 'session0', 'meta': {'audio_path': str(audio)}})
    j.archive_attempt('session0'); j.purge_attempt('session0')
    assert not audio.exists() and not sidecar.exists()


def test_purge_refuses_saved_retry_and_changed_preview(tmp_path):
    j = journal(tmp_path); seed(j); j.archive_attempt('session0')
    preview = j.purge_preview('session0')
    report = j.config.app_data.reports_dir / 'report-0.json'
    payload = read_json_fixture(report); payload['changed'] = True; private_json(report, payload)
    with pytest.raises(JournalError, match='preview changed'):
        j.purge_attempt('session0', preview['fingerprint'])
    private_json(j.config.app_data.reports_dir / 'retry.json', {'session_id': 'retry', 'meta': {'practice': {'retry_of_session_id': 'session0'}}})
    with pytest.raises(JournalError, match='saved retry'):
        j.purge_preview('session0')


def test_restart_rolls_back_purge_after_jsonl_replacement(tmp_path, monkeypatch):
    import app_backend.journal as module
    from assess_speaking import append_session_jsonl
    j = journal(tmp_path); seed(j); j.archive_attempt('session0')
    sessions = j.config.app_data.reports_dir / 'sessions.jsonl'
    append_session_jsonl(sessions, read_json_fixture(j.config.app_data.reports_dir / 'report-0.json'))
    before = sessions.read_bytes(); original = module.private_json
    def interrupt(path, value):
        if Path(path).name == 'trash.json':
            raise OSError('Injected process interruption')
        return original(path, value)
    monkeypatch.setattr(module, 'private_json', interrupt)
    monkeypatch.setattr(j, 'recover_purge', lambda: None)
    with pytest.raises(OSError): j.purge_attempt('session0')
    monkeypatch.setattr(module, 'private_json', original)
    restarted = journal(tmp_path)
    assert restarted.recovery_error is None and sessions.read_bytes() == before
    assert (j.config.app_data.reports_dir / 'report-0.json').exists()
    restarted.archive_attempt('session0', undo=True)


def test_missing_browser_audio_is_an_explicit_retake_on_opt_in_backup(tmp_path):
    j = journal(tmp_path)
    browser = {'sessions': [{'version': 1, 'id': 'fixture', 'createdAt': '2026-10-08', 'language': 'en', 'goal': 'B1', 'speaker': 'fixture', 'phase': 'review', 'revision': 1, 'preparationSec': 0, 'runtime': {'provider': 'ollama', 'model': 'fixture', 'whisper': 'tiny', 'feedbackLanguage': 'en'}, 'parts': [{'id': 'one', 'prompt': 'fixture', 'durationSec': 180, 'recorded': True}]}], 'recordings': []}
    tx = j.begin()['id']
    with pytest.raises(JournalError, match='no audio'): j.export(tx, browser)
    result = j.export(tx, browser, allow_missing=True)
    assert result['missing'] == ['rehearsal:fixture:0']
    with zipfile.ZipFile(j.directory / f"backup_{tx}.zip") as archive:
        part = json.loads(archive.read('manifest.json'))['browser']['sessions'][0]['parts'][0]
        assert part['recorded'] is False and part['missingRecording'] is True
    j.abort(tx)


def test_direct_api_purge_requires_reference_snapshot_and_exact_preview(tmp_path):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8871)
    with TestClient(create_app(config)) as client:
        j = client.app.state.journal; seed(j)
        assert client.post('/v1/journal/attempts/session0/archive').status_code == 200
        endpoint = '/v1/journal/attempts/session0/'
        assert client.post(endpoint + 'preview-purge', json={}).status_code == 409
        assert client.post(endpoint + 'preview-purge', json={'browser_references': ['session0']}).status_code == 409
        preview = client.post(endpoint + 'preview-purge', json={'browser_references': []}).json()
        assert 'rewrites' not in preview and 'cache_dirs' not in preview
        assert client.post(endpoint + 'purge', json={'browser_references': [], 'fingerprint': 'stale'}).status_code == 409
        assert client.post(endpoint + 'purge', json={'browser_references': [], 'fingerprint': preview['fingerprint']}).status_code == 200
        assert j.state() is None


def test_status_exposes_damaged_purge_without_restore_transaction(tmp_path):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    purge = config.app_data.root / 'maintenance/purge.json'
    purge.parent.mkdir(parents=True, exist_ok=True)
    purge.write_text('{damaged purge fixture')
    with TestClient(create_app(config)) as client:
        status = client.get('/v1/journal/status').json()
        assert status['transaction'] is None
        assert 'Recovery files are unavailable' in status['recovery_error']
        assert client.post('/v1/journal/begin').status_code == 409
    assert purge.read_text() == '{damaged purge fixture'


def test_shared_cache_file_survives_purge_without_stranding_recovery(tmp_path):
    j = journal(tmp_path); seed(j); j.archive_attempt('session0')
    cache = j.config.jobs_dir / 'asmt_fixture-stages'; cache.mkdir()
    retained = cache / 'shared.json'; retained.write_text('{"fixture":true}')
    private_json(j.config.jobs_dir / 'asmt_fixture.json', {'status': 'completed', 'payload': {'report': {'session_id': 'session0'}}, 'stage_cache_dir': str(cache)})
    private_json(j.config.jobs_dir / 'asmt_other.json', {'status': 'completed', 'payload': {'report': {'session_id': 'other'}, 'retained_artifact': str(retained)}})
    lease = j.begin()['id']; preview = j.purge_preview('session0')
    assert str(cache) not in preview['cache_dirs']
    j.purge_attempt('session0', preview['fingerprint']); j.abort(lease)
    assert retained.exists() and j.state() is None and not (j.directory / 'purge.json').exists()
    assert journal(tmp_path).recovery_error is None


def test_optional_empty_cache_cleanup_failure_does_not_block_committed_purge(tmp_path, monkeypatch):
    j = journal(tmp_path); seed(j); j.archive_attempt('session0')
    cache = j.config.jobs_dir / 'asmt_fixture-stages'; cache.mkdir(); (cache / 'asr.json').write_text('{}')
    private_json(j.config.jobs_dir / 'asmt_fixture.json', {'status': 'completed', 'payload': {'report': {'session_id': 'session0'}}, 'stage_cache_dir': str(cache)})
    original = Path.rmdir
    def refuse(path):
        if path == cache: raise OSError('Injected directory cleanup failure')
        return original(path)
    monkeypatch.setattr(Path, 'rmdir', refuse)
    lease = j.begin()['id']; j.purge_attempt('session0'); j.abort(lease)
    assert j.state() is None and not (j.directory / 'purge.json').exists()
    assert journal(tmp_path).recovery_error is None


@pytest.mark.parametrize('damage', ['transaction.json', 'purge.json', 'edit.json'])
@pytest.mark.parametrize('exit_action', ['restore', 'diagnostics', 'fresh'])
def test_damaged_journal_three_api_exits(tmp_path, exit_action, damage):
    import io
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    j = Journal(config, SimpleNamespace(_processes={}))
    seed(j)
    j.archive_attempt('session1')
    archive = backup(j).read_bytes()
    recording = config.app_data.recordings_dir / 'private.wav'
    recording.write_bytes(b'PRIVATE RECORDING')
    private_json(config.jobs_dir / 'asmt_private.json', {'status': 'completed', 'payload': {'report': 'PRIVATE JOB REPORT'}})
    config.log_file.write_text('INFO diagnostic fixture api_key=private-key\nERROR PRIVATE LOG REPORT Synthetic fixture.\n')
    damaged_path = j.directory / damage
    damaged_path.write_text('{damaged')
    before = {p: p.read_bytes() for p in config.app_data.root.rglob('*') if p.is_file()}
    with TestClient(create_app(config)) as client:
        assert client.get('/v1/health').status_code == 200
        assert len(client.get('/v1/history').json()['items']) == 1
        assert client.get('/v1/history/session1').status_code == 200
        assert client.get('/v1/journal/status').json()['recovery_error']
        blocked = client.post('/v1/maintenance/cleanup', json={'target': 'all_safe', 'dry_run': True})
        assert blocked.status_code == 409
        assert blocked.json()['detail']['detail'] == client.get('/v1/journal/status').json()['recovery_error']
        if exit_action == 'restore':
            response = client.post('/v1/journal/begin-restore', json={})
            assert response.status_code == 200, response.text
            tx = response.json()['id']
            assert client.post(f'/v1/journal/{tx}/restore', content=archive).status_code == 200
            assert client.post(f'/v1/journal/{tx}/commit', json={}).status_code == 200
            assert client.post(f'/v1/journal/{tx}/complete', json={}).status_code == 200
        elif exit_action == 'fresh':
            assert client.post('/v1/journal/start-fresh', json={}).status_code == 409
            response = client.post('/v1/journal/start-fresh', json={'confirmed': True})
            assert response.status_code == 200, response.text
            assert any(p.read_bytes() == b'{damaged' for p in j.directory.glob('quarantine-*/' + damage))
            for path, content in before.items():
                if path != damaged_path:
                    assert path.read_bytes() == content
        else:
            # A dedicated endpoint never honors private-content opt-ins or client payloads.
            response = client.post('/v1/journal/diagnostics', json={'include_reports': True, 'include_recordings': True,
                'include_uploads': True, 'client_snapshot': {'report': 'PRIVATE CLIENT REPORT'}})
            assert response.status_code == 200, response.text
            downloaded = client.get('/v1/support-bundles/' + response.json()['bundle_id'])
            assert downloaded.status_code == 200
            with zipfile.ZipFile(io.BytesIO(downloaded.content)) as bundle:
                assert {'backend_diagnostics.json', 'backend_state.json', 'logs/backend.log'} <= set(bundle.namelist())
                assert not any(name.startswith(('reports/', 'recordings/', 'uploads/')) for name in bundle.namelist())
                content = b'\n'.join(bundle.read(name) for name in bundle.namelist())
                for secret in (b'PRIVATE', b'Synthetic fixture.', b'synthetic waveform', b'do-not-export', b'private-key'):
                    assert secret not in content
            for path, content in before.items():
                assert path.read_bytes() == content
            assert client.get('/v1/journal/status').json()['recovery_error']
            assert client.post('/v1/journal/begin').status_code == 409
            return
        assert client.get('/v1/history/session1').status_code == 200
        assert not client.get('/v1/journal/status').json()['recovery_error']
        assert client.post('/v1/maintenance/cleanup', json={'target': 'all_safe', 'dry_run': False}).status_code == 200
    assert Journal(config, SimpleNamespace(_processes={})).recovery_error is None


@pytest.mark.parametrize('endpoint', ['begin-restore', 'diagnostics', 'start-fresh'])
def test_recovery_exits_refuse_other_maintenance(tmp_path, endpoint):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    from app_core.journal_lock import journal_guard
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    with TestClient(create_app(config)) as client:
        tx = client.post('/v1/journal/begin').json()['id']
        assert client.post('/v1/journal/' + endpoint, json={'confirmed': True}).status_code == 409
        client.post(f'/v1/journal/{tx}/abort')
        app_journal = client.app.state.journal
        app_journal.state_path.write_text('{damaged')
        with journal_guard(config.app_data.root, exclusive=True):
            assert client.post('/v1/journal/' + endpoint, json={'confirmed': True}).status_code == 409


@pytest.mark.parametrize('damage', ['transaction.json', 'purge.json', 'edit.json'])
def test_failed_or_cancelled_repair_stays_damaged_across_restart(tmp_path, damage):
    j = journal(tmp_path); seed(j)
    (j.directory / damage).write_text('{damaged')
    j = journal(tmp_path)
    tx = j.begin_restore()['id']
    with pytest.raises(JournalError):
        j.stage(tx, j.config.app_data.reports_dir / 'report-0.json')
    j.abort(tx)
    j = journal(tmp_path)
    with pytest.raises(JournalError):
        j.begin_mutation()
    j.start_fresh(True)
    guard = j.begin_mutation(); j.end_mutation(guard)
    assert len(read_history(j.config.app_data.reports_dir / 'history.csv')[1]) == 2


def test_repair_transaction_blocks_all_exits_across_restart(tmp_path):
    j = journal(tmp_path); seed(j)
    j.state_path.write_text('{damaged')
    tx = j.begin_restore()['id']
    j = journal(tmp_path)
    for action in (j.begin_restore, lambda: j.start_fresh(True)):
        with pytest.raises(JournalError, match='in progress'):
            action()
    with pytest.raises(JournalError, match='in progress'):
        with j.recovery_exit():
            pytest.fail('Diagnostics must not enter during restore')
    j.abort(tx)
    assert j.recovery_error


def test_interrupted_quarantine_never_replays_damaged_metadata(tmp_path, monkeypatch):
    import app_backend.journal as module
    j = journal(tmp_path); seed(j)
    j.state_path.write_text('{damaged')
    original = module.os.replace
    def refuse(source, target):
        if 'quarantine-' in str(target):
            raise OSError('Injected disk failure')
        return original(source, target)
    monkeypatch.setattr(module.os, 'replace', refuse)
    with pytest.raises(OSError):
        j.start_fresh(True)
    restarted = journal(tmp_path)
    assert restarted.recovery_error
    with pytest.raises(JournalError):
        restarted.begin_mutation()
    monkeypatch.setattr(module.os, 'replace', original)
    restarted.start_fresh(True)
    assert len(read_history(restarted.config.app_data.reports_dir / 'history.csv')[1]) == 2


@pytest.mark.parametrize('receipt', ['{broken', '{}', '["invalid-id"]'])
def test_fresh_start_quarantines_invalid_completion_receipts(tmp_path, receipt):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    j = Journal(config, SimpleNamespace(_processes={}))
    seed(j)
    tx = j.begin()['id']
    j.save({'id': tx, 'phase': 'complete'})
    (j.directory / 'completed.json').write_text(receipt)
    with TestClient(create_app(config)) as client:
        status = client.get('/v1/journal/status')
        assert status.status_code == 200
        assert status.json()['recovery_error']
        assert client.post('/v1/journal/start-fresh', json={'confirmed': True}).status_code == 200
        assert any(p.read_text() == receipt for p in j.directory.glob('quarantine-*/completed.json'))
        # A complete ordinary backup transaction must work, including release.
        new = client.post('/v1/journal/begin').json()['id']
        assert client.post(f'/v1/journal/{new}/export', json={'browser': EMPTY}).status_code == 200
        assert client.post(f'/v1/journal/{new}/abort').status_code == 200
        assert client.post('/v1/maintenance/cleanup', json={'target': 'all_safe', 'dry_run': True}).status_code == 200


def test_fresh_start_retains_browser_acknowledgement_across_restart(tmp_path):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    j = Journal(config, SimpleNamespace(_processes={}))
    j.state_path.write_text('{broken')
    browser_id = 'a' * 32
    request_id = 'b' * 32
    with TestClient(create_app(config)) as client:
        assert client.post('/v1/journal/start-fresh', json={'confirmed': True, 'fresh_request': 'invalid'}).status_code == 409
        assert client.get('/v1/journal/status').json()['recovery_error']
        assert client.post('/v1/journal/start-fresh', json={'confirmed': True, 'browser_transaction': browser_id, 'fresh_request': request_id}).status_code == 200
    # A renderer crash or lost response must still leave evidence that browser
    # cleanup can be retried without rolling back committed sessions/audio.
    with TestClient(create_app(config)) as client:
        status = client.get('/v1/journal/status').json()
        assert browser_id in status['completed']
        assert request_id in status['completed']
        assert status['transaction'] is None and status['recovery_error'] is None


def test_rejected_fresh_start_does_not_publish_its_request_receipt(tmp_path):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    j = Journal(config, SimpleNamespace(_processes={}))
    j.state_path.write_text('{broken')
    with TestClient(create_app(config)) as client:
        assert client.post('/v1/journal/start-fresh', json={'confirmed': True, 'fresh_request': 'a' * 32}).status_code == 200
        restore = client.post('/v1/journal/begin-restore', json={})
        assert restore.status_code == 200
        rejected = client.post('/v1/journal/start-fresh', json={'confirmed': True, 'fresh_request': 'b' * 32})
        assert rejected.status_code == 409
        status = client.get('/v1/journal/status').json()
        assert status['transaction']['id'] == restore.json()['id']
        assert 'a' * 32 in status['completed']
        assert 'b' * 32 not in status['completed']


def test_intact_startup_rollback_can_be_retried_after_transient_disk_failure(tmp_path, monkeypatch):
    import app_backend.journal as module
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    from assess_speaking import HISTORY_FIELDNAMES

    source = journal(tmp_path / 'source'); seed(source)
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    target = Journal(config, SimpleNamespace(_processes={}))
    history = config.app_data.reports_dir / 'history.csv'
    write_history(history, HISTORY_FIELDNAMES, [{field: {'session_id': 'original'}.get(field, '') for field in HISTORY_FIELDNAMES}])
    original_history = history.read_bytes()
    tx = target.begin()['id']
    target.stage(tx, backup(source)); target.commit(tx)
    target.save({**target.state(), 'phase': 'backend_committing'})
    snapshot = target.directory / tx / 'history-before.csv'
    original_copy = module.shutil.copyfile

    def disk_full(source, destination, *args, **kwargs):
        if Path(source) == snapshot:
            raise OSError('Injected full disk during startup rollback')
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.shutil, 'copyfile', disk_full)
    with TestClient(create_app(config)) as client:
        status = client.get('/v1/journal/status').json()
        assert status['recovery_error']
        assert status['transaction']['id'] == tx
        assert status['transaction']['phase'] == 'rolling_back'
        assert snapshot.read_bytes() == original_history
        assert client.post('/v1/maintenance/cleanup', json={'target': 'all_safe', 'dry_run': True}).status_code == 409
        assert client.post('/v1/journal/diagnostics').status_code == 200
        assert snapshot.exists() and not list(target.directory.glob('quarantine-*'))
        monkeypatch.setattr(module.shutil, 'copyfile', original_copy)
        assert client.post(f'/v1/journal/{tx}/abort').status_code == 200
        assert history.read_bytes() == original_history
        status = client.get('/v1/journal/status').json()
        assert status['transaction'] is None and status['recovery_error'] is None
        assert client.post('/v1/maintenance/cleanup', json={'target': 'all_safe', 'dry_run': True}).status_code == 200


@pytest.mark.parametrize('exit_name', ['restore', 'diagnostics', 'fresh'])
@pytest.mark.parametrize('finished', [True, False])
def test_recovery_exits_drain_finished_workers_before_checking_activity(tmp_path, exit_name, finished):
    j = journal(tmp_path)
    if exit_name != 'restore':
        j.state_path.write_text('{broken')
        j.state()
    calls = []
    j.jobs._processes['analysis'] = SimpleNamespace(is_alive=lambda: True)

    def drain():
        calls.append('drained')
        if finished:
            j.jobs._processes.clear()

    j.jobs.wait_for_finished_workers = drain

    def run_exit():
        if exit_name == 'restore':
            assert j.begin_restore()['phase'] == 'prepared'
        elif exit_name == 'fresh':
            assert j.start_fresh(True)['phase'] == 'fresh'
        else:
            with j.recovery_exit():
                pass

    if finished:
        run_exit()
    else:
        with pytest.raises(JournalError, match='active work'):
            run_exit()
    assert calls


@pytest.mark.parametrize('startup_failure', [False, True])
@pytest.mark.parametrize('damage', ['purge.json', 'edit.json', 'manifest.json', 'history-before.csv'])
@pytest.mark.parametrize('repair', [False, True])
@pytest.mark.parametrize('exit_action', ['restore', 'diagnostics', 'fresh'])
def test_failed_intact_recovery_keeps_all_exits(tmp_path, damage, repair, exit_action, startup_failure):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    app = None if startup_failure else create_app(config)
    j = Journal(config, SimpleNamespace(_processes={}))
    seed(j)
    archive = backup(j)
    archive_bytes = archive.read_bytes()
    tx = j.begin()['id']
    if damage in {'purge.json', 'edit.json'}:
        # Archive/purge holds a prepared transaction while writing this metadata.
        (j.directory / damage).write_text('{broken')
    else:
        j.stage(tx, archive); j.commit(tx)
        j.save({**j.state(), 'phase': 'rolling_back'})
        (j.directory / tx / damage).unlink()
    if repair:
        j.save({**j.state(), 'recovery_restore': True})
        private_json(j.recovery_marker, {'recovery_required': True})
    private_files = {p: p.read_bytes() for p in config.app_data.reports_dir.rglob('*') if p.is_file()}
    with TestClient(app or create_app(config), raise_server_exceptions=False) as client:
        status = client.get('/v1/journal/status').json()
        assert bool(status['recovery_error']) is startup_failure
        assert status['transaction']['id'] == tx
        assert client.get('/v1/history').status_code == 200
        assert client.post('/v1/maintenance/cleanup', json={'target': 'all_safe', 'dry_run': True}).status_code == 409
        # The retry is reachable but cannot repair missing/corrupt snapshots.
        assert client.post(f'/v1/journal/{tx}/abort').status_code == 409
        assert client.get('/v1/journal/status').json()['recovery_retryable']
        if exit_action == 'restore':
            response = client.post('/v1/journal/begin-restore', json={})
            assert response.status_code == 200, response.text
            new_tx = response.json()['id']
            assert client.post(f'/v1/journal/{new_tx}/restore', content=archive_bytes).status_code == 200
            for other in ('begin-restore', 'start-fresh', 'diagnostics'):
                assert client.post('/v1/journal/' + other, json={'confirmed': True}).status_code == 409
            assert client.post(f'/v1/journal/{new_tx}/commit').status_code == 200
            assert client.post(f'/v1/journal/{new_tx}/complete').status_code == 200
        elif exit_action == 'fresh':
            response = client.post('/v1/journal/start-fresh', json={'confirmed': True})
            assert response.status_code == 200, response.text
        else:
            before = {p: p.read_bytes() for p in j.directory.rglob('*') if p.is_file()}
            response = client.post('/v1/journal/diagnostics')
            assert response.status_code == 200, response.text
            assert client.get('/v1/support-bundles/' + response.json()['bundle_id']).status_code == 200
            assert all(p.read_bytes() == content for p, content in before.items())
            assert client.get('/v1/journal/status').json()['recovery_error']
        assert all(p.read_bytes() == content for p, content in private_files.items())
        if exit_action != 'diagnostics':
            status = client.get('/v1/journal/status').json()
            assert not status['recovery_error'] and status['transaction'] is None
            assert client.post('/v1/maintenance/cleanup', json={'target': 'all_safe', 'dry_run': True}).status_code == 200


def test_healthy_idle_api_controls_and_pending_preview_exclusion(tmp_path):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    with TestClient(create_app(config)) as client:
        status = client.get('/v1/journal/status').json()
        assert not status['recovery_error'] and not status['recovery_retryable'] and status['transaction'] is None
        assert client.post('/v1/journal/start-fresh', json={'confirmed': True}).status_code == 409
        assert client.post('/v1/journal/diagnostics').status_code == 200
        response = client.post('/v1/journal/begin-restore')
        assert response.status_code == 200
        tx = response.json()['id']
        for endpoint in ('begin', 'begin-restore', 'diagnostics', 'start-fresh'):
            assert client.post('/v1/journal/' + endpoint, json={'confirmed': True}).status_code == 409
        assert client.post(f'/v1/journal/{tx}/abort').status_code == 200
        assert client.post('/v1/maintenance/cleanup', json={'target': 'all_safe', 'dry_run': True}).status_code == 200

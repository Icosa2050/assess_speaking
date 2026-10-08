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
    path = backup(first)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert len(manifest['files']) == 1
        assert str(tmp_path) not in archive.read('manifest.json').decode()
        assert 'do-not-export' not in archive.read('manifest.json').decode()
    tx = second.begin()['id']
    assert second.stage(tx, path)['attempts'] == 2
    second.commit(tx)
    second.complete(tx)
    _, rows = read_history(second.config.app_data.reports_dir / 'history.csv')
    payloads = [json.loads(Path(row['report_path']).read_text()) for row in rows]
    assert payloads[0]['meta']['audio_path'] == payloads[1]['meta']['audio_path']
    assert Path(payloads[0]['meta']['audio_path']).read_bytes() == b'synthetic waveform'
    assert payloads[0]['eligibility']['status'] == 'unverified'
    assert payloads[0]['restored_from']['jobs_resumable'] is False
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

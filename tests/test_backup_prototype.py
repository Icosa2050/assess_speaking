import json
from pathlib import Path
import zipfile
import pytest
from benchmarking.backup_prototype import export_snapshot, inspect_archive, stage_restore


def snapshot(tmp_path):
    audio = tmp_path / 'recording.wav'; audio.write_bytes(b'functional audio placeholder')
    return {'backend': {'reports': [{'session_id': 'report-a', 'audio_path': str(audio), 'eligibility': {'version': 1, 'state': 'content_unverified'}, 'secret_ref': 'must-omit'}],
                        'index': [{'session_id': 'report-a', 'report_path': '/Users/private/reports/report-a.json'}]},
            'browser': {'version': 1, 'sessions': [{'id': 'rehearsal-a', 'parts': [{'reportId': 'report-a', 'recorded': True}]}],
                        'recordings': {'rehearsal-a:0': str(audio)}}}, audio


def test_two_store_snapshot_roundtrip_rebinds_shared_audio_and_preserves_eligibility(tmp_path):
    data, audio = snapshot(tmp_path); bundle = tmp_path / 'backup.zip'
    manifest = export_snapshot(bundle, data, {str(audio): audio})
    assert len(manifest['files']) == 1
    result = stage_restore(bundle, tmp_path / 'different-root')
    backend = result['snapshot']['backend']['reports'][0]
    assert backend['audio_path'] == result['snapshot']['browser']['recordings']['rehearsal-a:0']
    assert Path(backend['audio_path']).read_bytes() == audio.read_bytes()
    assert backend['eligibility']['state'] == 'content_unverified'
    assert result['snapshot']['browser']['sessions'][0]['parts'][0]['reportId'] == 'report-a'
    assert not result['jobs_resumable']
    encoded = json.dumps(manifest)
    assert '/Users/private' not in encoded and 'must-omit' not in encoded
    assert not any('recording.wav' in key for key in manifest['files'])


@pytest.mark.parametrize('attack', ['traversal', 'checksum', 'missing', 'duplicate', 'format'])
def test_invalid_archives_never_publish_restore_or_damage_existing_data(tmp_path, attack):
    data, audio = snapshot(tmp_path); source = tmp_path / 'source.zip'
    export_snapshot(source, data, {str(audio): audio})
    with zipfile.ZipFile(source) as original:
        contents = {name: original.read(name) for name in original.namelist()}
    manifest = json.loads(contents['manifest.json'])
    name = next(iter(manifest['files']))
    if attack == 'traversal': contents['../private'] = b'escape'
    if attack == 'checksum': contents[name] = b'corrupt'
    if attack == 'missing': del contents[name]
    if attack == 'format': manifest['version'] = 999
    contents['manifest.json'] = json.dumps(manifest).encode()
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as archive:
        for key, value in contents.items(): archive.writestr(key, value)
        if attack == 'duplicate': archive.writestr(name, contents[name])
    destination = tmp_path / 'new-root'
    with pytest.raises(ValueError): stage_restore(bad, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('.vostavo-restore-*'))


def test_existing_export_and_restore_destinations_are_preserved(tmp_path):
    data, audio = snapshot(tmp_path); bundle = tmp_path / 'backup.zip'
    export_snapshot(bundle, data, {str(audio): audio}); original = bundle.read_bytes()
    with pytest.raises(ValueError): export_snapshot(bundle, data, {str(audio): audio})
    assert bundle.read_bytes() == original
    destination = tmp_path / 'occupied'; destination.mkdir(); sentinel = destination / 'keep'; sentinel.write_text('unchanged')
    with pytest.raises(ValueError): stage_restore(bundle, destination)
    assert sentinel.read_text() == 'unchanged'


def test_missing_or_symlink_audio_is_not_silently_omitted(tmp_path):
    data, audio = snapshot(tmp_path); audio.unlink()
    with pytest.raises(ValueError): export_snapshot(tmp_path / 'backup.zip', data, {str(audio): audio})
    target = tmp_path / 'external.wav'; target.write_bytes(b'external'); audio.symlink_to(target)
    with pytest.raises(ValueError): export_snapshot(tmp_path / 'backup.zip', data, {str(audio): audio})
    assert not (tmp_path / 'backup.zip').exists()

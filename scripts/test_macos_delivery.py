#!/usr/bin/env python3
"""Acceptance test of the DMG's copied frozen backend, including real ASR.

No paid provider calls. Existing tiny weights are copied into isolated test state.
Native microphone/TCC and Developer ID Gatekeeper acceptance are separate gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid


def run(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dmg', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = {'artifact': str(args.dmg.resolve()), 'sha256': hashlib.sha256(args.dmg.read_bytes()).hexdigest(), 'checks': [], 'native_microphone': 'requires interactive acceptance', 'public_gatekeeper': 'requires Developer ID release'}
    with tempfile.TemporaryDirectory(prefix='vostavo-delivery-') as directory:
        work = Path(directory)
        mount = work / 'mounted'
        mount.mkdir()
        run(['hdiutil', 'attach', args.dmg.resolve(), '-readonly', '-nobrowse', '-mountpoint', mount], capture_output=True)
        try:
            installed = work / 'Applications/Vostavo.app'
            installed.parent.mkdir()
            run(['ditto', mount / 'Vostavo.app', installed])
            mounted_helper = mount / 'Vostavo.app/Contents/Helpers/VostavoBackend.app/Contents/MacOS/vostavo-backend'
            env = {key: os.environ[key] for key in ('HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG') if key in os.environ}
            env.update(PATH='/usr/bin:/bin:/usr/sbin:/sbin', HF_HUB_OFFLINE='1', XDG_CACHE_HOME=str(work / 'cache'), VOSTAVO_HOME=str(work / 'state'), VOSTAVO_CACHE_HOME=str(work / 'cache'))
            mounted = run([mounted_helper, '--self-test'], cwd='/', env=env, capture_output=True, text=True, timeout=90)
            assert json.loads(mounted.stdout)['ok']
            evidence['checks'].append('read-only mounted helper self-test')
        finally:
            run(['hdiutil', 'detach', mount], capture_output=True)
        run(['codesign', '--verify', '--deep', '--strict', installed], capture_output=True)
        helper = installed / 'Contents/Helpers/VostavoBackend.app/Contents/MacOS/vostavo-backend'
        # Prove runtime has no read access to checkout or Homebrew. This is an
        # acceptance harness restriction, not a shipped macOS sandbox entitlement.
        profile = work / 'isolation.sb'
        profile.write_text('(version 1)\n(allow default)\n' + ''.join(f'(deny file-read* (subpath {json.dumps(str(path))}))\n' for path in (root, Path('/opt/homebrew'), Path('/usr/local'), Path.home() / '.cache')))
        command = ['/usr/bin/sandbox-exec', '-f', str(profile), str(helper)]
        isolated = run([*command, '--self-test'], cwd='/', env=env, capture_output=True, text=True, timeout=90)
        assert json.loads(isolated.stdout)['ok']
        evidence['checks'].append('copied helper self-test with repository/Homebrew/global cache reads denied')
        source = Path.home() / '.cache/huggingface/hub/models--Systran--faster-whisper-tiny'
        if not source.is_dir():
            raise RuntimeError('Cache tiny once before this offline artifact test; no mock ASR fallback is allowed.')
        target = work / 'cache/huggingface/hub' / source.name
        target.parent.mkdir(parents=True)
        shutil.copytree(source, target, symlinks=False)
        env['HF_HUB_CACHE'] = str(target.parent)
        ready = work / 'ready.json'
        token = secrets.token_hex(32)
        log = (work / 'helper.log').open('wb')
        process = None

        def launch():
            child = subprocess.Popen([*command, '--desktop-owned', '--port', '0', '--ready-file', str(ready), '--app-data-dir', str(work / 'state'), '--cache-dir', str(work / 'cache')], cwd='/', env=env, stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True)
            child.stdin.write((token + '\n' + 'b' * 64 + '\n').encode())
            child.stdin.flush()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError('Frozen helper exited: ' + (work / 'helper.log').read_text())
                if ready.exists():
                    port = json.loads(ready.read_text())['port']
                    try:
                        request('/v1/health', port=port)
                        return child, port
                    except (OSError, urllib.error.URLError):
                        pass
                time.sleep(.2)
            child.kill()
            raise TimeoutError('Packaged backend startup')

        def request(path, *, port=None, method='GET', data=None, content_type=None, auth=True, extra=None, raw=False):
            headers = dict(extra or {})
            if auth:
                headers['X-Vostavo-Session'] = token
            if content_type:
                headers['Content-Type'] = content_type
            payload = data
            if isinstance(data, dict):
                payload = json.dumps(data).encode()
                headers['Content-Type'] = 'application/json'
            req = urllib.request.Request(f'http://127.0.0.1:{port or active_port}{path}', data=payload, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=30) as response:
                body = response.read()
                return (response.status, body) if raw else json.loads(body)

        def reject(path, code, **kwargs):
            try:
                request(path, **kwargs)
            except urllib.error.HTTPError as exc:
                assert exc.code == code, (path, exc.code)
            else:
                raise AssertionError('Unexpected access: ' + path)

        try:
            process, active_port = launch()
            reject('/v1/health', 401, auth=False)
            reject('/v1/health', 401, auth=False, extra={'X-Vostavo-Session': 'wrong'})
            reject('/v1/health', 401, auth=False, extra={'X-Vostavo-Session': 'b' * 64})
            reject('/v1/health', 403, extra={'Host': 'attacker.example'})
            reject('/docs', 404)
            assert request('/v1/samples')['items']
            assert request('/v1/history')['items'] == []
            evidence['checks'].append('fresh state, protected health, wrong token/host rejection, docs disabled, packaged samples')
            attempts = []
            for language in ('en', 'it'):
                audio = installed / f'Contents/Helpers/VostavoBackend.app/Contents/Resources/samples/cefr/{language}/B1/travel_story.wav'
                if not audio.exists():
                    candidates = list((installed / f'Contents/Helpers/VostavoBackend.app/Contents/Resources/samples/cefr/{language}/B1').glob('*.wav'))
                    audio = candidates[0]
                boundary = 'Vostavo' + uuid.uuid4().hex
                multipart = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + audio.read_bytes() + f'\r\n--{boundary}--\r\n'.encode())
                upload = request('/v1/uploads', method='POST', data=multipart, content_type=f'multipart/form-data; boundary={boundary}')
                created = request('/v1/assessments', method='POST', data=dict(audio_id=upload['audio_id'], whisper='tiny', provider='ollama', llm_model='delivery-no-llm', llm_base_url='http://127.0.0.1:1', expected_language=language, feedback_language=language, speaker_id='delivery-test', task_family='free_monologue', theme='travel', target_duration_sec=90, target_cefr='B1', dry_run=False))
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    status = request('/v1/assessments/' + created['assessment_id'])
                    if status['status'] in ('completed', 'failed', 'cancelled'):
                        break
                    time.sleep(.5)
                assert status['status'] == 'completed', status
                payload = status['payload']
                transcript = payload.get('transcript_full', '')
                # Real transcript is saved in the report; do not record learner text in evidence.
                assert transcript, payload.keys()
                rows = request('/v1/history')['items']
                assert len(rows) == len(attempts) + 1
                row = next(item for item in rows if item['learning_language'] == language)
                assert row['word_count'] > 0, row
                request('/v1/history/' + row['session_id'])
                code, body = request('/v1/history/' + row['session_id'] + '/audio', extra={'Range': 'bytes=0-15'}, raw=True)
                assert code == 206 and len(body) == 16
                media_path = '/v1/history/' + row['session_id'] + '/audio?session=' + 'b' * 64
                code, body = request(media_path, auth=False, extra={'Range': 'bytes=0-15'}, raw=True)
                assert code == 206 and len(body) == 16
                attempts.append({'language': language, 'word_count': row['word_count'], 'duration_sec': row['duration_sec']})
            evidence['attempts'] = attempts
            evidence['checks'].append('English and Italian real frozen ASR/spawned workers, uploads, reports, history, ranged audio')
            process.stdin.close()
            process.wait(timeout=15)
            # The macOS PyInstaller bootloader can exit before the Python
            # child finishes its signal-driven finally block.
            deadline = time.monotonic() + 5
            while ready.exists() and time.monotonic() < deadline:
                time.sleep(.1)
            assert not ready.exists()
            process, active_port = launch()
            assert len(request('/v1/history')['items']) == 2
            process.stdin.close()
            process.wait(timeout=15)
            evidence['checks'].append('owner EOF shutdown, readiness cleanup, persisted history after restart')
            run(['codesign', '--verify', '--deep', '--strict', installed], capture_output=True)
            evidence['checks'].append('signature remains valid after use')
        finally:
            if process and process.poll() is None:
                process.kill()
                process.wait()
            log.close()
        evidence['ok'] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

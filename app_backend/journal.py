"""Local learner archive and a durable, two-store maintenance transaction.

Accounts, credentials, worker checkpoints and provider configuration are never
restored. Browser recordings are transferred as bounded binary bodies, not JSON.
"""
from __future__ import annotations

import csv
from functools import wraps
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from uuid import uuid4
import zipfile

from app_core.cloud_policy import atomic_json, locked_file
from app_core.journal_lock import journal_guard, JournalMaintenanceError

LIMIT = 512 * 1024 * 1024
MANIFEST_LIMIT = 16 * 1024 * 1024
MEDIA = re.compile(r"media/[0-9a-f]{64}\.(wav|mp3|webm|ogg|opus|m4a|mp4|aac|flac|aiff|aif|bin)$")
ID = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class JournalError(ValueError):
    pass


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def private_json(path, value):
    atomic_json(Path(path), value)
    Path(path).chmod(0o600)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def read_history(path):
    if not path.exists():
        return [], []
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_history(path, fields, rows):
    with locked_file(path.with_suffix(".lock")):
        _write_history(path, fields, rows)


def _write_history(path, fields, rows):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix('.maintenance.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        os.chmod(temporary, 0o600)
        handle.write(text.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clean(value, assets):
    if isinstance(value, dict):
        return {key: clean(item, assets) for key, item in value.items()
                if key.lower().replace('_', '') not in {'secret', 'secretref', 'token', 'accesstoken', 'refreshtoken', 'idtoken', 'apikey', 'llmapikey', 'account', 'accountid', 'accountstate', 'baseurl', 'llmbaseurl', 'credential', 'credentials'}}
    if isinstance(value, list):
        return [clean(item, assets) for item in value]
    if isinstance(value, str):
        if MEDIA.fullmatch(value):
            return value
        if value in assets:
            return {'$media': assets[value]}
        # Paths inside embedded messages are private too, not just path fields.
        if value.startswith(('/', '~/', '\\\\')) or re.match(r'^[A-Za-z]:[\\/]', value):
            return '[private path omitted]'
        value = re.sub(r'https?://[^/\s:@]+:[^/\s@]+@[^\s]+', '[private endpoint omitted]', value)
        return re.sub(r'(?<!\w)/(?:Users|home|private|tmp|var|Volumes)/[^\s"<>]+', '[private path omitted]', value)
    return value


def rebind(value, root):
    if isinstance(value, dict):
        if set(value) == {'$media'}:
            return str(root / Path(value['$media']).name)
        return {key: rebind(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [rebind(item, root) for item in value]
    return value


def media_refs(value):
    if isinstance(value, dict):
        if '$media' in value:
            if set(value) != {'$media'} or not isinstance(value['$media'], str):
                raise JournalError('Invalid media reference.')
            yield value['$media']
        else:
            for item in value.values():
                yield from media_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from media_refs(item)


def validate_browser(browser, inventory):
    if not isinstance(browser, dict) or set(browser) != {'sessions', 'recordings'}:
        raise JournalError('The archive must include its rehearsal store.')
    sessions, recordings = browser['sessions'], browser['recordings']
    if not isinstance(sessions, list) or not isinstance(recordings, list) or len(sessions) > 10000:
        raise JournalError('Invalid rehearsal inventory.')
    ids = set()
    for session in sessions:
        if not isinstance(session, dict) or session.get('version') != 1 or not ID.fullmatch(str(session.get('id', ''))):
            raise JournalError('Invalid rehearsal identity.')
        if session['id'] in ids or not isinstance(session.get('parts'), list) or len(session['parts']) > 100:
            raise JournalError('Duplicate or invalid rehearsal.')
        required = {'createdAt', 'language', 'goal', 'speaker', 'phase', 'revision', 'preparationSec', 'runtime'}
        if not required.issubset(session) or session['language'] not in {'en', 'it'} or session['goal'] not in {'B1', 'B2', 'C1'} or session['phase'] not in {'ready', 'preparation', 'speaking', 'review'}:
            raise JournalError('Unsupported rehearsal fields.')
        if not isinstance(session['createdAt'], str) or not isinstance(session['speaker'], str) or len(session['speaker']) > 256 or not isinstance(session['revision'], int) or not isinstance(session['preparationSec'], int):
            raise JournalError('Invalid rehearsal metadata.')
        runtime = session['runtime']
        if not isinstance(runtime, dict) or any(not isinstance(runtime.get(key), str) for key in ('provider', 'model', 'whisper', 'feedbackLanguage')):
            raise JournalError('Invalid rehearsal runtime metadata.')
        for part in session['parts']:
            if not isinstance(part, dict) or any(not isinstance(part.get(key), str) for key in ('id', 'prompt')) or not isinstance(part.get('durationSec'), (int, float)) or not isinstance(part.get('recorded'), bool):
                raise JournalError('Invalid rehearsal part.')
        ids.add(session['id'])
    keys = set()
    for recording in recordings:
        if not isinstance(recording, dict) or set(recording) != {'key', 'file', 'type'}:
            raise JournalError('Invalid rehearsal recording.')
        key, _, index = str(recording['key']).rpartition(':')
        session = next((item for item in sessions if item['id'] == key), None)
        if session is None or not index.isdigit() or int(index) >= len(session['parts']) or recording['key'] in keys:
            raise JournalError('Unlinked or duplicate rehearsal recording.')
        if recording['file'] not in inventory or not isinstance(recording['type'], str) or len(recording['type']) > 128:
            raise JournalError('Missing rehearsal recording.')
        keys.add(recording['key'])
    for session in sessions:
        for index, part in enumerate(session['parts']):
            if not isinstance(part, dict) or (part.get('recorded') and f"{session['id']}:{index}" not in keys):
                raise JournalError('A recorded rehearsal part has no audio. Retake it or remove its recorded flag before backing up.')


def history_transaction(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.lock:
            if getattr(self.history_local, 'held', False):
                return method(self, *args, **kwargs)
            with locked_file((self.config.app_data.reports_dir / 'history.csv').with_suffix('.lock')), locked_file(self.config.app_data.reports_dir / 'sessions.lock'):
                self.history_local.held = True
                try:
                    return method(self, *args, **kwargs)
                finally:
                    self.history_local.held = False
    return guarded


class Journal:
    def __init__(self, config, jobs):
        self.config, self.jobs = config, jobs
        self.directory = config.app_data.root / 'maintenance'
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path = self.directory / 'transaction.json'
        self.lock = threading.RLock()
        self.history_local = threading.local()
        self.mutations = 0
        self.recovery_error = None
        try:
            prior = self.state()
            self.recover_purge()
            edit = read_json(self.directory / 'edit.json')
            if edit:
                self._rollback_edit(edit)
            if prior and prior['phase'] in {'backend_committing', 'rolling_back'}:
                self.abort(prior['id'])
            elif prior and (prior['phase'] == 'complete' or prior['id'] in read_json(self.directory / 'completed.json', [])):
                self._finish(prior['id'])
        except (OSError, JournalError, json.JSONDecodeError, KeyError, TypeError):
            # Keep health/history readable while refusing further writes. A
            # missing rollback snapshot must never trigger an empty replacement.
            self.recovery_error = 'Recovery files are unavailable. Check disk space and retained backup files before retrying recovery.'

    def state(self):
        try:
            state = read_json(self.state_path)
            if state and (not isinstance(state, dict) or not re.fullmatch('[0-9a-f]{32}', str(state.get('id', ''))) or state.get('phase') not in {'prepared', 'backend_staged', 'backend_committing', 'backend_committed', 'complete', 'rolling_back'}):
                raise JournalError('Invalid recovery journal.')
            return state
        except (json.JSONDecodeError, JournalError):
            self.recovery_error = 'The recovery journal is damaged. Original data has been retained; use a known backup for repair.'
            return {'id': 'damaged', 'phase': 'recovery_error'}

    def save(self, state):
        private_json(self.state_path, state)

    def check(self, transaction):
        state = self.state()
        if state and state['id'] == 'damaged':
            raise JournalError(self.recovery_error)
        if not state or state['id'] != transaction:
            raise JournalError('Maintenance session expired. Retry from Settings.')
        return state

    def begin_mutation(self):
        with self.lock:
            state = self.state()
            if self.recovery_error:
                raise JournalError(self.recovery_error)
            if state:
                raise JournalError('Backup or recovery is in progress. Finish it in Settings before changing recordings or reports.')
            guard = journal_guard(self.config.app_data.root)
            try:
                guard.__enter__()
            except JournalMaintenanceError as exc:
                raise JournalError(str(exc)) from exc
            self.mutations += 1
            return guard

    def end_mutation(self, guard=None):
        with self.lock:
            self.mutations -= 1
            if guard:
                guard.__exit__(None, None, None)

    def begin(self):
        try:
            with journal_guard(self.config.app_data.root, exclusive=True):
                return self._begin()
        except JournalMaintenanceError as exc:
            raise JournalError(str(exc)) from exc

    def _begin(self):
        with self.lock:
            drain = getattr(self.jobs, 'wait_for_finished_workers', None)
            if callable(drain):
                drain()
            if self.recovery_error or (self.directory / 'purge.json').exists() or (self.directory / 'edit.json').exists():
                raise JournalError(self.recovery_error or 'Finish interrupted journal recovery first.')
            if self.state() or self.mutations or any(p.is_alive() for p in self.jobs._processes.values()):
                raise JournalError('Finish or cancel active work and any pending recovery first.')
            for old in self.directory.glob('backup_*.zip'):
                if re.fullmatch(r'backup_[0-9a-f]{32}\.zip', old.name) and not old.is_symlink() and old.is_file() and time.time() - old.stat().st_mtime > 86400:
                    old.unlink()
            transaction = uuid4().hex
            work = self.directory / transaction
            work.mkdir(mode=0o700)
            state = {'id': transaction, 'phase': 'prepared', 'expires': time.time() + 300}
            self.save(state)
            return state

    def work(self, transaction):
        self.check(transaction)
        return self.directory / transaction

    def add_media(self, transaction, source, suffix='.bin'):
        with self.lock:
            return self._add_media(transaction, source, suffix)

    def _add_media(self, transaction, source, suffix):
        work = self.work(transaction)
        hashed = digest(source)
        suffix = suffix.lower() if suffix.lower() in {'.wav', '.mp3', '.webm', '.ogg', '.opus', '.m4a', '.mp4', '.aac', '.flac', '.aiff', '.aif'} else '.bin'
        name = f'media/{hashed}{suffix}'
        target = work / name
        target.parent.mkdir(exist_ok=True, mode=0o700)
        if not target.exists():
            if sum(p.stat().st_size for p in (work / 'media').glob('*') if p.is_file()) + source.stat().st_size > LIMIT:
                raise JournalError('Backup exceeds the 512 MB limit.')
            shutil.copyfile(source, target)
            target.chmod(0o600)
        return name

    @history_transaction
    def export(self, transaction, browser, *, allow_missing=False):
        with self.lock:
            state = self.check(transaction)
            if state['phase'] != 'prepared':
                raise JournalError('A restore is already staged.')
            work = self.work(transaction)
            fields, rows = read_history(self.config.app_data.reports_dir / 'history.csv')
            trashed = read_json(self.directory / 'trash.json', {})
            entries, missing = [], []
            for row, archived in [(row, False) for row in rows] + [(item['row'], True) for item in trashed.values()]:
                report_path = Path(row.get('report_path') or '')
                if not report_path.is_absolute():
                    report_path = self.config.app_data.reports_dir / report_path
                # Only app-managed regular reports can be exported by this API.
                report = None
                if report_path.is_file() and not report_path.is_symlink() and report_path.resolve().is_relative_to(self.config.app_data.reports_dir.resolve()):
                    report = read_json(report_path)
                if not isinstance(report, dict):
                    missing.append(f"report:{row.get('session_id')}")
                    if not allow_missing:
                        raise JournalError('Some reports or recordings are missing. Choose report-only backup to continue.')
                    continue
                assets = {}
                for value in self._strings(report):
                    path = Path(value)
                    if not path.is_absolute() and path.suffix.lower() in {'.wav', '.mp3', '.webm', '.ogg', '.m4a', '.mp4', '.flac'}:
                        path = self.config.app_data.reports_dir / path
                    roots = (self.config.app_data.recordings_dir, self.config.app_data.uploads_dir)
                    if path.is_absolute() and any(path.resolve().is_relative_to(root.resolve()) for root in roots):
                        if path.is_file() and not path.is_symlink() and path.suffix.lower() != '.json':
                            assets[value] = self.add_media(transaction, path, path.suffix)
                        elif value == (report.get('meta') or {}).get('audio_path'):
                            missing.append(f"audio:{row.get('session_id')}")
                audio_path = (report.get('meta') or {}).get('audio_path')
                if audio_path and audio_path not in assets and f"audio:{row.get('session_id')}" not in missing:
                    missing.append(f"audio:{row.get('session_id')}")
                if missing and not allow_missing:
                    raise JournalError('Some recordings are missing. Choose report-only backup to continue.')
                sanitized = clean(report, assets)
                report_hash = hashlib.sha256(json.dumps(sanitized, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                entries.append({'session_id': row['session_id'], 'row': clean(row, assets), 'report': sanitized, 'report_sha256': report_hash, 'source_report_sha256': digest(report_path), 'archived': archived})
            inventory = {str(p.relative_to(work)): {'sha256': digest(p), 'size': p.stat().st_size}
                         for p in (work / 'media').glob('*')} if (work / 'media').exists() else {}
            browser = json.loads(json.dumps(browser))
            if allow_missing:
                keys = {item['key'] for item in browser['recordings']}
                for session in browser['sessions']:
                    for index, part in enumerate(session['parts']):
                        if part.get('recorded') and f"{session['id']}:{index}" not in keys:
                            part.update(recorded=False, missingRecording=True)
                            missing.append(f"rehearsal:{session['id']}:{index}")
            validate_browser(browser, inventory)
            browser = clean(browser, {})
            manifest = {'version': 1, 'kind': 'vostavo-learner-backup', 'created_at': time.time(),
                        'fields': fields, 'attempts': entries, 'browser': browser, 'files': inventory,
                        'omissions': ['credentials', 'accounts', 'provider settings', 'jobs', 'stage caches'],
                        'missing': missing, 'jobs_resumable': False}
            encoded = json.dumps(manifest, ensure_ascii=False).encode()
            if len(encoded) + sum(item['size'] for item in inventory.values()) > LIMIT:
                raise JournalError('Backup exceeds the 512 MB restore limit.')
            if len(encoded) > MANIFEST_LIMIT:
                raise JournalError('Backup manifest exceeds 16 MB. Reduce the journal size first.')
            path = self.directory / f'backup_{transaction}.zip'
            try:
                with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr('manifest.json', encoded)
                    for name in inventory:
                        archive.write(work / name, name)
                path.chmod(0o600)
            except BaseException:  # quality: allow[broad-except] remove partial export and preserve original exception
                path.unlink(missing_ok=True)
                raise
            return {'id': transaction, 'filename': f'Vostavo-backup-{transaction}.zip', 'size_bytes': path.stat().st_size, 'missing': missing}

    @staticmethod
    def _strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from Journal._strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from Journal._strings(item)

    @history_transaction
    def stage(self, transaction, source):
        with self.lock:
            state = self.check(transaction)
            if state['phase'] != 'prepared':
                raise JournalError('Restore is already staged.')
            work = self.work(transaction)
            extracted = work / 'restore'
            extracted.mkdir(mode=0o700)
            try:
                with source.open('rb') as handle:
                    identity = hashlib.file_digest(handle, 'sha256').hexdigest()
                    handle.seek(0)
                    with zipfile.ZipFile(handle) as archive:
                        infos = archive.infolist()
                        names = [item.filename for item in infos]
                        if len(names) > 32000 or len(set(names)) != len(names) or names.count('manifest.json') != 1:
                            raise JournalError('Duplicate or excessive archive entries.')
                        if any(name != 'manifest.json' and not MEDIA.fullmatch(name) for name in names):
                            raise JournalError('Unexpected archive path.')
                        if sum(item.file_size for item in infos) > LIMIT or archive.getinfo('manifest.json').file_size > MANIFEST_LIMIT:
                            raise JournalError('Archive exceeds the restore limit.')
                        if shutil.disk_usage(work).free < 2 * sum(item.file_size for item in infos) + 64 * 1024 * 1024:
                            raise JournalError('Not enough disk space to stage this restore.')
                        manifest = json.loads(archive.read('manifest.json'))
                        if manifest.get('version') != 1 or manifest.get('kind') != 'vostavo-learner-backup' or manifest.get('jobs_resumable') is not False:
                            raise JournalError('Unsupported backup format.')
                        inventory = manifest.get('files')
                        if not isinstance(inventory, dict) or set(inventory) != set(names) - {'manifest.json'}:
                            raise JournalError('Archive inventory does not match its files.')
                        for name, expected in inventory.items():
                            info = archive.getinfo(name)
                            if not isinstance(expected, dict) or info.flag_bits & 1:
                                raise JournalError('Encrypted or invalid media inventory.')
                            if info.external_attr >> 16 & 0o170000 == 0o120000 or expected.get('size') != info.file_size:
                                raise JournalError('Invalid media inventory.')
                            target = extracted / name
                            target.parent.mkdir(exist_ok=True, mode=0o700)
                            with archive.open(name) as incoming, target.open('xb') as outgoing:
                                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                            target.chmod(0o600)
                            actual = digest(target)
                            if actual != expected.get('sha256') or Path(name).stem != actual:
                                raise JournalError('Media checksum does not match.')
                        if any(name not in inventory for name in media_refs(manifest)):
                            raise JournalError('Unlisted media reference.')
                        validate_browser(manifest['browser'], inventory)
                        if clean(manifest['attempts'], {}) != manifest['attempts'] or clean(manifest['browser'], {}) != manifest['browser']:
                            raise JournalError('Archive contains private paths or provider credentials.')
                fields, existing = read_history(self.config.app_data.reports_dir / 'history.csv')
                used = {row['session_id'] for row in existing} | set(read_json(self.directory / 'trash.json', {}))
                seen, skipped = set(), []
                if not isinstance(manifest.get('attempts'), list) or len(manifest['attempts']) > 10000:
                    raise JournalError('Invalid attempt inventory.')
                if not isinstance(manifest.get('fields'), list) or any(not isinstance(field, str) for field in manifest['fields']) or len(set(manifest['fields'])) != len(manifest['fields']):
                    raise JournalError('Invalid history fields.')
                if manifest['attempts'] and not {'session_id', 'report_path'}.issubset(manifest['fields']):
                    raise JournalError('History identity fields are missing.')
                for entry in manifest['attempts']:
                    session = entry.get('session_id', '')
                    if not ID.fullmatch(session) or session in seen or entry.get('row', {}).get('session_id') != session or not isinstance(entry.get('report'), dict):
                        raise JournalError('An attempt has a duplicate or invalid identity.')
                    if set(entry['row']) != set(manifest['fields']):
                        raise JournalError('History row does not match its fields.')
                    report_hash = hashlib.sha256(json.dumps(entry['report'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    if report_hash != entry.get('report_sha256') or not re.fullmatch('[0-9a-f]{64}', str(entry.get('source_report_sha256', ''))):
                        raise JournalError('Report checksum does not match.')
                    report_id = (entry['report'].get('report') or entry['report']).get('session_id')
                    if report_id != session:
                        raise JournalError('Report and history identities differ.')
                    seen.add(session)
                    if session in used:
                        skipped.append(session)
                from assess_speaking import HISTORY_FIELDNAMES
                if fields and fields != HISTORY_FIELDNAMES:
                    raise JournalError('The destination journal uses an unsupported history schema.')
                if any(field not in HISTORY_FIELDNAMES for field in manifest['fields']):
                    raise JournalError('Backup uses an unsupported history schema.')
                manifest['attempts'] = [{**entry, 'row': {field: entry['row'].get(field, '') for field in HISTORY_FIELDNAMES}} for entry in manifest['attempts'] if entry['session_id'] not in used]
                manifest['fields'] = HISTORY_FIELDNAMES
                private_json(work / 'manifest.json', manifest)
                history = self.config.app_data.reports_dir / 'history.csv'
                if history.exists():
                    shutil.copyfile(history, work / 'history-before.csv')
                private_json(work / 'trash-before.json', read_json(self.directory / 'trash.json', {}))
                state.update(phase='backend_staged', restored_from=identity, history_existed=history.exists(), history_before=digest(history) if history.exists() else None)
                self.save(state)
                return {'id': transaction, 'phase': state['phase'], 'browser': manifest['browser'], 'attempts': len(manifest['attempts']), 'skipped_attempts': skipped, 'missing': manifest.get('missing', [])}
            except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, AttributeError) as exc:
                shutil.rmtree(extracted, ignore_errors=True)
                raise JournalError('The backup format is invalid; nothing was restored.') from exc
            except BaseException:  # quality: allow[broad-except] remove only uncommitted staged data
                shutil.rmtree(extracted, ignore_errors=True)
                raise

    @history_transaction
    def commit(self, transaction):
        with self.lock:
            state = self.check(transaction)
            if state['phase'] != 'backend_staged':
                raise JournalError('Stage both stores before committing.')
            work = self.work(transaction)
            manifest = read_json(work / 'manifest.json')
            media = self.config.app_data.uploads_dir / f'restored_{transaction}'
            reports = self.config.app_data.reports_dir / f'restored_{transaction}'
            history = self.config.app_data.reports_dir / 'history.csv'
            if (digest(history) if history.exists() else None) != state['history_before']:
                raise JournalError('History changed after the preview. Cancel this restore and open the backup again.')
            state.update(phase='backend_committing', media=str(media), reports=str(reports))
            self.save(state)
            try:
                media.mkdir(mode=0o700)
                reports.mkdir(mode=0o700)
                required_media = set(media_refs(manifest['attempts']))
                for path in (work / 'restore/media').glob('*'):
                    if f'media/{path.name}' not in required_media:
                        continue
                    shutil.copyfile(path, media / path.name)
                    (media / path.name).chmod(0o600)
                fields, rows = read_history(self.config.app_data.reports_dir / 'history.csv')
                trash = read_json(self.directory / 'trash.json', {})
                for entry in manifest['attempts']:
                    payload = rebind(entry['report'], media)
                    payload['restored_from'] = {'archive_sha256': state['restored_from'], 'source_report_sha256': entry['source_report_sha256'], 'sanitized_report_sha256': entry['report_sha256'], 'jobs_resumable': False}
                    path = reports / f"{entry['session_id']}.json"
                    payload['report_path'] = str(path)
                    private_json(path, payload)
                    row = rebind(entry['row'], media)
                    row['report_path'] = str(path)
                    if entry.get('archived'):
                        trash[entry['session_id']] = {'row': row, 'archived_at': time.time()}
                    else:
                        rows.append(row)
                private_json(self.directory / 'trash.json', trash)
                _write_history(self.config.app_data.reports_dir / 'history.csv', fields or manifest['fields'], rows)
                state['history_after'] = digest(self.config.app_data.reports_dir / 'history.csv')
                state['phase'] = 'backend_committed'
                self.save(state)
            except BaseException:  # quality: allow[broad-except] durable rollback protects originals after partial publication
                self.abort(transaction)
                raise
            return {'phase': state['phase']}

    def complete(self, transaction):
        with self.lock:
            if transaction in read_json(self.directory / 'completed.json', []):
                state = self.state()
                if state and state['id'] == transaction:
                    self._finish(transaction)
                return {'phase': 'complete'}
            state = self.check(transaction)
            if state['phase'] != 'backend_committed':
                raise JournalError('Both stores must commit before completion.')
            # The client keeps its committed journal until it sees this acknowledgement.
            completed = read_json(self.directory / 'completed.json', [])
            private_json(self.directory / 'completed.json', list(dict.fromkeys([*completed, transaction])))
            state['phase'] = 'complete'
            self.save(state)
            self._finish(transaction)
            return {'phase': 'complete'}

    @history_transaction
    def abort(self, transaction):
        with self.lock:
            state = self.check(transaction)
            if transaction in read_json(self.directory / 'completed.json', []):
                self._finish(transaction)
                return {'phase': 'complete'}
            work = self.directory / transaction
            self.recover_purge()
            edit = read_json(self.directory / 'edit.json')
            if edit:
                self._rollback_edit(edit)
            if state['phase'] in {'backend_committing', 'backend_committed', 'rolling_back'}:
                state['phase'] = 'rolling_back'
                self.save(state)
                history = self.config.app_data.reports_dir / 'history.csv'
                manifest = read_json(work / 'manifest.json')
                imported = {entry['session_id'] for entry in manifest['attempts']}
                unchanged = history.exists() and digest(history) == state.get('history_after')
                if unchanged and state['history_existed']:
                    shutil.copyfile(work / 'history-before.csv', history.with_suffix('.rollback.tmp'))
                    os.replace(history.with_suffix('.rollback.tmp'), history)
                elif unchanged:
                    history.unlink(missing_ok=True)
                elif history.exists():
                    fields, rows = read_history(history)
                    retained = [row for row in rows if row['session_id'] not in imported]
                    if retained or state['history_existed']:
                        _write_history(history, fields, retained)
                    else:
                        history.unlink(missing_ok=True)
                trash = read_json(self.directory / 'trash.json', {})
                before = read_json(work / 'trash-before.json', {})
                private_json(self.directory / 'trash.json', {key: value for key, value in {**trash, **before}.items() if key not in imported})
                for key in ('media', 'reports'):
                    expected = (self.config.app_data.uploads_dir if key == 'media' else self.config.app_data.reports_dir) / f'restored_{transaction}'
                    if state.get(key) != str(expected):
                        raise JournalError('Invalid recovery journal. Originals have been retained.')
                    shutil.rmtree(expected, ignore_errors=False) if expected.exists() else None
            self._finish(transaction)
            return {'phase': 'aborted'}

    def _finish(self, transaction):
        if (self.directory / 'purge.json').exists() or (self.directory / 'edit.json').exists():
            raise JournalError('Finish file recovery before releasing journal maintenance.')
        self.recovery_error = None
        self.state_path.unlink(missing_ok=True)
        shutil.rmtree(self.directory / transaction, ignore_errors=True)

    @history_transaction
    def archive_attempt(self, session_id, *, undo=False):
        with self.lock:
            fields, rows = read_history(self.config.app_data.reports_dir / 'history.csv')
            trash = read_json(self.directory / 'trash.json', {})
            row = next((row for row in rows if row['session_id'] == session_id), None)
            edit = json.loads(json.dumps({'fields': fields, 'rows': rows, 'trash': trash}))
            private_json(self.directory / 'edit.json', edit)
            try:
                return self._edit_attempt(session_id, undo, fields, rows, trash, row)
            except BaseException:  # quality: allow[broad-except] restore both indexes after partial edits
                self._rollback_edit(edit)
                raise

    @history_transaction
    def _rollback_edit(self, edit):
        _write_history(self.config.app_data.reports_dir / 'history.csv', edit['fields'], edit['rows'])
        private_json(self.directory / 'trash.json', edit['trash'])
        (self.directory / 'edit.json').unlink(missing_ok=True)

    def _edit_attempt(self, session_id, undo, fields, rows, trash, row):
        if undo:
            if session_id not in trash or row:
                raise JournalError('This attempt is not archived or already exists.')
            rows.append(trash[session_id]['row'])
            # Retain an Undo source until the index publication succeeded.
            _write_history(self.config.app_data.reports_dir / 'history.csv', fields, rows)
            del trash[session_id]
            private_json(self.directory / 'trash.json', trash)
        else:
            if not row:
                raise JournalError('This attempt is not available.')
            for job in self.config.jobs_dir.glob('*.json'):
                data = read_json(job, {})
                if ((data.get('payload') or {}).get('report') or {}).get('session_id') == session_id and data.get('status') in {'queued', 'running', 'failed', 'cancelled'}:
                    raise JournalError('This attempt still has active or resumable recovery data.')
            trash[session_id] = {'row': row, 'archived_at': time.time()}
            private_json(self.directory / 'trash.json', trash)
            _write_history(self.config.app_data.reports_dir / 'history.csv', fields, [item for item in rows if item['session_id'] != session_id])
        (self.directory / 'edit.json').unlink(missing_ok=True)
        return {'archived': list(trash), 'retained_media': True}

    def archived_payload(self, session_id):
        try:
            item = read_json(self.directory / 'trash.json', {}).get(session_id)
        except (OSError, ValueError):
            return None
        if not item:
            return None
        from app_core.services import load_report_payload
        path = Path(item['row']['report_path'])
        if not path.is_absolute():
            path = self.config.app_data.reports_dir / path
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.config.app_data.reports_dir.resolve()):
            return load_report_payload(path)
        return None

    @staticmethod
    def report_identity(payload):
        return (payload.get('report') or payload).get('session_id') if isinstance(payload, dict) else None

    def _jsonl_rows(self, path):
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if len(line) > MANIFEST_LIMIT:
                    raise JournalError('A journal entry is too large to safely remove.')
                if line.strip():
                    yield line, json.loads(line)

    @history_transaction
    def purge_preview(self, session_id):
        trash = read_json(self.directory / 'trash.json', {})
        if session_id not in trash:
            raise JournalError('Archive this attempt before permanently removing it.')
        roots = (self.config.app_data.reports_dir.resolve(), self.config.jobs_dir.resolve())
        candidates, references, completed_jobs, caches = set(), set(), [], set()
        jobs = list(self.config.jobs_dir.glob('*.json'))
        reports = list(self.config.app_data.reports_dir.rglob('*.json'))
        if len(reports) + len(jobs) > 20000:
            raise JournalError('Too many retained files to prove a safe purge. Keep this attempt archived.')
        for job in jobs:
            data = read_json(job, {})
            if self.report_identity(data.get('payload')) == session_id:
                if data.get('status') != 'completed':
                    raise JournalError('Retained recovery data still references this attempt. Finish its recovery first.')
                completed_jobs.append(job.resolve())
                cache = Path(data.get('stage_cache_dir') or self.config.jobs_dir / (job.stem + '-stages'))
                if cache.exists():
                    if cache.is_symlink() or not cache.resolve().is_relative_to(self.config.jobs_dir.resolve()):
                        raise JournalError('Recovery cache is outside the managed journal. Keep this attempt archived.')
                    caches.add(cache.resolve())
            else:
                if session_id in set(self._strings(data)):
                    raise JournalError('Another retained job references this attempt. Keep it archived.')
                references.update(self._strings(data))
                cache = Path(data.get('stage_cache_dir') or self.config.jobs_dir / (job.stem + '-stages'))
                references.add(str(cache.resolve()))
        for cache in caches:
            if str(cache) in references:
                raise JournalError('A recovery cache is shared with another job. Keep this attempt archived.')
            for path in cache.rglob('*'):
                if path.is_symlink():
                    raise JournalError('A recovery cache contains an unmanaged link. Keep it archived.')
                if path.is_file():
                    candidates.add(path.resolve())
        candidates.update(completed_jobs)
        for report in reports:
            if report.is_symlink():
                raise JournalError('A retained report is a link. Keep this attempt archived.')
            if report.parent.resolve() == self.config.app_data.uploads_dir.resolve() and re.fullmatch(r'aud_[0-9a-f]{32}\.json', report.name):
                continue
            payload = read_json(report, {})
            if ((payload.get('meta') or {}).get('practice') or {}).get('retry_of_session_id') == session_id:
                raise JournalError('A saved retry references this attempt. Keep the parent archived.')
            if self.report_identity(payload) == session_id:
                candidates.add(report.resolve())
                for value in self._strings(payload):
                    path = Path(value)
                    if not path.is_absolute() and path.suffix.lower() in {'.wav', '.webm', '.m4a', '.mp4', '.mp3', '.flac', '.ogg', '.opus', '.aac', '.aiff', '.aif'}:
                        path = self.config.app_data.reports_dir / path
                    if path.is_absolute() and path.is_file() and not path.is_symlink() and any(path.resolve().is_relative_to(root.resolve()) for root in (self.config.app_data.uploads_dir, self.config.app_data.recordings_dir)) and path.suffix.lower() != '.json':
                        candidates.add(path.resolve())
            else:
                references.update(self._strings(payload))
        _, active = read_history(self.config.app_data.reports_dir / 'history.csv')
        for row in active + [item['row'] for key, item in trash.items() if key != session_id]:
            references.update(self._strings(row))
        rewrites = []
        sessions = self.config.app_data.reports_dir / 'sessions.jsonl'
        if sessions.exists():
            count = 0
            for _, payload in self._jsonl_rows(sessions):
                if ((payload.get('meta') or {}).get('practice') or {}).get('retry_of_session_id') == session_id:
                    raise JournalError('A retained retry references this attempt. Keep the parent archived.')
                if self.report_identity(payload) == session_id:
                    count += 1
                else:
                    references.update(self._strings(payload))
            if count:
                rewrites.append(str(sessions))
        for value in list(references):
            if value.startswith('/'):
                references.add(str(Path(value).resolve()))
            elif len(value) < 1024 and Path(value).suffix.lower() in {'.wav', '.webm', '.m4a', '.mp4', '.mp3', '.flac', '.ogg', '.opus', '.aac', '.aiff', '.aif'}:
                references.add(str((self.config.app_data.reports_dir / value).resolve()))
        removable = []
        for path in candidates:
            if str(path) in references:
                continue
            if path.is_symlink() or not any(path.is_relative_to(root) for root in roots):
                raise JournalError('Only app-managed regular files can be removed.')
            removable.append(str(path))
        retained = len(candidates) - len(removable)
        for path in list(removable):
            sidecar = Path(path).with_suffix('.json')
            if sidecar.parent.resolve() == self.config.app_data.uploads_dir.resolve() and re.fullmatch(r'aud_[0-9a-f]{32}\.json', sidecar.name) and sidecar.is_file() and not sidecar.is_symlink():
                removable.append(str(sidecar))
        inventory = [{'path': path, 'size': Path(path).stat().st_size, 'sha256': digest(path)} for path in sorted(set(removable + rewrites))]
        fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
        removable_paths = set(removable)
        empty_caches = [str(path) for path in caches if all(str(child.resolve()) in removable_paths for child in path.rglob('*') if child.is_file())]
        return {'files': sorted(set(removable + rewrites)), 'rewrites': rewrites, 'cache_dirs': empty_caches, 'retained_files': retained, 'size_bytes': sum(item['size'] for item in inventory), 'fingerprint': fingerprint}

    @history_transaction
    def purge_attempt(self, session_id, fingerprint=None):
        preview = self.purge_preview(session_id)
        if fingerprint is not None and fingerprint != preview['fingerprint']:
            raise JournalError('Removal preview changed. Review it again before permanently removing files.')
        token = uuid4().hex
        state = {'id': token, 'session_id': session_id, 'trash': read_json(self.directory / 'trash.json', {}),
                 'moves': [{'source': source, 'target': str(Path(source).with_name('.' + Path(source).name + '.purge_' + token))} for source in preview['files']],
                 'rewrites': {}, 'cache_dirs': preview['cache_dirs'], 'phase': 'moving'}
        try:
            for source in preview['rewrites']:
                path = Path(source)
                if shutil.disk_usage(path.parent).free < path.stat().st_size + 64 * 1024 * 1024:
                    raise OSError('Not enough space to safely rewrite the retained journal.')
                temporary = path.with_name('.' + path.name + '.filtered_' + token)
                with temporary.open('x', encoding='utf-8') as handle:
                    temporary.chmod(0o600)
                    for line, payload in self._jsonl_rows(path):
                        if self.report_identity(payload) != session_id:
                            handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                state['rewrites'][source] = {'temporary': str(temporary), 'sha256': digest(temporary)}
            private_json(self.directory / 'purge.json', state)
            for move in state['moves']:
                os.replace(move['source'], move['target'])
                replacement = state['rewrites'].get(move['source'])
                if replacement:
                    os.replace(replacement['temporary'], move['source'])
            trash = dict(state['trash'])
            del trash[session_id]
            private_json(self.directory / 'trash.json', trash)
            state['phase'] = 'complete'
            private_json(self.directory / 'purge.json', state)
            self.recover_purge()
        except BaseException:  # quality: allow[broad-except] recover interrupted moves before their durable commit decision
            if (self.directory / 'purge.json').exists():
                self.recover_purge()
            else:
                for replacement in state['rewrites'].values():
                    Path(replacement['temporary']).unlink(missing_ok=True)
            raise
        return {'removed_files': len(preview['files']), 'retained_files': preview['retained_files']}

    @history_transaction
    def recover_purge(self):
        state = read_json(self.directory / 'purge.json')
        if not state:
            return
        if not re.fullmatch('[0-9a-f]{32}', str(state.get('id', ''))) or state.get('phase') not in {'moving', 'complete'}:
            raise JournalError('Invalid purge recovery journal.')
        roots = (self.config.app_data.reports_dir.resolve(), self.config.jobs_dir.resolve())
        for move in state['moves']:
            source, target = Path(move['source']), Path(move['target'])
            if source.is_symlink() or target.is_symlink() or not any(source.resolve().is_relative_to(root) for root in roots) or target.parent != source.parent or target.name != '.' + source.name + '.purge_' + state['id']:
                raise JournalError('Invalid purge recovery location; files were retained.')
            replacement = state.get('rewrites', {}).get(str(source))
            if replacement:
                temporary = Path(replacement['temporary'])
                if temporary.parent != source.parent or temporary.name != '.' + source.name + '.filtered_' + state['id']:
                    raise JournalError('Invalid retained journal replacement.')
            if state['phase'] == 'complete':
                target.unlink(missing_ok=True)
            elif target.exists():
                if source.exists():
                    if not replacement or digest(source) != replacement['sha256']:
                        raise JournalError('A purge recovery file conflicts with another file. Both were retained.')
                    source.unlink()
                os.replace(target, source)
            if replacement:
                Path(replacement['temporary']).unlink(missing_ok=True)
        if state['phase'] != 'complete':
            private_json(self.directory / 'trash.json', state['trash'])
        else:
            for directory in state.get('cache_dirs', []):
                path = Path(directory)
                if path.is_symlink() or not path.resolve().is_relative_to(self.config.jobs_dir.resolve()):
                    raise JournalError('Invalid recovery cache location.')
                if path.exists():
                    for child in sorted(path.rglob('*'), key=lambda item: len(item.parts), reverse=True):
                        if child.is_dir():
                            try:
                                child.rmdir()
                            except OSError:
                                pass  # Retain nonempty or inaccessible folders after committed removal.
                    try:
                        path.rmdir()
                    except OSError:
                        pass  # Optional empty-folder cleanup cannot strand the journal lease.
        (self.directory / 'purge.json').unlink(missing_ok=True)

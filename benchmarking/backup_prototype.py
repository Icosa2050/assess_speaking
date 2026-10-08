"""E0 archive experiment for frozen snapshots and disposable roots only.

No live journal/IndexedDB writer integration, maintenance lease or production
restore endpoint is supplied here. Those are E1/E2 acceptance prerequisites.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

LIMIT = 512 * 1024 * 1024
MANIFEST_LIMIT = 16 * 1024 * 1024
ALLOWED = re.compile(r'files/[a-f0-9]{64}\.(wav|webm|mp4|m4a|json|csv|bin)$')
OMISSIONS = ['credentials', 'account_state', 'secret_references', 'jobs', 'stage_caches']


def scrub(value, paths):
    if isinstance(value, dict):
        return {key: scrub(item, paths) for key, item in value.items()
                if not key.startswith(('/', '\\')) and not any(part in key.lower() for part in ('secret', 'token', 'api_key', 'account', 'base_url'))}
    if isinstance(value, list):
        return [scrub(item, paths) for item in value]
    if isinstance(value, str):
        if value in paths:
            return {'archive_file': paths[value]}
        if value.startswith(('/', '\\')) or re.match(r'^[A-Za-z]:[\\/]', value):
            return '[omitted external path]'
    return value


def export_snapshot(destination: Path, snapshot: dict, assets: dict[str, Path]):
    """Snapshot has backend reports/index and browser sessions/recording keys.

    Assets are referenced by their original path strings; export replaces those
    references with checksum identities. Caller must already quiesce both stores.
    """
    if destination.exists():
        raise ValueError('Refusing to overwrite an existing export')
    paths, entries, total = {}, {}, 0
    for original, path in assets.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Only regular snapshot assets are supported')
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as source:
            while block := source.read(1024 * 1024):
                digest.update(block); size += len(block)
                if size > LIMIT:
                    raise ValueError('Snapshot expansion limit exceeded')
        total += size
        if total > LIMIT:
            raise ValueError('Snapshot expansion limit exceeded')
        suffix = path.suffix.lower().lstrip('.')
        if suffix not in ('wav', 'webm', 'mp4', 'm4a'):
            raise ValueError('Snapshot assets must be audio; reports/indexes belong in structured metadata')
        name = f'files/{digest.hexdigest()}.{suffix}'
        paths[original] = name
        entries[name] = {'sha256': digest.hexdigest(), 'size': size, 'source': path}
    manifest = {'version': 1, 'prototype_only': True, 'restored_jobs_resumable': False,
                'omissions': OMISSIONS, 'snapshot': scrub(snapshot, paths),
                'files': {name: {key: row[key] for key in ('sha256', 'size')} for name, row in entries.items()}}
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    if len(encoded) > MANIFEST_LIMIT:
        raise ValueError('Manifest is too large')
    with tempfile.NamedTemporaryFile(prefix='.vostavo-export-', dir=destination.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json', encoded)
            for name, row in entries.items():
                archive.write(row['source'], name)
        # Validate a stable snapshot: a changed asset cannot publish a valid ZIP.
        inspect_archive(temporary)
        # Hard-link publication is exclusive and preserves a racing destination.
        import os
        os.link(temporary, destination)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def inspect_archive(path: Path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or 'manifest.json' not in names or len(names) > 32000:
            raise ValueError('Duplicate, missing or excessive archive entries')
        if any(name != 'manifest.json' and not ALLOWED.fullmatch(name) for name in names):
            raise ValueError('Invalid relative archive path')
        if archive.getinfo('manifest.json').file_size > MANIFEST_LIMIT or sum(item.file_size for item in infos) > LIMIT:
            raise ValueError('Archive expansion limit exceeded')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('version') != 1 or manifest.get('prototype_only') is not True:
            raise ValueError('Unsupported prototype format')
        if set(manifest.get('files', {})) != set(names) - {'manifest.json'}:
            raise ValueError('Manifest inventory differs from archive')
        for name, entry in manifest['files'].items():
            digest = hashlib.sha256(); size = 0
            with archive.open(name) as source:
                while block := source.read(1024 * 1024):
                    digest.update(block); size += len(block)
            if size != entry['size'] or digest.hexdigest() != entry['sha256']:
                raise ValueError('Checksum mismatch')
        def references(value):
            if isinstance(value, dict):
                if 'archive_file' in value and value['archive_file'] not in manifest['files']:
                    raise ValueError('Missing recording reference')
                for item in value.values(): references(item)
            elif isinstance(value, list):
                for item in value: references(item)
        references(manifest['snapshot'])
        return manifest


def stage_restore(path: Path, destination: Path):
    """Unpack into a NEW disposable root; never mutate an existing journal."""
    manifest = inspect_archive(path)
    if destination.exists():
        raise ValueError('Restore conflicts with an existing root')
    staged = Path(tempfile.mkdtemp(prefix='.vostavo-restore-', dir=destination.parent))
    try:
        with zipfile.ZipFile(path) as archive:
            for name in manifest['files']:
                target = staged / PurePosixPath(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        def rebind(value):
            if isinstance(value, dict):
                if set(value) == {'archive_file'}: return str(destination / value['archive_file'])
                return {key: rebind(item) for key, item in value.items()}
            if isinstance(value, list): return [rebind(item) for item in value]
            return value
        with path.open('rb') as original:
            source_hash = hashlib.file_digest(original, 'sha256').hexdigest()
        result = {'restored_from': source_hash,
                  'snapshot': rebind(manifest['snapshot']), 'jobs_resumable': False}
        (staged / 'restored-snapshot.json').write_text(json.dumps(result, ensure_ascii=False))
        staged.rename(destination)
        return result
    finally:
        if staged.exists(): shutil.rmtree(staged)

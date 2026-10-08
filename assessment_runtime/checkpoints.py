"""Immutable stage artifacts and atomic manifests for explicit assessment recovery."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
import re
from pathlib import Path
from uuid import uuid4

from app_core.cloud_policy import atomic_json, locked_file


class StageCache:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = root / 'manifest.json'

    def _read(self):
        if not self.manifest.exists():
            return {'version': 1, 'attempt_id': uuid4().hex, 'created_at': datetime.now().isoformat(timespec='seconds'), 'stages': {}}
        result = json.loads(self.manifest.read_text())
        if result.get('version') != 1 or not isinstance(result.get('stages'), dict) or not isinstance(result.get('attempt_id'), str):
            raise ValueError('Unsupported assessment checkpoint version.')
        return result

    def run(self, name: str, identity: dict, produce):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        with locked_file(self.root / 'checkpoint.lock'):
            manifest = self._read()
            saved = manifest['stages'].get(name)
            if saved and saved['key'] == key:
                if Path(saved['file']).name != saved['file']:
                    raise ValueError('Invalid assessment checkpoint artifact path.')
                path = self.root / saved['file']
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != saved['sha256']:
                    raise ValueError('Assessment checkpoint integrity failed; repeat the stage explicitly.')
                return json.loads(raw)
        value = produce()
        # Unique files ensure interrupted writes never change a previously published artifact.
        artifact = self.root / (uuid4().hex + '.json')
        atomic_json(artifact, value)
        with locked_file(self.root / 'checkpoint.lock'):
            manifest = self._read()
            manifest['stages'][name] = {'key': key, 'file': artifact.name, 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
            atomic_json(self.manifest, manifest)
        return value

    def attempt_id(self) -> str:
        with locked_file(self.root / 'checkpoint.lock'):
            value = self._read()
            atomic_json(self.manifest, value)
            return value['attempt_id']

    def created_at(self) -> str | None:
        with locked_file(self.root / 'checkpoint.lock'):
            return self._read().get('created_at')

    def seed_attempt(self, attempt_id: str, created_at: str | None = None) -> None:
        """Preserve an older retained take when its job predates checkpoints."""
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', attempt_id):
            raise ValueError('The retained take has an invalid identifier.')
        with locked_file(self.root / 'checkpoint.lock'):
            if self.manifest.exists():
                return
            value = self._read()
            value['attempt_id'] = attempt_id
            if created_at:
                datetime.fromisoformat(created_at)
                value['created_at'] = created_at
            atomic_json(self.manifest, value)

    def reject_reply(self, name: str, text_hash: str) -> bool:
        """Unpublish a validation failure without deleting its audit artifact."""
        with locked_file(self.root / 'checkpoint.lock'):
            value = self._read()
            saved = value['stages'].get(name)
            if not saved:
                return False
            if Path(saved['file']).name != saved['file']:
                raise ValueError('Invalid assessment checkpoint artifact path.')
            raw = (self.root / saved['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != saved['sha256']:
                raise ValueError('Assessment checkpoint integrity failed.')
            text = str(json.loads(raw).get('text') or '')
            if hashlib.sha256(text.encode()).hexdigest() != text_hash:
                return False
            del value['stages'][name]
            atomic_json(self.manifest, value)
            return True

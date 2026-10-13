"""Explicit cloud settings and durable per-request spending reservations."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import tempfile
import threading
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

_LOCK = threading.RLock()
_FILE_LOCKS: dict[str, threading.RLock] = {}


class CloudPolicyError(RuntimeError):
    pass


class CloudSettings(BaseModel):
    version: int = 1
    asr_provider: str = 'local'
    asr_connection_id: str = ''
    asr_model: str = 'whisper-large-v3'
    openrouter_modes: dict[str, str] = Field(default_factory=dict)
    fallback_connection_id: str = ''
    paid_fallback_enabled: bool = False
    monthly_budget_usd: float = Field(default=5, gt=0, le=1000, allow_inf_nan=False)
    max_output_tokens: int = Field(default=4096, ge=256, le=8192)

    @model_validator(mode='after')
    def valid_policy(self):
        if self.version != 1 or self.asr_provider not in {'local', 'groq'}:
            raise ValueError('Unsupported cloud settings version or transcription provider.')
        if self.asr_provider == 'groq' and (not self.asr_connection_id or self.asr_model not in {'whisper-large-v3', 'whisper-large-v3-turbo'}):
            raise ValueError('Choose a saved Groq connection and supported transcription model.')
        if any(mode not in {'free', 'paid'} for mode in self.openrouter_modes.values()):
            raise ValueError('Choose free-only or paid OpenRouter access.')
        if self.paid_fallback_enabled and (not self.fallback_connection_id or self.openrouter_modes.get(self.fallback_connection_id) != 'paid'):
            raise ValueError('Paid fallback requires an explicitly selected paid connection.')
        return self


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        if hasattr(os, 'O_DIRECTORY'):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def locked_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        thread_lock = _FILE_LOCKS.setdefault(str(path.resolve()), threading.RLock())
    with thread_lock, path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            handle.write(b'\0')
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_settings(root: Path) -> CloudSettings:
    path = root / 'cloud-settings.json'
    return CloudSettings.model_validate_json(path.read_text()) if path.exists() else CloudSettings()


def write_settings(root: Path, settings: CloudSettings) -> None:
    with locked_file(root / 'cloud-settings.lock'):
        atomic_json(root / 'cloud-settings.json', settings.model_dump())


def money(value) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise CloudPolicyError('Provider price or cost is unavailable.') from None
    if not result.is_finite() or result < 0:
        raise CloudPolicyError('Provider price or cost is invalid.')
    return result


class SpendingLedger:
    def __init__(self, root: Path):
        self.path = root / 'cloud-spending.json'
        self.lock = root / 'cloud-spending.lock'

    def _read(self):
        if not self.path.exists():
            return {'version': 1, 'requests': {}}
        value = json.loads(self.path.read_text())
        if value.get('version') != 1 or not isinstance(value.get('requests'), dict):
            raise CloudPolicyError('Spending ledger needs recovery before paid requests.')
        return value

    def reserve(self, amount: Decimal, budget: float, connection_id: str) -> str:
        amount = money(amount)
        month = datetime.now(UTC).strftime('%Y-%m')
        with locked_file(self.lock):
            value = self._read()
            # Unresolved requests remain reserved across month boundaries.
            used = sum((money(row['amount']) for row in value['requests'].values()
                        if row['month'] == month or row['status'] == 'pending'), Decimal(0))
            if used + amount > money(budget):
                raise CloudPolicyError('Monthly app budget exhausted; retained work can be retried after reconciliation.')
            request_id = uuid4().hex
            value['requests'][request_id] = {'month': month, 'amount': str(amount), 'status': 'pending', 'connection_id': connection_id}
            atomic_json(self.path, value)
            return request_id

    def reconcile(self, request_id: str, cost, *, source: str = 'provider') -> None:
        if cost is None:
            return
        actual = money(cost)
        with locked_file(self.lock):
            value = self._read()
            row = value['requests'].get(request_id)
            if not row or row['status'] != 'pending':
                raise CloudPolicyError('Choose an unresolved spending reservation.')
            row.update(amount=str(actual), status='reported', reconciliation_source=source,
                       reconciled_at=datetime.now(UTC).isoformat())
            atomic_json(self.path, value)

    def summary(self) -> dict:
        with locked_file(self.lock):
            rows = self._read()['requests']
        month = datetime.now(UTC).strftime('%Y-%m')
        return {'month': month, 'spent_usd': float(sum((money(row['amount']) for row in rows.values() if row['month'] == month and row['status'] == 'reported'), Decimal(0))),
                'reserved_usd': float(sum((money(row['amount']) for row in rows.values() if row['status'] == 'pending'), Decimal(0))),
                'unresolved_requests': [key for key, row in rows.items() if row['status'] == 'pending']}

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Optional, Protocol

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - handled when Redis store is requested
    redis = None  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class JobRecord:
    job_id: str
    chat_id: int
    message_id: int
    file_id: str
    status: str
    created_at: str
    updated_at: str
    error: str = ""
    report_path: str = ""


class JobStore(Protocol):
    def create(self, *, chat_id: int, message_id: int, file_id: str) -> JobRecord: ...
    def update(self, job_id: str, *, status: str, error: str = "", report_path: str = "") -> None: ...
    def as_dict(self, job_id: str) -> Optional[dict]: ...
    def enqueue_telegram(self, *, job_id: str, chat_id: int, message_id: int, file_id: str) -> None: ...
    def dequeue_telegram(self, timeout_sec: int = 1) -> Optional[dict]: ...
    def acknowledge_telegram(self, *, raw_payload: str) -> None: ...
    def requeue_processing_telegram(self) -> int: ...


class InMemoryJobStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(self, *, chat_id: int, message_id: int, file_id: str) -> JobRecord:
        now = _utc_now_iso()
        record = JobRecord(
            job_id=uuid.uuid4().hex,
            chat_id=chat_id,
            message_id=message_id,
            file_id=file_id,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[record.job_id] = record
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, *, status: str, error: str = "", report_path: str = "") -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.status = status
            record.updated_at = _utc_now_iso()
            if error:
                record.error = error
            if report_path:
                record.report_path = report_path

    def as_dict(self, job_id: str) -> Optional[dict]:
        record = self.get(job_id)
        if not record:
            return None
        return asdict(record)

    def enqueue_telegram(self, *, job_id: str, chat_id: int, message_id: int, file_id: str) -> None:
        # No-op in local mode: queueing is handled by app.state.executor.
        _ = (job_id, chat_id, message_id, file_id)

    def dequeue_telegram(self, timeout_sec: int = 1) -> Optional[dict]:
        _ = timeout_sec
        return None

    def acknowledge_telegram(self, *, raw_payload: str) -> None:
        _ = raw_payload

    def requeue_processing_telegram(self) -> int:
        return 0


class RedisJobStore:
    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "assess_speaking",
        job_ttl_sec: int = 604800,
        client=None,
    ):
        if client is None and redis is None:
            raise RuntimeError("redis package is not installed. Install dependencies via requirements.txt.")
        self._client = client or redis.Redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = key_prefix
        self.job_ttl_sec = job_ttl_sec

    def _job_key(self, job_id: str) -> str:
        return f"{self.key_prefix}:job:{job_id}"

    @property
    def _telegram_pending_queue_key(self) -> str:
        return f"{self.key_prefix}:queue:telegram:pending"

    @property
    def _telegram_processing_queue_key(self) -> str:
        return f"{self.key_prefix}:queue:telegram:processing"

    def _set_job(self, job_id: str, mapping: dict[str, str]) -> None:
        self._client.hset(self._job_key(job_id), mapping=mapping)
        if self.job_ttl_sec > 0:
            self._client.expire(self._job_key(job_id), self.job_ttl_sec)

    def create(self, *, chat_id: int, message_id: int, file_id: str) -> JobRecord:
        job_id = uuid.uuid4().hex
        now = _utc_now_iso()
        record = JobRecord(
            job_id=job_id,
            chat_id=chat_id,
            message_id=message_id,
            file_id=file_id,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        self._set_job(
            job_id,
            {
                "job_id": record.job_id,
                "chat_id": str(record.chat_id),
                "message_id": str(record.message_id),
                "file_id": record.file_id,
                "status": record.status,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "error": record.error,
                "report_path": record.report_path,
            },
        )
        return record

    def update(self, job_id: str, *, status: str, error: str = "", report_path: str = "") -> None:
        if not self._client.exists(self._job_key(job_id)):
            return
        data = {
            "status": status,
            "updated_at": _utc_now_iso(),
        }
        if error:
            data["error"] = error
        if report_path:
            data["report_path"] = report_path
        self._set_job(job_id, data)

    def as_dict(self, job_id: str) -> Optional[dict]:
        data = self._client.hgetall(self._job_key(job_id))
        if not data:
            return None
        return {
            "job_id": data.get("job_id", job_id),
            "chat_id": int(data.get("chat_id", "0")),
            "message_id": int(data.get("message_id", "0")),
            "file_id": data.get("file_id", ""),
            "status": data.get("status", ""),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "error": data.get("error", ""),
            "report_path": data.get("report_path", ""),
        }

    def enqueue_telegram(self, *, job_id: str, chat_id: int, message_id: int, file_id: str) -> None:
        payload = {
            "job_id": job_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "file_id": file_id,
        }
        # LPUSH + BRPOPLPUSH yields FIFO semantics while preserving in-flight items.
        self._client.lpush(self._telegram_pending_queue_key, json.dumps(payload))

    def dequeue_telegram(self, timeout_sec: int = 1) -> Optional[dict]:
        raw = self._client.brpoplpush(
            self._telegram_pending_queue_key,
            self._telegram_processing_queue_key,
            timeout=timeout_sec,
        )
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.acknowledge_telegram(raw_payload=raw)
            return None
        if not isinstance(payload, dict):
            self.acknowledge_telegram(raw_payload=raw)
            return None
        payload["_raw_payload"] = raw
        return payload

    def acknowledge_telegram(self, *, raw_payload: str) -> None:
        self._client.lrem(self._telegram_processing_queue_key, 1, raw_payload)

    def requeue_processing_telegram(self) -> int:
        items = self._client.lrange(self._telegram_processing_queue_key, 0, -1)
        if not items:
            return 0
        self._client.delete(self._telegram_processing_queue_key)
        for raw in items:
            self._client.rpush(self._telegram_pending_queue_key, raw)
        return len(items)

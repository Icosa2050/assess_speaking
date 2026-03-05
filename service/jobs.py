from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Optional


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


class InMemoryJobStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(self, *, chat_id: int, message_id: int, file_id: str) -> JobRecord:
        now = datetime.now(UTC).isoformat(timespec="seconds")
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
            record.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
            if error:
                record.error = error
            if report_path:
                record.report_path = report_path

    def as_dict(self, job_id: str) -> Optional[dict]:
        record = self.get(job_id)
        if not record:
            return None
        return asdict(record)

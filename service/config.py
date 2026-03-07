from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServiceConfig:
    telegram_bot_token: str
    telegram_webhook_secret: Optional[str]
    whisper_model: str
    llm_model: str
    feedback_enabled: bool
    train_dir: Path
    target_cefr: Optional[str]
    report_dir: Path
    temp_dir: Path
    max_workers: int
    redis_url: Optional[str]
    redis_key_prefix: str
    job_ttl_sec: int

    @classmethod
    def from_env(cls, *, strict: bool = True) -> "ServiceConfig":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if strict and not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required to run the Telegram service.")
        webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET")
        webhook_secret = webhook_secret.strip() if webhook_secret else None
        target_cefr = os.getenv("ASSESS_TARGET_CEFR")
        target_cefr = target_cefr.strip().upper() if target_cefr else None
        redis_url = os.getenv("SERVICE_REDIS_URL")
        redis_url = redis_url.strip() if redis_url else None

        return cls(
            telegram_bot_token=token,
            telegram_webhook_secret=webhook_secret,
            whisper_model=os.getenv("ASSESS_WHISPER_MODEL", "large-v3").strip(),
            llm_model=os.getenv("ASSESS_LLM_MODEL", "llama3.1").strip(),
            feedback_enabled=_env_bool("ASSESS_ENABLE_FEEDBACK", False),
            train_dir=Path(os.getenv("ASSESS_TRAIN_DIR", "training")),
            target_cefr=target_cefr,
            report_dir=Path(os.getenv("SERVICE_REPORT_DIR", "reports/service")),
            temp_dir=Path(os.getenv("SERVICE_TEMP_DIR", "tmp/service")),
            max_workers=max(1, int(os.getenv("SERVICE_MAX_WORKERS", "2"))),
            redis_url=redis_url,
            redis_key_prefix=os.getenv("SERVICE_REDIS_PREFIX", "assess_speaking"),
            job_ttl_sec=max(60, int(os.getenv("SERVICE_JOB_TTL_SEC", "604800"))),
        )

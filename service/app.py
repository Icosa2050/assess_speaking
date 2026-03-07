from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request

from assess_speaking import extract_rubric_json, run_assessment
from service.config import ServiceConfig
from service.jobs import InMemoryJobStore, JobStore, RedisJobStore
from service.telegram_client import TelegramClient


def extract_telegram_media(update: dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[str]]:
    message = (
        update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
    )
    if not isinstance(message, dict):
        return None, None, None

    chat = message.get("chat") or {}
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    message_id = message.get("message_id")

    voice = message.get("voice")
    if isinstance(voice, dict) and voice.get("file_id"):
        return chat_id, message_id, voice["file_id"]

    audio = message.get("audio")
    if isinstance(audio, dict) and audio.get("file_id"):
        return chat_id, message_id, audio["file_id"]

    document = message.get("document")
    if isinstance(document, dict):
        file_id = document.get("file_id")
        mime = str(document.get("mime_type", ""))
        if file_id and mime.startswith("audio/"):
            return chat_id, message_id, file_id

    return chat_id, message_id, None


def build_result_message(assessment: dict[str, Any]) -> str:
    metrics = assessment.get("metrics", {})
    report = assessment.get("report") or {}
    overall = ""
    final_score = ""
    requires_human_review = False
    if isinstance(report, dict):
        scores = report.get("scores") or {}
        if isinstance(scores, dict) and scores.get("final") is not None:
            final_score = str(scores["final"])
        requires_human_review = bool(report.get("requires_human_review", False))
    llm_rubric = assessment.get("llm_rubric")
    if isinstance(llm_rubric, str):
        rubric = extract_rubric_json(llm_rubric) or {}
        if "overall" in rubric:
            overall = str(rubric["overall"])

    lines = [
        "Valutazione completata.",
        f"WPM: {metrics.get('wpm', 'n/a')}",
        f"Parole: {metrics.get('word_count', 'n/a')}",
        f"Pause (totale s): {metrics.get('pause_total_sec', 'n/a')}",
    ]
    if final_score:
        lines.insert(1, f"Punteggio finale: {final_score}")
    if overall:
        insert_at = 2 if final_score else 1
        lines.insert(insert_at, f"Punteggio complessivo (LLM): {overall}")
    if requires_human_review:
        lines.append("Revisione umana consigliata.")
    return "\n".join(lines)


def _run_telegram_job(app: FastAPI, job_id: str, chat_id: int, message_id: int, file_id: str) -> None:
    cfg: ServiceConfig = app.state.config
    jobs: JobStore = app.state.jobs
    telegram: TelegramClient = app.state.telegram
    jobs.update(job_id, status="processing")
    try:
        with TemporaryDirectory(dir=str(cfg.temp_dir)) as td:
            tmp_dir = Path(td)
            remote_file_path = telegram.get_file_path(file_id)
            local_suffix = Path(remote_file_path).suffix or ".bin"
            local_input = tmp_dir / f"input{local_suffix}"
            telegram.download_file(remote_file_path, local_input)

            assessment = run_assessment(
                local_input,
                cfg.whisper_model,
                cfg.llm_model,
                feedback_enabled=cfg.feedback_enabled,
                train_dir=cfg.train_dir,
                target_cefr=cfg.target_cefr,
            )

            report_path = cfg.report_dir / f"telegram_{job_id}.json"
            report_payload = {
                "job_id": job_id,
                "chat_id": chat_id,
                "assessment": assessment,
            }
            report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            telegram.send_message(
                chat_id,
                build_result_message(assessment),
                reply_to_message_id=message_id,
            )
            telegram.send_document(
                chat_id,
                report_path,
                caption="Report completo in JSON.",
                reply_to_message_id=message_id,
            )
            jobs.update(job_id, status="done", report_path=str(report_path.resolve()))
    except Exception as exc:
        jobs.update(job_id, status="failed", error=str(exc))
        try:
            telegram.send_message(
                chat_id,
                f"Elaborazione fallita: {exc}",
                reply_to_message_id=message_id,
            )
        except Exception:
            pass


def _redis_worker_loop(app: FastAPI, stop_event: threading.Event) -> None:
    jobs: JobStore = app.state.jobs
    while not stop_event.is_set():
        payload = jobs.dequeue_telegram(timeout_sec=1)
        if not payload:
            continue
        raw_payload = str(payload.get("_raw_payload", ""))
        job_id = str(payload.get("job_id", ""))
        if not job_id:
            if raw_payload:
                jobs.acknowledge_telegram(raw_payload=raw_payload)
            continue
        try:
            chat_id = int(payload["chat_id"])
            message_id = int(payload["message_id"])
            file_id = str(payload["file_id"])
        except (KeyError, TypeError, ValueError):
            jobs.update(job_id, status="failed", error="Invalid payload in Redis queue.")
            if raw_payload:
                jobs.acknowledge_telegram(raw_payload=raw_payload)
            continue
        try:
            _run_telegram_job(app, job_id, chat_id, message_id, file_id)
        finally:
            if raw_payload:
                jobs.acknowledge_telegram(raw_payload=raw_payload)


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    cfg = config or ServiceConfig.from_env(strict=False)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)

    if cfg.redis_url:
        jobs: JobStore = RedisJobStore(
            cfg.redis_url,
            key_prefix=cfg.redis_key_prefix,
            job_ttl_sec=cfg.job_ttl_sec,
        )
        queue_backend = "redis"
        executor: Optional[ThreadPoolExecutor] = None
    else:
        jobs = InMemoryJobStore()
        queue_backend = "in_memory"
        executor = ThreadPoolExecutor(max_workers=cfg.max_workers)

    @asynccontextmanager
    async def lifespan(app_obj: FastAPI):
        if app_obj.state.queue_backend == "redis":
            app_obj.state.recovered_jobs = app_obj.state.jobs.requeue_processing_telegram()
            workers = []
            for idx in range(cfg.max_workers):
                worker = threading.Thread(
                    target=_redis_worker_loop,
                    args=(app_obj, app_obj.state.stop_event),
                    daemon=True,
                    name=f"redis-worker-{idx+1}",
                )
                worker.start()
                workers.append(worker)
            app_obj.state.worker_threads = workers
        yield
        app_obj.state.stop_event.set()
        for worker in app_obj.state.worker_threads:
            worker.join(timeout=2.0)
        if app_obj.state.executor:
            app_obj.state.executor.shutdown(wait=False)

    app = FastAPI(title="assess_speaking service", version="0.1.0", lifespan=lifespan)
    app.state.config = cfg
    app.state.jobs = jobs
    app.state.executor = executor
    app.state.queue_backend = queue_backend
    app.state.stop_event = threading.Event()
    app.state.worker_threads = []
    app.state.recovered_jobs = 0
    app.state.telegram = TelegramClient(cfg.telegram_bot_token) if cfg.telegram_bot_token else None

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "telegram_configured": bool(app.state.telegram),
            "max_workers": cfg.max_workers,
            "queue_backend": app.state.queue_backend,
            "recovered_jobs": app.state.recovered_jobs,
        }

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        payload = app.state.jobs.as_dict(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return payload

    @app.post("/webhooks/telegram")
    async def telegram_webhook(
        request: Request,
        secret_header: Optional[str] = Header(default=None, alias="X-Telegram-Bot-Api-Secret-Token"),
    ) -> dict[str, Any]:
        if not app.state.telegram:
            raise HTTPException(status_code=503, detail="Telegram is not configured (missing TELEGRAM_BOT_TOKEN).")
        if cfg.telegram_webhook_secret and secret_header != cfg.telegram_webhook_secret:
            raise HTTPException(status_code=401, detail="Invalid webhook secret.")

        update = await request.json()
        chat_id, message_id, file_id = extract_telegram_media(update)
        if chat_id is None or message_id is None:
            return {"ok": True, "queued": False, "reason": "unsupported_update"}

        if not file_id:
            try:
                app.state.telegram.send_message(
                    chat_id,
                    "Inviami un messaggio vocale o un file audio per iniziare la valutazione.",
                    reply_to_message_id=message_id,
                )
            except Exception:
                pass
            return {"ok": True, "queued": False, "reason": "no_audio"}

        record = app.state.jobs.create(chat_id=chat_id, message_id=message_id, file_id=file_id)
        if app.state.queue_backend == "redis":
            app.state.jobs.enqueue_telegram(
                job_id=record.job_id,
                chat_id=chat_id,
                message_id=message_id,
                file_id=file_id,
            )
        else:
            app.state.executor.submit(_run_telegram_job, app, record.job_id, chat_id, message_id, file_id)
        return {"ok": True, "queued": True, "job_id": record.job_id}

    return app


app = create_app()

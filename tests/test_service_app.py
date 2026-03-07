import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from service.app import create_app, extract_telegram_media
from service.config import ServiceConfig


class _DummyExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        return None

    def shutdown(self, wait=False):
        return None


def _config(tmp_path: Path, token: str = "test-token", secret: str | None = None) -> ServiceConfig:
    return ServiceConfig(
        telegram_bot_token=token,
        telegram_webhook_secret=secret,
        whisper_model="large-v3",
        llm_model="llama3.1",
        feedback_enabled=False,
        train_dir=tmp_path / "training",
        target_cefr=None,
        report_dir=tmp_path / "reports",
        temp_dir=tmp_path / "tmp",
        max_workers=1,
        redis_url=None,
        redis_key_prefix="assess_speaking",
        job_ttl_sec=3600,
    )


class TelegramParsingTests(unittest.TestCase):
    def test_extract_telegram_media_from_voice_message(self):
        update = {
            "message": {
                "message_id": 123,
                "chat": {"id": 55},
                "voice": {"file_id": "voice-file-id"},
            }
        }
        chat_id, message_id, file_id = extract_telegram_media(update)
        self.assertEqual(chat_id, 55)
        self.assertEqual(message_id, 123)
        self.assertEqual(file_id, "voice-file-id")

    def test_extract_telegram_media_without_audio(self):
        update = {"message": {"message_id": 7, "chat": {"id": 42}, "text": "ciao"}}
        chat_id, message_id, file_id = extract_telegram_media(update)
        self.assertEqual(chat_id, 42)
        self.assertEqual(message_id, 7)
        self.assertIsNone(file_id)


class TelegramWebhookTests(unittest.TestCase):
    def test_health_includes_queue_backend(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(Path(td))
            app = create_app(cfg)
            with TestClient(app) as client:
                resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["queue_backend"], "in_memory")
            self.assertEqual(resp.json()["recovered_jobs"], 0)

    def test_webhook_rejects_invalid_secret(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(Path(td), secret="expected-secret")
            app = create_app(cfg)
            with TestClient(app) as client:
                resp = client.post("/webhooks/telegram", json={"update_id": 1})
            self.assertEqual(resp.status_code, 401)

    def test_webhook_requires_token_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(Path(td), token="")
            app = create_app(cfg)
            with TestClient(app) as client:
                resp = client.post("/webhooks/telegram", json={"update_id": 1})
            self.assertEqual(resp.status_code, 503)

    def test_webhook_queues_voice_job(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(Path(td))
            app = create_app(cfg)
            dummy = _DummyExecutor()
            app.state.executor = dummy

            payload = {
                "update_id": 99,
                "message": {
                    "message_id": 18,
                    "chat": {"id": 77},
                    "voice": {"file_id": "voice-id"},
                },
            }
            with TestClient(app) as client:
                resp = client.post("/webhooks/telegram", json=payload)

            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body["queued"])
            self.assertTrue(body["job_id"])
            self.assertEqual(len(dummy.calls), 1)


if __name__ == "__main__":
    unittest.main()

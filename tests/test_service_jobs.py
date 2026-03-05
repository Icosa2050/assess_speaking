import json
import unittest

from service.jobs import RedisJobStore


class _FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.lists = {}
        self.expiries = {}

    def hset(self, key, mapping):
        self.hashes[key] = {**self.hashes.get(key, {}), **mapping}

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def expire(self, key, ttl):
        self.expiries[key] = ttl

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def blpop(self, key, timeout=1):
        queue = self.lists.get(key, [])
        if not queue:
            return None
        value = queue.pop(0)
        return key, value


class RedisJobStoreTests(unittest.TestCase):
    def test_create_update_and_read_job(self):
        fake = _FakeRedis()
        store = RedisJobStore(
            "redis://unused",
            key_prefix="test",
            job_ttl_sec=600,
            client=fake,
        )

        record = store.create(chat_id=77, message_id=12, file_id="abc")
        payload = store.as_dict(record.job_id)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["chat_id"], 77)
        self.assertEqual(payload["status"], "queued")

        store.update(record.job_id, status="done", report_path="/tmp/report.json")
        payload2 = store.as_dict(record.job_id)
        self.assertEqual(payload2["status"], "done")
        self.assertEqual(payload2["report_path"], "/tmp/report.json")

    def test_enqueue_and_dequeue_telegram_payload(self):
        fake = _FakeRedis()
        store = RedisJobStore(
            "redis://unused",
            key_prefix="test",
            job_ttl_sec=600,
            client=fake,
        )

        store.enqueue_telegram(job_id="j1", chat_id=1, message_id=2, file_id="f1")
        payload = store.dequeue_telegram(timeout_sec=1)
        self.assertEqual(payload["job_id"], "j1")
        self.assertEqual(payload["chat_id"], 1)
        self.assertEqual(payload["message_id"], 2)
        self.assertEqual(payload["file_id"], "f1")

    def test_dequeue_ignores_invalid_json(self):
        fake = _FakeRedis()
        store = RedisJobStore(
            "redis://unused",
            key_prefix="test",
            job_ttl_sec=600,
            client=fake,
        )
        fake.rpush("test:queue:telegram", "{invalid")
        payload = store.dequeue_telegram(timeout_sec=1)
        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()


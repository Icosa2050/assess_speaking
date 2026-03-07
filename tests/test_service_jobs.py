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

    def exists(self, key):
        return key in self.hashes

    def expire(self, key, ttl):
        self.expiries[key] = ttl

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def brpoplpush(self, source, destination, timeout=1):
        _ = timeout
        source_queue = self.lists.get(source, [])
        if not source_queue:
            return None
        value = source_queue.pop()
        self.lists.setdefault(destination, []).insert(0, value)
        return value

    def lrem(self, key, count, value):
        queue = self.lists.get(key, [])
        removed = 0
        new_queue = []
        for item in queue:
            if item == value and (count == 0 or removed < count):
                removed += 1
                continue
            new_queue.append(item)
        self.lists[key] = new_queue
        return removed

    def lrange(self, key, start, end):
        queue = self.lists.get(key, [])
        if end == -1:
            end = len(queue) - 1
        return queue[start : end + 1]

    def delete(self, key):
        self.lists.pop(key, None)
        self.hashes.pop(key, None)


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
        self.assertIn("_raw_payload", payload)
        store.acknowledge_telegram(raw_payload=payload["_raw_payload"])
        self.assertEqual(fake.lists["test:queue:telegram:processing"], [])

    def test_dequeue_ignores_invalid_json(self):
        fake = _FakeRedis()
        store = RedisJobStore(
            "redis://unused",
            key_prefix="test",
            job_ttl_sec=600,
            client=fake,
        )
        fake.lpush("test:queue:telegram:pending", "{invalid")
        payload = store.dequeue_telegram(timeout_sec=1)
        self.assertIsNone(payload)
        self.assertEqual(fake.lists["test:queue:telegram:processing"], [])

    def test_requeue_processing_moves_inflight_jobs_back_to_pending(self):
        fake = _FakeRedis()
        store = RedisJobStore(
            "redis://unused",
            key_prefix="test",
            job_ttl_sec=600,
            client=fake,
        )
        fake.lists["test:queue:telegram:processing"] = ["a", "b"]
        count = store.requeue_processing_telegram()
        self.assertEqual(count, 2)
        self.assertEqual(fake.lists.get("test:queue:telegram:processing"), None)
        self.assertEqual(fake.lists["test:queue:telegram:pending"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

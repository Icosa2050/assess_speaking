import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from app_backend.contracts import ErrorCode
from app_shell import backend_client


class _FakeResponse:
    def __init__(
        self,
        payload=None,
        *,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = {} if payload is None else payload
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._payload


def _http_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "http://backend.local/test"),
    )


class BackendClientTests(unittest.TestCase):
    def test_backend_base_url_reuses_state_or_starts_backend(self):
        with mock.patch("app_shell.backend_client.get_backend_state", return_value={"base_url": "http://ready"}) as get_state, \
                mock.patch("app_shell.backend_client.ensure_local_backend") as ensure_backend:
            self.assertEqual(backend_client.backend_base_url(log_dir="reports"), "http://ready")
        get_state.assert_called_once_with(log_dir="reports")
        ensure_backend.assert_not_called()

        with mock.patch("app_shell.backend_client.get_backend_state", return_value=None), \
                mock.patch("app_shell.backend_client.ensure_local_backend", return_value={"base_url": "http://started"}) as ensure_backend:
            self.assertEqual(backend_client.backend_base_url(log_dir="reports"), "http://started")
        ensure_backend.assert_called_once_with(log_dir="reports")

    def test_request_wraps_structured_backend_status_errors(self):
        response = _http_response(
            422,
            {
                "detail": {
                    "code": ErrorCode.VALIDATION.value,
                    "detail": "bad cleanup target",
                }
            },
        )

        with mock.patch("app_shell.backend_client.backend_base_url", return_value="http://backend.local"), \
                mock.patch("app_shell.backend_client.httpx.request", return_value=response):
            with self.assertRaises(RuntimeError) as raised:
                backend_client._request("POST", "/v1/maintenance/cleanup", json={"target": "bad"})

        payload = json.loads(str(raised.exception))
        self.assertEqual(payload["code"], ErrorCode.VALIDATION.value)
        self.assertEqual(payload["detail"], "bad cleanup target")

    def test_request_wraps_unreachable_backend_errors(self):
        request = httpx.Request("GET", "http://backend.local/v1/health")
        with mock.patch("app_shell.backend_client.backend_base_url", return_value="http://backend.local"), \
                mock.patch(
                    "app_shell.backend_client.httpx.request",
                    side_effect=httpx.ConnectError("connection refused", request=request),
                ):
            with self.assertRaises(RuntimeError) as raised:
                backend_client._request("GET", "/v1/health")

        payload = json.loads(str(raised.exception))
        self.assertEqual(payload["code"], ErrorCode.BACKEND_UNAVAILABLE.value)
        self.assertIn("connection refused", payload["detail"])

    def test_endpoint_helpers_parse_responses_and_forward_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            audio_path.write_bytes(b"audio")
            download_dir = Path(tmpdir) / "downloads"
            download_dir.mkdir()

            responses = [
                _FakeResponse({"audio_id": "aud_1", "stored_path": str(audio_path), "sha1": "abc", "original_name": "renamed.wav"}),
                _FakeResponse({"assessment_id": "asmt_1", "status": "queued"}),
                _FakeResponse({"assessment_id": "asmt_1", "status": "running", "phase": "transcribing", "progress": 0.4}),
                _FakeResponse({"assessment_id": "asmt_1", "status": "cancelled", "phase": "cancelled", "progress": 1.0}),
                _FakeResponse({"items": [{"session_id": "sess_1"}]}),
                _FakeResponse({"payload": {"report": {"session_id": "sess_1"}}}),
                _FakeResponse({"items": [{"sample_id": "sample_1"}]}),
                _FakeResponse(
                    {
                        "app_data_root": tmpdir,
                        "cache_root": tmpdir,
                        "areas": {
                            "tmp": {
                                "path": tmpdir,
                                "size_bytes": 4,
                                "file_count": 1,
                            }
                        },
                    }
                ),
                _FakeResponse({"target": "tmp", "dry_run": True, "deleted_file_count": 1, "freed_bytes": 4, "warnings": []}),
                _FakeResponse({"bundle_id": "bundle_123", "filename": "bundle_123.zip", "size_bytes": 8, "expires_at": "2026-04-23T12:00:00+00:00"}),
                _FakeResponse(
                    content=b"zip-bytes",
                    headers={"content-disposition": 'attachment; filename="support.zip"'},
                ),
            ]

            with mock.patch("app_shell.backend_client._request", side_effect=responses) as request:
                upload = backend_client.upload_audio_path(audio_path, filename="renamed.wav", log_dir="logs")
                created = backend_client.create_assessment({"audio_id": upload.audio_id}, log_dir="logs")
                status = backend_client.get_assessment_status(created.assessment_id, log_dir="logs")
                cancelled = backend_client.cancel_assessment(created.assessment_id, log_dir="logs")
                history = backend_client.load_history(log_dir="logs")
                detail = backend_client.load_history_detail("sess_1", log_dir="logs")
                samples = backend_client.load_samples(log_dir="logs")
                storage = backend_client.get_maintenance_storage(log_dir="logs")
                cleanup = backend_client.post_maintenance_cleanup({"target": "tmp", "dry_run": True}, log_dir="logs")
                bundle = backend_client.create_support_bundle({"client_snapshot": {}}, log_dir="logs")
                downloaded = backend_client.download_support_bundle(bundle.bundle_id, destination=download_dir, log_dir="logs")

            self.assertEqual(downloaded.name, "support.zip")
            self.assertEqual(downloaded.read_bytes(), b"zip-bytes")

        self.assertEqual(upload.original_name, "renamed.wav")
        self.assertEqual(status.phase, "transcribing")
        self.assertEqual(cancelled.status.value, "cancelled")
        self.assertEqual(history, [{"session_id": "sess_1"}])
        self.assertEqual(detail["report"]["session_id"], "sess_1")
        self.assertEqual(samples, [{"sample_id": "sample_1"}])
        self.assertEqual(storage.areas["tmp"].file_count, 1)
        self.assertEqual(cleanup.freed_bytes, 4)
        self.assertEqual(bundle.filename, "bundle_123.zip")
        self.assertEqual(request.call_count, 11)

    def test_backend_path_segments_are_percent_encoded(self):
        captured: list[str] = []

        def fake_request(method: str, url: str, **kwargs):
            captured.append(url)
            if url.endswith("/cancel"):
                payload = {"assessment_id": "asmt/1 ?", "status": "cancelled", "phase": "cancelled", "progress": 1.0}
            elif "/history/" in url:
                payload = {"payload": {"report": {"session_id": "sess/1 ?"}}}
            elif "/support-bundles/" in url:
                return httpx.Response(200, content=b"zip", request=httpx.Request(method, url))
            else:
                payload = {"assessment_id": "asmt/1 ?", "status": "running", "phase": "transcribing", "progress": 0.4}
            return httpx.Response(200, json=payload, request=httpx.Request(method, url))

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "app_shell.backend_client.backend_base_url",
            return_value="http://backend.local",
        ), mock.patch("app_shell.backend_client.httpx.request", side_effect=fake_request):
            backend_client.get_assessment_status("asmt/1 ?")
            backend_client.cancel_assessment("asmt/1 ?")
            backend_client.load_history_detail("sess/1 ?")
            backend_client.download_support_bundle("bundle/1 ?", destination=Path(tmpdir))

        self.assertEqual(
            captured,
            [
                "http://backend.local/v1/assessments/asmt%2F1%20%3F",
                "http://backend.local/v1/assessments/asmt%2F1%20%3F/cancel",
                "http://backend.local/v1/history/sess%2F1%20%3F",
                "http://backend.local/v1/support-bundles/bundle%2F1%20%3F",
            ],
        )

    def test_download_support_bundle_parses_encoded_content_disposition_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir)
            response = _FakeResponse(
                content=b"zip-bytes",
                headers={"content-disposition": "attachment; filename*=UTF-8''support%20bundle.zip"},
            )

            with mock.patch("app_shell.backend_client._request", return_value=response):
                downloaded = backend_client.download_support_bundle("bundle_123", destination=download_dir)

            self.assertEqual(downloaded.name, "support bundle.zip")
            self.assertEqual(downloaded.read_bytes(), b"zip-bytes")

    def test_load_history_detail_rejects_invalid_payload_shape(self):
        with mock.patch("app_shell.backend_client._request", return_value=_FakeResponse({"payload": []})):
            with self.assertRaises(RuntimeError) as raised:
                backend_client.load_history_detail("sess_bad")

        payload = json.loads(str(raised.exception))
        self.assertEqual(payload["code"], ErrorCode.RUNTIME.value)
        self.assertIn("sess_bad", payload["detail"])

    def test_collection_helpers_tolerate_non_mapping_payloads(self):
        with mock.patch("app_shell.backend_client._request", side_effect=[_FakeResponse([]), _FakeResponse([])]):
            self.assertEqual(backend_client.load_history(), [])
            self.assertEqual(backend_client.load_samples(), [])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import lms


class CanvasClientTests(unittest.TestCase):
    def test_check_response_sanitizes_error_body(self):
        client = lms.CanvasClient(base_url="https://canvas.example.edu/", token="token123")
        mock_response = mock.Mock()
        mock_response.status_code = 403
        mock_response.text = "x" * 250
        http_error = type("HTTPError", (Exception,), {})
        mock_response.raise_for_status.side_effect = http_error("forbidden")

        with mock.patch.object(lms, "requests", mock.Mock(HTTPError=http_error)):
            with self.assertRaises(RuntimeError) as ctx:
                client._check_response(mock_response)

        self.assertIn("HTTP 403", str(ctx.exception))
        self.assertIn("x" * 200, str(ctx.exception))
        self.assertNotIn("x" * 220, str(ctx.exception))

    def test_upload_submission_uses_course_and_assignment_in_url(self):
        requests_module = mock.Mock()
        start_response = mock.Mock()
        start_response.raise_for_status.return_value = None
        start_response.json.return_value = {
            "upload_url": "https://uploads.example.edu/files",
            "upload_params": {"key": "value"},
        }
        upload_response = mock.Mock()
        upload_response.status_code = 201
        upload_response.headers = {"Location": "https://canvas.example.edu/api/v1/files/555"}
        finalize_response = mock.Mock()
        finalize_response.raise_for_status.return_value = None
        finalize_response.json.return_value = {"id": 555}
        submit_response = mock.Mock()
        submit_response.raise_for_status.return_value = None

        requests_module.post.side_effect = [start_response, upload_response, submit_response]
        requests_module.get.return_value = finalize_response

        with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
            fh.write(b"{}")
            attachment = Path(fh.name)
        self.addCleanup(lambda: attachment.unlink(missing_ok=True))

        with mock.patch.object(lms, "requests", requests_module):
            client = lms.CanvasClient(base_url="https://canvas.example.edu/", token="token123")
            ok = client.upload_submission(
                course_id=77,
                assignment_id=42,
                submission_data={"comment": "Suggested score: 93.5"},
                attachment_path=attachment,
            )

        self.assertTrue(ok)
        self.assertEqual(
            requests_module.post.call_args_list[0].args[0],
            "https://canvas.example.edu/api/v1/courses/77/assignments/42/submissions/self/files",
        )
        self.assertEqual(
            requests_module.post.call_args_list[2].args[0],
            "https://canvas.example.edu/api/v1/courses/77/assignments/42/submissions",
        )
        self.assertEqual(
            requests_module.post.call_args_list[2].kwargs["headers"],
            {"Authorization": "Bearer token123"},
        )
        self.assertEqual(
            requests_module.post.call_args_list[2].kwargs["data"],
            [
                ("submission[submission_type]", "online_upload"),
                ("submission[file_ids][]", "555"),
                ("comment[text_comment]", "Suggested score: 93.5"),
            ],
        )
        self.assertEqual(
            requests_module.get.call_args.args[0],
            "https://canvas.example.edu/api/v1/files/555",
        )

    def test_upload_submission_accepts_empty_upload_params(self):
        requests_module = mock.Mock()
        start_response = mock.Mock()
        start_response.raise_for_status.return_value = None
        start_response.json.return_value = {
            "upload_url": "https://uploads.example.edu/files",
            "upload_params": {},
        }
        upload_response = mock.Mock()
        upload_response.status_code = 200
        upload_response.raise_for_status.return_value = None
        upload_response.json.return_value = {"id": 777}
        submit_response = mock.Mock()
        submit_response.raise_for_status.return_value = None

        requests_module.post.side_effect = [start_response, upload_response, submit_response]

        with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
            fh.write(b"{}")
            attachment = Path(fh.name)
        self.addCleanup(lambda: attachment.unlink(missing_ok=True))

        with mock.patch.object(lms, "requests", requests_module):
            client = lms.CanvasClient(base_url="https://canvas.example.edu/", token="token123")
            ok = client.upload_submission(
                course_id=77,
                assignment_id=42,
                submission_data={},
                attachment_path=attachment,
            )

        self.assertTrue(ok)
        self.assertEqual(requests_module.post.call_args_list[1].kwargs["data"], {})

    @unittest.skipIf(lms.requests is None, "requests is required for LMS tests")
    def test_upload_submission_wraps_timeout(self):
        client = lms.CanvasClient(
            base_url="https://canvas.example.edu/",
            token="token123",
            timeout_sec=5.0,
        )
        with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
            fh.write(b"{}")
            attachment = Path(fh.name)
        self.addCleanup(lambda: attachment.unlink(missing_ok=True))

        with mock.patch.object(lms.requests, "post", side_effect=lms.requests.Timeout("slow")):
            with self.assertRaises(RuntimeError) as ctx:
                client.upload_submission(
                    course_id=77,
                    assignment_id=42,
                    submission_data={},
                    attachment_path=attachment,
                )

        self.assertIn("timed out after 5.0s", str(ctx.exception))


class UploadHelpersTests(unittest.TestCase):
    def test_build_moodle_submission_data_includes_score_and_resources(self):
        payload = lms.build_moodle_submission_data(
            score=81.5,
            resources=[
                {"title": "Listening Drill", "url": "https://example.org/listening"},
                {"url": "https://example.org/fluency"},
            ],
        )

        self.assertEqual(payload["attachments"], [])
        self.assertIn("Suggested score: 81.5", payload["comment"])
        self.assertIn("Suggested training resources:", payload["comment"])
        self.assertIn("- Listening Drill: https://example.org/listening", payload["comment"])
        self.assertIn("- Resource: https://example.org/fluency", payload["comment"])

    @mock.patch.object(lms.CanvasClient, "upload_submission", return_value=True)
    def test_upload_to_canvas_passes_course_id_and_resource_comment(self, mock_upload):
        with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
            fh.write(b"{}")
            attachment = Path(fh.name)
        self.addCleanup(lambda: attachment.unlink(missing_ok=True))

        resources = [
            {"title": "Grammar Drill", "url": "https://example.org/grammar"},
            {"url": "https://example.org/vocab"},
        ]

        result = lms.upload_to_canvas(
            base_url="https://canvas.example.edu",
            token="abc",
            course_id=12,
            assignment_id=34,
            score=88.0,
            attachment_path=attachment,
            resources=resources,
        )

        self.assertTrue(result)
        kwargs = mock_upload.call_args.kwargs
        self.assertEqual(kwargs["course_id"], 12)
        self.assertEqual(kwargs["assignment_id"], 34)
        self.assertEqual(kwargs["submission_data"]["score"], 88.0)
        comment = kwargs["submission_data"]["comment"]
        self.assertIn("Suggested score: 88.0", comment)
        self.assertIn("Suggested training resources:", comment)
        self.assertIn("- Grammar Drill: https://example.org/grammar", comment)
        self.assertIn("- Resource: https://example.org/vocab", comment)

    @mock.patch.object(lms.CanvasClient, "upload_submission", return_value=True)
    def test_upload_to_canvas_omits_comment_without_resources(self, mock_upload):
        with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
            fh.write(b"{}")
            attachment = Path(fh.name)
        self.addCleanup(lambda: attachment.unlink(missing_ok=True))

        lms.upload_to_canvas(
            base_url="https://canvas.example.edu",
            token="abc",
            course_id=7,
            assignment_id=9,
            score=None,
            attachment_path=attachment,
            resources=None,
        )

        submission_data = mock_upload.call_args.kwargs["submission_data"]
        self.assertEqual(submission_data, {})
        self.assertNotIn("comment", submission_data)


if __name__ == "__main__":
    unittest.main()

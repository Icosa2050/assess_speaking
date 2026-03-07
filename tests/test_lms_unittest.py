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
        mock_response.raise_for_status.side_effect = lms.requests.HTTPError("forbidden")

        with self.assertRaises(RuntimeError) as ctx:
            client._check_response(mock_response)

        self.assertIn("HTTP 403", str(ctx.exception))
        self.assertIn("x" * 200, str(ctx.exception))
        self.assertNotIn("x" * 220, str(ctx.exception))

    def test_upload_submission_uses_course_and_assignment_in_url(self):
        requests_module = mock.Mock()
        mock_response = mock.Mock()
        mock_response.raise_for_status.return_value = None
        requests_module.post.return_value = mock_response

        with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
            fh.write(b"{}")
            attachment = Path(fh.name)
        self.addCleanup(lambda: attachment.unlink(missing_ok=True))

        with mock.patch.object(lms, "requests", requests_module):
            client = lms.CanvasClient(base_url="https://canvas.example.edu/", token="token123")
            ok = client.upload_submission(
                course_id=77,
                assignment_id=42,
                submission_data={"score": 93.5},
                attachment_path=attachment,
            )

        self.assertTrue(ok)
        self.assertEqual(
            requests_module.post.call_args.args[0],
            "https://canvas.example.edu/api/v1/courses/77/assignments/42/submissions",
        )
        self.assertEqual(requests_module.post.call_args.kwargs["headers"], {"Authorization": "Bearer token123"})
        self.assertEqual(requests_module.post.call_args.kwargs["data"], {"submission": {"score": 93.5}})
        self.assertIn("submission[attachment]", requests_module.post.call_args.kwargs["files"])


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
        self.assertIn("Score 81.5", payload["comment"])
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
            score=75.0,
            attachment_path=attachment,
            resources=None,
        )

        submission_data = mock_upload.call_args.kwargs["submission_data"]
        self.assertEqual(submission_data["score"], 75.0)
        self.assertNotIn("comment", submission_data)


if __name__ == "__main__":
    unittest.main()

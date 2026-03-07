"""Utility functions for interfacing with common Learning Management Systems.

At the moment the project ships as a command‑line tool that processes a
single audio file and returns a JSON report.  Adding LMS support means the
tool can upload those reports so instructors can see them in their course
management system.

Only a very small subset of features is implemented – just enough to
demonstrate how integration would work.  The :class:`CanvasClient` and
``MoodleClient`` are intentionally lightweight wrappers around the REST
APIs that do not require any special SDK.

Both clients expose a single :func:`upload_submission` method that takes a
JSON payload and a *file path* to attach.  The payload is sent as part of
the ``submission`` field for Canvas and as part of ``file_submission`` for
Moodle.  The functions raise :class:`RuntimeError` if the HTTP request
fails.

Usage example (Canvas):

>>> from lms import CanvasClient
>>> client = CanvasClient(base_url="https://canvas.example.edu", token="abc123")
>>> client.upload_submission(course_id=7, assignment_id=42, submission_data={"score": 85}, attachment_path="report.json")
True

The tool can be extended with more LMS providers by following the same
pattern.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover – handled in functions
    requests = None  # type: ignore

__all__ = [
    "CanvasClient",
    "MoodleClient",
    "build_canvas_submission_data",
    "build_moodle_submission_data",
    "upload_to_canvas",
    "upload_to_moodle",
]


class _BaseClient:
    """Shared functionality for LMS API clients.

    Sub‑classes are expected to implement :meth:`_build_url`.
    """

    def __init__(self, base_url: str, token: str):
        # Normalise base URL – drop trailing slash to simplify joins.
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _check_response(self, resp: requests.Response) -> None:
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover – exercised via tests
            raise RuntimeError(f"{self.__class__.__name__} request failed: {resp.text}") from exc


class CanvasClient(_BaseClient):
    """Minimal client for https://canvas.instructure.com API.

    The implementation follows the Canvas v1 API – https://api.instructure.com.
    """

    def upload_submission(
        self,
        *,
        course_id: int,
        assignment_id: int,
        submission_data: Dict[str, Any],
        attachment_path: Path,
    ) -> bool:
        if requests is None:
            raise RuntimeError("The 'requests' library is required for LMS uploads but is not installed.")
        url = f"{self.base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions"
        data = {"submission": {**submission_data}}
        with attachment_path.open("rb") as fp:
            files = {"submission[attachment]": fp}
            resp = requests.post(url, headers=self._auth_headers(), data=data, files=files)
        self._check_response(resp)
        return True


class MoodleClient(_BaseClient):  # pragma: no cover – simple wrapper
    """Very small wrapper around Moodle 3.x web service API.
    Uses the ``core_files_upload`` and ``mod_assign_save_submission`` services.
    """

    def upload_submission(
        self,
        *,
        assignment_id: int,
        submission_data: Dict[str, Any],
        attachment_path: Path,
    ) -> bool:
        if requests is None:
            raise RuntimeError("The 'requests' library is required for LMS uploads but is not installed.")
        # Upload file first
        upload_url = f"{self.base_url}/webservice/upload.php"
        with attachment_path.open("rb") as fp:
            files = {"file": fp}
            params = {"token": self.token, "wstoken": self.token}
            upload_resp = requests.post(upload_url, files=files, data=params)
            self._check_response(upload_resp)
            upload_json = upload_resp.json()
            file_id = upload_json[0].get("fileid")

        # Now submit the assignment with the uploaded file id
        service_url = f"{self.base_url}/webservice/rest/server.php"
        data = {
            "wstoken": self.token,
            "wsfunction": "mod_assign_save_submission",
            "moodlewsrestformat": "json",
            "assignmentid": assignment_id,
            "submission": {
                "attachments": submission_data.get("attachments", []) + [file_id],
                "comment": submission_data.get("comment", ""),
            },
        }
        resp = requests.post(service_url, data=data)
        self._check_response(resp)
        return True


def _format_resources_comment(*, prefix_lines: list[str], resources: list | None) -> str:
    lines = list(prefix_lines)
    if resources:
        lines.append("Suggested training resources:")
        for resource in resources:
            title = resource.get("title", "Resource")
            url = resource.get("url", "")
            lines.append(f"- {title}: {url}")
    return "\n".join(line for line in lines if line)


def build_canvas_submission_data(*, score: float, resources: list | None = None) -> Dict[str, Any]:
    submission_data: Dict[str, Any] = {"score": score}
    comment = _format_resources_comment(prefix_lines=[], resources=resources)
    if comment:
        submission_data["comment"] = comment
    return submission_data


def build_moodle_submission_data(*, score: float, resources: list | None = None) -> Dict[str, Any]:
    comment = _format_resources_comment(prefix_lines=[f"Score {score}"], resources=resources)
    return {"attachments": [], "comment": comment}


def upload_to_canvas(
    *,
    base_url: str,
    token: str,
    course_id: int,
    assignment_id: int,
    score: float,
    attachment_path: Path,
    resources: list | None = None,
):
    client = CanvasClient(base_url, token)
    return client.upload_submission(
        course_id=course_id,
        assignment_id=assignment_id,
        submission_data=build_canvas_submission_data(score=score, resources=resources),
        attachment_path=attachment_path,
    )


def upload_to_moodle(
    *,
    base_url: str,
    token: str,
    assignment_id: int,
    score: float,
    attachment_path: Path,
    resources: list | None = None,
):
    client = MoodleClient(base_url, token)
    return client.upload_submission(
        assignment_id=assignment_id,
        submission_data=build_moodle_submission_data(score=score, resources=resources),
        attachment_path=attachment_path,
    )

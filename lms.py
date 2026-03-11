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

The optional score is treated as submission metadata, not as an authoritative
gradebook writeback. It is included in the comment payload so instructors can
see the suggested score alongside the uploaded report.

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

    def __init__(self, base_url: str, token: str, timeout_sec: float = 20.0):
        # Normalise base URL – drop trailing slash to simplify joins.
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_sec = float(timeout_sec)

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _check_response(self, resp: requests.Response) -> None:
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover – exercised via tests
            body = (resp.text or "").strip()
            preview = body[:200] or "No response body"
            if len(body) > 200:
                preview += "..."
            raise RuntimeError(
                f"{self.__class__.__name__} request failed (HTTP {resp.status_code}): {preview}"
            ) from exc

    def _post(self, url: str, **kwargs):
        try:
            return requests.post(url, timeout=self.timeout_sec, **kwargs)
        except requests.Timeout as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(f"{self.__class__.__name__} request timed out after {self.timeout_sec:.1f}s") from exc

    def _get(self, url: str, **kwargs):
        try:
            return requests.get(url, timeout=self.timeout_sec, **kwargs)
        except requests.Timeout as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(f"{self.__class__.__name__} request timed out after {self.timeout_sec:.1f}s") from exc


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
        start_url = f"{self.base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/self/files"
        start_resp = self._post(
            start_url,
            headers=self._auth_headers(),
            data={"name": attachment_path.name, "size": attachment_path.stat().st_size},
        )
        self._check_response(start_resp)
        upload_spec = start_resp.json()
        upload_url = upload_spec.get("upload_url")
        upload_params = upload_spec.get("upload_params")
        if not upload_url or upload_params is None:
            raise RuntimeError("Canvas upload initiation did not return upload_url/upload_params.")

        with attachment_path.open("rb") as fp:
            upload_resp = self._post(
                upload_url,
                data=upload_params or {},
                files={"file": fp},
                allow_redirects=False,
            )

        if 300 <= upload_resp.status_code < 400:
            location = upload_resp.headers.get("Location")
            if not location:
                raise RuntimeError("Canvas upload did not return a completion URL.")
            finalize_resp = self._get(location, headers=self._auth_headers())
            self._check_response(finalize_resp)
            file_payload = finalize_resp.json()
        elif upload_resp.status_code == 201 and upload_resp.headers.get("Location"):
            finalize_resp = self._get(upload_resp.headers["Location"], headers=self._auth_headers())
            self._check_response(finalize_resp)
            file_payload = finalize_resp.json()
        else:
            self._check_response(upload_resp)
            file_payload = upload_resp.json()

        file_id = file_payload.get("id")
        if file_id is None:
            raise RuntimeError("Canvas upload completed without a file id.")

        submit_url = f"{self.base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions"
        submit_data = [
            ("submission[submission_type]", "online_upload"),
            ("submission[file_ids][]", str(file_id)),
        ]
        if submission_data.get("comment"):
            submit_data.append(("comment[text_comment]", submission_data["comment"]))
        resp = self._post(submit_url, headers=self._auth_headers(), data=submit_data)
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
            upload_resp = self._post(upload_url, files=files, data=params)
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
        resp = self._post(service_url, data=data)
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


def build_canvas_submission_data(*, score: float | None, resources: list | None = None) -> Dict[str, Any]:
    submission_data: Dict[str, Any] = {}
    if score is not None:
        submission_data["score"] = score
    prefix_lines = [f"Suggested score: {score}"] if score is not None else []
    comment = _format_resources_comment(prefix_lines=prefix_lines, resources=resources)
    if comment:
        submission_data["comment"] = comment
    return submission_data


def build_moodle_submission_data(*, score: float | None, resources: list | None = None) -> Dict[str, Any]:
    prefix_lines = [f"Suggested score: {score}"] if score is not None else []
    comment = _format_resources_comment(prefix_lines=prefix_lines, resources=resources)
    return {"attachments": [], "comment": comment}


def upload_to_canvas(
    *,
    base_url: str,
    token: str,
    course_id: int,
    assignment_id: int,
    score: float | None,
    attachment_path: Path,
    resources: list | None = None,
    timeout_sec: float = 20.0,
):
    client = CanvasClient(base_url, token, timeout_sec=timeout_sec)
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
    score: float | None,
    attachment_path: Path,
    resources: list | None = None,
    timeout_sec: float = 20.0,
):
    client = MoodleClient(base_url, token, timeout_sec=timeout_sec)
    return client.upload_submission(
        assignment_id=assignment_id,
        submission_data=build_moodle_submission_data(score=score, resources=resources),
        attachment_path=attachment_path,
    )

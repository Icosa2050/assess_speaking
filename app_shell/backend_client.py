from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from app_backend.contracts import AssessmentCreateResponse, AssessmentStatusResponse, ErrorCode, ErrorResponse, UploadResponse
from app_backend.lifecycle import ensure_local_backend, get_backend_state

def backend_base_url(*, log_dir: str | Path | None = None) -> str:
    state = get_backend_state(log_dir=log_dir)
    if state is None:
        state = ensure_local_backend(log_dir=log_dir)
    return str(state["base_url"])


def _request(
    method: str,
    path: str,
    *,
    log_dir: str | Path | None = None,
    timeout_sec: float = 10.0,
    **kwargs: Any,
) -> httpx.Response:
    base_url = backend_base_url(log_dir=log_dir)
    try:
        response = httpx.request(method, f"{base_url.rstrip('/')}{path}", timeout=timeout_sec, **kwargs)
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        detail = ""
        code = ErrorCode.BACKEND_UNAVAILABLE
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                detail_payload = payload.get("detail")
                if isinstance(detail_payload, dict):
                    detail = str(detail_payload.get("detail") or exc)
                    raw_code = str(detail_payload.get("code") or "").strip()
                    if raw_code in {item.value for item in ErrorCode}:
                        code = ErrorCode(raw_code)
                else:
                    detail = str(detail_payload or exc)
        except (ValueError, json.JSONDecodeError):
            detail = str(exc)
        raise RuntimeError(ErrorResponse(code=code, detail=detail or str(exc)).model_dump_json()) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            ErrorResponse(
                code=ErrorCode.BACKEND_UNAVAILABLE,
                detail=f"The local backend is not reachable: {exc}",
            ).model_dump_json()
        ) from exc


def upload_audio_path(audio_path: str | Path, *, filename: str = "", log_dir: str | Path | None = None) -> UploadResponse:
    path = Path(audio_path)
    upload_name = filename or path.name or "audio.wav"
    with path.open("rb") as handle:
        response = _request(
            "POST",
            "/v1/uploads",
            log_dir=log_dir,
            timeout_sec=30.0,
            files={"file": (upload_name, handle, "application/octet-stream")},
        )
    return UploadResponse(**response.json())


def create_assessment(request: dict[str, Any], *, log_dir: str | Path | None = None) -> AssessmentCreateResponse:
    response = _request("POST", "/v1/assessments", log_dir=log_dir, json=request, timeout_sec=10.0)
    return AssessmentCreateResponse(**response.json())


def get_assessment_status(assessment_id: str, *, log_dir: str | Path | None = None) -> AssessmentStatusResponse:
    response = _request("GET", f"/v1/assessments/{assessment_id}", log_dir=log_dir, timeout_sec=10.0)
    return AssessmentStatusResponse(**response.json())


def cancel_assessment(assessment_id: str, *, log_dir: str | Path | None = None) -> AssessmentStatusResponse:
    response = _request("POST", f"/v1/assessments/{assessment_id}/cancel", log_dir=log_dir, timeout_sec=10.0)
    return AssessmentStatusResponse(**response.json())


def load_history(*, log_dir: str | Path | None = None) -> list[dict[str, Any]]:
    response = _request("GET", "/v1/history", log_dir=log_dir, timeout_sec=10.0)
    payload = response.json()
    items = payload.get("items") if isinstance(payload, dict) else []
    return list(items or [])


def load_history_detail(session_id: str, *, log_dir: str | Path | None = None) -> dict[str, Any]:
    response = _request("GET", f"/v1/history/{session_id}", log_dir=log_dir, timeout_sec=10.0)
    payload = response.json()
    if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
        return payload["payload"]
    raise RuntimeError(
        ErrorResponse(code=ErrorCode.RUNTIME, detail=f"History payload {session_id} is invalid.").model_dump_json()
    )


def load_samples(*, log_dir: str | Path | None = None) -> list[dict[str, Any]]:
    response = _request("GET", "/v1/samples", log_dir=log_dir, timeout_sec=10.0)
    payload = response.json()
    items = payload.get("items") if isinstance(payload, dict) else []
    return list(items or [])

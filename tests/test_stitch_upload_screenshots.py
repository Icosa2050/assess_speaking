import json
from pathlib import Path

from scripts import stitch_upload_screenshots as stitch


def test_detect_mime_type_accepts_supported_assets(tmp_path: Path) -> None:
    for filename, expected in {
        "home.jpg": "image/jpeg",
        "home.jpeg": "image/jpeg",
        "home.png": "image/png",
        "home.webp": "image/webp",
        "screen.html": "text/html",
        "screen.htm": "text/html",
    }.items():
        path = tmp_path / filename
        path.write_bytes(b"asset")
        assert stitch.detect_mime_type(path) == expected


def test_detect_mime_type_rejects_unsupported_assets(tmp_path: Path) -> None:
    path = tmp_path / "screen.gif"
    path.write_bytes(b"asset")
    try:
        stitch.detect_mime_type(path)
    except ValueError as exc:
        assert ".gif" in str(exc)
        assert ".jpg" in str(exc)
    else:
        raise AssertionError("expected unsupported extension failure")


def test_build_batch_create_body_uses_image_or_document_slots() -> None:
    image_body = stitch.build_batch_create_body(
        project_id="123",
        encoded_content="abc",
        mime_type="image/jpeg",
        title="Current Home",
        create_screen_instances=True,
    )
    image_screen = image_body["requests"][0]["screen"]
    assert image_body["parent"] == "projects/123"
    assert image_body["createScreenInstances"] is True
    assert image_screen["screenType"] == "IMAGE"
    assert image_screen["title"] == "Current Home"
    assert image_screen["screenshot"] == {
        "fileContentBase64": "abc",
        "mimeType": "image/jpeg",
    }
    assert "htmlCode" not in image_screen

    html_body = stitch.build_batch_create_body(
        project_id="123",
        encoded_content="abc",
        mime_type="text/html",
        title=None,
        create_screen_instances=False,
    )
    html_screen = html_body["requests"][0]["screen"]
    assert html_body["createScreenInstances"] is False
    assert html_screen["screenType"] == "DOCUMENT"
    assert html_screen["htmlCode"]["mimeType"] == "text/html"
    assert "title" not in html_screen


def test_build_auth_headers_uses_api_key_without_manifest_leak() -> None:
    headers = stitch.build_auth_headers(api_key="secret-key", access_token=None, quota_project=None)
    assert headers["X-Goog-Api-Key"] == "secret-key"
    assert "Authorization" not in headers


def test_build_auth_headers_uses_oauth_when_api_key_is_absent() -> None:
    headers = stitch.build_auth_headers(
        api_key=None,
        access_token="access-token",
        quota_project="cloud-project",
    )
    assert headers["Authorization"] == "Bearer access-token"
    assert headers["X-Goog-User-Project"] == "cloud-project"
    assert "X-Goog-Api-Key" not in headers


def test_build_auth_headers_rejects_missing_credentials() -> None:
    try:
        stitch.build_auth_headers(api_key=None, access_token="access-token", quota_project=None)
    except ValueError as exc:
        assert "STITCH_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing credential failure")


def test_dry_run_writes_manifest_without_secret_or_base64(tmp_path: Path) -> None:
    asset = tmp_path / "home.jpg"
    asset.write_bytes(b"fake-jpeg")
    manifest_path = tmp_path / "manifest.json"

    exit_code = stitch.main(
        [
            "--project-id",
            "123",
            "--manifest",
            str(manifest_path),
            "--dry-run",
            "--title-prefix",
            "Vostavo current",
            str(asset),
        ]
    )

    assert exit_code == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["projectId"] == "123"
    assert manifest["dryRun"] is True
    assert manifest["uploads"][0]["title"] == "Vostavo current home"
    serialized = json.dumps(manifest)
    assert "fake-jpeg" not in serialized
    assert "secret" not in serialized
    assert "fileContentBase64" not in serialized

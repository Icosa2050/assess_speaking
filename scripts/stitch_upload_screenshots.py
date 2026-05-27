from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SUPPORTED_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".html": "text/html",
    ".htm": "text/html",
}
DEFAULT_BASE_URL = "https://stitch.googleapis.com/mcp"


@dataclass(frozen=True)
class UploadResult:
    source_path: str
    title: str
    mime_type: str
    status: str
    screen_names: list[str]
    error: str | None = None


def detect_mime_type(path: Path) -> str:
    mime_type = SUPPORTED_MIME_TYPES.get(path.suffix.lower())
    if not mime_type:
        supported = ", ".join(sorted(SUPPORTED_MIME_TYPES))
        raise ValueError(f"Unsupported file extension {path.suffix!r}. Supported: {supported}")
    return mime_type


def build_batch_create_body(
    *,
    project_id: str,
    encoded_content: str,
    mime_type: str,
    title: str | None,
    create_screen_instances: bool,
) -> dict[str, Any]:
    file_obj = {"fileContentBase64": encoded_content, "mimeType": mime_type}
    is_html = mime_type == "text/html"
    screen: dict[str, Any] = {
        "screenType": "DOCUMENT" if is_html else "IMAGE",
        "isCreatedByClient": True,
    }
    if title:
        screen["title"] = title
    if is_html:
        screen["htmlCode"] = file_obj
    else:
        screen["screenshot"] = file_obj
    return {
        "parent": f"projects/{project_id}",
        "requests": [{"screen": screen}],
        "createScreenInstances": create_screen_instances,
    }


def rest_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/mcp"):
        normalized = normalized[:-4]
    return normalized


def build_auth_headers(
    *,
    api_key: str | None,
    access_token: str | None,
    quota_project: str | None,
) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-Goog-Api-Key"] = api_key
    elif access_token and quota_project:
        headers["Authorization"] = f"Bearer {access_token}"
        headers["X-Goog-User-Project"] = quota_project
    else:
        raise ValueError("Set STITCH_API_KEY, or set both STITCH_ACCESS_TOKEN and GOOGLE_CLOUD_PROJECT.")
    return headers


def default_title(path: Path, prefix: str) -> str:
    title = path.stem
    if title[:3].isdigit() and "-" in title:
        title = title.split("-", 1)[1]
    title = title.replace("-", " ").replace("_", " ").strip()
    return f"{prefix} {title}".strip()


def _extract_screen_names(payload: dict[str, Any]) -> list[str]:
    screen_names: list[str] = []
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        screen = result.get("screen")
        if not isinstance(screen, dict):
            continue
        name = screen.get("name")
        if isinstance(name, str) and name:
            screen_names.append(name)
    return screen_names


def upload_asset(
    *,
    project_id: str,
    path: Path,
    title: str,
    base_url: str,
    headers: dict[str, str],
    create_screen_instances: bool,
    timeout_seconds: int,
) -> UploadResult:
    try:
        mime_type = detect_mime_type(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except (OSError, ValueError) as exc:
        return UploadResult(str(path), title, "unknown", "error", [], str(exc))

    body = build_batch_create_body(
        project_id=project_id,
        encoded_content=encoded,
        mime_type=mime_type,
        title=title,
        create_screen_instances=create_screen_instances,
    )
    url = f"{rest_base_url(base_url)}/v1/projects/{project_id}/screens:batchCreate"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return UploadResult(str(path), title, mime_type, "error", [], f"HTTP {exc.code}: {text}")
    except (OSError, json.JSONDecodeError) as exc:
        return UploadResult(str(path), title, mime_type, "error", [], str(exc))

    return UploadResult(str(path), title, mime_type, "uploaded", _extract_screen_names(payload))


def write_manifest(
    *,
    manifest_path: Path,
    project_id: str,
    dry_run: bool,
    uploads: list[UploadResult],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "projectId": project_id,
        "dryRun": dry_run,
        "uploads": [
            {
                "sourcePath": item.source_path,
                "title": item.title,
                "mimeType": item.mime_type,
                "status": item.status,
                "screenNames": item.screen_names,
                "error": item.error,
            }
            for item in uploads
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload local screenshots or HTML screens to Stitch.")
    parser.add_argument("paths", nargs="+", help="Image or HTML assets to upload.")
    parser.add_argument("--project-id", required=True, help="Bare Stitch project ID, without projects/ prefix.")
    parser.add_argument("--manifest", required=True, help="Where to write the non-secret upload manifest.")
    parser.add_argument("--title-prefix", default="Vostavo current", help="Prefix for generated Stitch titles.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Stitch MCP base URL.")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--no-create-screen-instances", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [Path(path) for path in args.paths]
    uploads: list[UploadResult] = []
    try:
        if args.dry_run:
            for path in paths:
                uploads.append(
                    UploadResult(
                        source_path=str(path),
                        title=default_title(path, args.title_prefix),
                        mime_type=detect_mime_type(path),
                        status="dry_run",
                        screen_names=[],
                    )
                )
        else:
            headers = build_auth_headers(
                api_key=os.environ.get("STITCH_API_KEY"),
                access_token=os.environ.get("STITCH_ACCESS_TOKEN"),
                quota_project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            )
            for path in paths:
                uploads.append(
                    upload_asset(
                        project_id=args.project_id,
                        path=path,
                        title=default_title(path, args.title_prefix),
                        base_url=args.base_url,
                        headers=headers,
                        create_screen_instances=not args.no_create_screen_instances,
                        timeout_seconds=args.timeout_seconds,
                    )
                )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    write_manifest(
        manifest_path=Path(args.manifest),
        project_id=args.project_id,
        dry_run=args.dry_run,
        uploads=uploads,
    )
    errors = [item for item in uploads if item.status == "error"]
    if errors:
        for item in errors:
            sys.stderr.write(f"{item.source_path}: {item.error}\n")
        return 1
    sys.stdout.write(f"Wrote Stitch upload manifest to {args.manifest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

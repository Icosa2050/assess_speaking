# Stitch Upload And Design System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small repo-local workflow that uploads current app screenshots to Stitch, creates a Vostavo design-system source of truth, and uses Stitch outputs as reviewed design references before React screen work.

**Architecture:** Keep this outside the product runtime. Add a Python upload CLI under `scripts/` that mirrors the Stitch SDK upload endpoint without adding npm dependencies to the app, records a non-secret manifest, and leaves all UI implementation for later PAL-reviewed screen slices. Store the visual system as a repo-level `DESIGN.md` so Stitch, PAL, and future coding agents share the same design contract.

**Tech Stack:** Python 3 standard library, repository `.venv`, Stitch REST endpoint `projects/{projectId}/screens:batchCreate`, Google DESIGN.md lint via `npx @google/design.md`, existing screenshot artifacts under `docs/ux-audit-screenshots/`.

---

## Source Notes

- Stitch MCP is enabled and exposes project, screen, variant, and design-system tools.
- Stitch MCP does not expose screenshot upload directly.
- `@google/stitch-sdk@0.3.5` includes upload support through a private REST endpoint, with supported input extensions `.png`, `.jpg`, `.jpeg`, `.webp`, `.html`, `.htm`.
- Keep `STITCH_API_KEY` in the environment for upload runs. Do not read or print Codex config secrets.

## File Structure

- `scripts/stitch_upload_screenshots.py`: CLI and reusable upload helpers. Reads local image/HTML files, validates MIME type, POSTs to Stitch, writes a non-secret manifest.
- `tests/test_stitch_upload_screenshots.py`: Unit tests for MIME detection, REST body shape, auth headers, dry-run behavior, and manifest redaction.
- `DESIGN.md`: Vostavo design-system source of truth for Stitch and coding agents.
- `docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.json`: Generated upload manifest for the current screenshot set.
- `docs/ux-audit-screenshots/2026-05-26/stitch-generation-manifest.json`: Generated record of Stitch project/design-system/screen IDs and critique status.

## Task 1: Upload Tool Core

**Files:**
- Create: `scripts/stitch_upload_screenshots.py`
- Create: `tests/test_stitch_upload_screenshots.py`

- [ ] **Step 1: Write failing tests for MIME detection and request body shape**

Add this test skeleton:

```python
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
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_stitch_upload_screenshots.py -q
```

Expected: import or attribute failures because `scripts/stitch_upload_screenshots.py` does not exist yet.

- [ ] **Step 3: Implement MIME and body helpers**

Create `scripts/stitch_upload_screenshots.py` with these helpers first:

```python
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
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_stitch_upload_screenshots.py -q
```

Expected: MIME/body tests pass. Remaining tests from later steps may still be absent.

## Task 2: Upload CLI, Auth, Dry Run, Manifest

**Files:**
- Modify: `scripts/stitch_upload_screenshots.py`
- Modify: `tests/test_stitch_upload_screenshots.py`

- [ ] **Step 1: Add failing tests for redacted manifests and dry-run uploads**

Append:

```python
import json


def test_build_auth_headers_uses_api_key_without_manifest_leak() -> None:
    headers = stitch.build_auth_headers(api_key="secret-key", access_token=None, quota_project=None)
    assert headers["X-Goog-Api-Key"] == "secret-key"
    assert "Authorization" not in headers


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
```

- [ ] **Step 2: Implement auth, dry-run, manifest, and CLI**

Add these functions to `scripts/stitch_upload_screenshots.py`:

```python
@dataclass(frozen=True)
class UploadResult:
    source_path: str
    title: str
    mime_type: str
    status: str
    screen_names: list[str]
    error: str | None = None


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
    mime_type = detect_mime_type(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
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
    except OSError as exc:
        return UploadResult(str(path), title, mime_type, "error", [], str(exc))
    screen_names = [
        result.get("screen", {}).get("name", "")
        for result in payload.get("results", [])
        if result.get("screen", {}).get("name")
    ]
    return UploadResult(str(path), title, mime_type, "uploaded", screen_names)


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
```

Then wire `main(argv)` with `argparse`:

```python
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
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_stitch_upload_screenshots.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run a dry-run manifest against the current screenshot set**

Run:

```bash
.venv/bin/python scripts/stitch_upload_screenshots.py \
  --project-id DRY_RUN_PROJECT \
  --manifest docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.dry-run.json \
  --dry-run \
  docs/ux-audit-screenshots/2026-05-24/*.jpg
```

Expected: manifest exists, contains 10 uploads, and contains no `fileContentBase64`, `STITCH_API_KEY`, `Authorization`, or bearer token material.

## Task 3: Real Upload And Manifest

**Files:**
- Modify: `docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.json`

- [ ] **Step 1: Create or choose the Stitch project**

Use native Stitch MCP:

```text
create_project({ "title": "Vostavo learner UX redesign" })
```

Record the returned bare project ID in a shell variable for the rest of the task:

```bash
PROJECT_ID="12345678901234567890"
```

- [ ] **Step 2: Upload the current screenshot set**

Run with `STITCH_API_KEY` set in the shell environment:

```bash
STITCH_API_KEY="$STITCH_API_KEY" .venv/bin/python scripts/stitch_upload_screenshots.py \
  --project-id "$PROJECT_ID" \
  --manifest docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.json \
  --title-prefix "Vostavo current" \
  docs/ux-audit-screenshots/2026-05-24/*.jpg
```

Expected: 10 uploads with status `uploaded`.

- [ ] **Step 3: Verify manifest redaction**

Run:

```bash
rg -n "fileContentBase64|STITCH_API_KEY|Authorization|Bearer|sk-" docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.json
```

Expected: no output, exit code 1.

- [ ] **Step 4: Verify Stitch can list uploaded screens**

Use native Stitch MCP:

```text
list_screens({ "projectId": "$PROJECT_ID" })
```

Expected: uploaded screens are visible or project canvas screen instances are visible through `get_project`.

## Task 4: Vostavo DESIGN.md

**Files:**
- Create: `DESIGN.md`

- [ ] **Step 1: Write the design-system source**

Create `DESIGN.md` with the canonical Vostavo visual contract:

```markdown
---
version: alpha
name: Vostavo Learner Coach
description: A calm, motivating speaking-practice interface for adult language learners.
colors:
  ink: "#243033"
  muted: "#5B676B"
  surface: "#F8FAF8"
  surface-soft: "#EDF3EF"
  surface-raised: "#FFFFFF"
  primary: "#276A73"
  primary-hover: "#1F5961"
  progress: "#4F8F63"
  focus: "#6B5CA5"
  caution: "#B85C38"
  border-soft: "#CED8D2"
typography:
  display:
    fontFamily: Inter
    fontSize: 40px
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: 0
  headline:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: 0
  title:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: 650
    lineHeight: 1.35
    letterSpacing: 0
  body:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: 0
  label:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: 0
rounded:
  sm: 4px
  md: 8px
  full: 999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface-raised}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: 12px
  card:
    backgroundColor: "{colors.surface-raised}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: 24px
---

## Overview

Vostavo should feel like a calm speaking coach: focused, steady, and encouraging. The product is a practice companion first and a local-AI control panel second. The interface should reduce performance anxiety by making the next practice action obvious and keeping technical setup out of the main learning loop.

## Colors

Use a balanced light palette, not a beige editorial theme and not a single-hue blue dashboard. `primary` is for the main learner action. `progress` is for growth and completed states. `focus` is for selected practice focus. `caution` is for setup or review warnings.

## Typography

Use Inter for all UI text. Avoid viewport-scaled type. Prompts and scores may use `display` or `headline`, but dense tool panels, settings, and setup forms should use compact `title`, `body`, and `label` levels.

## Layout

Lead every screen with the learner's next action. Home should prioritize starting practice. Speak should prioritize prompt and recording. Review and History should prioritize feedback, next focus, and retry/new-session actions. Runtime setup is reachable but secondary.

## Elevation & Depth

Use soft surfaces and thin borders. Avoid nested cards, decorative orbs, bokeh, and broad gradients. Use stable dimensions for toolbars, status rows, recording controls, and route navigation so text and state changes do not shift layout.

## Shapes

Use 8px radius or less for cards and controls, except pills/chips where `full` is appropriate. Keep icon buttons square enough to read as controls.

## Components

Primary buttons start or continue practice. Secondary buttons navigate or reveal setup details. Empty states always provide a next step. System status should be a compact row, not a primary screen destination when everything is healthy.

## Do's and Don'ts

Do preserve localization, semantic IDs, route guards, and existing app architecture. Do make setup progressive and learner-friendly. Do keep technical labels available in advanced sections. Do not hide runtime errors. Do not make Runtime Setup a peer of Speak/Review once configured. Do not use landing-page hero composition inside the application shell.
```

- [ ] **Step 2: Lint the design system**

Run:

```bash
npx @google/design.md@0.2.0 lint DESIGN.md
```

Expected: zero errors. Warnings are acceptable only if reviewed and documented in the final note.

- [ ] **Step 3: Upload DESIGN.md into Stitch**

Use native Stitch MCP:

```text
upload_design_md({ "projectId": "$PROJECT_ID", "designMdBase64": "$DESIGN_MD_BASE64" })
create_design_system_from_design_md({ "projectId": "$PROJECT_ID", "deviceType": "DESKTOP", "selectedScreenInstance": "$UPLOADED_HOME_SCREEN_INSTANCE" })
```

Expected: a Vostavo design-system asset is created for the project.

## Task 5: Generate North-Star Screens

**Files:**
- Create: `docs/ux-audit-screenshots/2026-05-26/stitch-generation-manifest.json`

- [ ] **Step 1: Generate Home, Speak, and Review from the uploaded-current context**

Use native Stitch MCP `generate_screen_from_text` three times with `modelId: "GEMINI_3_1_PRO"` and the Vostavo design system asset:

```text
Generate a desktop Home screen for Vostavo. Use the uploaded current Home screenshot only as product-structure reference, not as visual style. Apply the Vostavo Learner Coach design system. The screen should make Start practicing the primary action, collapse healthy system checks into one compact row, keep Runtime Setup reachable only when missing or from Settings, and avoid marketing hero layout.
```

```text
Generate a desktop Speak screen for Vostavo. Use the uploaded current Speak screenshot as structure reference. Apply the Vostavo Learner Coach design system. The prompt and recording action must dominate. Metadata is compact. Submit-disabled guidance is explicit before audio exists. Preserve a serious assessment feel without technical intimidation.
```

```text
Generate a desktop Review empty/completed-state screen for Vostavo. Use the uploaded current Review screenshot as structure reference. Apply the Vostavo Learner Coach design system. Empty state must route learners to Start session or Speak based on readiness. Completed state should prioritize band, next focus, and retry/new-session actions.
```

- [ ] **Step 2: Generate one variant set**

Use `generate_variants` on each screen:

```json
{
  "variantOptions": {
    "variantCount": 3,
    "creativeRange": "EXPLORE",
    "aspects": ["LAYOUT", "COLOR_SCHEME", "TEXT_CONTENT"]
  }
}
```

Variant labels to request in the prompt:

```text
Create three variants: Calm Coach, Progress Journey, and Local AI Simplified. Keep the design system tokens recognizable and keep the learner practice loop primary.
```

- [ ] **Step 3: Save generation manifest**

Create `docs/ux-audit-screenshots/2026-05-26/stitch-generation-manifest.json`:

```json
{
  "createdAt": "2026-05-26T00:00:00Z",
  "projectId": "$PROJECT_ID",
  "designSystemAssetId": "$DESIGN_SYSTEM_ASSET_ID",
  "sourceUploadManifest": "docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.json",
  "screens": [
    {
      "name": "home",
      "sourceCurrentScreenshot": "docs/ux-audit-screenshots/2026-05-24/01-home-default.jpg",
      "stitchScreenIds": []
    },
    {
      "name": "speak",
      "sourceCurrentScreenshot": "docs/ux-audit-screenshots/2026-05-24/04-speak-ready.jpg",
      "stitchScreenIds": []
    },
    {
      "name": "review",
      "sourceCurrentScreenshot": "docs/ux-audit-screenshots/2026-05-24/05-review-empty.jpg",
      "stitchScreenIds": []
    }
  ],
  "reviewStatus": "pending"
}
```

Before committing, replace shell-variable marker values in the JSON with the concrete IDs returned by Stitch. Do not include API keys, auth headers, or base64 image payloads.

Current execution note from 2026-05-27:

- Native `generate_screen_from_text` and `generate_variants` still fail through the remote Stitch MCP transport after roughly one minute.
- `apply_design_system` successfully returned screen `e1b78a8967c248e6b15edb745bb66e1f`, but the downloaded screenshot is blank and the HTML artifact is empty.
- `scripts/stitch_generate_review.mjs` now records the SDK fallback attempt with a fresh-client poll. The SDK path also fails with no new screen; details are in `docs/ux-audit-screenshots/2026-05-26/stitch-sdk-generation-manifest.json`.
- Treat Stitch as useful for `DESIGN.md`, upload, and design-system tokens, but not yet reliable for Codex-driven north-star generation.

## Task 6: Multi-Model Review And Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-05-24-learner-ux-flow-refinement.md`
- Modify: `docs/ux-audit-screenshots/2026-05-26/stitch-generation-manifest.json`

- [ ] **Step 1: Review Stitch outputs with vision-capable models**

Use direct OpenRouter vision or native PAL if image handling works. Compare current screenshots against Stitch outputs with:

```text
Evaluate these current Vostavo screens and Stitch redesign variants for a language learner. Rank the variants by ease of use, motivational quality, setup clarity, and implementation risk. Identify any design ideas that conflict with localization, semantic IDs, route guards, or the existing React/CSS-module architecture.
```

Models to prefer:

```text
anthropic/claude-opus-4.7
google/gemini-3.1-pro-preview
x-ai/grok-4.3
```

- [ ] **Step 2: Amend the learner UX plan**

Add a short section to `docs/superpowers/plans/2026-05-24-learner-ux-flow-refinement.md`:

```markdown
## Stitch Design-System Pass

The selected visual direction is `Calm Coach`.

Implementation constraints:
- Keep `DESIGN.md` as the design-system source of truth.
- Use Stitch outputs as references only; do not import generated React/CSS directly.
- Preserve localization keys, semantic IDs, and route guards.
- Apply visual changes only through the existing file-bounded UX tasks.
```

- [ ] **Step 3: Verify docs do not contain secrets**

Run:

```bash
rg -n "fileContentBase64|STITCH_API_KEY|Authorization|Bearer|sk-" DESIGN.md docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.json docs/ux-audit-screenshots/2026-05-26/stitch-generation-manifest.json docs/superpowers/plans/2026-05-24-learner-ux-flow-refinement.md
```

Expected: no output, exit code 1.

## Verification Matrix

- Upload-tool unit tests:
  - `.venv/bin/python -m pytest tests/test_stitch_upload_screenshots.py -q`
- Upload dry run:
  - `.venv/bin/python scripts/stitch_upload_screenshots.py --project-id DRY_RUN_PROJECT --manifest docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.dry-run.json --dry-run docs/ux-audit-screenshots/2026-05-24/*.jpg`
- DESIGN.md lint:
  - `npx @google/design.md@0.2.0 lint DESIGN.md`
- Secret scan:
  - `rg -n "fileContentBase64|STITCH_API_KEY|Authorization|Bearer|sk-" DESIGN.md docs/ux-audit-screenshots`
- No frontend code is touched in this plan. Frontend tests are not required until the later PAL-reviewed implementation slices.

## Commit Slices

1. `tooling: add Stitch screenshot upload helper`
   - `scripts/stitch_upload_screenshots.py`
   - `tests/test_stitch_upload_screenshots.py`

2. `docs: add Vostavo design system`
   - `DESIGN.md`

3. `docs: record Stitch UX generation artifacts`
   - upload manifest
   - generation manifest
   - learner UX plan amendment

## Self-Review

- Spec coverage: upload tool, DESIGN.md, Stitch upload, Stitch generation, multi-model review, and handoff are all represented.
- Placeholder scan: implementation steps use concrete file paths, commands, and sample content. Placeholder IDs remain only in generation manifest examples and must be replaced before commit.
- Type consistency: upload helper names are consistent across tests and implementation snippets.
- Scope check: no product runtime code or React screen files are touched in this plan.

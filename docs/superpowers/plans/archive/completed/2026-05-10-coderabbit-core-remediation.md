# CodeRabbit Core Remediation Implementation Plan

> Archive status, 2026-05-20: completed in code and retained as historical evidence. The original checklist was not backfilled before the later `app_core` and Streamlit-removal migration, so do not execute this file as an active plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 26 CodeRabbit findings across assessment runtime, backend support/lifecycle, app shell, and frontend library code without broad refactors.

**Architecture:** Apply small, boundary-focused hardening where untrusted or stale data enters the system: JSON loading, filesystem traversal, process state, API path construction, upload buffers, and frontend shared-data parsing. Keep existing module ownership intact, avoid new framework dependencies, and land each group with focused tests before implementation.

**Tech Stack:** Python 3 with unittest/pytest via `.venv`, FastAPI/Pydantic backend contracts, Streamlit app shell helpers, TypeScript/React/Vitest frontend.

---

## PAL Review Notes

Two PAL passes shaped this plan:

- Gemini agreed with the risk areas but over-expanded some tasks into deep app-store and environment validation. Those broader suggestions are intentionally excluded.
- Claude/OpenRouter recommended the final split used here: separate maintenance from support bundles, lifecycle from jobs, app shell path/runtime from UI/services, and frontend API types from setup-content validation.

Decisions locked in:

- Do not add pydantic/jsonschema/zod for these fixes.
- Do not introduce process groups, watchdogs, or repo-wide status migrations.
- For support bundle symlinks, skip symlinked files rather than following them.
- For legacy Windows cache paths, stop creating the extra `Cache` segment going forward; do not migrate old directories in this remediation.
- Keep every task to 5 touched files or fewer, including tests.

## File Map

### Task 1: Assessment Runtime Content Safety

**Files:**
- Modify: `assessment_runtime/theme_library.py`
- Modify: `assessment_runtime/data/session_setup_content.json`
- Modify: `assessment_runtime/runner.py`
- Test: `tests/test_theme_library.py`
- Test: `tests/test_assessment_runner.py`

**Findings Covered:** Critical theme JSON file/parse crash, critical missing theme JSON keys, German umlaut text, short `top_3_priorities`.

### Task 2: Backend Cleanup Accounting

**Files:**
- Modify: `app_backend/maintenance.py`
- Test: `tests/test_app_backend_config.py`

**Findings Covered:** Cleanup stat TOCTOU, deleted count/freed bytes overreporting failed deletes.

### Task 3: Support Bundle Filesystem Hardening

**Files:**
- Modify: `app_backend/support_bundle.py`
- Modify: `app_backend/app.py`
- Test: `tests/test_app_backend_support_bundle.py`
- Test: `tests/test_app_backend_api.py`

**Findings Covered:** Unreadable text file aborts, optional tree bad file aborts, symlink traversal, `bundle_id` traversal.

### Task 4: Backend Lifecycle Process Safety

**Files:**
- Modify: `app_backend/lifecycle.py`
- Test: `tests/test_app_backend_lifecycle.py`

**Findings Covered:** Spawned backend not cleaned up on timeout, recycled PID termination risk.

### Task 5: Backend Job Terminal-State Correctness

**Files:**
- Modify: `app_backend/jobs.py`
- Create: `tests/test_app_backend_jobs.py`

**Findings Covered:** Terminal status casing, cancel overwriting a job that finished during termination.

### Task 6: App Shell Runtime And Data Roots

**Files:**
- Modify: `app_shell/runtime_resolver.py`
- Modify: `app_shell/app_data.py`
- Test: `tests/test_runtime_resolver.py`
- Test: `tests/test_app_shell_app_data.py`

**Findings Covered:** Provider API key env fallback, cache root with custom app data home, legacy Windows cache path.

### Task 7: App Shell UI Diagnostics

**Files:**
- Modify: `app_shell/review_components.py`
- Modify: `app_shell/diagnostics.py`
- Test: `tests/test_app_shell_review_components.py`
- Test: `tests/test_app_shell_diagnostics.py`

**Findings Covered:** Translated status text comparison, writable diagnostic reporting `reports_dir` instead of `temp_dir`.

### Task 8: App Shell Service Guardrails

**Files:**
- Modify: `app_shell/services.py`
- Test: `tests/test_app_shell_services.py`

**Findings Covered:** Support bundle export dereferencing invalid creation result, uploaded file assuming `getvalue` or `getbuffer`.

### Task 9: Frontend API Contract Safety

**Files:**
- Modify: `frontend/src/lib/api/types.ts`
- Modify: `frontend/src/lib/api/client.ts`
- Create: `frontend/src/lib/api/client.test.ts`

**Findings Covered:** `HistoryRow` broad `unknown` fields, unencoded dynamic path params.

### Task 10: Frontend Store And Setup Content Guards

**Files:**
- Modify: `frontend/src/lib/state/appStore.ts`
- Modify: `frontend/src/lib/setup/sessionSetupContent.ts`
- Create: `frontend/src/lib/state/appStore.test.ts`
- Create: `frontend/src/lib/setup/sessionSetupContent.test.ts`

**Findings Covered:** Failed/cancelled job `nextError` undefined, shared JSON cast without validation, arbitrary `task_family` cast.

---

## Task 1: Assessment Runtime Content Safety

- [ ] **Step 1: Add failing theme content loader tests**

Add tests to `tests/test_theme_library.py`:

```python
def test_load_session_setup_content_falls_back_when_file_is_missing(self):
    missing = Path("/tmp/does-not-exist/session_setup_content.json")

    payload = theme_library._load_session_setup_content(path=missing)

    self.assertIn("default_theme_library", payload)
    self.assertIn("practice_brief_templates", payload)
    self.assertIn("en", payload["practice_brief_templates"])

def test_load_session_setup_content_fills_missing_required_keys(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session_setup_content.json"
        path.write_text('{"default_theme_library": {}}', encoding="utf-8")

        payload = theme_library._load_session_setup_content(path=path)

    self.assertEqual(payload["default_theme_library"], {})
    self.assertIn("practice_brief_templates", payload)
    self.assertIn("en", payload["practice_brief_templates"])
```

- [ ] **Step 2: Add failing runner priority padding test**

Add a test to `tests/test_assessment_runner.py` near the history persistence test. Reuse the existing mocking style and make the fake report contain only one priority:

```python
def test_execute_assessment_run_pads_short_priority_lists(self, mock_run_assessment, _mock_progress):
    payload = _assessment_payload()
    payload["report"]["coaching"]["top_3_priorities"] = ["Use clearer connectors"]
    mock_run_assessment.return_value = payload

    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.execute_assessment_run(
            runner.AssessmentRunRequest(
                audio=Path("tests/audio/test1.m4a"),
                whisper_model="tiny",
                llm_model="demo",
                provider="openrouter",
                target_cefr="B1",
                theme="Travel",
                task_family="free_monologue",
                speaker_id="speaker",
                target_duration_sec=60,
                expected_language="en",
                language_profile_key="en",
                feedback_language="en",
                dry_run=True,
                log_dir=Path(tmpdir),
            )
        )

    self.assertIsNotNone(result.report_path)
    history = (Path(tmpdir) / "history.csv").read_text(encoding="utf-8")
    self.assertIn("Use clearer connectors", history)
```

- [ ] **Step 3: Run the focused tests and confirm failure**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_theme_library.py tests/test_assessment_runner.py -q
```

Expected: fails because `_load_session_setup_content` does not accept `path` and runner indexes missing priorities.

- [ ] **Step 4: Implement safe theme content loading**

In `assessment_runtime/theme_library.py`, add logging, a minimal fallback, and key validation:

```python
import logging

logger = logging.getLogger(__name__)
_REQUIRED_SESSION_SETUP_KEYS = ("default_theme_library", "practice_brief_templates")
_FALLBACK_SESSION_SETUP_CONTENT = {
    "default_theme_library": {
        "en": {"label": "English", "themes": []},
        "it": {"label": "Italiano", "themes": []},
    },
    "practice_brief_templates": {
        "en": {
            "travel_narrative": "Speak about '{theme}' in a clear sequence.",
            "personal_experience": "Explain '{theme}' as a personal experience.",
            "opinion_monologue": "Give your opinion about '{theme}'.",
            "free_monologue": "Speak in English about '{theme}'.",
            "picture_description": "Describe '{theme}'.",
            "default_duration_minutes": "Aim to speak for about {minutes} minutes.",
            "default_duration_seconds": "Aim to speak for about {seconds} seconds.",
            "success_focus": [],
        }
    },
}

def _load_session_setup_content(path: Path = _SESSION_SETUP_CONTENT_PATH) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load session setup content from %s: %s", path, exc)
        return deepcopy(_FALLBACK_SESSION_SETUP_CONTENT)
    if not isinstance(payload, dict):
        logger.warning("Session setup content from %s is not a JSON object.", path)
        return deepcopy(_FALLBACK_SESSION_SETUP_CONTENT)
    normalized = deepcopy(_FALLBACK_SESSION_SETUP_CONTENT)
    for key in _REQUIRED_SESSION_SETUP_KEYS:
        if isinstance(payload.get(key), dict):
            normalized[key] = payload[key]
        else:
            logger.warning("Session setup content from %s is missing key %s.", path, key)
    return normalized
```

Keep module constants:

```python
_SESSION_SETUP_CONTENT = _load_session_setup_content()
DEFAULT_THEME_LIBRARY = deepcopy(_SESSION_SETUP_CONTENT["default_theme_library"])
```

- [ ] **Step 5: Implement priority padding**

In `assessment_runtime/runner.py`, compute the priorities once before `append_history`:

```python
priorities = [str(item) for item in (coaching_obj.get("top_3_priorities") or []) if str(item).strip()]
padded_priorities = priorities[:3] + [""] * max(0, 3 - len(priorities[:3]))
```

Then use:

```python
"top_priority_1": padded_priorities[0],
"top_priority_2": padded_priorities[1],
"top_priority_3": padded_priorities[2],
```

- [ ] **Step 6: Fix German text in shared JSON**

In `assessment_runtime/data/session_setup_content.json`, replace only the German ASCII transliterations:

```json
"travel_narrative": "Sprich über '{theme}' in einer klaren Reihenfolge: Anfang, Entwicklung, Schluss.",
"personal_experience": "Erzähle '{theme}' als persönliche Erfahrung mit konkreten Details.",
"opinion_monologue": "Beziehe zu '{theme}' Stellung und stütze deine Meinung mit mindestens zwei Argumenten.",
"free_monologue": "Sprich auf Deutsch über '{theme}' mit einer einfachen, klaren Struktur.",
"picture_description": "Beschreibe '{theme}', erkläre den Kontext und vermute, was danach passiert.",
"Verbinde deine Ideen mit klaren Übergängen."
```

- [ ] **Step 7: Run focused tests**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_theme_library.py tests/test_assessment_runner.py -q
```

Expected: pass.

## Task 2: Backend Cleanup Accounting

- [ ] **Step 1: Add failing cleanup count tests**

In `tests/test_app_backend_config.py`, add:

```python
def test_execute_cleanup_counts_only_successful_non_dry_run_deletes(self):
    with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
        config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8765)
        deleted = config.app_data.temp_dir / "deleted.tmp"
        locked = config.app_data.temp_dir / "locked.tmp"
        vanished = config.app_data.temp_dir / "vanished.tmp"
        deleted.parent.mkdir(parents=True, exist_ok=True)
        deleted.write_text("gone", encoding="utf-8")
        locked.write_text("stay", encoding="utf-8")

        original_unlink = Path.unlink

        def flaky_unlink(path):
            if path == locked:
                raise OSError("locked")
            return original_unlink(path)

        with mock.patch("app_backend.maintenance.cleanup_candidates", return_value=[deleted, locked, vanished]), mock.patch.object(Path, "unlink", autospec=True, side_effect=flaky_unlink):
            result = execute_cleanup(config, CleanupTarget.TMP, dry_run=False)

    self.assertEqual(result.deleted_file_count, 1)
    self.assertEqual(result.freed_bytes, 4)
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_config.py::BackendConfigTests::test_execute_cleanup_counts_only_successful_non_dry_run_deletes -q
```

Expected: fails because current code counts all candidates and stats before deletion.

- [ ] **Step 3: Implement per-file stat/delete accounting**

In `app_backend/maintenance.py`, rewrite `execute_cleanup` to count successful deletes:

```python
deleted_file_count = 0
freed_bytes = 0
for path in candidates:
    try:
        size_bytes = path.stat().st_size
    except OSError:
        continue
    if dry_run:
        deleted_file_count += 1
        freed_bytes += size_bytes
        continue
    try:
        path.unlink()
    except OSError:
        continue
    deleted_file_count += 1
    freed_bytes += size_bytes
```

Return those variables instead of `len(candidates)` and a precomputed sum.

- [ ] **Step 4: Run focused cleanup tests**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_config.py -q
```

Expected: pass.

## Task 3: Support Bundle Filesystem Hardening

- [ ] **Step 1: Add failing support bundle path validation tests**

In `tests/test_app_backend_support_bundle.py`, add:

```python
def test_support_bundle_path_rejects_path_traversal_ids(self):
    with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
        config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8773)

        for bundle_id in ("../escape", "bundle_123/escape", "/tmp/bundle_123", "bundle_123.zip"):
            with self.subTest(bundle_id=bundle_id):
                with self.assertRaises(ValueError):
                    support_bundle_path(config, bundle_id)
```

- [ ] **Step 2: Add failing unreadable/bad optional tree tests**

In `tests/test_app_backend_support_bundle.py`, add tests that monkeypatch `Path.read_text` and `zipfile.ZipFile.write` for one source path only:

```python
def test_create_support_bundle_skips_unreadable_text_file(self):
    with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
        config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8774)
        notes = config.app_data.reports_dir / "notes.txt"
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text("secret text", encoding="utf-8")
        original_read_text = Path.read_text

        def flaky_read_text(path, *args, **kwargs):
            if path == notes:
                raise OSError("unreadable")
            return original_read_text(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", autospec=True, side_effect=flaky_read_text):
            response = create_support_bundle(
                config,
                SupportBundleCreateRequest(include_reports=True, client_snapshot={}, client_diagnostics=[]),
            )

    self.assertTrue(response.bundle_id.startswith("bundle_"))
```

Add a symlink test guarded for platforms that support symlinks:

```python
def test_create_support_bundle_skips_symlinked_files(self):
    with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
        config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8775)
        outside = Path(app_dir).parent / "outside-secret.txt"
        outside.write_text("do not bundle", encoding="utf-8")
        link = config.app_data.reports_dir / "linked.txt"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symlinks are not available")

        response = create_support_bundle(
            config,
            SupportBundleCreateRequest(include_reports=True, client_snapshot={}, client_diagnostics=[]),
        )

        with zipfile.ZipFile(support_bundle_path(config, response.bundle_id)) as archive:
            self.assertNotIn("reports/linked.txt", archive.namelist())
```

- [ ] **Step 3: Add failing API invalid bundle id test**

In `tests/test_app_backend_api.py`, add a GET test for `/v1/support-bundles/..%2Fescape` expecting a 400 or 404 JSON error, not filesystem escape or 500.

- [ ] **Step 4: Run focused tests and confirm failure**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_support_bundle.py tests/test_app_backend_api.py -q
```

Expected: fails on traversal and unreadable-file behavior.

- [ ] **Step 5: Implement local validation and per-file IO guards**

In `app_backend/support_bundle.py`:

```python
BUNDLE_ID_PATTERN = re.compile(r"^bundle_[A-Za-z0-9]+$")

def support_bundle_path(runtime_config: BackendRuntimeConfig, bundle_id: str) -> Path:
    candidate = str(bundle_id or "").strip()
    if not BUNDLE_ID_PATTERN.fullmatch(candidate):
        raise ValueError("Invalid support bundle id.")
    bundle_root = support_bundle_dir(runtime_config).resolve()
    bundle_path = (bundle_root / f"{candidate}.zip").resolve()
    bundle_path.relative_to(bundle_root)
    return bundle_path
```

Update `_iter_files`:

```python
return [path for path in root.rglob("*") if not path.is_symlink() and path.is_file()]
```

Update text and optional tree archive helpers so a single bad file logs and skips:

```python
try:
    content = source.read_text(encoding="utf-8")
except (OSError, UnicodeDecodeError) as exc:
    logger.warning("Skipping support bundle text file %s: %s", source, exc)
    return
```

Wrap `archive.write` with `try/except OSError` in `_add_optional_tree`.

In `app_backend/app.py`, catch `ValueError` from `support_bundle_path` in the download route and raise `_http_error(400, ErrorCode.VALIDATION, str(exc))`.

- [ ] **Step 6: Run focused tests**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_support_bundle.py tests/test_app_backend_api.py -q
```

Expected: pass.

## Task 4: Backend Lifecycle Process Safety

- [ ] **Step 1: Add failing timeout cleanup test**

In `tests/test_app_backend_lifecycle.py`, add:

```python
class FakePopen:
    def __init__(self):
        self.pid = 4242
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.terminated and not self.killed:
            raise subprocess.TimeoutExpired("backend", timeout)
        return 0

def test_ensure_local_backend_cleans_spawned_process_after_timeout(self):
    fake_process = FakePopen()
    with tempfile.TemporaryDirectory() as root_dir:
        config = _build_config(Path(root_dir).resolve(), port=9004)
        with mock.patch("app_backend.lifecycle.get_backend_state", return_value=None), mock.patch(
            "app_backend.lifecycle.build_backend_runtime_config", return_value=config
        ), mock.patch("app_backend.lifecycle._backend_command", return_value=["python", "scripts/run_backend.py"]), mock.patch(
            "app_backend.lifecycle.subprocess.Popen", return_value=fake_process
        ), mock.patch("app_backend.lifecycle.read_backend_state", return_value={"base_url": config.base_url}), mock.patch(
            "app_backend.lifecycle.is_backend_healthy", return_value=False
        ), mock.patch("app_backend.lifecycle.time.time", side_effect=[100.0, 100.2]), mock.patch(
            "app_backend.lifecycle.time.sleep"
        ):
            with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                ensure_local_backend(startup_timeout_sec=0.1)

    self.assertTrue(fake_process.terminated)
    self.assertTrue(fake_process.killed)
```

- [ ] **Step 2: Add failing recycled PID guard test**

Update `_safe_terminate_pid` tests:

```python
def test_safe_terminate_pid_skips_process_when_metadata_does_not_match(self):
    with mock.patch("app_backend.lifecycle._pid_matches_backend_command", return_value=False), mock.patch(
        "app_backend.lifecycle.os.kill"
    ) as mock_kill:
        _safe_terminate_pid(123, expected_command_marker="scripts/run_backend.py")

    mock_kill.assert_not_called()
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_lifecycle.py -q
```

Expected: fails because `Popen` is not retained and `_safe_terminate_pid` has no metadata check.

- [ ] **Step 4: Implement process cleanup without process groups**

In `app_backend/lifecycle.py`, assign the process:

```python
process = subprocess.Popen(...)
```

Add:

```python
def _terminate_spawned_backend(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            return
```

Call `_terminate_spawned_backend(process)` before raising the timeout error.

- [ ] **Step 5: Implement stale PID identity guard**

Add a small command marker check that avoids new dependencies:

```python
def _pid_matches_backend_command(pid: int, expected_command_marker: str) -> bool:
    if not expected_command_marker:
        return True
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and expected_command_marker in result.stdout
```

Change `_safe_terminate_pid` to accept `expected_command_marker: str = ""` and skip `SIGTERM` when the marker does not match. In `get_backend_state`, call:

```python
_safe_terminate_pid(pid, expected_command_marker="scripts/run_backend.py")
```

- [ ] **Step 6: Run focused tests**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_lifecycle.py -q
```

Expected: pass.

## Task 5: Backend Job Terminal-State Correctness

- [ ] **Step 1: Create focused jobs tests**

Create `tests/test_app_backend_jobs.py` with:

```python
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from app_backend.config import build_backend_runtime_config
from app_backend.jobs import JobManager, prunable_job_metadata_files


class BackendJobsTests(unittest.TestCase):
    def test_prunable_job_metadata_files_handles_mixed_case_terminal_statuses(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8766)
            now = datetime(2026, 4, 22, 10, 0, tzinfo=UTC)
            old_completed = config.jobs_dir / "old-completed.json"
            old_completed.write_text(
                json.dumps({"status": "Completed", "completed_at": (now - timedelta(days=31)).isoformat()}),
                encoding="utf-8",
            )

            candidates = prunable_job_metadata_files(config.jobs_dir, now=now)

        self.assertEqual(candidates, [old_completed])
```

Add a cancel race test using a fake process and a job file that becomes completed before the cancel write:

```python
def test_cancel_does_not_overwrite_completed_job_after_process_join(self):
    with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
        config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
        manager = JobManager(config)
        assessment_id = "asmt_race"
        job_file = config.jobs_dir / f"{assessment_id}.json"
        job_file.write_text(json.dumps({"assessment_id": assessment_id, "status": "running", "phase": "running", "progress": 0.5}), encoding="utf-8")
        process = mock.Mock()
        process.is_alive.return_value = True

        def complete_on_join(timeout=None):
            job_file.write_text(json.dumps({"assessment_id": assessment_id, "status": "completed", "phase": "done", "progress": 1.0}), encoding="utf-8")

        process.join.side_effect = complete_on_join
        manager._processes[assessment_id] = process

        status = manager.cancel(assessment_id)

    self.assertEqual(status.status.value, "completed")
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_jobs.py -q
```

Expected: cancel race fails because current code writes cancelled after join.

- [ ] **Step 3: Normalize terminal status comparisons at read boundary**

In `app_backend/jobs.py`, add:

```python
TERMINAL_JOB_STATUSES_NORMALIZED = {status.lower() for status in TERMINAL_JOB_STATUSES}
```

Use it in `prunable_job_metadata_files` and cancel re-read checks:

```python
if str(payload.get("status") or "").strip().lower() not in TERMINAL_JOB_STATUSES_NORMALIZED:
    continue
```

- [ ] **Step 4: Re-read status after cancel join**

In `JobManager.cancel`, after `process.join(timeout=1.0)`, re-read the job file:

```python
payload = _read_json(_job_file(self._config.jobs_dir, assessment_id))
if str(payload.get("status") or "").strip().lower() in TERMINAL_JOB_STATUSES_NORMALIZED:
    self._processes.pop(assessment_id, None)
    return self.get_status(assessment_id)
```

Only write cancelled when the re-read payload is still non-terminal.

- [ ] **Step 5: Run focused tests**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_jobs.py tests/test_app_backend_config.py tests/test_app_backend_api.py -q
```

Expected: pass.

## Task 6: App Shell Runtime And Data Roots

- [ ] **Step 1: Update runtime resolver env fallback test**

In `tests/test_runtime_resolver.py`, replace `test_resolve_runtime_config_ignores_environment_fallback` with:

```python
@mock.patch("app_shell.runtime_resolver.get_secret", return_value="")
def test_resolve_runtime_config_uses_provider_environment_fallback(self, _mock_get_secret):
    prefs = AppPreferences(
        connections=[
            ProviderConnection(
                connection_id="conn-3",
                provider_kind="openrouter",
                label="OpenRouter",
                base_url="https://openrouter.ai/api/v1",
                default_model="google/gemini-3.1-pro-preview",
                secret_ref="connection:conn-3",
                is_default=True,
            )
        ],
        active_connection_id="conn-3",
    )
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key", "LLM_API_KEY": ""}, clear=False):
        runtime = resolve_runtime_config(prefs)

    self.assertEqual(runtime.api_key, "env-key")
```

- [ ] **Step 2: Add app data cache path tests**

In `tests/test_app_shell_app_data.py`, update the custom-home expectation:

```python
def test_resolve_cache_root_uses_custom_app_data_home_when_cache_home_is_absent(self):
    with tempfile.TemporaryDirectory() as data_home, mock.patch.dict(
        os.environ,
        {APP_DATA_HOME_ENV_VAR: data_home},
        clear=True,
    ):
        self.assertEqual(resolve_cache_root(), Path(data_home).resolve() / "cache")
```

Add the Windows legacy branch test:

```python
def test_legacy_windows_cache_home_does_not_append_cache_segment(self):
    with mock.patch("app_shell.app_data.os.name", "nt"), mock.patch.dict(
        os.environ,
        {"LOCALAPPDATA": "C:/Users/test/AppData/Local"},
        clear=True,
    ):
        self.assertEqual(
            app_data._legacy_platform_cache_home(),
            Path("C:/Users/test/AppData/Local"),
        )
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_runtime_resolver.py tests/test_app_shell_app_data.py -q
```

Expected: fails on old env fallback and cache root behavior.

- [ ] **Step 4: Implement provider env fallback**

In `app_shell/runtime_resolver.py`, add `os`, `logging`, and a local helper:

```python
_PROVIDER_API_KEY_ENV_NAMES = {
    "openrouter": ("OPENROUTER_API_KEY", "LLM_API_KEY"),
    "ollama": ("OLLAMA_API_KEY", "LLM_API_KEY"),
    "lmstudio": ("LLM_API_KEY",),
    "openai_compatible": ("LLM_API_KEY",),
}

def _provider_env_api_key(provider: str) -> str:
    for name in _PROVIDER_API_KEY_ENV_NAMES.get(provider, ("LLM_API_KEY",)):
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return ""
```

Change `_connection_api_key` to accept `provider: str`, warn when `secret_ref` is present but missing, and fall back:

```python
def _connection_api_key(connection: ProviderConnection, provider: str) -> str:
    if connection.secret_ref:
        secret = get_secret(connection.secret_ref)
        if secret:
            return secret
        logger.warning("Saved secret %s is unavailable; checking provider environment variables.", connection.secret_ref)
    return _provider_env_api_key(provider)
```

- [ ] **Step 5: Implement cache root semantics**

In `app_shell/app_data.py`:

```python
if _env_override(APP_DATA_HOME_ENV_VAR, LEGACY_APP_DATA_HOME_ENV_VAR) is not None:
    return resolve_app_data_root() / "cache"
```

Keep `APP_CACHE_HOME_ENV_VAR` as the highest priority override.

Change Windows legacy cache home:

```python
if os.name == "nt":
    return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
```

- [ ] **Step 6: Run focused tests**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_runtime_resolver.py tests/test_app_shell_app_data.py -q
```

Expected: pass.

## Task 7: App Shell UI Diagnostics

- [ ] **Step 1: Add translated-status regression test**

In `tests/test_app_shell_review_components.py`, add a test that makes translated labels differ from English and still expects warning promotion:

```python
def test_report_status_summary_promotes_warning_without_comparing_translated_text(self):
    translations = {
        "review.status_done": "Fertig",
        "review.status_short_done": "Erledigt",
        "review.status_short_unstable": "Prüfen",
        "review.warnings": "Warnung: {value}",
        "review.warning_codes.llm_unavailable": "LLM nicht verfügbar",
    }

    with mock.patch("app_shell.review_components.t", side_effect=lambda key, **kwargs: translations.get(key, key).format(**kwargs)):
        status = report_status_summary({"warnings": ["llm_unavailable"]})

    self.assertEqual(status.level, "warning")
    self.assertEqual(status.short_label, "Prüfen")
```

- [ ] **Step 2: Add diagnostics path expectation**

In `tests/test_app_shell_diagnostics.py`, update the writable diagnostic assertions so both ok and error detail args use `paths.temp_dir`.

Example:

```python
self.assertEqual(by_key["app_data"].detail_args["path"], str(paths.temp_dir))
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_review_components.py tests/test_app_shell_diagnostics.py -q
```

Expected: diagnostics expectation fails before implementation.

- [ ] **Step 4: Stop comparing translated text**

In `app_shell/review_components.py`, track whether the base status was done:

```python
is_done = not summary.get("requires_human_review") and not failed_gates
...
if warning_messages:
    level = "warning"
    messages.append(t("review.warnings", value=" ".join(warning_messages)))
    if is_done:
        short_label = t("review.status_short_unstable")
```

- [ ] **Step 5: Report the probed writable path**

In `app_shell/diagnostics.py`, use `paths.temp_dir` in `_app_data_writable_diagnostic` `detail_args` for both ok and error branches.

- [ ] **Step 6: Run focused tests**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_review_components.py tests/test_app_shell_diagnostics.py -q
```

Expected: pass.

## Task 8: App Shell Service Guardrails

- [ ] **Step 1: Add invalid support bundle creation result test**

In `tests/test_app_shell_services.py`, add:

```python
def test_export_support_bundle_archive_rejects_missing_bundle_id(self):
    state = build_default_state()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
        "app_shell.services.create_support_bundle_archive",
        return_value=mock.Mock(bundle_id=""),
    ), mock.patch("app_shell.services.backend_client.download_support_bundle") as download:
        with self.assertRaisesRegex(RuntimeError, "Support bundle could not be created"):
            services.export_support_bundle_archive(state, destination=tmpdir)

    download.assert_not_called()
```

- [ ] **Step 2: Add upload file duck-typing tests**

In `tests/test_app_shell_services.py`, add:

```python
def test_store_uploaded_audio_accepts_plain_file_like_read(self):
    uploaded = io.BytesIO(b"audio")
    uploaded.name = "attempt.wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        path, digest = services.store_uploaded_audio(uploaded, target_dir=tmpdir)

    self.assertIsNotNone(path)
    self.assertEqual(digest, hashlib.sha1(b"audio").hexdigest())

def test_store_uploaded_audio_rejects_objects_without_byte_reader(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with self.assertRaisesRegex(TypeError, "Uploaded file"):
            services.store_uploaded_audio(object(), target_dir=tmpdir)
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_services.py -q
```

Expected: fails on missing bundle id and plain file-like upload.

- [ ] **Step 4: Implement support bundle result validation**

In `app_shell/services.py`:

```python
created = create_support_bundle_archive(...)
bundle_id = str(getattr(created, "bundle_id", "") or "").strip()
if not bundle_id:
    raise RuntimeError("Support bundle could not be created.")
return backend_client.download_support_bundle(bundle_id, destination=destination, log_dir=...)
```

- [ ] **Step 5: Implement a small upload bytes helper**

In `app_shell/services.py`:

```python
def _uploaded_file_bytes(uploaded_file) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return bytes(uploaded_file.getvalue())
    if hasattr(uploaded_file, "getbuffer"):
        return bytes(uploaded_file.getbuffer())
    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        if isinstance(data, bytes):
            return data
    raise TypeError("Uploaded file must provide bytes via getvalue(), getbuffer(), or read().")
```

Use it in `store_uploaded_audio`.

- [ ] **Step 6: Run focused tests**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_services.py -q
```

Expected: pass.

## Task 9: Frontend API Contract Safety

- [ ] **Step 1: Add API client path encoding tests**

Create `frontend/src/lib/api/client.test.ts`:

```ts
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import { createApiClient } from "./client";

const jsonResponse = (payload: unknown) =>
  new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

describe("api client", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ payload: {} })),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("encodes dynamic history path segments", async () => {
    const client = createApiClient("http://localhost:8771");

    await client.getHistoryDetail("session/one two");

    expect(fetch).toHaveBeenCalledWith(
      "http://localhost:8771/v1/history/session%2Fone%20two",
      expect.any(Object),
    );
  });
});
```

Add similar assertions for `getAssessmentStatus`, `cancelAssessment`, `downloadSupportBundle`, runtime connection ids, and whisper model names.

- [ ] **Step 2: Narrow `HistoryRow` types**

In `frontend/src/lib/api/types.ts`, replace `unknown` fields with explicit API contract fields:

```ts
export interface HistoryRow {
  timestamp: string;
  session_id: string;
  speaker_id: string;
  learning_language: string;
  theme: string;
  task_family: string;
  overall: number | null;
  wpm: number | null;
  report_path: string;
  requires_human_review: boolean;
  duration_pass: boolean;
  topic_pass: boolean;
  language_pass: boolean;
  min_words_pass: boolean;
  top_priorities: string[];
  grammar_error_categories: string[];
  coherence_issue_categories: string[];
  final_score: number | null;
  band: string;
}
```

If typecheck reveals a backend nullable boolean contract, use `boolean | null` for only that field and update `HistoryRoute.tsx` accordingly.

- [ ] **Step 3: Run frontend tests and confirm failure**

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm run test -- src/lib/api/client.test.ts
```

Expected: encoding tests fail before implementation.

- [ ] **Step 4: Implement dynamic segment encoding**

In `frontend/src/lib/api/client.ts`, add:

```ts
const pathSegment = (value: string): string => encodeURIComponent(value);
```

Use it for every dynamic segment:

```ts
`/v1/assessments/${pathSegment(assessmentId)}`
`/v1/assessments/${pathSegment(assessmentId)}/cancel`
`/v1/history/${pathSegment(sessionId)}`
`/v1/support-bundles/${pathSegment(bundleId)}`
`/v1/runtime/settings/connections/${pathSegment(connectionId)}`
`/v1/runtime/whisper-models/${pathSegment(modelSize)}`
```

- [ ] **Step 5: Run focused frontend checks**

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm run test -- src/lib/api/client.test.ts
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm run typecheck
```

Expected: pass.

## Task 10: Frontend Store And Setup Content Guards

- [ ] **Step 1: Add app store terminal-error test**

Create `frontend/src/lib/state/appStore.test.ts`:

```ts
import { describe, expect, it } from "vitest";

import { createAppStore } from "./appStore";

describe("app store recording jobs", () => {
  it("does not store undefined when a failed job omits an error", () => {
    const store = createAppStore({
      recording: {
        audioPath: "/tmp/audio.wav",
        inputDigest: "abc",
        inputMethod: "upload",
        status: "assessing",
        assessmentState: "running",
        error: "",
        job: {
          assessmentId: "asmt-1",
          status: "running",
          phase: "running",
          progress: 0.5,
          error: "",
          reportPath: "",
        },
      },
    });

    store.getState().setRecordingJob({ status: "failed", error: undefined });

    expect(store.getState().recording.error).toBe("");
  });
});
```

- [ ] **Step 2: Add setup content validation tests**

Create `frontend/src/lib/setup/sessionSetupContent.test.ts`:

```ts
import { describe, expect, it } from "vitest";

import { isValidTaskFamily, parseSessionSetupContent } from "./sessionSetupContent";

describe("session setup content", () => {
  it("rejects shared content without required keys", () => {
    expect(() => parseSessionSetupContent({ default_theme_library: {} })).toThrow(
      /practice_brief_templates/,
    );
  });

  it("guards task families before assigning them", () => {
    expect(isValidTaskFamily("opinion_monologue")).toBe(true);
    expect(isValidTaskFamily("made_up_family")).toBe(false);
  });
});
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm run test -- src/lib/state/appStore.test.ts src/lib/setup/sessionSetupContent.test.ts
```

Expected: setup parser exports do not exist and app store can store undefined.

- [ ] **Step 4: Fix app store error fallback**

In `frontend/src/lib/state/appStore.ts`:

```ts
nextError = nextJob.error ?? "";
```

Keep this local to the failed/cancelled branch.

- [ ] **Step 5: Add narrow setup content parsing**

In `frontend/src/lib/setup/sessionSetupContent.ts`, import the runtime option list:

```ts
import { TASK_FAMILY_OPTIONS, type CefrLevel, type DurationOption, type TaskFamily } from "@/lib/state/sessionDraft";
```

Add:

```ts
const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

export const isValidTaskFamily = (value: string): value is TaskFamily =>
  TASK_FAMILY_OPTIONS.includes(value as TaskFamily);
```

Add `parseSessionSetupContent(raw: unknown): SessionSetupContent` that verifies `default_theme_library` and `practice_brief_templates` are records. Export it for tests and replace:

```ts
const content = sharedContent as SessionSetupContent;
```

with:

```ts
const content = parseSessionSetupContent(sharedContent);
```

When normalizing theme entries, replace the direct cast:

```ts
const taskFamily = String(theme.task_family || "free_monologue").trim();
...
task_family: isValidTaskFamily(taskFamily) ? taskFamily : "free_monologue",
```

- [ ] **Step 6: Run focused frontend checks**

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm run test -- src/lib/state/appStore.test.ts src/lib/setup/sessionSetupContent.test.ts
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm run typecheck
```

Expected: pass.

## Full Verification

After all tasks pass individually, run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest \
  tests/test_theme_library.py \
  tests/test_assessment_runner.py \
  tests/test_app_backend_config.py \
  tests/test_app_backend_support_bundle.py \
  tests/test_app_backend_api.py \
  tests/test_app_backend_lifecycle.py \
  tests/test_app_backend_jobs.py \
  tests/test_runtime_resolver.py \
  tests/test_app_shell_app_data.py \
  tests/test_app_shell_review_components.py \
  tests/test_app_shell_diagnostics.py \
  tests/test_app_shell_services.py \
  -q
```

Then run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm run test
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm run typecheck
```

Finally run the repository Flutter-error helper only if the implementation unexpectedly touches Flutter-facing files or generated screen code:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.tools/testing/flutter_errors_lib_only.py
```

## Self-Review

- Spec coverage: all 26 CodeRabbit findings are mapped to exactly one task above.
- File limit: every task touches 5 files or fewer, including tests.
- Scope check: no broad screen redesign, no new validation framework, no process-group rewrite, no branch/worktree changes.
- Localization: only existing shared German copy changes; no new visible UI strings are introduced without localization.
- Execution mode: use zsh commands and the repository `.venv` for Python.

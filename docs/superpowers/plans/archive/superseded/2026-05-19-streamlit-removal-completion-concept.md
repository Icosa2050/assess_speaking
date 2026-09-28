# Streamlit Removal Completion Concept Implementation Plan

> Archive status, 2026-05-20: superseded by `docs/superpowers/plans/2026-05-16-streamlit-retirement.md` and the current `app_core` implementation. The review findings in this concept are stale; do not execute this file.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Streamlit and `streamlit-webrtc` as installed product dependencies while preserving the useful runtime, review, history, setup, support, and regression coverage in the React/FastAPI/Tauri lane.

**Architecture:** Treat the old Streamlit lane as deleted product code, not as a dormant compatibility path. Pure local-app support modules move under `app_core`, backend and launchers import only `app_core`/`app_backend`, React Playwright replaces valuable legacy behavior, and then Streamlit files/tests/packages are removed in one narrow final deletion slice.

**Tech Stack:** Python 3.12 via `.venv`, FastAPI backend, React 19 + TypeScript + Vite, Playwright, Vitest, Tauri v2, pytest, zsh on macOS.

---

## Review Findings Blocking Removal

1. `requirements.txt` still installs `streamlit==1.55.0` and `streamlit-webrtc==0.47.6`.
2. `streamlit_app.py`, `pages/`, `app_shell/page_helpers.py`, `app_shell/review_components.py`, and `app_shell/visual_system.py` are still present and import Streamlit directly.
3. `tests/test_app_shell_pages.py`, `tests/test_app_shell_page_helpers.py`, and `tests/e2e/conftest.py` still depend on Streamlit testing/server fixtures.
4. The legacy pytest E2E fixture currently calls `scripts/run_app.py` without `--legacy-streamlit`, but `run_app.py` is now backend-only by default. That means the old pytest E2E lane is not a reliable safety net anymore.
5. `app_core.state` and `app_core.services` still import pure support modules from `app_shell`. This does not necessarily import Streamlit, but it keeps the "core" layer coupled to a package that still contains Streamlit UI modules.
6. React Playwright coverage currently has local guest and settings support smoke tests, but not the full two-attempt review/history progression, live runtime setup parity, or real-audio progression replacement described in the retirement plan.
7. `scripts/run_app.py`, `app_shell/bootstrap.py`, and `tests/test_run_app.py` still preserve a `--legacy-streamlit` command path. That is fine for a transition checkpoint, but it is incompatible with claiming Streamlit has been removed.

## Removal Concept

The clean target is:

- `requirements.txt` contains no Streamlit packages.
- No default command path can try `python -m streamlit`.
- Product entrypoints are React/Vite, Tauri, FastAPI, and CLI only.
- `app_core` contains pure local app support: state, i18n, app data paths, runtime providers, runtime connections, runtime resolver, secret store, and services.
- `app_shell` is either deleted or reduced to non-product historical notes during the deletion commit. Do not keep modules named `app_shell.state` or `app_shell.i18n` after Streamlit package removal; keeping them invites accidental imports.
- Legacy Streamlit tests are deleted only after React/Vitest/Playwright coverage protects the useful behavior they used to cover.
- Hosted/EU deployment becomes straightforward: serve React static assets, run FastAPI in the chosen EU region, store runtime/user data in regional infrastructure, and do not deploy Streamlit.

## Test Replacement Matrix

Worth replacing before deletion:

- `tests/e2e/test_app_shell_e2e.py`
  - Replace with React Playwright for two-attempt upload/record -> review -> history progression.
  - Keep remove-recording in Vitest if `SpeakRoute.test.tsx` already covers it; otherwise add one Playwright assertion in the same flow.
  - Replace localized setup assertions with route/Vitest coverage plus one Playwright smoke.
- `tests/e2e/test_app_shell_runtime_setup_e2e.py`
  - Replace with opt-in React Playwright for live Ollama/LM Studio model detection and failure feedback.
- `tests/e2e/test_app_shell_real_history_e2e.py`
  - Replace with opt-in React Playwright real-audio progression using the same environment guard.

Safe to delete after replacement:

- `tests/test_app_shell_pages.py`
- `tests/test_app_shell_page_helpers.py`
- Streamlit-specific assertions in `tests/test_app_shell_state.py`
- Streamlit-only portions of `tests/test_app_shell_review_components.py`
- Streamlit pytest server fixtures in `tests/e2e/conftest.py`

Migrate rather than delete:

- `tests/test_app_shell_services.py` -> `tests/test_app_core_services.py`
- `tests/test_app_shell_i18n.py` -> `tests/test_app_core_i18n.py`
- `tests/test_runtime_connections.py` and `tests/test_runtime_resolver.py` imports -> `app_core.*`
- `tests/test_app_backend_api.py` imports -> `app_core.*`
- `tests/test_run_app.py` -> backend/bootstrap-only launcher contract

---

## Task 1: Add Hard Removal Guardrails

**Files:**
- Modify: `tests/test_app_core_imports.py`
- Create: `tests/test_streamlit_removal_contract.py`
- Test: `tests/test_app_core_imports.py`, `tests/test_streamlit_removal_contract.py`

- [ ] **Step 1: Add a failing guard for core importing shell**

Add this test to `tests/test_app_core_imports.py`:

```python
def test_app_core_modules_do_not_import_app_shell_support_modules():
    script = """
import builtins
import importlib

original_import = builtins.__import__

def block_app_shell_import(name, *args, **kwargs):
    if name == "app_shell" or name.startswith("app_shell."):
        raise AssertionError(f"app_core must not import {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_app_shell_import
importlib.import_module("app_core.state")
importlib.import_module("app_core.i18n")
importlib.import_module("app_core.services")
"""
    _run_no_streamlit_import_guard(script)
```

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_imports.py::test_app_core_modules_do_not_import_app_shell_support_modules -q
```

Expected now: FAIL because `app_core.state` imports `app_shell.app_data` and `app_core.services` imports several `app_shell.*` support modules.

- [ ] **Step 2: Add a failing package-level Streamlit contract**

Create `tests/test_streamlit_removal_contract.py`:

```python
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = ROOT / root
        if path.is_file():
            files.append(path)
        elif path.exists():
            files.extend(sorted(item for item in path.rglob("*.py") if item.is_file()))
    return files


def test_no_product_python_imports_streamlit():
    offenders: list[str] = []
    for path in _python_files("app_backend", "app_core", "assessment_runtime", "scripts", "assess_speaking.py"):
        text = path.read_text(encoding="utf-8")
        if "import streamlit" in text or "from streamlit" in text or "streamlit_webrtc" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_requirements_do_not_install_streamlit():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "streamlit==" not in requirements
    assert "streamlit-webrtc" not in requirements
```

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_streamlit_removal_contract.py -q
```

Expected now: FAIL because `requirements.txt` still contains Streamlit packages.

---

## Task 2: Move Pure Support Modules From `app_shell` To `app_core`

**Files:**
- Create: `app_core/app_data.py`
- Create: `app_core/runtime_providers.py`
- Create: `app_core/runtime_connections.py`
- Create: `app_core/runtime_resolver.py`
- Modify: `app_core/state.py`
- Test: `tests/test_app_core_imports.py`

- [ ] **Step 1: Move app data helpers**

Copy the pure contents of `app_shell/app_data.py` to `app_core/app_data.py`.

Then update `app_core/state.py`:

```python
from app_core.app_data import build_app_data_paths
from app_core.runtime_providers import DEFAULT_PROVIDER
```

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_state.py tests/test_app_core_imports.py -q
```

Expected: state/import tests pass except the full core-no-shell guard until the remaining service imports move.

- [ ] **Step 2: Move runtime provider helpers**

Copy `app_shell/runtime_providers.py` to `app_core/runtime_providers.py`.

Update imports in:

```python
from app_core.runtime_providers import ...
```

for `app_core/state.py`, `app_core/services.py`, `app_core/runtime_connections.py`, and `app_core/runtime_resolver.py`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_runtime_connections.py tests/test_runtime_resolver.py tests/test_app_core_imports.py -q
```

Expected: runtime tests pass.

- [ ] **Step 3: Move runtime connection serialization**

Copy `app_shell/runtime_connections.py` to `app_core/runtime_connections.py`, then update it to import:

```python
from app_core.state import DEFAULT_MODEL, ProviderConnection
from app_core.runtime_providers import connection_secret_ref, default_base_url, default_connection_label, normalize_provider
```

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_runtime_connections.py -q
```

Expected: runtime connection tests pass.

- [ ] **Step 4: Move runtime resolver**

Copy `app_shell/runtime_resolver.py` to `app_core/runtime_resolver.py`, then update it to import from `app_core.state`, `app_core.runtime_providers`, and the secret-store module after Task 3.

Temporarily leave `app_shell.secret_store` import only if Task 3 has not moved secrets yet.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_runtime_resolver.py tests/test_app_core_imports.py -q
```

Expected: resolver tests pass; the full no-`app_shell` guard may still fail until Task 3.

---

## Task 3: Move Secret Store And Bootstrap/Data Path Runtime

**Files:**
- Create: `app_core/secret_store.py`
- Create: `app_core/bootstrap.py`
- Modify: `app_backend/config.py`
- Modify: `app_backend/lifecycle.py`
- Modify: `scripts/run_app.py`
- Test: `tests/test_secret_store.py`, `tests/test_run_app.py`, `tests/test_app_backend_lifecycle.py`

- [ ] **Step 1: Move secret store**

Copy `app_shell/secret_store.py` to `app_core/secret_store.py`.

Update imports in `app_core/services.py`, `app_core/runtime_resolver.py`, and `app_backend/app.py` to use:

```python
from app_core.secret_store import ...
```

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_secret_store.py tests/test_runtime_resolver.py tests/test_app_backend_api.py -q
```

Expected: secret/runtime/backend tests pass.

- [ ] **Step 2: Move backend-safe bootstrap helpers**

Create `app_core/bootstrap.py` from the backend-safe parts of `app_shell/bootstrap.py`:

- `PROJECT_ROOT`
- `ASSESS_SCRIPT`
- `BACKEND_ENTRYPOINT`
- environment constants
- `RuntimeMetadata`
- `_set_default_env`
- `_set_compat_envs`
- `_has_explicit_override`
- `is_within_project_checkout`
- `build_runtime_metadata`
- `bootstrap_app_environment`

Do not move `STREAMLIT_ENTRYPOINT` or `build_streamlit_launch_command`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_run_app.py tests/test_app_backend_config.py tests/test_app_backend_lifecycle.py -q
```

Expected: launcher/backend config tests pass after imports are updated.

- [ ] **Step 3: Update backend and scripts imports**

Update these files to import app data/bootstrap from `app_core`:

- `app_backend/config.py`
- `app_backend/lifecycle.py`
- `app_backend/support_bundle.py`
- `scripts/run_backend.py`
- `scripts/run_app.py`

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_config.py tests/test_app_backend_lifecycle.py tests/test_app_backend_support_bundle.py tests/test_run_backend.py tests/test_run_app.py -q
```

Expected: backend launch/support tests pass.

---

## Task 4: Migrate Test Imports To Core Names

**Files:**
- Modify: `tests/test_app_shell_services.py`
- Modify: `tests/test_app_shell_i18n.py`
- Modify: `tests/test_app_shell_state.py`
- Modify: `tests/test_runtime_connections.py`
- Modify: `tests/test_runtime_resolver.py`

- [ ] **Step 1: Rename service test imports**

In `tests/test_app_shell_services.py`, replace:

```python
from app_shell.services import ...
from app_shell.state import ...
```

with:

```python
from app_core.services import ...
from app_core.state import ...
```

Also update mock patch targets from `"app_shell.services..."` to `"app_core.services..."`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_services.py -q
```

Expected: service tests pass.

- [ ] **Step 2: Rename pure state/i18n imports**

In `tests/test_app_shell_i18n.py` and the pure parts of `tests/test_app_shell_state.py`, import from:

```python
from app_core.i18n import flatten_keys, load_locale, locale_key_map, t
from app_core.state import AppPreferences, AppState, DraftSession, RecordingState, RecordingStatus, ReviewState, build_default_state
```

Move the Streamlit session wrapper test into a legacy deletion bucket instead of keeping it as a core test.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_i18n.py tests/test_app_shell_state.py -q
```

Expected: pure tests pass with no Streamlit import.

- [ ] **Step 3: Rename runtime tests**

Update `tests/test_runtime_connections.py` and `tests/test_runtime_resolver.py` to import from `app_core.runtime_connections`, `app_core.runtime_resolver`, and `app_core.state`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_runtime_connections.py tests/test_runtime_resolver.py tests/test_app_core_imports.py -q
```

Expected: runtime tests pass and `test_app_core_modules_do_not_import_app_shell_support_modules` passes.

---

## Task 5: Add React E2E Replacements For Valuable Streamlit Coverage

**Files:**
- Create: `frontend/tests/e2e/reviewHistoryFlow.spec.ts`
- Create: `frontend/tests/e2e/runtimeSetupLive.spec.ts`
- Create: `frontend/tests/e2e/realAudioHistory.spec.ts`
- Modify: `frontend/playwright.config.ts`
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`

- [ ] **Step 1: Confirm/remove-recording Vitest coverage**

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
npm test src/routes/tests/SpeakRoute.test.tsx
```

If no assertion covers attached recording removal, add a Vitest case in `frontend/src/routes/tests/SpeakRoute.test.tsx` that:

```ts
await user.upload(screen.getByTestId("speak.upload_audio"), file);
expect(screen.getByTestId("speak.recording_ready")).toBeInTheDocument();
await user.click(screen.getByTestId("speak.remove_recording"));
expect(screen.queryByTestId("speak.recording_ready")).not.toBeInTheDocument();
```

Run the file again.

- [ ] **Step 2: Add deterministic review/history Playwright flow**

Create `frontend/tests/e2e/reviewHistoryFlow.spec.ts`.

The test must:

- configure a dry-run/local runtime connection through backend API or seeded state;
- navigate Home -> Session Setup -> Speak;
- upload or attach first sample audio;
- submit and reach Review;
- assert report summary, transcript/notes, validation gates, and score mode are visible;
- use Try Again;
- upload/attach second sample audio;
- assert Progress delta appears;
- open History and assert the second attempt detail is selected.

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
NODE_ENV=development npx playwright test -c playwright.config.ts reviewHistoryFlow.spec.ts
```

Expected: deterministic flow passes without Streamlit.

- [ ] **Step 3: Add live runtime setup replacement**

Create `frontend/tests/e2e/runtimeSetupLive.spec.ts` with this guard:

```ts
test.skip(process.env.RUN_VOSTAVO_LOCAL_RUNTIME_E2E !== "1", "Set RUN_VOSTAVO_LOCAL_RUNTIME_E2E=1 to run live runtime setup tests.");
```

Cover:

- mobile viewport `390x844`;
- Ollama local model detection and sanitized base URL;
- LM Studio local model detection or visible connection-failure feedback for invalid model.

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
RUN_VOSTAVO_LOCAL_RUNTIME_E2E=1 NODE_ENV=development npx playwright test -c playwright.config.ts runtimeSetupLive.spec.ts
```

Expected: skips without env flag; passes when services are available.

- [ ] **Step 4: Add real-audio history progression replacement**

Create `frontend/tests/e2e/realAudioHistory.spec.ts` with this guard:

```ts
test.skip(process.env.RUN_VOSTAVO_REAL_E2E !== "1", "Set RUN_VOSTAVO_REAL_E2E=1 to run real-audio history progression.");
```

Assert:

- weaker real audio saves a report;
- stronger real audio saves a report;
- second score is greater than first score;
- second band is at least first band;
- Progress delta appears;
- History opens with second attempt detail.

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
RUN_VOSTAVO_REAL_E2E=1 OPENROUTER_API_KEY="$OPENROUTER_API_KEY" NODE_ENV=development npx playwright test -c playwright.config.ts realAudioHistory.spec.ts
```

Expected: skips unless prerequisites exist; passes with real audio and key.

---

## Task 6: Remove Legacy Streamlit Launcher Contract

**Files:**
- Modify: `scripts/run_app.py`
- Modify: `tests/test_run_app.py`
- Modify: `frontend/src-tauri/src/main.rs`
- Modify: `README.md`
- Modify: `app_core/bootstrap.py`

- [ ] **Step 1: Delete `--legacy-streamlit` from run_app**

In `scripts/run_app.py`, remove:

- `--legacy-streamlit`
- `streamlit_args`
- `STREAMLIT_ENTRYPOINT`
- `build_streamlit_launch_command`
- code that adds `entrypoint` or `command` to payload
- `subprocess.run(payload["command"])`

The default `main([])` behavior should remain: start/reuse backend, print JSON, return `0`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_run_app.py -q
```

Expected: tests pass after Step 2 updates expectations.

- [ ] **Step 2: Rewrite launcher tests**

In `tests/test_run_app.py`, delete tests named like:

- `test_main_legacy_streamlit_launches_streamlit_command`
- `test_launcher_payload_forwards_legacy_streamlit_args`
- `test_parse_args_strips_separator_from_legacy_streamlit_args`

Keep tests that assert:

- dry-run prints backend diagnostics;
- default starts/reuses backend and does not invoke `subprocess.run`;
- `--desktop-bootstrap` prints bridge env for compatibility or is replaced by `scripts/bootstrap_backend.py`;
- Tauri uses `scripts/bootstrap_backend.py`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_run_app.py -q
```

Expected: launcher tests pass.

- [ ] **Step 3: Remove Streamlit launcher helper from bootstrap**

In `app_core/bootstrap.py`, do not include `STREAMLIT_ENTRYPOINT` or `build_streamlit_launch_command`.

If `app_shell/bootstrap.py` still exists, it should be deleted in Task 7.

Run:

```zsh
rg -n "build_streamlit_launch_command|STREAMLIT_ENTRYPOINT|legacy-streamlit" scripts app_core app_backend frontend/src-tauri tests
```

Expected: no matches outside this plan file.

---

## Task 7: Delete Streamlit UI Files And Tests

**Files:**
- Modify: `requirements.txt`
- Delete: `streamlit_app.py`
- Delete: `pages/`
- Delete: `app_shell/page_helpers.py`, `app_shell/review_components.py`, `app_shell/visual_system.py`
- Delete: `tests/test_app_shell_pages.py`, `tests/test_app_shell_page_helpers.py`, Streamlit pytest E2E files/fixtures

- [ ] **Step 1: Delete package pins**

Remove these lines from `requirements.txt`:

```text
streamlit==1.55.0
streamlit-webrtc==0.47.6
```

Keep `watchdog` only if another tool needs it. If not, remove it in the same dependency cleanup.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_streamlit_removal_contract.py -q
```

Expected: requirements guard passes.

- [ ] **Step 2: Delete Streamlit app and page files**

Delete:

```text
streamlit_app.py
pages/00_Setup.py
pages/01_Session_Setup.py
pages/02_Speak.py
pages/03_Review.py
pages/04_History.py
pages/05_Library.py
pages/06_Settings.py
pages/07_Scoring_Guide.py
app_shell/page_helpers.py
app_shell/review_components.py
app_shell/visual_system.py
```

Run:

```zsh
rg -n "import streamlit|from streamlit|streamlit_webrtc|st\\." app_backend app_core scripts assessment_runtime
```

Expected: no matches.

- [ ] **Step 3: Delete Streamlit tests and fixtures**

Delete:

```text
tests/test_app_shell_pages.py
tests/test_app_shell_page_helpers.py
tests/e2e/test_app_shell_e2e.py
tests/e2e/test_app_shell_runtime_setup_e2e.py
tests/e2e/test_app_shell_real_history_e2e.py
```

Then remove Streamlit server fixture code from `tests/e2e/conftest.py`, or delete the file if no pytest E2E tests still use it.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_streamlit_removal_contract.py -q
```

Expected: Streamlit removal contract passes.

---

## Task 8: Documentation And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md`
- Modify: `docs/LOCAL_BACKEND_ARCHITECTURE.md`
- Modify: `docs/IMPLEMENTATION_PLAN.md`
- Modify: `docs/SHARED_FRONTEND_PARITY_INVENTORY.md`

- [ ] **Step 1: Update canonical local commands**

Document:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python scripts/run_backend.py --host 127.0.0.1 --port 8800
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
NODE_ENV=development VITE_LOCAL_API_BASE_URL=http://127.0.0.1:8800 npm run dev -- --host 127.0.0.1 --port 4173 --strictPort
```

Document Tauri bootstrap as:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python scripts/bootstrap_backend.py
```

- [ ] **Step 2: Update hosted/EU concept**

Use this wording:

```markdown
Hosted deployment should serve the React frontend as static assets and expose the FastAPI backend in the selected deployment region. If EU residency becomes a product/compliance requirement, choose EU-region compute, storage, logs, and secret handling for the FastAPI deployment. Streamlit is not part of the hosted architecture.
```

- [ ] **Step 3: Final searches**

Run:

```zsh
rg -n "Streamlit|streamlit|streamlit-webrtc|streamlit_app.py|pages/" README.md docs scripts tests app_backend app_core frontend
```

Expected: only historical plan notes remain, or no matches if plans are excluded.

- [ ] **Step 4: Final verification**

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
npm test
npm run typecheck
NODE_ENV=development npx playwright test -c playwright.config.ts
```

Expected:

- Python tests pass.
- Vitest passes.
- TypeScript typecheck passes.
- Playwright browser lane passes.
- `requirements.txt` installs without Streamlit.
- No product code imports Streamlit.

## Execution Order

Recommended order:

1. Task 1: guardrails first, so removal cannot regress quietly.
2. Tasks 2-4: pure module migration, so backend/core no longer depend on `app_shell`.
3. Task 5: React replacements, so no product behavior is lost.
4. Task 6: remove legacy launcher contract.
5. Task 7: delete packages/files/tests.
6. Task 8: docs and full verification.

Do not delete Streamlit package pins before Task 5 is green unless the team accepts losing the old E2E safety net for the affected flows.

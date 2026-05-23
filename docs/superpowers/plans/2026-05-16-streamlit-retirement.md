# Streamlit Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire Streamlit as a product dependency after the React/Vite frontend, FastAPI backend, and Tauri shell have equivalent coverage for the useful legacy flows.

**Architecture:** Streamlit is retired from the product runtime. Valuable legacy E2E coverage moved into the React browser lane, shared Python state/services moved into `app_core`, launchers now use the FastAPI/backend bootstrap path, and deletion is guarded by contract tests.

**Tech Stack:** Python 3.12 via `.venv`, FastAPI backend, React 19 + TypeScript + Vite, Playwright, Vitest, Tauri v2, pytest.

---

## PAL Review Summary

PAL agreed with removing Streamlit, but both reviewers rejected a big-bang delete.

The important correction was sequencing: decouple shared Python modules from `app_shell` before making Streamlit optional in launchers. Otherwise a backend-only path can still import `streamlit` indirectly through `app_shell.state` or `app_shell.i18n`.

Recommended order:

1. Add React/Playwright replacements for valuable legacy behavior.
2. Extract pure shared Python state/services out of `app_shell`.
3. Change launchers so Streamlit is no longer in product launch/bootstrap paths.
4. Delete Streamlit UI/tests/dependencies.

## Current Evidence

- `requirements.txt` no longer pins Streamlit or `streamlit-webrtc`.
- `frontend/playwright.config.ts` already starts `scripts/run_backend.py` on `127.0.0.1:8800` and Vite on `127.0.0.1:4173`; it does not need Streamlit.
- `frontend/src-tauri/src/main.rs` calls `scripts/bootstrap_backend.py` for backend bridge env vars.
- `frontend/src/components/speak/RecorderPanel.tsx` already uses browser `MediaRecorder` and `navigator.mediaDevices.getUserMedia`.
- `app_core` now owns pure app-data, bootstrap, state, i18n, diagnostics, backend-client, provider, runtime, secret-store, scoring-guide, and service modules.
- `streamlit_app.py`, `pages/`, `app_shell/`, Streamlit-only tests, and Streamlit E2E fixtures have been deleted. `tests/test_streamlit_removal_contract.py` guards against reintroducing deleted files, product imports, and product dependencies.

## Test Replacement Matrix

Replace these because they protect product behavior:

- `tests/e2e/test_app_shell_e2e.py`
  - Replace two-attempt upload -> review -> history progression with React Playwright.
  - Replace remove-recording behavior with React Playwright or keep existing Vitest if the E2E value is low.
  - Replace localized runtime setup screen assertions with React route/Vitest plus one Playwright smoke.
  - Do not literally replace Streamlit browse-file button behavior; React upload and MediaRecorder coverage supersede it.
- `tests/e2e/test_app_shell_real_history_e2e.py`
  - Replace with an opt-in React Playwright real-audio E2E using the same environment guard.
- `tests/e2e/test_app_shell_runtime_setup_e2e.py`
  - Replace with React Playwright tests for live Ollama and LM Studio model detection/test feedback on a small viewport.

Delete these once replacements and backend/unit coverage are in place:

- `tests/test_app_shell_pages.py`
  - Streamlit `AppTest` page rendering and widget navigation.
- `tests/test_app_shell_page_helpers.py`
  - Streamlit navigation, `st.switch_page`, and rendering helpers.
- `tests/test_app_shell_review_components.py`
  - Streamlit rendering helpers. Keep/replace only pure mapping logic if still used by non-Streamlit code.
- Streamlit-specific parts of `tests/test_app_shell_state.py`
  - Replace pure dataclass/default-state coverage under the new shared module.
- Streamlit-specific pytest E2E fixtures in `tests/e2e/conftest.py`
  - Replace with frontend Playwright web-server fixtures already owned by `frontend/playwright.config.ts`.

Migrated backend/runtime coverage:

- `tests/test_app_core_services.py`
  - Covers the new pure service module.
- `tests/test_app_backend_api.py`
  - Kept and updated to pure backend dependencies.
- `tests/test_app_core_i18n.py`
  - Covers the new pure i18n module.
- `tests/test_run_app.py`
  - Covers backend/bootstrap defaults and rejects the removed legacy Streamlit flag.
- `frontend/src/routes/tests/SpeakRoute.test.tsx`
  - Keep; it already covers MediaRecorder, upload, cancel, failed/completed assessment lifecycle.

---

## Task 1: Add React E2E Parity For Legacy Business Flows

**Files:**
- Create: `frontend/tests/e2e/reviewHistoryFlow.spec.ts`
- Create: `frontend/tests/e2e/runtimeSetupLive.spec.ts`
- Modify: `frontend/playwright.config.ts`
- Modify: `README.md`
- Test: `frontend/tests/e2e/*.spec.ts`

- [x] **Step 1: Add a deterministic React two-attempt E2E**

Create `frontend/tests/e2e/reviewHistoryFlow.spec.ts` with a Playwright flow that covers:

1. seed or configure a runtime connection through backend APIs or route stubs;
2. Home -> Session Setup -> Speak;
3. upload first sample audio;
4. submit and reach Review;
5. assert notes/transcript/score summary are visible;
6. use Try again;
7. upload second sample audio;
8. assert Progress delta appears;
9. open History and assert the second attempt detail is selected.

Prefer backend APIs and existing test sample files over UI setup boilerplate. If local model scoring would make the test slow or flaky, add a test-only dry-run backend mode to `frontend/playwright.config.ts` rather than mocking the whole app in the browser.

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/reviewHistoryFlow.spec.ts
```

Expected: the new flow passes against backend + Vite without Streamlit.

- [x] **Step 2: Add or confirm remove-recording coverage**

First check whether `frontend/src/routes/tests/SpeakRoute.test.tsx` is enough for remove-recording:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
npm test src/routes/tests/SpeakRoute.test.tsx
```

If the Vitest coverage already asserts record/upload switching, attached recording state, and remove button behavior, do not add a duplicate Playwright test. If there is no E2E guard for the final user-facing route state, add one short test to `frontend/tests/e2e/reviewHistoryFlow.spec.ts`.

- [x] **Step 3: Port live local runtime setup E2E**

Create `frontend/tests/e2e/runtimeSetupLive.spec.ts` with the same opt-in gate as the legacy test:

```ts
test.skip(process.env.RUN_VOSTAVO_LOCAL_RUNTIME_E2E !== "1", "Set RUN_VOSTAVO_LOCAL_RUNTIME_E2E=1 to run live local runtime setup tests.");
```

Cover:

1. mobile viewport such as `390x844`;
2. Runtime Setup route;
3. Ollama local provider model detection and sanitized base URL;
4. LM Studio local provider model detection and visible connection-failure feedback for an invalid model.

Run only when local services are available:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
RUN_VOSTAVO_LOCAL_RUNTIME_E2E=1 NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/runtimeSetupLive.spec.ts
```

Expected: tests skip without the env flag and pass when the relevant local services are running.

- [x] **Step 4: Port the real-audio progression test as opt-in**

Create an opt-in React Playwright test, either in `frontend/tests/e2e/reviewHistoryFlow.spec.ts` or `frontend/tests/e2e/realAudioHistory.spec.ts`, guarded by:

```ts
test.skip(process.env.RUN_VOSTAVO_REAL_E2E !== "1" && process.env.RUN_STREAMLIT_REAL_E2E !== "1", "Set RUN_VOSTAVO_REAL_E2E=1 to run the real audio E2E test.");
```

Keep the same behavioral assertions as the legacy test:

- first weaker audio saves a report;
- second stronger audio saves a report;
- second score is greater than first score;
- second band is at least first band;
- Progress delta appears on the second review;
- History opens with the second attempt detail.

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
RUN_VOSTAVO_REAL_E2E=1 OPENROUTER_API_KEY="$OPENROUTER_API_KEY" NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/realAudioHistory.spec.ts
```

Expected: skips unless prerequisites are present; passes when the two real audio files and OpenRouter key are available.

- [x] **Step 5: Document the React E2E replacement lane**

Update `README.md` so the primary browser regression commands point to `frontend/tests/e2e/*.spec.ts`, and document that the old pytest/Streamlit E2E suite is retired.

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
npm test
npm run typecheck
npx playwright test -c playwright.config.ts
```

Expected: Vitest, typecheck, and frontend Playwright pass.

---

## Task 2: Extract Pure Runtime State And I18n From `app_shell`

**Files:**
- Create: `app_core/state.py`
- Create: `app_core/i18n.py`
- Modify: `app_shell/state.py`
- Modify: `app_shell/i18n.py`
- Test: `tests/test_app_shell_state.py`, `tests/test_app_shell_i18n.py`

- [x] **Step 1: Create pure state dataclasses**

Move the dataclasses, enums, constants, and `build_default_state()` from `app_shell/state.py` to `app_core/state.py`.

`app_core/state.py` must not import `streamlit`.

During the migration, keep Streamlit session helpers in `app_shell/state.py` as a wrapper around the pure `app_core.state` dataclasses. Remove that wrapper once the Streamlit package is deleted.

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_state.py -q
```

Expected: existing state tests still pass while imports migrate.

- [x] **Step 2: Create pure i18n loader**

Move `load_locale`, `t`, `flatten_keys`, and `locale_key_map` to `app_core/i18n.py`.

`app_core/i18n.py` must accept locale explicitly and must not read `st.session_state`.

During the migration, keep `app_shell/i18n.py` as a wrapper that supplies the Streamlit current locale to `app_core.i18n`. Remove that wrapper once Streamlit screens are gone.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_i18n.py -q
```

Expected: locale tests pass and the pure module can be imported without Streamlit installed.

- [x] **Step 3: Add no-Streamlit import guards**

Add a focused test that imports the new pure modules after blocking `streamlit` from import.

Suggested test target: `tests/test_app_core_imports.py`.

The test should assert:

```python
import app_core.state
import app_core.i18n
```

does not import `streamlit`.

Implementation note: this guard also covers the backend service import path
(`app_shell.services` and `app_backend.app`) so the backend no longer imports
`streamlit` through state-only types.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_imports.py -q
```

Expected: pure imports pass without touching Streamlit.

---

## Task 3: Move Pure Services Out Of `app_shell`

**Files:**
- Create: `app_core/services.py`
- Delete after migration: `app_shell/services.py`
- Modify: `app_backend/app.py`
- Create: `tests/test_app_core_services.py`
- Test: `tests/test_app_backend_api.py`

- [x] **Step 1: Split service functions by dependency**

Move backend-safe functions from `app_shell/services.py` to `app_core/services.py`. Start with functions used by `app_backend/app.py`, including:

- runtime connection builders and mutators;
- runtime connection tests;
- theme/workspace preference helpers;
- history/report serialization;
- whisper model status/download wrappers;
- assessment request helpers.

`app_core/services.py` may import `app_core.state`, `app_core.app_data`, `app_core.runtime_providers`, `app_core.runtime_resolver`, and `app_core.secret_store`, but it must not import Streamlit UI modules or `app_shell.state`.

- [x] **Step 2: Remove `app_shell.services` after compatibility migration**

During migration, this compatibility shape was acceptable:

```python
from app_core.services import *  # transitional re-export during migration
```

Final state: `app_shell.services` was removed with the rest of the Streamlit shell, and contract tests guard against product imports from `app_shell`.

- [x] **Step 3: Migrate backend imports**

Update `app_backend/app.py` to import pure service functions from `app_core.services` and pure state from `app_core.state`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_api.py tests/test_app_core_services.py -q
```

Expected: backend API and service tests pass.

- [x] **Step 4: Rename tests only after imports are pure**

The service tests now live at `tests/test_app_core_services.py`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_services.py tests/test_app_backend_api.py -q
```

Expected: tests pass from the new module names.

---

## Task 4: Refactor Launcher And Desktop Bootstrap Away From Streamlit

**Files:**
- Create: `scripts/bootstrap_backend.py`
- Modify: `scripts/run_app.py`
- Create: `app_core/bootstrap.py`
- Modify: `frontend/src-tauri/src/main.rs`
- Test: `tests/test_run_app.py`

- [x] **Step 1: Add a backend-bootstrap script**

Create `scripts/bootstrap_backend.py` that owns the current `--desktop-bootstrap` behavior:

- parse `--app-data-dir`, `--cache-dir`, and `--log-dir`;
- call `build_desktop_bootstrap_env(...)`;
- print `KEY=value` lines for Tauri.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python scripts/bootstrap_backend.py --app-data-dir /tmp/vostavo-bootstrap-check
```

Expected: output includes `VOSTAVO_DESKTOP_API_BASE_URL=`.

- [x] **Step 2: Make `run_app.py` non-Streamlit by default**

Change `scripts/run_app.py` so default behavior starts or reports backend state, not Streamlit.

Final cleanup removed the transitional `--legacy-streamlit` flag entirely. `scripts/run_app.py` rejects it so no product path can call a Streamlit launch command.

Keep `--desktop-bootstrap` during one transition release, but implement it by delegating to the new bootstrap helper.

- [x] **Step 3: Update Tauri bootstrap call**

Change `frontend/src-tauri/src/main.rs` to call `scripts/bootstrap_backend.py` instead of `scripts/run_app.py --desktop-bootstrap`.

Run:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend/src-tauri
cargo test
```

Expected: Rust tests pass.

- [x] **Step 4: Rewrite launcher tests**

Update `tests/test_run_app.py`:

- default `run_app.py --dry-run` reports backend/bootstrap payload;
- `run_app.py --legacy-streamlit` is rejected;
- `run_app.py --desktop-bootstrap` remains backward-compatible or clearly deprecated;
- no default test expects `python -m streamlit run`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_run_app.py tests/test_app_backend_lifecycle.py -q
```

Expected: launcher and backend lifecycle tests pass.

---

## Task 5: Delete Streamlit UI Lane

**Files:**
- Modify: `requirements.txt`
- Delete: `streamlit_app.py`
- Delete: `pages/`
- Delete or shrink: `app_shell/page_helpers.py`, `app_shell/review_components.py`, `app_shell/visual_system.py`
- Delete or replace: `tests/test_app_shell_pages.py`, `tests/test_app_shell_page_helpers.py`, Streamlit E2E fixtures/tests

- [x] **Step 1: Confirm deletion gates**

Before deleting files, run:

```zsh
rg -n "import streamlit|from streamlit|streamlit_webrtc|st\\." app_backend app_core scripts frontend tests
```

Expected: no product imports or dependencies match; remaining matches are historical docs or removal-contract tests.

- [x] **Step 2: Remove packages**

Remove from `requirements.txt`:

```text
streamlit==1.55.0
streamlit-webrtc==0.47.6
```

Run:

```zsh
./scripts/setup_env.sh .venv
```

Expected: environment installs without Streamlit.

- [x] **Step 3: Delete legacy UI files**

Delete the Streamlit entrypoint, page files, and Streamlit-only helpers.

`app_shell.app_data`, runtime providers/resolver, and secret-store responsibilities moved to `app_core`.

- [x] **Step 4: Delete or replace legacy tests**

Delete the old tests only after the Task 1 React replacements pass:

- delete `tests/test_app_shell_pages.py`;
- delete `tests/test_app_shell_page_helpers.py`;
- delete Streamlit-only parts of `tests/e2e/conftest.py`;
- delete Streamlit pytest E2E files after their React replacements exist;
- migrate remaining pure state/i18n/service tests to `app_core`.

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend && npm test && npm run typecheck && npx playwright test -c playwright.config.ts
```

Expected: Python, Vitest, typecheck, and frontend Playwright all pass without Streamlit installed.

---

## Task 6: Update Product And Hosting Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md`
- Modify: `docs/IMPLEMENTATION_PLAN.md`
- Modify: `docs/PLANNING_ALIGNMENT_META_PLAN.md`
- Test: documentation search checks

- [x] **Step 1: Update launch instructions**

Document these as canonical:

- local backend: `./scripts/python.sh scripts/run_backend.py --host 127.0.0.1 --port 8800`;
- React dev frontend: `NODE_ENV=development VITE_LOCAL_API_BASE_URL=http://127.0.0.1:8800 npm run dev -- --host 127.0.0.1 --port 4173 --strictPort`;
- Tauri desktop bootstrap: `scripts/bootstrap_backend.py`;
- Streamlit: removed.

- [x] **Step 2: Update hosted/EU direction**

Clarify that hosted deployment uses React static assets plus the FastAPI backend, not Streamlit.

Do not make legal claims such as "must host in the EU" without a separate compliance decision. Phrase it as an architecture-ready path for EU regional hosting.

- [x] **Step 3: Search for stale Streamlit guidance**

Run:

```zsh
rg -n "Streamlit|streamlit|streamlit-webrtc|run_app.py|streamlit_app.py" README.md docs scripts tests frontend app_backend app_core
```

Expected: matches are historical notes, removal-contract checks, or explicit statements that Streamlit is removed; no primary run instructions point to Streamlit after removal.

---

## Final Verification Checklist

Run from repo root:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
npm test
npm run typecheck
NODE_ENV=development npx playwright test -c playwright.config.ts
```

Optional/live:

```zsh
cd /Users/bernhard/Development/assess_speaking-codex-v6/frontend
RUN_VOSTAVO_LOCAL_RUNTIME_E2E=1 NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/runtimeSetupLive.spec.ts
RUN_VOSTAVO_REAL_E2E=1 OPENROUTER_API_KEY="$OPENROUTER_API_KEY" NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/realAudioHistory.spec.ts
```

Verification status on 2026-05-19:

- Passed: `./scripts/run_tests.sh tests/test_streamlit_removal_contract.py tests/test_app_core_imports.py tests/test_app_core_services.py tests/test_app_backend_api.py tests/test_run_app.py -q`
- Passed: `npm --prefix frontend test`
- Passed: `npm --prefix frontend run typecheck`
- Passed: `cargo test` from `frontend/src-tauri`
- Passed Chromium install and launch check: `npm --prefix frontend exec playwright install chromium`, then `chromium.launch({ headless: true })`
- Blocked in this Codex Desktop session: `playwright test -c playwright.config.ts` was rejected before process start by the local approval policy, so the full frontend Playwright gate still needs a permitted local run.

Completion criteria:

- `rg "import streamlit|from streamlit|streamlit_webrtc|st\\."` has no matches outside archived notes or removed legacy code.
- `requirements.txt` no longer installs Streamlit.
- React/Vite and Tauri are the only product UI lanes.
- Backend imports no `app_shell.state` or Streamlit-bound modules.
- Valuable legacy E2E behavior is covered by frontend Playwright or Vitest.

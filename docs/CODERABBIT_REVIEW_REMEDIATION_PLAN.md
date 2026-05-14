# CodeRabbit Review Remediation Plan

Last updated: 2026-05-14

Status: planning artifact for the broad CodeRabbit review on branch `codex/recorder-ux-20260311`.

## Summary

CodeRabbit could not review the whole branch in one pass because the diff exceeded its 150-file service limit. The review was split by directory. Completed CodeRabbit scopes:

| Scope | Issues |
|---|---:|
| `frontend` | 64 |
| `app_backend` | 10 |
| `app_shell` | 16 |
| `tests` | 8 |
| `assessment_runtime` | 3 |
| `scripts` | 2 |
| `pages` | 0 |
| `benchmarking` | 0 |

Known remaining CodeRabbit review coverage:

- `docs`
- `locales`
- `.taskmaster`
- `samples`
- `prompts`
- `branding`
- `corpora`
- `.github`
- root files such as `.env.example`, `.gitignore`, `README.md`, `requirements.txt`, `assess_speaking.py`, `streamlit_app.py`, and `assess_core`

This plan turns the reviewed findings into implementation slices. Each slice is file-based and touches no more than four files, matching the repo instructions. CodeRabbit severity is used as an input, not as an automatic truth: every finding must be verified against current code before editing.

## Working Rules

- Do not implement a CodeRabbit suggestion blindly. Verify the finding against the current code and tests first.
- Keep every implementation slice to four files or fewer.
- Prefer focused fixes over broad refactors.
- Add or update tests in the same slice when behavior changes.
- Preserve existing uncommitted user work. Do not revert unrelated changes.
- For frontend screen edits, discuss the approach with PAL first.
- Use localization keys for user-facing strings.
- Keep Maestro semantic labels intact when touching UI controls.

## Verification Baseline

Use these commands as the baseline after each relevant group:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest -q
cd frontend && npm test
cd frontend && npm run typecheck
cd frontend && npm run build
```

When Rust/Tauri files change, also run:

```zsh
cd frontend/src-tauri && cargo test
```

When browser behavior changes, run the relevant Playwright suite or a manual smoke in the in-app browser.

## Phase 0: Finish CodeRabbit Coverage

Goal: complete review coverage before declaring the CodeRabbit audit closed.

### Task 0.1: Remaining Non-Code Review Slices

Files reviewed only:

- `docs/**`
- `locales/**`
- `.taskmaster/**`
- `README.md`

Run after rate-limit cooldown:

```zsh
coderabbit review --agent --base main --dir docs -c AGENTS.md
coderabbit review --agent --base main --dir locales -c AGENTS.md
coderabbit review --agent --base main --dir .taskmaster -c AGENTS.md
coderabbit review --agent --base main --dir . -c AGENTS.md
```

If the root `.` review exceeds the file limit, split it into root-only path groups by temporarily using `--dir` where possible, or run the remaining small directories one by one.

Acceptance:

- Record issue counts for every remaining scope.
- Add any new security or correctness issues to the relevant phase below.

### Task 0.2: Remaining Asset And Corpus Review Slices

Files reviewed only:

- `samples/**`
- `prompts/**`
- `branding/**`
- `corpora/**`
- `.github/**`

Run:

```zsh
coderabbit review --agent --base main --dir samples -c AGENTS.md
coderabbit review --agent --base main --dir prompts -c AGENTS.md
coderabbit review --agent --base main --dir branding -c AGENTS.md
coderabbit review --agent --base main --dir corpora -c AGENTS.md
coderabbit review --agent --base main --dir .github -c AGENTS.md
```

Acceptance:

- Capture new findings.
- Do not change binary assets unless a finding is verified and scoped.

## Phase 1: Security And Secret Safety

Goal: remove the highest-risk data exposure and injection issues first.

### Task 1.1: Harden Tauri Script Escaping

Files:

- `frontend/src-tauri/src/main.rs`
- Optional test file in the same Rust module, if local pattern supports inline tests

Findings covered:

- `escape_js` does not escape backticks, double quotes, Unicode line/paragraph separators, or `</script>`.

Implementation:

- Escape backslash first.
- Escape single quotes, double quotes, backticks, newlines, carriage returns, U+2028, and U+2029.
- Neutralize `</script>` or the broader `</` sequence.
- Add unit coverage if the Tauri crate has a test pattern available.

Validation:

```zsh
cd frontend/src-tauri && cargo test
```

### Task 1.2: Redact Secrets From Job Metadata

Files:

- `app_backend/jobs.py`
- `tests/test_app_backend_jobs.py`
- `tests/test_app_backend_api.py`

Findings covered:

- `request.model_dump()` can persist `llm_api_key` or similar sensitive fields into job JSON.

Implementation:

- Create a sanitized request dump before writing job metadata.
- Exclude known secret fields such as `llm_api_key`.
- Use a helper if multiple write paths exist.
- Add a regression test that submits a request with a fake key and asserts it is absent from the stored job file.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_jobs.py tests/test_app_backend_api.py -q
```

### Task 1.3: Restrict Support Bundle Optional Binaries

Files:

- `app_backend/support_bundle.py`
- `tests/test_app_backend_support_bundle.py`
- `app_backend/app.py`
- `tests/test_app_backend_api.py`

Findings covered:

- Optional support-bundle trees may include arbitrary binary files.
- Bundle id / optional tree hardening needs to remain intact.

Implementation:

- Keep existing text/json sanitization.
- Add a small allowlist for safe binary extensions that are explicitly user-selected, such as `.wav`, `.mp3`, `.m4a`, `.png`, `.jpg`, `.jpeg`.
- Skip other binary extensions and record the skip in manifest/redaction stats.
- Preserve symlink-skipping behavior.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_support_bundle.py tests/test_app_backend_api.py -q
```

## Phase 2: Backend Process And Filesystem Robustness

Goal: prevent avoidable backend crashes, races, and incorrect filesystem accounting.

### Task 2.1: Fix Local Port Allocation Race

Files:

- `app_backend/config.py`
- `tests/test_app_backend_config.py`

Findings covered:

- `allocate_local_port()` binds and closes the socket, creating a TOCTOU race.

Implementation options:

- Preferred: return or reserve a socket through the startup path if that matches existing server ownership.
- Conservative: add bind retry logic where the backend server actually binds, and stop treating a closed ephemeral socket as a reservation.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_config.py -q
```

### Task 2.2: Harden Backend Lifecycle Startup

Files:

- `app_backend/lifecycle.py`
- `tests/test_app_backend_lifecycle.py`

Findings covered:

- `subprocess.Popen` errors can crash or hide poor diagnostics.
- Backend subprocess may use the wrong Python executable.
- Timeout cleanup and PID termination need clearer behavior.
- Unix-only termination path may need a platform guard.

Implementation:

- Resolve `.venv/bin/python` under `PROJECT_ROOT`, falling back to `sys.executable`.
- Catch `FileNotFoundError`, `PermissionError`, and `OSError` around `Popen`.
- Preserve clear diagnostics for startup failure.
- Guard PID termination and document platform behavior.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_lifecycle.py -q
```

### Task 2.3: Correct Maintenance Cleanup Accounting

Files:

- `app_backend/maintenance.py`
- `tests/test_app_backend_config.py`

Findings covered:

- `freed_bytes` can raise on `stat()` or overreport when deletion fails.

Implementation:

- Account bytes only for files that are successfully deleted, or wrap stat and deletion consistently.
- Preserve dry-run behavior separately from real cleanup.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_config.py -q
```

### Task 2.4: Harden History Detail Loading

Files:

- `app_backend/app.py`
- `tests/test_app_backend_api.py`

Findings covered:

- `_find_history_payload()` can raise on file disappearance, permissions, encoding errors, or CSV parse errors.

Implementation:

- Wrap CSV open/read in `try/except`.
- Catch `OSError`, `UnicodeDecodeError`, and `csv.Error`.
- Return `None` so the endpoint returns the existing controlled 404-style error instead of 500.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_backend_api.py -q
```

## Phase 3: App Shell Backend Client Safety

Goal: make app-shell HTTP path construction and download handling safe.

### Task 3.1: URL Encode Backend Client Path Segments

Files:

- `app_shell/backend_client.py`
- `tests/test_app_shell_backend_client.py`

Findings covered:

- `assessment_id`, `session_id`, and `bundle_id` are interpolated into URL paths without percent encoding.

Implementation:

- Use `urllib.parse.quote(value, safe="")` for path segments.
- Cover `get_assessment_status`, `cancel_assessment`, `load_history_detail`, and `download_support_bundle`.
- Add tests with IDs containing `/`, spaces, and `?`.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_backend_client.py -q
```

### Task 3.2: Robust Support Bundle Download Filename Parsing

Files:

- `app_shell/backend_client.py`
- `tests/test_app_shell_backend_client.py`

Findings covered:

- `content-disposition` parsing is fragile and can mishandle quoted or encoded filenames.

Implementation:

- Use a standard parser from the standard library if sufficient.
- Support `filename=` and `filename*=`.
- Fall back to `{bundle_id}.zip`.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_backend_client.py -q
```

### Task 3.3: Validate Support Bundle Creation Result

Files:

- `app_shell/services.py`
- `tests/test_app_shell_services.py`

Findings covered:

- Code assumes `create_support_bundle_archive()` always returns an object with `bundle_id`.

Implementation:

- Check that the creation result is not `None`.
- Check `bundle_id` exists and is non-empty before download.
- Return or raise a clear user-facing error following existing service patterns.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_services.py -q
```

## Phase 4: Runtime And Data Root Correctness

Goal: keep local runtime settings predictable across environments.

### Task 4.1: Restore Provider API Key Fallback Visibility

Files:

- `app_shell/runtime_resolver.py`
- `tests/test_runtime_resolver.py`

Findings covered:

- Missing secret refs can silently resolve to an empty API key.

Implementation:

- Try saved secret first.
- Fall back to provider-specific environment variables where current behavior expects it.
- Log or surface missing-secret state without breaking optional local providers.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_runtime_resolver.py -q
```

### Task 4.2: Validate Whisper Cache Root

Files:

- `app_shell/bootstrap.py`
- `tests/test_app_shell_app_data.py`

Findings covered:

- `whisper_cache_dir` lacks the same checkout validation as app-data and cache dirs.

Implementation:

- Apply the existing repository-check rule to `whisper_cache_dir`.
- Require explicit override or reject accidental project-checkout writes.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_app_data.py -q
```

### Task 4.3: Make Bootstrap Typing Python-Compatible

Files:

- `app_shell/bootstrap.py`
- Tests only if existing coverage needs update

Findings covered:

- PEP 695 generic syntax requires Python 3.12+.

Implementation:

- Replace bracket generic syntax with `TypeVar`.
- Keep runtime behavior unchanged.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_app_data.py tests/test_run_app.py -q
```

## Phase 5: Frontend API And State Correctness

Goal: keep API clients, app state, and frontend contracts from hiding runtime failures.

### Task 5.1: Fix Abort Timeout Cleanup

Files:

- `frontend/src/lib/api/client.ts`
- `frontend/src/lib/api/client.test.ts`

Findings covered:

- `mergeSignals()` can leak a timeout if an external signal is already aborted.

Implementation:

- Register the internal abort cleanup before checking the external signal.
- Add a unit test with an already-aborted signal.

Validation:

```zsh
cd frontend && npm test -- src/lib/api/client.test.ts
cd frontend && npm run typecheck
```

### Task 5.2: Tighten Frontend History Types

Files:

- `frontend/src/lib/api/types.ts`
- `frontend/src/routes/HistoryRoute.tsx`
- `frontend/src/routes/tests/HistoryRoute.test.tsx`

Findings covered:

- `HistoryRow` has broad `unknown` fields for values that now have backend-normalized shapes.

Implementation:

- Narrow numeric fields to `number | null` where backend behavior supports it.
- Narrow booleans to `boolean | null` if open/pending states exist.
- Keep `band` compatible with the backend value actually emitted.
- Update normalizers and tests accordingly.

Validation:

```zsh
cd frontend && npm test -- src/routes/tests/HistoryRoute.test.tsx
cd frontend && npm run typecheck
```

### Task 5.3: Preserve Error State For Failed/Cancelled Jobs

Files:

- `frontend/src/lib/state/appStore.ts`
- `frontend/src/lib/state/appStore.test.ts`

Findings covered:

- Failed/cancelled job updates can overwrite error state with `undefined`.

Implementation:

- Use `nextJob.error ?? state.recording.error ?? ""`.
- Add focused tests for partial failed and cancelled updates.

Validation:

```zsh
cd frontend && npm test -- src/lib/state/appStore.test.ts
```

### Task 5.4: Surface Setup And Settings API Failures

Files:

- `frontend/src/routes/SetupRoute.tsx`
- `frontend/src/routes/SettingsRoute.tsx`
- `frontend/src/lib/i18n.ts`
- `frontend/src/routes/tests/SettingsRoute.test.tsx`

Findings covered:

- Save failures can be labeled as test failures.
- Delete/default failures can be swallowed or not surfaced.

Implementation:

- Add distinct statuses/messages for save, default, and delete failure paths.
- Keep successful state updates inside `try`.
- Put cleanup in `finally`.
- Add localized strings.

Validation:

```zsh
cd frontend && npm test -- src/routes/tests/SettingsRoute.test.tsx
cd frontend && npm run typecheck
```

## Phase 6: Frontend Accessibility And Responsive Layout

Goal: make the new React surfaces keyboard- and screen-reader-friendly.

### Task 6.1: App Shell Focus And Mobile Layout

Files:

- `frontend/src/components/shell/AppShell.tsx`
- `frontend/src/components/shell/AppShell.module.css`
- `frontend/src/App.tsx`

Findings covered:

- Locale buttons and nav links lack visible focus states.
- Shell grid is not responsive on narrow screens.
- Navigation landmark label is not descriptive enough.
- `localizedRoutes` dependency should include stable translation dependency.

Implementation:

- Move shell interaction styles into CSS module.
- Add `:hover`, `:focus-visible`, active-state classes, and mobile breakpoint.
- Change nav label to "Main navigation" or localized equivalent.
- Fix `useMemo` dependencies.

Validation:

```zsh
cd frontend && npm run typecheck
cd frontend && npm test
```

### Task 6.2: Home Focus States

Files:

- `frontend/src/routes/HomeRoute.tsx`
- Optional route test if behavior changes

Findings covered:

- Home primary/secondary buttons lack keyboard focus indicators.

Implementation:

- Add visible focus handling with CSS classes or component-local focus state.
- Preserve semantic IDs.

Validation:

```zsh
cd frontend && npm run typecheck
```

### Task 6.3: History List Accessibility

Files:

- `frontend/src/components/history/HistoryList.tsx`
- `frontend/src/components/history/HistoryDetailPanel.tsx`
- `frontend/src/routes/tests/HistoryRoute.test.tsx`

Findings covered:

- Table headers need `scope="col"`.
- Long jump-button labels can overflow on small screens.
- `HistoryDetailPanel` has redundant fallbacks and error truthiness ambiguity.

Implementation:

- Add column scopes.
- Split recent-attempt button text into structured spans with responsive wrapping/truncation.
- Make `error` prop `string | null` or explicitly check `error.length`.

Validation:

```zsh
cd frontend && npm test -- src/routes/tests/HistoryRoute.test.tsx
cd frontend && npm run typecheck
```

### Task 6.4: Theme Form Validation Accessibility

Files:

- `frontend/src/components/setup/ThemeForm.tsx`
- `frontend/src/routes/tests/SessionSetupRoute.test.tsx`

Findings covered:

- Error list is not linked to fields.
- Error item keys can collide.
- CEFR and duration select casts lack runtime validation.

Implementation:

- Add `role="alert"` or `aria-live`.
- Give field-specific errors stable IDs.
- Set `aria-invalid` and `aria-describedby` on affected controls.
- Add guards for CEFR and duration values.

Validation:

```zsh
cd frontend && npm test -- src/routes/tests/SessionSetupRoute.test.tsx
cd frontend && npm run typecheck
```

## Phase 7: Frontend Data And Content Guards

Goal: stop invalid JSON/content assumptions from leaking into UI state.

### Task 7.1: Validate Session Setup Content In TypeScript

Files:

- `frontend/src/lib/setup/sessionSetupContent.ts`
- `frontend/src/lib/setup/sessionSetupContent.test.ts`

Findings covered:

- Arbitrary `task_family` casts.
- Missing English practice-brief templates can produce undefined access.

Implementation:

- Add type guards for task-family values.
- Add required English template validation.
- Default invalid task families to `free_monologue` or skip invalid entries based on current UX expectations.

Validation:

```zsh
cd frontend && npm test -- src/lib/setup/sessionSetupContent.test.ts
cd frontend && npm run typecheck
```

### Task 7.2: Add German Theme Library Parity

Files:

- `assessment_runtime/data/session_setup_content.json`
- `assessment_runtime/theme_library.py`
- `tests/test_theme_library.py`

Findings covered:

- German practice-brief templates exist without matching `default_theme_library.de`.
- Duplicate `save_workspace_prefs` implementation in `theme_library.py`.

Implementation:

- Add `de` theme library with same shape as `it` and `en`.
- Remove duplicate `save_workspace_prefs`, preserving the correct implementation.
- Add tests for language key parity and one authoritative save function behavior.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_theme_library.py -q
```

### Task 7.3: Align ASR Provider Constants

Files:

- `assessment_runtime/asr.py`
- `tests/test_asr.py`

Findings covered:

- `KNOWN_ASR_PROVIDERS` does not mirror the internal registry.

Implementation:

- Prefer `available_asr_providers()` as source of truth.
- Either remove `KNOWN_ASR_PROVIDERS` if unused, or compute it from the registry.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_asr.py -q
```

## Phase 8: Tauri And Cross-Platform Tooling

Goal: avoid desktop build surprises across platforms.

### Task 8.1: Respect Existing Tauri Icons

Files:

- `frontend/src-tauri/build.rs`
- Optional icon asset files if verified necessary

Findings covered:

- Build script always writes the default icon and can overwrite user icons.
- Default icon is a placeholder.

Implementation:

- Ensure the icon directory exists.
- Only write a generated/default icon if no icon exists.
- Replace placeholder icon with a real project icon only if that is in scope and available.

Validation:

```zsh
cd frontend/src-tauri && cargo test
```

### Task 8.2: Cross-Platform Playwright Backend Launch

Files:

- `frontend/playwright.config.ts`

Findings covered:

- Python detection misses Windows venvs.
- Hardcoded `/tmp` paths break on Windows.

Implementation:

- Check both `.venv/bin/python` and `.venv/Scripts/python.exe`.
- Use `os.tmpdir()` and `path.join()` for app-data/cache temp dirs.

Validation:

```zsh
cd frontend && npm run typecheck
```

### Task 8.3: Tauri Backend Bootstrap Failure Clarity

Files:

- `frontend/src-tauri/src/main.rs`

Findings covered:

- `bootstrap_from_repo_launcher().ok()` can hide desktop-bridge bootstrap errors.
- Hardcoded `python3` can fail on systems where the executable is `python`.

Implementation:

- Try environment bridge first, then launcher bootstrap.
- If both fail, surface a clear diagnostic.
- Resolve Python from configured env or candidate names.

Validation:

```zsh
cd frontend/src-tauri && cargo test
```

## Phase 9: Tests And Test Infrastructure

Goal: make tests deterministic and honest.

### Task 9.1: Complete LocalStorage Test Polyfill

Files:

- `frontend/src/test/renderWithProviders.tsx`
- Frontend test file if adding coverage

Findings covered:

- Existing `localStorage` may retain state across tests.
- Polyfill lacks `length` and `key(index)`.

Implementation:

- If real storage exists, call `clear()` before each render helper use.
- If installing a shim, implement `length` and `key(index)`.
- Document that `appState` is ignored when a custom store is passed.

Validation:

```zsh
cd frontend && npm test
```

### Task 9.2: Clean Python Test Helpers

Files:

- `tests/test_app_shell_backend_client.py`
- `tests/test_run_app.py`
- `tests/test_run_backend.py`
- `tests/test_app_backend_lifecycle.py`

Findings covered:

- Fake response corrupts falsy payloads.
- `run_app` test mutates env without restoring.
- `run_backend` has unused imports/locals and may read logs before flush.
- Lifecycle test uses `os.sys.executable`.

Implementation:

- Default fake payload only when `payload is None`.
- Use `addCleanup` or context helpers to restore env.
- Flush handlers before reading log files.
- Import and use `sys.executable`.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_shell_backend_client.py tests/test_run_app.py tests/test_run_backend.py tests/test_app_backend_lifecycle.py -q
```

### Task 9.3: Localize E2E Selectors

Files:

- `tests/e2e/test_app_shell_e2e.py`

Findings covered:

- Hardcoded "Browse files" selector breaks for non-English E2E locales.

Implementation:

- Replace hardcoded selector with `app_shell_text(ui_locale, "speak.browse_files")` or the correct key.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/e2e/test_app_shell_e2e.py -q
```

## Phase 10: UI Cleanup And Lower-Risk Polish

Goal: address lower-risk issues after correctness and safety are handled.

### Task 10.1: Review Warning Labels And Duplicate Text

Files:

- `frontend/src/components/review/WarningsPanel.tsx`
- `frontend/src/components/review/ReviewSummary.tsx`
- `frontend/src/lib/i18n.ts`
- `frontend/src/routes/tests/ReviewRoute.test.tsx`

Findings covered:

- Regex-stripping translated labels is fragile.
- Review summary repeats the same translation key for label and heading.

Implementation:

- Add dedicated label/eyebrow keys.
- Remove `.replace(/: $/, "")` patterns.
- Update tests and localization entries.

Validation:

```zsh
cd frontend && npm test -- src/routes/tests/ReviewRoute.test.tsx
cd frontend && npm run typecheck
```

### Task 10.2: Practice Brief Card Polish

Files:

- `frontend/src/components/setup/PracticeBriefCard.tsx`
- `frontend/src/routes/tests/SessionSetupRoute.test.tsx`

Findings covered:

- Duplicate React keys for repeated success-focus strings.
- Nested card styling.
- Theme fallback repeats preview title.

Implementation:

- Use stable keys with index fallback.
- Replace nested `cardStyle` with lightweight detail item style.
- Add a distinct no-theme placeholder or render nothing when empty.

Validation:

```zsh
cd frontend && npm test -- src/routes/tests/SessionSetupRoute.test.tsx
cd frontend && npm run typecheck
```

### Task 10.3: Library Route Resilience

Files:

- `frontend/src/routes/LibraryRoute.tsx`
- `frontend/src/components/library/SampleTrialGrid.tsx`
- `frontend/src/routes/tests/LibraryGuideRoutes.test.tsx`

Findings covered:

- Missing loading/error UI for samples query.
- Selected theme row keys can collide.
- Sample title casing should normalize uppercase inputs.

Implementation:

- Add loading and error states with localized strings.
- Use stable IDs or include index as fallback in keys.
- Lowercase before title-casing sample labels.

Validation:

```zsh
cd frontend && npm test -- src/routes/tests/LibraryGuideRoutes.test.tsx
cd frontend && npm run typecheck
```

### Task 10.4: Shell Runtime Status Translation Fallbacks

Files:

- `app_shell/runtime_status.py`
- `tests/test_runtime_status.py`
- `frontend/src/routes/HistoryRoute.tsx`
- `frontend/src/routes/tests/HistoryRoute.test.tsx`

Findings covered:

- Placeholder-detection via `startswith("[")` is fragile in Python shell status helpers and React history labels.

Implementation:

- Compare against exact placeholder patterns where current translator behavior requires it.
- Prefer explicit translation-existence helper if available.
- Add fallback strings that never expose placeholder text to users.

Validation:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_runtime_status.py -q
cd frontend && npm test -- src/routes/tests/HistoryRoute.test.tsx
```

## Final Regression Pass

After all accepted findings are implemented:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest -q
cd frontend && npm test
cd frontend && npm run typecheck
cd frontend && npm run build
cd frontend/src-tauri && cargo test
```

Then run targeted manual smoke for:

- Home to Settings to Runtime Setup return flow
- Library sample to Session Setup to Speak upload to Review to History
- Support bundle creation with runtime health enabled

## Commit Strategy

Use small commits by phase or by task group:

1. Security and secret safety
2. Backend lifecycle and filesystem robustness
3. App shell backend-client hardening
4. Runtime/settings frontend error handling
5. Accessibility and responsive UI
6. Content validation and localization parity
7. Test infrastructure cleanup

Do not mix unrelated phases in one commit unless the change is mechanically required by shared tests.

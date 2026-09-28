# Playwright Flow Expansion Plan

Last updated: 2026-07-22
Status: Focused React/Vite flows implemented; full Chromium runner recheck is
blocked by the current macOS sandbox before app assertions

> Status note, 2026-05-20: partially implemented. The review/history,
> live-runtime-setup, and real-audio replacement specs now exist under
> `frontend/tests/e2e`; the full Playwright runner gate still needs a deliberate
> full local run.

> Status note, 2026-06-03: `frontend/tests/e2e/reviewHistoryFlow.spec.ts`
> now asserts actionable empty Review/History states, the Home → Session Setup
> → Speak path, the compact Speak session summary, and the Review next-step
> card. Local execution was attempted with
> `npx playwright test -c playwright.config.ts tests/e2e/reviewHistoryFlow.spec.ts`
> but Chromium failed before app assertions with macOS Mach bootstrap permission
> errors in the sandbox. Direct probes also showed system Chrome and Edge abort
> under Playwright before page load, while Firefox and WebKit are not installed
> in the local Playwright cache.

> Status note, 2026-06-04: fresh verification supersedes the 2026-06-03 local
> blocker for the focused review/history lane. Backend and frontend localhost
> smokes passed, the in-app Browser loaded and captured the frontend, and
> `NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/reviewHistoryFlow.spec.ts`
> passed after one transient first-run click actionability timeout.

> Status note, 2026-06-06: `frontend/tests/e2e/visualRefreshSmoke.spec.ts`
> now covers the visual-refresh browser lane. From `frontend/`,
> `env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-05 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts`
> passed locally, generated Home/Speak/Review/History desktop and 390px mobile
> screenshots, asserted no visible serif typography, verified the primary
> visual artifacts, and caught then fixed the History mobile page-overflow
> regression.

> Status note, 2026-06-09: `frontend/tests/e2e/visualRefreshSmoke.spec.ts`
> is now the first-class focused visual-smoke gate. From `frontend/`,
> `env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-05 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts`
> passed locally. The gate now captures and checks 390px mobile screenshots for
> Home, Session Setup, Speak, clean Review, failed-gate Review, History, and
> expanded History, and it runs no-serif plus no-horizontal-overflow assertions
> for each mobile route state.

> Status note, 2026-06-09 later: the focused visual-smoke spec was updated for
> the Session Setup newbie wizard (`setup.wizard`, `setup.recommended_start`,
> and `setup.runtime_callout`). The exact command attempted from `frontend/` was
> `env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-09 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts`.
> It was blocked before app assertions because Chromium/Chrome could not launch
> in the current sandbox. Bundled Chromium failed with
> `bootstrap_check_in org.chromium.Chromium.MachPortRendezvousServer... Permission denied (1100)`.
> System Chrome also aborted before page load, and direct Chrome produced no DOM
> output. Treat 2026-06-09 screenshot refresh as open until browser launch works
> again in this environment.

> Status note, 2026-06-09 final: Chrome/Playwright is working again for the
> focused visual-smoke gate. From `frontend/`,
> `env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-09 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts`
> passed locally. The gate now also asserts the learner-confidence
> simplification: History no longer renders duplicate `history-priority-*`
> cards, and a 360px History overflow check captures
> `visual-refresh-smoke-history-narrow-mobile.png`. It also captures
> `visual-refresh-smoke-speak-ready-mobile.png` after audio is attached, proving
> the optional-context handoff and hidden native file input do not create mobile
> overflow.

> Status note, 2026-07-22: `libraryGuideFlow.spec.ts` and
> `reviewChangeTaskFlow.spec.ts` now cover the Library/Guide practice-support
> journey and changing tasks from Review. The focused Review change-task flow
> passed in real Chromium on 2026-07-21 after installing the pinned Playwright
> browser under Node 22. The deterministic history specs now open
> `speak.optional_context` before filling label/notes, matching the native
> disclosure behavior while preserving metadata assertions. The Node 22
> command
> `/Users/bernhard/.nvm/versions/node/v22.19.0/bin/node node_modules/playwright/cli.js test -c playwright.config.ts --list`
> collects 11 tests in 9 files. A full run with the same command minus `--list`
> reached the runner but all eight browser-enabled tests failed at Chromium
> launch with
> `bootstrap_check_in org.chromium.Chromium.MachPortRendezvousServer... Permission denied (1100)`;
> three opt-in live/runtime tests were skipped. This is an execution-environment
> blocker, not an app assertion failure. The required smallest rerun on
> 2026-07-22 used
> `/Users/bernhard/.nvm/versions/node/v22.19.0/bin/node node_modules/playwright/cli.js test -c playwright.config.ts tests/e2e/reviewHistoryFlow.spec.ts`;
> both collected tests failed in 1 ms at the same Chromium launch check, before
> page creation or app assertions. An independently managed browser loaded the
> correctly wired localhost Home route and navigated to Runtime Setup.

## Summary

This plan now defines the next Playwright browser flows for the shared
React/Vite frontend.

The old pytest-playwright Streamlit suite remains useful only as legacy
evidence while `docs/superpowers/plans/2026-05-16-streamlit-retirement.md`
replaces valuable behavior and then deletes Streamlit tests. New browser work
should not target `streamlit_app.py` or `pages/` unless it is explicitly a
temporary retirement gate.

This plan is subordinate to:
1. `docs/PLANNING_ALIGNMENT_META_PLAN.md`
2. `docs/IMPLEMENTATION_PLAN.md`
3. `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md`
4. `docs/superpowers/plans/2026-05-16-streamlit-retirement.md`

## Goal

Expand browser-level regression coverage for the product UI that will ship:
React/Vite locally, Tauri for desktop packaging, and the same frontend shape
for future hosted mode.

The goal is:
1. replace valuable legacy Streamlit browser behavior before deleting it
2. catch browser-only regressions in React navigation, layout, upload,
   recording, polling, Settings, and History behavior
3. keep browser automation aligned with the local FastAPI backend contract
4. avoid new Streamlit-only browser tests

## Non-Goals

This plan does not:
1. recreate Streamlit widget behavior one-for-one
2. replace Playwright with Maestro for the current web shell
3. create a second product plan
4. define mobile or companion-app automation
5. open hosted auth or hosted persistence work

## Current Baseline

The repo already has a shared-frontend browser foundation:
1. `frontend/playwright.config.ts` starts Vite on `127.0.0.1:4173`
2. the same config starts the local FastAPI backend on `127.0.0.1:8800`
3. frontend smoke and local-guest regression flows cover Home, Runtime Setup,
   Session Setup, Speak, Review, History, Library, Guide, Settings, support
   bundles, change-task navigation, and Settings return/setup navigation
4. frontend Vitest route tests cover localized screen states and component
   behavior

The legacy Streamlit pytest-playwright suite still protects some behavior, but
it is not the primary browser lane anymore. Treat it as a checklist of behavior
to replace under the Streamlit retirement plan.

## Replacement Priorities

### Subtask 1: Review and History progression

Files:
1. `frontend/tests/e2e/reviewHistoryFlow.spec.ts` `[CREATE]`
2. `frontend/tests/e2e/reviewChangeTaskFlow.spec.ts` `[CREATE]`
3. `frontend/tests/e2e/libraryGuideFlow.spec.ts` `[CREATE]`
4. `frontend/playwright.config.ts`
5. `frontend/src/routes/tests/ReviewRoute.test.tsx`

Deliverables:
1. deterministic two-attempt upload or seeded-audio flow
2. Review summary assertions for score, transcript, warnings, and next focus
3. Try-again behavior that returns to Speak without losing setup state
4. History detail selection for the latest attempt
5. progress-delta assertion on the second attempt when fixture data supports it
6. Review change-task navigation preserves the intended setup handoff
7. Library and Guide frame content as practice support

Timing:
1. implement before deleting `tests/e2e/test_app_shell_e2e.py`

### Subtask 2: Runtime setup live checks

Files:
1. `frontend/tests/e2e/runtimeSetupLive.spec.ts` `[CREATE]`
2. `frontend/src/routes/SetupRoute.tsx`
3. `frontend/src/components/setup/RuntimeConnectionForm.tsx`
4. `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`

Deliverables:
1. opt-in small-viewport checks for live Ollama and LM Studio detection
2. visible sanitized base URL and failure feedback
3. skip behavior unless `RUN_VOSTAVO_LOCAL_RUNTIME_E2E=1`
4. no dependence on Streamlit runtime setup widgets

Timing:
1. implement before deleting `tests/e2e/test_app_shell_runtime_setup_e2e.py`

### Subtask 3: Real-audio progression

Files:
1. `frontend/tests/e2e/realAudioHistory.spec.ts` `[CREATE]`
2. `frontend/playwright.config.ts`
3. `frontend/src/routes/SpeakRoute.tsx`
4. `frontend/src/routes/ReviewRoute.tsx`
5. `frontend/src/routes/HistoryRoute.tsx`

Deliverables:
1. opt-in real-audio flow guarded by `RUN_VOSTAVO_REAL_E2E=1`
2. first weaker audio and second stronger audio produce persisted reports
3. second score is greater than or equal to the first where fixtures guarantee it
4. History opens with the second attempt selected

Timing:
1. implement before deleting `tests/e2e/test_app_shell_real_history_e2e.py`

### Subtask 4: Recorder and upload confidence

Files:
1. `frontend/tests/e2e/reviewHistoryFlow.spec.ts`
2. `frontend/src/components/speak/RecorderPanel.tsx`
3. `frontend/src/routes/SpeakRoute.tsx`
4. `frontend/src/routes/tests/SpeakRoute.test.tsx`

Deliverables:
1. confirm Vitest coverage for record/upload switching and remove-recording
   behavior
2. add a short Playwright assertion only if route-level behavior is not already
   protected
3. treat browser `MediaRecorder` behavior as the product path; do not recreate
   `st.audio_input` behavior

Timing:
1. pair with Subtask 1 if the removal behavior is user-visible in the route

### Subtask 5: Settings support and maintenance

Files:
1. `frontend/tests/e2e/settingsSupportFlow.spec.ts`
2. `frontend/src/routes/SettingsRoute.tsx`
3. `frontend/src/components/setup/RuntimeConnectionForm.tsx`
4. `frontend/src/routes/tests/SettingsRoute.test.tsx`

Deliverables:
1. saved-connection state that the learner sees
2. storage summary visibility
3. cleanup preview/run affordances
4. support bundle creation with safe default exclusions
5. origin-preserving return navigation

Timing:
1. keep green while Streamlit Settings tests are removed

## Legacy Streamlit Coverage Rule

The following Streamlit browser tests should not grow. Replace or delete them
through the retirement plan:
1. `tests/e2e/test_app_shell_e2e.py`
2. `tests/e2e/test_app_shell_real_history_e2e.py`
3. `tests/e2e/test_app_shell_runtime_setup_e2e.py`
4. Streamlit-specific fixtures in `tests/e2e/conftest.py`

Deletion gate:
1. frontend Playwright replacement passes
2. frontend Vitest route coverage passes
3. backend API and service tests pass
4. no remaining product path depends on Streamlit widgets

## Verification Commands

Run from repo root:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest
npm --prefix frontend test
npm --prefix frontend run typecheck
cd frontend
/Users/bernhard/.nvm/versions/node/v22.19.0/bin/node node_modules/playwright/cli.js test -c playwright.config.ts
```

Optional/live:

```zsh
cd frontend
RUN_VOSTAVO_LOCAL_RUNTIME_E2E=1 NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/runtimeSetupLive.spec.ts
RUN_VOSTAVO_REAL_E2E=1 OPENROUTER_API_KEY="$OPENROUTER_API_KEY" NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/realAudioHistory.spec.ts
```

## Future Re-Evaluation

If the repo later reaches a real companion-app stage:
1. revisit Maestro under `docs/MOBILE_COMPANION_STRATEGY.md`
2. keep that decision separate from the current frontend Playwright plan

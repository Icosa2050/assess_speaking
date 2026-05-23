# Playwright Flow Expansion Plan

Last updated: 2026-05-17
Status: Accepted supporting browser automation plan, rebased for React/Vite as
the primary browser lane and Streamlit as legacy replacement coverage

> Status note, 2026-05-20: partially implemented. The review/history,
> live-runtime-setup, and real-audio replacement specs now exist under
> `frontend/tests/e2e`; the full Playwright runner gate still needs a permitted
> local run.

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
   History, Settings, support bundles, and Settings return/setup navigation
4. frontend Vitest route tests cover localized screen states and component
   behavior

The legacy Streamlit pytest-playwright suite still protects some behavior, but
it is not the primary browser lane anymore. Treat it as a checklist of behavior
to replace under the Streamlit retirement plan.

## Replacement Priorities

### Subtask 1: Review and History progression

Files:
1. `frontend/tests/e2e/reviewHistoryFlow.spec.ts` `[CREATE]`
2. `frontend/playwright.config.ts`
3. `frontend/src/routes/ReviewRoute.tsx`
4. `frontend/src/routes/HistoryRoute.tsx`
5. `frontend/src/routes/tests/ReviewRoute.test.tsx`

Deliverables:
1. deterministic two-attempt upload or seeded-audio flow
2. Review summary assertions for score, transcript, warnings, and next focus
3. Try-again behavior that returns to Speak without losing setup state
4. History detail selection for the latest attempt
5. progress-delta assertion on the second attempt when fixture data supports it

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
./scripts/run_tests.sh
cd frontend
npm test
npm run typecheck
NODE_ENV=development npx playwright test -c playwright.config.ts
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

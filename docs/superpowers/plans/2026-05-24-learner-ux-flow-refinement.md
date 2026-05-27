# Learner UX Flow Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Vostavo easier and more motivating for language learners by decluttering global chrome, moving UI language to Settings, clarifying setup, and turning Speak/Review/History into a guided practice loop.

**Architecture:** Keep the current React Router route structure, Zustand app store, React Query API access, localization system, and semantic IDs. Implement every visible screen change as a PAL-reviewed, file-bounded slice. Split localization JSON changes into separate five-file subtasks when a UI slice needs new strings.

**Tech Stack:** React 19, React Router 7, Zustand, React Query, Vite, Vitest, Playwright, local backend API.

---

## Baseline Evidence

Screenshots captured on 2026-05-24 from the running app:

- `docs/ux-audit-screenshots/2026-05-24/01-home-default.jpg`
- `docs/ux-audit-screenshots/2026-05-24/02-runtime-setup-default.jpg`
- `docs/ux-audit-screenshots/2026-05-24/03-session-setup-default.jpg`
- `docs/ux-audit-screenshots/2026-05-24/04-speak-ready.jpg`
- `docs/ux-audit-screenshots/2026-05-24/05-review-empty.jpg`
- `docs/ux-audit-screenshots/2026-05-24/06-history.jpg`
- `docs/ux-audit-screenshots/2026-05-24/07-library.jpg`
- `docs/ux-audit-screenshots/2026-05-24/08-guide.jpg`
- `docs/ux-audit-screenshots/2026-05-24/09-settings.jpg`
- `docs/ux-audit-screenshots/2026-05-24/10-home-configured.jpg`

PAL review:

- `anthropic/claude-opus-4.7`: agreed the app is coherent but visually flat and motivationally weak; recommended moving locale to Settings first, then tightening setup, Speak, Review, History, and Library.
- `qwen/qwen3.7-max`: agreed the locale move is technically low-risk; emphasized empty-state CTAs and localized coach-style microcopy.
- `google/gemini-3.1-pro-preview`: agreed with the locale move; added the risk that technical terms like Whisper, Ollama, model, and Base URL can intimidate non-developer learners.

Direct OpenRouter visual second pass:

- Raw model output is saved at `docs/ux-audit-screenshots/2026-05-24/openrouter-vision-second-pass.json`.
- `anthropic/claude-opus-4.7`, `google/gemini-3.1-pro-preview`, and `x-ai/grok-4.3` all inspected the full screenshot set directly through OpenRouter image input.
- All three models agreed the first-run Home screen buries the practice loop under technical setup.
- All three models agreed the global UI-language banner should move to Settings because it is a set-once preference and conflicts with the learning-language selector.
- All three models agreed Runtime Setup should not be a peer of Speak/Review in the main practice navigation. Opus recommended demotion rather than hiding: keep the route reachable through Settings, Home setup warnings, and deep links.
- Opus recommended deferring Scoring Guide changes because it is a reference page, not a primary flow blocker.

## Product Direction

The app should read as a practice companion first and a local-AI control panel second.

Primary learner loop:

1. Pick a session goal.
2. Speak.
3. Review feedback.
4. Choose the next exercise from progress/history/library.

Technical setup remains necessary, but it should use progressive disclosure and avoid competing with the daily practice path.

## Plan Amendments From Direct Visual Review

These amendments supersede the earlier ordering in this file where they conflict.

- Add Home simplification as the first product-facing implementation slice after the locale move. Home should show a primary Start practicing / Start new session CTA, with startup checks collapsed or summarized as one system-status row.
- Keep Runtime Setup as a route, but remove it from primary practice navigation once runtime is configured. It remains reachable from Settings, Home warning states, and direct links.
- Treat Scoring Guide as a lower-priority reference page. Do not include it in the first polish wave except for navigation grouping.
- Add an explicit non-goal for this wave: do not change runtime detection, Whisper caching, provider contracts, color tokens, or the overall card/button design system.
- Preserve semantic IDs and route guards in every navigation and Home change.

## Stitch Design-System Pass

The selected local design direction is `Calm Coach`, captured in the repo-level `DESIGN.md`.

Implementation constraints:

- Keep `DESIGN.md` as the design-system source of truth.
- Use Stitch outputs as references only; do not import generated React/CSS directly.
- Preserve localization keys, semantic IDs, route guards, and the existing app architecture.
- Apply visual changes only through the existing file-bounded UX tasks.

Stitch setup status from 2026-05-26:

- Project `12105866389178941937` (`Vostavo learner UX redesign`) contains the current screenshot set and uploaded `DESIGN.md`.
- Screenshot upload succeeded through `scripts/stitch_upload_screenshots.py`; the non-secret manifest is `docs/ux-audit-screenshots/2026-05-24/stitch-upload-manifest.json`.
- `upload_design_md` succeeded. Stitch design-system creation initially returned `invalid argument` with the full screen instance, then succeeded with the minimal SDK-documented `{id, sourceScreen}` shape; the design-system asset is `813fdb63a0e347419f5bdfc69e41c4db`.
- Generation attempts still hit transport disconnects. A one-screen `apply_design_system` trial created Stitch screen `e1b78a8967c248e6b15edb745bb66e1f`, but its downloaded screenshot is blank and its HTML artifact is empty, so it is not reviewable.
- The SDK fallback (`@google/stitch-sdk` 0.3.5 from a temporary `/tmp` install, 300s timeout) also failed with a remote MCP socket close after about one minute; a fresh-client poll found no new screen.
- Generation status is recorded in `docs/ux-audit-screenshots/2026-05-26/stitch-generation-manifest.json`.

## Task 1: Move UI Language Out Of Global Chrome

### Task 1A: Localize Settings Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [ ] Add or confirm Settings strings for UI language as a first-class preference.
- [ ] Keep existing locale keys intact; add new keys first and remove obsolete keys only after the code move is verified.
- [ ] Run `npm --prefix frontend test -- SettingsRoute.test.tsx`.

### Task 1B: Move Locale Control To Settings

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/shell/AppShell.tsx`
- Modify: `frontend/src/components/shell/AppShell.module.css`
- Modify: `frontend/src/routes/SettingsRoute.tsx`
- Modify: `frontend/src/routes/tests/SettingsRoute.test.tsx`

- [ ] Remove `localeLabel`, `localeOptions`, `activeLocale`, and `onLocaleChange` from `AppShell`.
- [ ] Remove the header locale button group and related CSS.
- [ ] Render the UI-language control in Settings, using `useAppStore((state) => state.setUiLocale)`.
- [ ] Preserve or deliberately relocate semantic IDs for automation.
- [ ] Run `npm --prefix frontend test -- SettingsRoute.test.tsx HomeSetupRoutes.test.tsx`.
- [ ] Capture before/after screenshots of Home and Settings.

## Task 2: Reframe Global Navigation Around Practice

### Task 2A: Make Home Launch Practice First

**Files:**
- Modify: `frontend/src/routes/HomeRoute.tsx`
- Modify: `frontend/src/routes/HomeRoute.module.css`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `locales/en.json`
- Modify: one additional locale file in a separate follow-up if the English copy lands first

- [ ] Move the primary Home focus from startup checks to Start practicing / Start new session.
- [ ] Collapse startup checks behind a single System status summary unless runtime is missing or a check fails.
- [ ] Remove duplicate Other surfaces content if the sidebar already exposes those routes.
- [ ] Preserve runtime-missing behavior: Home must still give a clear route to Runtime Setup when no runtime is configured.
- [ ] Run `npm --prefix frontend test -- HomeSetupRoutes.test.tsx`.
- [ ] Capture Home default and Home configured screenshots after the change.

### Task 2B: Localize Navigation Grouping Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [ ] Add copy for learner-facing navigation groups such as Practice, Progress, Resources, and Settings.
- [ ] Keep technical setup labels available for CTAs and Settings links.

### Task 2C: Reduce Navigation Overload

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/shell/AppShell.tsx`
- Modify: `frontend/src/components/shell/AppShell.module.css`
- Modify: `frontend/src/routes/HomeRoute.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`

- [ ] Stop presenting all nine routes with equal priority.
- [ ] Keep Home, Session Setup, Speak, Review, History, Library, and Settings discoverable.
- [ ] Move Runtime Setup out of primary practice navigation once runtime is configured; keep it reachable from Home warnings, Settings, and direct links.
- [ ] Keep Scoring Guide as a secondary resource link rather than a primary practice step.
- [ ] Run `npm --prefix frontend test -- HomeSetupRoutes.test.tsx LibraryGuideRoutes.test.tsx`.
- [ ] Get PAL review before patching this slice because it changes screen navigation semantics.

## Task 3: Make Runtime Setup Progressive

### Task 3A: Localize Learner-Friendly Runtime Labels

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [ ] Add learner-facing labels for Speech recognition, AI tutor, Advanced connection details, and Local AI health.
- [ ] Keep technical labels available inside advanced sections.

### Task 3B: Hide Technical Runtime Details By Default

**Files:**
- Modify: `frontend/src/routes/SetupRoute.tsx`
- Modify: `frontend/src/components/setup/RuntimeConnectionForm.tsx`
- Modify: `frontend/src/components/setup/ConnectionStatusPanel.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `frontend/src/routes/tests/SettingsRoute.test.tsx`

- [ ] Keep local providers visible first.
- [ ] Present Whisper as speech recognition and the LLM provider as AI tutor in the default view.
- [ ] Put Base URL, raw model IDs, OpenRouter metadata, and diagnostics behind explicit advanced toggles.
- [ ] Preserve saved-key replacement behavior and OpenRouter validation.
- [ ] Run `npm --prefix frontend test -- HomeSetupRoutes.test.tsx SettingsRoute.test.tsx`.

## Task 4: Make Session Setup Feel Like Choosing A Practice Goal

### Task 4A: Localize Goal-Oriented Setup Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [ ] Add copy for learner profile, session goal, suggested challenge, and practice focus.
- [ ] Keep all theme and task-family labels localized.

### Task 4B: Restructure Session Setup Sections

**Files:**
- Modify: `frontend/src/routes/SessionSetupRoute.tsx`
- Modify: `frontend/src/components/setup/ThemeForm.tsx`
- Modify: `frontend/src/components/setup/PracticeBriefCard.tsx`
- Modify: `frontend/src/routes/tests/SessionSetupRoute.test.tsx`

- [ ] Group speaker/language/level as Learner profile.
- [ ] Group theme/duration/custom theme as Session goal.
- [ ] Make the prompt preview the main confidence-building element, not a secondary form artifact.
- [ ] Preserve current route guard behavior into Runtime Setup when no runtime is configured.
- [ ] Run `npm --prefix frontend test -- SessionSetupRoute.test.tsx`.

## Task 5: Make Speak The Primary Practice Moment

### Task 5A: Localize Recording Guidance

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [ ] Add copy for recording readiness, submit-disabled guidance, microphone confidence, and retry encouragement.

### Task 5B: Improve Speak Hierarchy

**Files:**
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `frontend/src/components/speak/RecorderPanel.tsx`
- Modify: `frontend/src/components/speak/AssessmentStatusPanel.tsx`
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`

- [ ] Make the prompt and recording action visually dominant.
- [ ] Collapse repeated metadata into one compact session summary.
- [ ] Add explicit helper text explaining why Submit is disabled before audio exists.
- [ ] Consider a lightweight microphone-level indicator only after confirming browser support and testability.
- [ ] Run `npm --prefix frontend test -- SpeakRoute.test.tsx`.
- [ ] Get PAL review before patching because this touches the highest-risk screen.

## Task 6: Turn Review And History Into Next-Step Surfaces

### Task 6A: Localize Empty-State And Progress Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [ ] Add copy for no review yet, first session CTA, next exercise, progress summary, and human-review guidance.

### Task 6B: Improve Review And History Empty States

**Files:**
- Modify: `frontend/src/routes/ReviewRoute.tsx`
- Modify: `frontend/src/routes/HistoryRoute.tsx`
- Modify: `frontend/src/routes/tests/ReviewRoute.test.tsx`
- Modify: `frontend/src/routes/tests/HistoryRoute.test.tsx`

- [ ] Replace dead-end Review empty state with Start session / Go to Speak CTAs based on setup readiness.
- [ ] Make completed Review prioritize band, next focus, and retry/new-session actions.
- [ ] Make empty History point to Session Setup and explain what progress will appear after attempts.
- [ ] Run `npm --prefix frontend test -- ReviewRoute.test.tsx HistoryRoute.test.tsx`.

## Task 7: Reframe Library And Guide As Practice Support

### Task 7A: Localize Practice-Support Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [ ] Add copy that presents Library as pick your next exercise and Guide as understand your feedback.

### Task 7B: Adjust Library And Guide Framing

**Files:**
- Modify: `frontend/src/routes/LibraryRoute.tsx`
- Modify: `frontend/src/routes/GuideRoute.tsx`
- Modify: `frontend/src/routes/tests/LibraryGuideRoutes.test.tsx`

- [ ] Make Library CTAs read as practice actions, not content-management actions.
- [ ] Keep custom theme management available but visually secondary.
- [ ] Make Guide sections link conceptually to Review gates and next-focus coaching.
- [ ] Run `npm --prefix frontend test -- LibraryGuideRoutes.test.tsx`.

## Task 8: Browser Flow And Screenshot Regression

**Files:**
- Modify: `frontend/tests/e2e/runtimeSetupLive.spec.ts`
- Modify: `frontend/tests/e2e/reviewHistoryFlow.spec.ts`
- Modify: `frontend/playwright.config.ts`
- Modify: `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`

- [ ] Add/update browser coverage for first-run Home to Runtime Setup to Session Setup to Speak.
- [ ] Add/update coverage for Review empty state and History empty state.
- [ ] Save fresh screenshots after each major UX slice under a dated audit folder.
- [ ] Run `./scripts/run_e2e.sh` or the smallest equivalent Playwright lane available on the branch.

## Verification Matrix

- Frontend unit tests: `npm --prefix frontend test`
- Frontend typecheck: `npm --prefix frontend run typecheck`
- Browser lane: `./scripts/run_e2e.sh`
- Backend smoke when runtime setup changes: `./scripts/run_tests.sh tests/test_app_backend_config.py tests/test_runtime_status.py tests/test_runtime_connections.py`
- PAL review required before changes to `SetupRoute.tsx`, `SpeakRoute.tsx`, `SettingsRoute.tsx`, or navigation semantics.

## Recommended Order

1. Task 1: Move UI language to Settings.
2. Task 2A: Make Home launch practice first and collapse startup checks.
3. Task 2B/2C: Reframe global navigation and demote Runtime Setup from the primary practice list.
4. Task 6: Fix Review/History empty states.
5. Task 3: Runtime setup progressive disclosure.
6. Task 4: Session setup goal framing.
7. Task 5: Speak hierarchy.
8. Task 7: Library framing. Defer Guide content changes unless navigation grouping needs it.
9. Task 8: E2E/screenshot regression.

This order removes obvious cognitive load before touching the more sensitive practice screens.

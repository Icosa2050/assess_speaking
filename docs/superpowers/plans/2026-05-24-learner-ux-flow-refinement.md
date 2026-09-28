# Learner UX Flow Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Vostavo easier and more motivating for language learners by decluttering global chrome, moving UI language to Settings, clarifying setup, and turning Speak/Review/History into a guided practice loop.

**Architecture:** Keep the current React Router route structure, Zustand app store, React Query API access, localization system, and semantic IDs. Implement every visible screen change as a PAL-reviewed, file-bounded slice. Split localization JSON changes into separate five-file subtasks when a UI slice needs new strings.

**Tech Stack:** React 19, React Router 7, Zustand, React Query, Vite, Vitest, Playwright, local backend API.

---

## Status

Implementation complete on 2026-07-22. The learner-facing slices, focused
browser flows, and screenshot evidence are present. Full Playwright execution
remains an environment-sensitive stabilization gate tracked in
`docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`.

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

## 2026-06-01 Refinement: Active Execution Source

This section supersedes the older recommended order where it conflicts. The
separate V2 critique/backlog has been archived under
`docs/superpowers/plans/archive/superseded/2026-05-24-learner-ux-flow-v2.md`;
do not execute it directly.

Current local implementation status:

- A local Home/shell cleanup slice is implemented in the current branch.
- The large header language button cluster is removed from global chrome; Settings owns UI-language changes.
- It removes duplicate Home "Other surfaces" copy when the sidebar already exposes those routes.
- It gives Startup checks explicit loading, setup-needed, ready, issue-summary, and unavailable states so an empty checklist no longer looks unfinished.
- The sidebar now groups learner navigation as Practice, Progress, and Discover, with Settings separated.
- Runtime Setup remains a canonical route for direct links and recovery, but is filtered out of primary navigation once setup is complete.
- The existing `/runtime-setup` route now has a Setup Guide readiness panel with speech recognition, AI tutor connection, microphone setup, and practice-session handoff rows.
- Review and History no-attempt empty states now route learners back to Session Setup instead of ending in dead copy.
- Verification passed on 2026-06-01: `npm --prefix frontend test`, `npm --prefix frontend run typecheck`, `.venv/bin/python -m pytest tests/test_app_core_i18n.py`, and `git diff --check`.

PAL review on 2026-05-30, using `anthropic/claude-opus-4.7`, `qwen/qwen3.7-max`, and `google/gemini-3.1-pro-preview`, converged on the same direction:

- The next learner-facing screen should be a Setup Guide, implemented by evolving the existing `/runtime-setup` route rather than adding a new route.
- Home should contain only an aggregate readiness summary and primary next action; detailed diagnostics belong on the Setup Guide.
- Setup Guide must be state-aware and non-blocking. Users should still be able to browse History, Library, Guide, and Settings.
- The guide offers a practice-session handoff after the verifiable speech-recognition and AI-tutor checks pass. It does not claim to run a separate technical audio check or pedagogical placement test.
- Settings remains the home for UI language, saved providers, credentials, support bundle, and advanced runtime configuration.

The refined learner flow is:

1. Practice Home
   - If runtime is not ready: primary action goes to `/runtime-setup`.
   - If runtime is ready: primary action goes to `/session-setup`.
   - Startup status is a compact summary, not a detailed checklist.
2. Runtime Setup / Setup Guide
   - Shows readiness rows for speech recognition, AI tutor connection, microphone setup, and the practice-session handoff.
   - Each row has one status, one short recovery hint, and one action.
   - Complex configuration opens focused existing controls; the guide does not duplicate the full Settings page.
3. Session Setup
   - Practice language, CEFR level, goal, theme, and prompt preview.
4. Speak
5. Review
6. History / Library / Guide / Settings as supporting surfaces.

### Completed Local Slices

These slices are implemented in the current learner UX baseline and should be treated as
the baseline for any next work.

#### Slice 1: Home/Shell Cleanup Code

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/shell/AppShell.tsx`
- Modify: `frontend/src/components/shell/AppShell.module.css`
- Modify: `frontend/src/routes/HomeRoute.tsx`
- Modify: `frontend/src/routes/HomeRoute.module.css`

- [x] Remove shell-level UI-language controls and leave UI language in Settings.
- [x] Remove duplicate sidebar/Home "Other surfaces" copy.
- [x] Keep route links discoverable through grouped navigation.
- [x] Keep Startup checks from rendering an empty checklist.
- [x] Preserve semantic IDs through relocated navigation links.

#### Slice 2: Home/Shell Locale Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Add localized strings for diagnostics loading, setup-needed, ready, issue-summary, and unavailable states.
- [x] Keep technical runtime labels available for the Setup Guide and Settings.
- [x] Do not add new untranslated Home copy in code.

#### Slice 3: Home/Shell Tests And Screenshot Evidence

**Files:**
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `frontend/src/routes/tests/LibraryGuideRoutes.test.tsx`
- Add: `docs/ux-audit-screenshots/2026-05-26/home-unconfigured-after-shell-cleanup.png`

- [x] Assert the shell no longer renders the old five-button language cluster.
- [x] Assert Home no longer renders duplicate "Other surfaces" text.
- [x] Assert empty diagnostics render a real setup/ready/unavailable state.
- [x] Update Library/Guide navigation tests if they used removed Home duplicate links.
- [x] Capture browser screenshots after the cleanup.

#### Slice 4: Add Setup Guide Readiness Model

**Files:**
- Create: `frontend/src/lib/setup/readiness.ts`
- Create: `frontend/src/lib/setup/readiness.test.ts`
- Modify: `frontend/src/lib/api/types.ts` only if existing diagnostic/runtime types cannot express the rows.

- [x] Derive four row states: speech recognition, AI tutor connection, microphone setup, and practice-session handoff.
- [x] Use the existing runtime and diagnostics API data where possible.
- [x] Represent each row as `loading`, `ready`, `setup`, or `unavailable`.
- [x] Enable the practice-session handoff only from verifiable speech-recognition and AI-tutor readiness.

#### Slice 5: Add Setup Guide Row UI

**Files:**
- Create: `frontend/src/components/setup/ReadinessRow.tsx`
- Create: `frontend/src/components/setup/SetupReadinessPanel.tsx`
- Create: `frontend/src/components/setup/SetupReadinessPanel.module.css`
- Create: `frontend/src/components/setup/SetupReadinessPanel.test.tsx`

- [x] Render each row with status, short explanation, recovery hint, and one action.
- [x] Keep microphone setup honest and actionable without treating it as a remotely verifiable completion gate.
- [x] Keep the component presentational; do not put API calls in the row component.
- [x] Preserve accessible button labels and status text.

#### Slice 6: Integrate Setup Guide Into Existing Runtime Setup

**Files:**
- Modify: `frontend/src/routes/SetupRoute.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `frontend/src/routes/tests/SettingsRoute.test.tsx`
- Modify: `frontend/src/routes/HomeRoute.tsx` only if the Home CTA needs final wiring.

- [x] Put the readiness guide at the top of `/runtime-setup`.
- [x] Keep existing runtime connection controls available below the guide, with row actions scrolling to the relevant section.
- [x] Route first-run Home setup action to `/runtime-setup`.
- [x] Keep configured users able to proceed directly to `/session-setup`.
- [x] Verify route guards still prevent Speak without a setup draft.

#### Slice 7: Move Full UI Language Management To Settings

**Files:**
- Modify: `frontend/src/routes/SettingsRoute.tsx`
- Modify: `frontend/src/routes/tests/SettingsRoute.test.tsx`
- Modify: `frontend/src/components/shell/AppShell.tsx`
- Modify: `frontend/src/components/shell/AppShell.module.css`
- Modify: `frontend/src/App.tsx`

- [x] Treat the header select as temporary or convenience-only.
- [x] Make Settings the durable place for UI-language preference.
- [x] Avoid a custom settings drawer until the Settings route proves too slow or too hidden.
- [x] Preserve document `lang` updates and semantic IDs.

### Completed Execution Sequence

The implementation followed this sequence. Any future screen changes must
remain file-bounded and receive PAL review before editing `SetupRoute.tsx`,
`SpeakRoute.tsx`, `SettingsRoute.tsx`, or navigation semantics.

1. Session Setup was made goal-oriented around learner profile, session goal,
   and prompt preview.
2. Speak was established as the primary practice moment.
3. Review gained score/band-first hierarchy, next focus, next exercise, and
   retry/change-task actions.
4. Library and Guide were reframed as practice support.
5. Focused browser-flow and screenshot regression coverage was added.

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

- [x] Add or confirm Settings strings for UI language as a first-class preference.
- [x] Keep existing locale keys intact; add new keys first and remove obsolete keys only after the code move is verified.
- [x] Run `npm --prefix frontend test -- SettingsRoute.test.tsx`.

### Task 1B: Move Locale Control To Settings

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/shell/AppShell.tsx`
- Modify: `frontend/src/components/shell/AppShell.module.css`
- Modify: `frontend/src/routes/SettingsRoute.tsx`
- Modify: `frontend/src/routes/tests/SettingsRoute.test.tsx`

- [x] Remove `localeLabel`, `localeOptions`, `activeLocale`, and `onLocaleChange` from `AppShell`.
- [x] Remove the header locale button group and related CSS.
- [x] Render the UI-language control in Settings, using `useAppStore((state) => state.setUiLocale)`.
- [x] Preserve or deliberately relocate semantic IDs for automation.
- [x] Run `npm --prefix frontend test -- SettingsRoute.test.tsx HomeSetupRoutes.test.tsx`.
- [x] Capture before/after screenshots of Home and Settings.

## Task 2: Reframe Global Navigation Around Practice

### Task 2A: Make Home Launch Practice First

**Files:**
- Modify: `frontend/src/routes/HomeRoute.tsx`
- Modify: `frontend/src/routes/HomeRoute.module.css`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `locales/en.json`
- Modify: one additional locale file in a separate follow-up if the English copy lands first

- [x] Move the primary Home focus from startup checks to Start practicing / Start new session.
- [x] Collapse startup checks behind a single System status summary unless runtime is missing or a check fails.
- [x] Remove duplicate Other surfaces content if the sidebar already exposes those routes.
- [x] Preserve runtime-missing behavior: Home must still give a clear route to Runtime Setup when no runtime is configured.
- [x] Run `npm --prefix frontend test -- HomeSetupRoutes.test.tsx`.
- [x] Capture Home default and Home configured screenshots after the change.

### Task 2B: Localize Navigation Grouping Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Add copy for learner-facing navigation groups such as Practice, Progress, Resources, and Settings.
- [x] Keep technical setup labels available for CTAs and Settings links.

### Task 2C: Reduce Navigation Overload

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/shell/AppShell.tsx`
- Modify: `frontend/src/components/shell/AppShell.module.css`
- Modify: `frontend/src/routes/HomeRoute.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`

- [x] Stop presenting all nine routes with equal priority.
- [x] Keep Home, Session Setup, Speak, Review, History, Library, and Settings discoverable.
- [x] Move Runtime Setup out of primary practice navigation once runtime is configured; keep it reachable from Home warnings, Settings, and direct links.
- [x] Keep Scoring Guide as a secondary resource link rather than a primary practice step.
- [x] Run `npm --prefix frontend test -- HomeSetupRoutes.test.tsx LibraryGuideRoutes.test.tsx`.
- [x] Get PAL review before patching this slice because it changes screen navigation semantics.

## Task 3: Make Runtime Setup Progressive

The implemented path is the Setup Guide described in the 2026-06-01 refinement
above. The older Task 3B checklist remains useful only as historical
progressive-disclosure guidance; do not execute it literally over the completed
Setup Guide slice.

### Task 3A: Localize Learner-Friendly Runtime Labels

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Superseded by the 2026-06-01 Setup Guide slices: learner-facing Speech recognition, AI tutor, microphone setup, and practice-session handoff labels are present in the guided readiness panel.
- [x] Keep technical labels available in Runtime Setup and Settings for direct configuration and recovery.

### Task 3B: Hide Technical Runtime Details By Default

**Files:**
- Modify: `frontend/src/routes/SetupRoute.tsx`
- Modify: `frontend/src/components/setup/RuntimeConnectionForm.tsx`
- Modify: `frontend/src/components/setup/ConnectionStatusPanel.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `frontend/src/routes/tests/SettingsRoute.test.tsx`

- [x] Superseded by the Setup Guide path for this learner-flow wave; local/provider recovery remains reachable through Runtime Setup and Settings.
- [x] Present Whisper as speech recognition and the LLM provider as AI tutor in the guided readiness panel.
- [x] Defer raw provider-form advanced toggles outside this active learner-flow wave; do not execute this older checklist literally over the completed Setup Guide slice.
- [x] Preserve saved-key replacement behavior and OpenRouter validation by leaving existing Runtime Setup and Settings controls intact.
- [x] Covered by the Home/Setup/Settings verification bundle from the completed Setup Guide slices.

## Task 4: Make Session Setup Feel Like Choosing A Practice Goal

### Task 4A: Localize Goal-Oriented Setup Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Add copy for learner profile, session goal, speaking brief, and session receipt.
- [x] Keep all theme and task-family labels localized.

### Task 4B: Restructure Session Setup Sections

**Files:**
- Modify: `frontend/src/routes/SessionSetupRoute.tsx`
- Modify: `frontend/src/components/setup/ThemeForm.tsx`
- Modify: `frontend/src/components/setup/PracticeBriefCard.tsx`
- Modify: `frontend/src/routes/tests/SessionSetupRoute.test.tsx`

- [x] Group speaker/language/level as Learner profile.
- [x] Group theme/duration/custom theme as Session goal.
- [x] Make the prompt preview the main confidence-building element, not a secondary form artifact.
- [x] Preserve current route guard behavior into Runtime Setup when no runtime is configured.
- [x] Run `npm --prefix frontend test -- src/routes/tests/SessionSetupRoute.test.tsx`.
- [x] Save desktop/mobile screenshots:
  `docs/ux-audit-screenshots/2026-05-26/session-setup-goal-oriented-desktop.png`,
  `docs/ux-audit-screenshots/2026-05-26/session-setup-goal-oriented-mobile.png`.

## Task 5: Make Speak The Primary Practice Moment

### Task 5A: Localize Recording Guidance

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Add copy for recording readiness, submit-disabled guidance, runtime detail, and retry guidance; microphone-level confidence was deliberately deferred after PAL review.

### Task 5B: Improve Speak Hierarchy

**Files:**
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `frontend/src/components/speak/RecorderPanel.tsx`
- Modify: `frontend/src/components/speak/AssessmentStatusPanel.tsx`
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`

- [x] Make the prompt and recording action visually dominant.
- [x] Collapse repeated metadata into one compact session summary.
- [x] Add explicit helper text explaining why Submit is disabled before audio exists.
- [x] Defer a microphone-level indicator after PAL confirmed browser/media support and testability risk.
- [x] Run `npm --prefix frontend test -- src/routes/tests/SpeakRoute.test.tsx`.
- [x] Get PAL review before patching because this touches the highest-risk screen.

## Task 6: Turn Review And History Into Next-Step Surfaces

### Task 6A: Localize Empty-State And Progress Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Add basic empty-state copy for no review yet and first-session History CTA.
- [x] Add copy for next exercise, progress summary, and human-review guidance.

### Task 6B: Improve Review And History Empty States

**Files:**
- Modify: `frontend/src/routes/ReviewRoute.tsx`
- Modify: `frontend/src/routes/HistoryRoute.tsx`
- Modify: `frontend/src/routes/tests/ReviewRoute.test.tsx`
- Modify: `frontend/src/routes/tests/HistoryRoute.test.tsx`

- [x] Replace dead-end Review empty state with a Start session CTA to Session Setup.
- [x] Make completed Review prioritize band, next focus, next exercise, and retry/change-task actions.
- [x] Make empty History point to Session Setup and explain what progress will appear after attempts.
- [x] Run `npm --prefix frontend test -- ReviewRoute.test.tsx HistoryRoute.test.tsx`.
- [x] Capture screenshot evidence for Review and History empty states.

## Task 7: Reframe Library And Guide As Practice Support

### Task 7A: Localize Practice-Support Copy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Add copy that presents Library as pick your next exercise and Guide as understand your feedback.

### Task 7B: Adjust Library And Guide Framing

**Files:**
- Modify: `frontend/src/routes/LibraryRoute.tsx`
- Modify: `frontend/src/routes/GuideRoute.tsx`
- Modify: `frontend/src/routes/tests/LibraryGuideRoutes.test.tsx`

- [x] Make Library CTAs read as practice actions, not content-management actions.
- [x] Keep custom theme management available but visually secondary.
- [x] Make Guide sections link conceptually to Review gates and next-focus coaching.
- [x] Run `npm --prefix frontend test -- src/routes/tests/LibraryGuideRoutes.test.tsx`.

## Task 8: Browser Flow And Screenshot Regression

**Files:**
- Modify: `frontend/tests/e2e/runtimeSetupLive.spec.ts`
- Modify: `frontend/tests/e2e/reviewHistoryFlow.spec.ts`
- Add: `frontend/tests/e2e/libraryGuideFlow.spec.ts`
- Add: `frontend/tests/e2e/reviewChangeTaskFlow.spec.ts`
- Modify: `frontend/tests/e2e/realAudioHistory.spec.ts`

- [x] Add/update browser coverage for Home to Session Setup to Speak and the existing first-run Home to Runtime Setup smoke path.
- [x] Add/update coverage for Review empty state and History empty state.
- [x] Add coverage for Library/Guide practice-support framing and the Review change-task handoff.
- [x] Open the native optional-context disclosure before history flows fill labels or notes.
- [x] Save fresh screenshots after each major UX slice under a dated audit folder. Current 2026-06-05 evidence covers Home, Runtime Setup, Session Setup, Speak, Review empty, History empty, Library, Guide, Settings, and mobile Home under `docs/ux-audit-screenshots/2026-06-05/`.
- [x] Run `./scripts/run_e2e.sh` or the smallest equivalent Playwright lane available on the branch. Fresh 2026-06-04 run of `NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/reviewHistoryFlow.spec.ts` passed locally after one transient first-run click actionability timeout on `review-guard-cta`; a direct Playwright probe and the in-app Browser both clicked the CTA through to `/session-setup`.

## Verification Matrix

- Frontend unit tests: `npm --prefix frontend test`
- Frontend typecheck: `npm --prefix frontend run typecheck`
- Browser lane: `./scripts/run_e2e.sh`
- Backend smoke when runtime setup changes: `./scripts/run_tests.sh tests/test_app_backend_config.py tests/test_runtime_status.py tests/test_runtime_connections.py`
- PAL review required before changes to `SetupRoute.tsx`, `SpeakRoute.tsx`, `SettingsRoute.tsx`, or navigation semantics.

## Closure Gate

The implementation contract is closed. Preserve these gates when the flow is
changed again:

1. Run the full frontend unit suite and typecheck.
2. Run backend tests and locale parity.
3. Collect the Playwright suite and run the smallest affected browser flow.
4. Revisit screenshot evidence only after a changed screen stabilizes.
5. Record browser-launch restrictions separately from app assertion failures.

The full Playwright runner is not a reason to reopen completed UX tasks when the
current environment prevents Chromium from launching before app code.

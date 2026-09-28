# Session Setup Newbie Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Session Setup possible for a total beginner by adding a guided
quick-start path, beginner copy, and an explicit ready-to-practice handoff while
preserving the existing session draft, runtime guard, localization, and
automation semantics.

**Architecture:** Keep the existing `/session-setup` route. Do not create new
routes, URL step params, placement tests, backend contracts, or new npm
dependencies. Treat the change as a guided setup frame around the existing
learner profile, session goal, and practice brief model.

**Tech Stack:** React 19, React Router 7, Zustand app store, React Query runtime
readiness query, existing i18n JSON files, Vitest, Testing Library, Playwright
visual smoke.

---

## Status

Implemented on 2026-06-09; final screenshot evidence was captured by the later
focused visual-smoke pass.

- Session Setup now uses a PAL-reviewed two-step beginner frame with
  `setup.wizard`, a recommended-start path, beginner copy, optional detail
  controls, native advanced-topic disclosure, and an explicit runtime handoff
  callout.
- The recommended path fills a safe Italian B1 practice, 90 seconds, the first
  matching library theme, and then shows the prompt/start state.
- The custom path still preserves `speakerId`, language/CEFR/theme/duration,
  custom-theme save-for-reuse, `applySetup`, `/speak`, and `/runtime-setup`
  routing contracts.
- Verified passing: focused setup/guard route tests, full frontend tests,
  frontend typecheck, backend i18n parity, localhost backend/frontend HTTP
  smoke, and `git diff --check`.
- Attempted visual-smoke command from `frontend/`:

```zsh
env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-09 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

Result on 2026-06-09: blocked before app assertions because Chromium/Chrome
cannot launch in this sandbox. The bundled Chromium failure was
`bootstrap_check_in org.chromium.Chromium.MachPortRendezvousServer... Permission
denied (1100)`. System Chrome also aborted before page load, and the direct
Chrome probe produced no DOM output.

---

## Baseline Evidence

Current code shape:

- `SessionSetupRoute.tsx` owns local state for `speakerId`, language, CEFR,
  theme mode/title/custom theme, save-for-reuse, duration, and validation
  errors.
- `ThemeForm.tsx` renders one form-like section with `Learner profile` and
  `Session goal` fieldsets.
- `PracticeBriefCard.tsx` renders `Today's speaking brief`, success focus, and
  the session receipt.
- `SessionSetupRoute.test.tsx` already covers the important contracts:
  grouped setup sections, semantic IDs, preview updates, runtime handoff,
  custom-theme persistence, validation announcements, and invalid select values.
- `speakerId` is not just display copy. Backend and frontend history use it as
  the stable learner/history key, so implementation must keep the state and API
  field as `speakerId`/`speaker_id` while making the visible label friendlier.

Existing visual-smoke path:

- `frontend/tests/e2e/visualRefreshSmoke.spec.ts` already covers Home ->
  Session Setup -> Speak and captures/checks Session Setup at 390px mobile.
- The exact focused command currently documented for the visual lane is:

```zsh
env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-05 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

## External Reference Scan

Use local-AI setup apps as interaction references, not content references:

- [Locally AI by LM Studio](https://apps.apple.com/us/app/locally-ai-by-lm-studio/id6741426692)
  sets the product bar as local, private, offline, no-login, and low-friction.
  Session Setup should feel similarly safe: no jargon first, no technical
  detour, and clear privacy/history framing for the learner name.
- [OpenClaw Desktop](https://getopenclawdesktop.com/) presents a real first-run
  wizard and makes the first screen a welcome/purpose step before exposing
  gateway/channel details. Its docs also frame onboarding as a guided flow for
  choosing a provider, setting a key, and configuring the gateway.
- [HammerLock AI](https://www.hammerlockai.com/get-app) uses a numbered first
  launch path and a recommended local setup. The useful pattern is "pick the
  safe default, then reveal alternatives," not "show every configuration field
  immediately."
- [HammerAI Desktop](https://www.hammerai.com/desktop) emphasizes "works out of
  the box" and hides local model complexity behind automatic configuration.
  For Vostavo, the analogous product promise is "enter a name, use a recommended
  session, and start speaking."

## PAL Review

PAL reviewed the proposed three-step wizard with the current code shape in
mind. The strongest critique was that Session Setup is recurring, unlike a
one-time AI-provider wizard, so a forced three-step flow would add friction for
repeat practice.

Refined decision:

- Use a two-step guided setup frame, not a heavy three-step wizard.
- Keep the practice brief visible from the practice-choice step onward, because
  the live preview is already a strength.
- Make the recommended starter path explicit: the learner enters only a name or
  nickname, clicks the recommended CTA, fields are filled from existing defaults,
  and the route jumps to the preview/start state.
- Keep existing controls mounted only when they are relevant to the active path;
  update tests to drive the beginner path and the full customize path. If a
  hidden-mounted approach is chosen later, it must use `aria-hidden` or `inert`
  correctly.
- Preserve `speakerId` semantics. Visible copy can say "Learner name or
  nickname," with helper copy that it separates this learner's history.

## Product Direction

The screen should answer three beginner questions:

1. Who is practicing?
2. What should I practice today?
3. What happens when I continue?

The fastest path should be:

1. Enter learner name or nickname.
2. Select "Use recommended practice."
3. See the prompt and one readiness note.
4. Start speaking, or continue to Runtime Setup if the device still needs setup.

The custom path should still be available for repeat users:

1. Choose language, level, theme, and duration.
2. Open "Make my own topic" only when needed.
3. Watch the brief update as choices change.
4. Continue with the same existing `/speak` or `/runtime-setup` guard.

## File-Bounded Tasks

### Task 1: English Copy And Test Contract

**Files:**
- Modify: `locales/en.json`
- Modify: `frontend/src/routes/tests/SessionSetupRoute.test.tsx`

- [x] Add failing tests for the guided beginner path:
  - wizard/guided setup frame is visible with semantic ID
    `setup.wizard`.
  - learner name or nickname field still uses existing semantic ID
    `setup.speaker_id`.
  - recommended starter CTA uses semantic ID `setup.recommended_start`.
  - recommended starter fills existing default language, `B1`, first available
    matching theme, and `90` seconds.
  - recommended starter advances to the preview/start state.
- [x] Add failing tests for the full customize path:
  - learners can choose details without using the recommended CTA.
  - custom topic remains available behind `setup.advanced_topic`.
  - existing custom-theme persistence still works.
- [x] Add failing tests for runtime handoff copy:
  - ready runtime shows a start-speaking label and routes to `/speak`.
  - missing runtime shows a setup-needed label/callout and routes to
    `/runtime-setup`.
- [x] Add English keys for starter CTA, beginner helper copy, step labels,
  CEFR hint, duration hint, advanced-topic summary, runtime-ready callout, and
  runtime-setup-needed callout.

### Task 2: Guided Route Frame And Recommended Starter

**Files:**
- Modify: `frontend/src/routes/SessionSetupRoute.tsx`
- Modify: `frontend/src/components/setup/ThemeForm.tsx`
- Modify: `frontend/src/components/setup/PracticeBriefCard.tsx`
- Modify: `frontend/src/routes/tests/SessionSetupRoute.test.tsx`

- [x] Add local step state for the guided frame. Keep it local React state; do
  not add query params or store state.
- [x] Render a lightweight step/progress header with semantic ID
  `setup.wizard`.
- [x] Implement `setup.recommended_start` by using existing safe defaults:
  current/default language, `B1` when available, first available matching
  library theme, `90` seconds, and existing task family from that theme.
- [x] After the recommended CTA, jump to the preview/start state with a clear
  "Change details" action that reopens customization.
- [x] Keep `applySetup`, custom-theme saving, and runtime-readiness branching
  unchanged except for visible labels/callout text.

### Task 3: Beginner-Friendly Detail Controls

**Files:**
- Modify: `frontend/src/components/setup/ThemeForm.tsx`
- Modify: `frontend/src/components/setup/PracticeBriefCard.tsx`
- Modify: `frontend/src/routes/SessionSetupRoute.tsx`
- Modify: `frontend/src/routes/tests/SessionSetupRoute.test.tsx`

- [x] Change visible copy from "Speaker ID" to beginner product copy while
  preserving the underlying `speakerId` state and `setup.speaker_id` semantic ID.
- [x] Add localized helper text explaining that the name/nickname separates
  this learner's history.
- [x] Add CEFR and duration hints in the full customize path.
- [x] Move custom topic controls under a native `<details>` disclosure with
  semantic ID `setup.advanced_topic`.
- [x] Keep the practice brief visible when selecting theme/level/duration so the
  learner sees the exact prompt before speaking.

### Task 4: Runtime Handoff Callout

**Files:**
- Modify: `frontend/src/routes/SessionSetupRoute.tsx`
- Modify: `frontend/src/components/setup/ThemeForm.tsx`
- Modify: `frontend/src/components/setup/PracticeBriefCard.tsx`
- Modify: `frontend/src/routes/tests/SessionSetupRoute.test.tsx`

- [x] Add a final callout with semantic ID `setup.runtime_callout`.
- [x] If runtime is ready, label the primary action as starting speaking.
- [x] If runtime is missing, label the primary action as saving the practice
  setup and continuing to device/runtime setup.
- [x] Preserve the current guard: ready runtime navigates to `/speak`; missing
  runtime sets return target to setup and navigates to `/runtime-setup`.
- [x] Make the runtime callout concise and non-technical; detailed provider
  setup remains on Runtime Setup.

### Task 5: Locale Fan-Out

**Files:**
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Translate the finalized English keys after the English copy is stable.
- [x] Keep the copy native and concise, not a literal technical translation.
- [x] Do not add user-visible fallback strings in React code.
- [x] Run frontend tests plus backend i18n parity after fan-out.

### Task 6: Visual Smoke And Evidence

**Files:**
- Modify: `frontend/tests/e2e/visualRefreshSmoke.spec.ts`
- Modify: `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`
- Modify: `docs/PLAN_STATUS.md`
- Modify: `docs/superpowers/plans/README.md`
- Add or modify: `docs/ux-audit-screenshots/2026-06-09/README.md`

- [x] Update the focused visual smoke to assert the new Session Setup guided
  frame, recommended starter CTA, and runtime callout.
- [x] Keep the existing 390px mobile no-serif and no-horizontal-overflow checks.
- [x] Capture refreshed Session Setup mobile evidence through the existing
  `VISUAL_REFRESH_SCREENSHOT_DIR` path.
- [x] Document the exact command, date, and result.
- [x] Update plan/status docs only after the implementation and verification
  pass.

Capture note: the original 2026-06-09 attempt was blocked before app code, then
the later focused visual-smoke pass produced
`docs/ux-audit-screenshots/2026-06-09/visual-refresh-smoke-session-setup-mobile.png`.

## Verification Commands

Use zsh from the repo root.

Focused route tests:

```zsh
npm --prefix frontend test -- src/routes/tests/SessionSetupRoute.test.tsx
```

Frontend baseline:

```zsh
npm --prefix frontend test
npm --prefix frontend run typecheck
```

Backend i18n parity:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
```

Focused visual smoke:

```zsh
env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-09 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

Diff hygiene:

```zsh
git diff --check
```

## Acceptance

- A total beginner can start with only a learner name/nickname and a
  recommended session.
- Repeat users can still customize language, level, theme, custom topic, and
  duration without a heavy multi-route wizard.
- The prompt preview remains visible and useful before the learner records.
- Runtime setup handoff is explicit and honest without exposing provider jargon.
- Existing draft state, custom-theme persistence, route guards, semantic IDs,
  and localization contracts are preserved.
- Session Setup passes focused route tests, frontend baseline, typecheck,
  backend i18n parity, focused Playwright visual smoke, and `git diff --check`.

## Non-Goals

- No new placement test.
- No backend API or assessment contract changes.
- No URL step params or route split.
- No dependency additions.
- No broad restyling of the app shell, Review, History, Speak, Library, Guide,
  or Settings in this slice.
- No change to runtime detection, Whisper caching, provider setup, or saved
  credential semantics.

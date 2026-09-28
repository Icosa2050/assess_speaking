# Visual Smoke And Speak Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the focused browser visual-smoke path a first-class regression gate, then add a small confidence-oriented Speak polish slice and bounded learner-copy pass.

**Architecture:** Keep browser hardening in the existing Playwright smoke, keep Speak polish additive and state-derived, and keep copy changes localized through existing locale JSON files. Do not change backend assessment behavior, recording behavior, navigation, runtime setup, or package dependencies.

**Tech Stack:** React 19, TypeScript, Vite, Playwright, Vitest, existing i18n JSON, existing semantic IDs and CSS patterns.

---

## Status

Implemented on 2026-06-09.

- Focused visual-smoke gate passed with the documented command and now covers
  Home, Session Setup, Speak, clean Review, failed-gate Review, History, and
  expanded History at 390px mobile width with no-serif and no-horizontal-overflow
  assertions.
- Speak now has a PAL-reviewed non-interactive confidence rail derived from
  existing recording/assessment state.
- Source and non-English locale copy for the touched Speak/Review/History
  surfaces was tightened while preserving key parity and placeholder tokens.

## Decision Inputs

- User recommendation: testing hardening first, then Speak polish, because the browser lane was weak and recording/submitting is the active learner confidence moment.
- Product assessment: History and Review remain the weakest surfaces, but Speak is the better next slice if the edit is tightly scoped to confidence during record/submit.
- PAL critique: Speak-first is defensible as a confidence-moment slice. Keep it additive: a non-interactive status rail driven from existing state, not a recorder redesign. Failed-gate visual coverage should assert stable structure and state, not brittle prose.
- Competitor scan:
  - Babbel Speak and Duolingo Video Call emphasize low-pressure guided speaking and realistic conversation.
  - ELSA and Yoodli emphasize immediate coaching feedback and progress signals.
  - Praktika and Talkpal use persona, roleplay, and adaptive modes for retention.
- Vostavo opportunity: stay local-first and transparent, but make the learner feel guided before submission and coached after assessment. Do not chase avatars, streaks, badges, confetti, or broad gamification in this slice.

## Non-Goals

- No new npm dependencies.
- No changes to backend scoring, upload, polling, runtime settings, or persisted history contracts.
- No new global visual system, navigation redesign, avatar, mascot, streak, XP, badge, or marketing-style hero.
- No semantic ID removal or rename.
- No broad all-route redesign; each subtask stays under five files.

## File-Based Tasks

### Task 1: Visual-Smoke Gate Hardening

**Files:**
- Modify: `frontend/tests/e2e/visualRefreshSmoke.spec.ts`
- Modify: `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`
- Modify: `docs/ux-audit-screenshots/2026-06-05/README.md`
- Modify: `docs/PLAN_STATUS.md`
- Modify: `docs/superpowers/plans/README.md`

- [x] **Step 1: Add failing Playwright assertions for every mobile route audit.**

Add a helper contract in `visualRefreshSmoke.spec.ts` that is first used for Home and History:

```ts
const expectMobileRouteAudit = async (page: Page, filename: string) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await expectNoVisibleSerifTypography(page);
  await expectNoPageHorizontalOverflow(page);
  await captureScreenshot(page, filename);
  await page.setViewportSize({ width: 1280, height: 720 });
};
```

Run:

```zsh
cd frontend
env NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

Expected before implementation completion: FAIL on routes that were not yet adapted or failed-gate coverage not present.

- [x] **Step 2: Add failed-gate Review fixture and structural assertions.**

Add a second report payload where at least two checks fail and `requires_human_review` is true. Assert stable structure:

```ts
await expect(page.getByTestId("review-quality-summary")).toContainText("3 of 5");
await expect(page.getByTestId("review-gates-disclosure")).toHaveAttribute("open", "");
await expect(page.getByTestId("review-failed-gates")).toBeVisible();
await expect(page.getByTestId("review-gate-topic")).toBeVisible();
await expect(page.getByTestId("review-gate-content-validity")).toBeVisible();
await expect(page.getByTestId("review-next-step-card")).toBeVisible();
```

Do not assert full localized warning sentences in Playwright.

- [x] **Step 3: Capture screenshots for all key mobile route states.**

Use the reusable mobile helper for:

- Home
- Session Setup
- Speak
- Review clean
- Review failed-gate
- History
- History expanded

Screenshots must land under `docs/ux-audit-screenshots/2026-06-05/` when `VISUAL_REFRESH_SCREENSHOT_DIR` is set.

- [x] **Step 4: Document the exact first-class gate command.**

Record this command in the Playwright plan and screenshot inventory:

```zsh
cd frontend
env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-05 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

Acceptance:

- The command is documented as the focused visual-smoke gate.
- Failed Review is visually covered, not only the clean 5/5 state.
- Every key mobile route screenshot path also gets no-serif and no-horizontal-overflow checks.

### Task 2: Speak Confidence Rail

**Files:**
- Create: `frontend/src/components/speak/SpeakStatusRail.tsx`
- Create: `frontend/src/components/speak/SpeakStatusRail.test.tsx`
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`
- Modify: `locales/en.json`

- [x] **Step 1: Write failing component tests for state-derived guidance.**

Test the rail with localized copy and phases:

```ts
expect(screen.getByTestId("speak.status_rail")).toHaveAttribute("data-semantic-id", "speak.status_rail");
expect(screen.getByTestId("speak.status_rail_step_brief")).toHaveAttribute("data-step-state", "complete");
expect(screen.getByTestId("speak.status_rail_step_record")).toHaveAttribute("data-step-state", "active");
expect(screen.getByTestId("speak.status_rail_step_review")).toHaveAttribute("data-step-state", "upcoming");
```

Run:

```zsh
npm --prefix frontend test -- src/components/speak/SpeakStatusRail.test.tsx
```

Expected: FAIL because the component does not exist yet.

- [x] **Step 2: Implement the rail as a pure presentational component.**

The component takes a phase only:

```ts
type SpeakConfidencePhase = "prepare" | "record" | "submit" | "assess" | "failed";
```

It renders four stable steps:

- `speak.status_rail_step_brief`
- `speak.status_rail_step_record`
- `speak.status_rail_step_submit`
- `speak.status_rail_step_review`

Use `data-step-state="complete" | "active" | "upcoming" | "attention"` for tests and automation. Keep the rail non-interactive and do not add extra live-region announcements.

- [x] **Step 3: Integrate the rail in SpeakRoute without changing recorder behavior.**

Derive the phase from existing state:

```ts
const confidencePhase =
  lifecycleState === "failed" || lifecycleState === "cancelled"
    ? "failed"
    : lifecycleState === "queued" || lifecycleState === "running"
      ? "assess"
      : hasAttachment
        ? "submit"
        : "record";
```

Render the rail below the speaking brief and above the recorder/assessment grid.

- [x] **Step 4: Add focused route expectations.**

Extend `SpeakRoute.test.tsx` to prove:

- idle state highlights recording;
- uploaded audio highlights submit;
- queued/running state highlights review/assessment;
- existing recorder and submit semantic IDs remain present.

Acceptance:

- Speak feels more guided without moving the prompt, recorder, or assessment panel.
- Runtime detail remains secondary.
- No recorder internals, backend calls, or route guards change.

### Task 3: Learner Copy And Locale Propagation

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] **Step 1: Polish source copy for Speak, Review, and History.**

English copy should become calmer and more coach-like:

- Speak: reduce technical wording around submission and queued/running states.
- Review: emphasize next action, open checks, and evidence without sounding like a raw report.
- History: keep progress-story language precise; avoid overclaiming improvement.

- [x] **Step 2: Propagate keys and placeholder tokens across all locales.**

Keep interpolation tokens identical across all five files. Do not add layout-affecting long strings to the rail without checking the 390px smoke.

- [x] **Step 3: Verify locale parity and UI selectors.**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
npm --prefix frontend test -- src/routes/tests/SpeakRoute.test.tsx src/components/speak/SpeakStatusRail.test.tsx
```

Acceptance:

- No hardcoded learner-facing strings are added.
- Existing text-based route tests are updated only where the source copy intentionally changed.
- Locale files keep matching key sets and placeholder names.

## Final Product Review Criteria

Review the finished work against these criteria:

- Testing hardening: Is the visual-smoke command documented, runnable, and broad enough to catch mobile overflow and failed Review gates?
- Speak polish: Does the screen feel guided during record/submit without becoming busy or childish?
- Wizard/user guidance: Does the learner understand the current step and next step without reading technical runtime detail?
- Retention: Does the flow encourage another attempt through calm progress and coach language rather than rewards?
- Competition: Does the app lean into local-first transparent assessment rather than imitating avatars or gamified calls?
- Risk: Are evidence sections, failed gates, and raw reports still accessible for trust?

## Verification Commands

Use zsh from the repo root:

```zsh
npm --prefix frontend test -- src/components/speak/SpeakStatusRail.test.tsx
npm --prefix frontend test -- src/routes/tests/SpeakRoute.test.tsx src/routes/tests/ReviewRoute.test.tsx
npm --prefix frontend run typecheck
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
cd frontend
env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-05 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

Finish with:

```zsh
git diff --check
```

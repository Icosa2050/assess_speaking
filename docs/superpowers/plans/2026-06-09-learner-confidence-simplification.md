# Learner Confidence Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve learner confidence and retention by reducing duplicate progress/status signals in History and Speak, not by adding another layer of UI.

**Architecture:** Ship this as two independent, file-bounded slices. History is first because it is read-only and currently repeats the same progress facts in multiple sections. Speak is second because it touches the submit-critical path and must consolidate existing signals instead of adding a new submit handoff.

**Tech Stack:** React 19, TypeScript, Vite, Vitest, Testing Library, Playwright, existing i18n JSON, existing semantic IDs and CSS patterns.

---

## Status

Implemented on 2026-06-09.

- History no longer repeats latest/new/resolved priority cards below the top progress story.
- `HistoryProgressStory` now uses a safer 360px layout and gives the next-practice cue full row width.
- Speak now moves label/notes into an optional context disclosure after audio is attached.
- Speak queued/running status now uses one assessment-wait helper instead of two overlapping wait paragraphs.
- The native upload input is visually hidden after the browser smoke exposed it as an attached-audio mobile overflow source; the custom upload affordance and `speak.upload_input` automation hook remain.
- Verification passed: focused History/Speak tests, full frontend tests, frontend typecheck, backend i18n parity, focused Playwright visual smoke, and `git diff --check`.
- Refreshed visual evidence lives under `docs/ux-audit-screenshots/2026-06-09/`, including `visual-refresh-smoke-history-narrow-mobile.png`.

## PAL Review

PAL consensus was requested on 2026-06-09.

- `gpt-5.2` was listed by PAL but failed with a project access `403`, so it provided no useful review.
- `anthropic/claude-opus-4.7` reviewed the proposal critically and rejected the original "add progressive submit handoff" framing.
- Accepted critique: Speak already has several overlapping confidence/status surfaces: `SpeakStatusRail`, `speak.handoff_hint`, `speak.recording_ready_checkpoint`, `speak.submit_disabled_help`, phase messages, and queued/running helper copy.
- Accepted critique: History currently repeats latest/resolved/focus information through `HistoryProgressStory` and the `history-trends` priority trio.
- Plan change: implement deletion and consolidation. Do not add another Speak card or History narrative block.

## Learner-Value Hypotheses

- History simplification should make the learner understand the next practice target in the first viewport instead of scanning several report sections.
- Speak consolidation should reduce hesitation after a recording is attached by making the submit action and optional context clearly distinct.
- The changes are successful only if mobile screenshots become shorter/clearer at 360-390px and all existing route semantics remain stable.

## Non-Goals

- No backend, API, assessment, upload, polling, or history persistence changes.
- No new npm dependencies.
- No route additions, query params, telemetry system, gamification, badges, streaks, avatars, or confetti.
- No removal or rename of existing semantic IDs that automation already relies on.
- No hardcoded learner-facing strings.
- No broad redesign of Home, Session Setup, Review, Library, Guide, Runtime Setup, or Settings in this plan.

## Phase 1: History Progress Story De-Duplication

### Task 1: Remove Duplicate Priority Trio From History Trends

**Files:**
- Modify: `frontend/src/routes/tests/HistoryRoute.test.tsx`
- Modify: `frontend/src/routes/HistoryRoute.tsx`
- Modify: `frontend/tests/e2e/visualRefreshSmoke.spec.ts`

- [x] **Step 1: Write failing route assertions that the story owns learner priorities.**

In `frontend/src/routes/tests/HistoryRoute.test.tsx`, extend the existing progress-story test to prove the top story remains and the duplicate priority cards are gone:

```ts
expect(screen.getByTestId("history-progress-story")).toBeVisible();
expect(screen.getByTestId("history-progress-story-observed-focus")).toHaveTextContent(
  "Tighten the ending",
);
expect(screen.getByTestId("history-progress-story-next-practice")).toBeVisible();
expect(screen.queryByTestId("history-priority-latest")).not.toBeInTheDocument();
expect(screen.queryByTestId("history-priority-new")).not.toBeInTheDocument();
expect(screen.queryByTestId("history-priority-resolved")).not.toBeInTheDocument();
expect(screen.getByTestId("history-chart-score")).toBeVisible();
expect(screen.getByTestId("history-chart-pace")).toBeVisible();
```

Run:

```zsh
npm --prefix frontend test -- src/routes/tests/HistoryRoute.test.tsx
```

Expected before implementation: FAIL because `history-priority-*` cards are still rendered.

- [x] **Step 2: Remove the duplicate priority trio from `HistoryRoute`.**

In `frontend/src/routes/HistoryRoute.tsx`, remove these derived values when no longer used:

```ts
const priorities = {
  latest: latestRecord?.topPriorities ?? [],
  previous: filteredRecords.length > 1 ? filteredRecords[filteredRecords.length - 2].topPriorities : [],
};
const newPriorities = priorities.latest.filter((item) => !priorities.previous.includes(item));
const resolvedPriorities = priorities.previous.filter((item) => !priorities.latest.includes(item));
```

Then remove the `history-priority-latest`, `history-priority-new`, and `history-priority-resolved` block inside `history-trends`. Keep `history-trends`, score sparkline, pace sparkline, and `history-trends-empty`.

- [x] **Step 3: Update Playwright smoke to stop expecting the removed duplicate cards.**

In `frontend/tests/e2e/visualRefreshSmoke.spec.ts`, keep expectations for:

```ts
await expect(page.getByTestId("history-progress-story")).toBeVisible();
await expect(page.getByTestId("history-progress-story-observed-focus")).toContainText(
  "Add one clearer closing sentence",
);
await expect(page.getByRole("img", { name: "Score trend" })).toBeVisible();
await expect(page.getByRole("img", { name: "Speaking pace trend" })).toBeVisible();
```

Add explicit negative checks only if the removed IDs were previously asserted in the smoke:

```ts
await expect(page.getByTestId("history-priority-latest")).toHaveCount(0);
await expect(page.getByTestId("history-priority-new")).toHaveCount(0);
await expect(page.getByTestId("history-priority-resolved")).toHaveCount(0);
```

- [x] **Step 4: Verify the focused History route test passes.**

Run:

```zsh
npm --prefix frontend test -- src/routes/tests/HistoryRoute.test.tsx
```

Expected: PASS.

### Task 2: Compress HistoryProgressStory For Mobile And Long Copy

**Files:**
- Modify: `frontend/src/components/history/HistoryProgressStory.test.tsx`
- Modify: `frontend/src/components/history/HistoryProgressStory.tsx`
- Modify: `frontend/tests/e2e/visualRefreshSmoke.spec.ts`

- [x] **Step 1: Add component tests for the retained story contract.**

In `frontend/src/components/history/HistoryProgressStory.test.tsx`, add a test that proves the story keeps exactly one top score anchor and one next-practice cue for comparable attempts:

```ts
render(
  <HistoryProgressStory
    records={[
      record({
        finalScore: 3.8,
        scoreLabel: "3.8",
        sessionId: "older",
        timestamp: "2026-06-06T12:00:00Z",
        topPriorities: ["Stay closer to the travel theme"],
      }),
      record({
        finalScore: 4.1,
        scoreLabel: "4.1",
        sessionId: "newer",
        timestamp: "2026-06-07T12:00:00Z",
        topPriorities: ["Add one clearer closing sentence"],
      }),
    ]}
    translate={translate}
  />,
);

expect(screen.getByTestId("history-progress-story-score")).toBeVisible();
expect(screen.getByTestId("history-progress-story-pace")).toBeVisible();
expect(screen.getByTestId("history-progress-story-observed-focus")).toBeVisible();
expect(screen.getByTestId("history-progress-story-no-longer-flagged")).toBeVisible();
expect(screen.getByTestId("history-progress-story-next-practice")).toBeVisible();
```

- [x] **Step 2: Make the story grid resistant to 360px and long translated text.**

In `frontend/src/components/history/HistoryProgressStory.tsx`, change the grid columns from a small 140px auto-fit layout to a wider one-column-first layout:

```ts
const storyGridStyle = {
  display: "grid",
  gap: "0.75rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 220px), 1fr))",
} as const;
```

Keep `overflowWrap: "anywhere"` on values that can contain learner themes or priorities.

- [x] **Step 3: Make next practice visually stronger without adding new copy.**

In `HistoryProgressStory.tsx`, render `history-progress-story-next-practice` as a full-width item when present:

```tsx
{story.nextPractice ? (
  <div
    data-semantic-id="history-progress-story-next-practice"
    data-testid="history-progress-story-next-practice"
    style={{ ...chipStyle, gridColumn: "1 / -1" }}
  >
    <span style={chipLabelStyle}>
      <Icon name="play" size={17} />
      {translate("history.story_next_practice_label")}
    </span>
    <p style={chipValueStyle}>
      {story.nextPractice.focus
        ? translate("history.story_next_practice_focus", {
            focus: story.nextPractice.focus,
            theme: story.nextPractice.theme,
          })
        : translate("history.story_next_practice_repeat", {
            theme: story.nextPractice.theme,
          })}
    </p>
  </div>
) : null}
```

Do not add new translation keys in this task. Reuse `history.story_next_practice_label`, `history.story_next_practice_focus`, and `history.story_next_practice_repeat`.

- [x] **Step 4: Add a 360px mobile guard to the visual smoke helper.**

In `frontend/tests/e2e/visualRefreshSmoke.spec.ts`, add a narrow viewport constant:

```ts
const narrowMobileViewport = { width: 360, height: 780 };
```

After the History mobile checks, set the narrow viewport and verify no horizontal overflow:

```ts
await page.setViewportSize(narrowMobileViewport);
await expect(page.getByTestId("history-progress-story")).toBeVisible();
await expectNoPageHorizontalOverflow(page);
await captureScreenshot(page, "visual-refresh-smoke-history-narrow-mobile.png");
await page.setViewportSize(desktopViewport);
```

- [x] **Step 5: Verify focused History story tests pass.**

Run:

```zsh
npm --prefix frontend test -- src/components/history/HistoryProgressStory.test.tsx src/routes/tests/HistoryRoute.test.tsx
```

Expected: PASS.

## Phase 2: Speak Submit-Moment Consolidation

### Task 3: Move Optional Context Behind Disclosure After Audio Is Attached

**Files:**
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`
- Modify: `frontend/src/components/speak/AssessmentStatusPanel.tsx`
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `locales/en.json`

- [x] **Step 1: Write failing tests for optional context disclosure.**

In `frontend/src/routes/tests/SpeakRoute.test.tsx`, add expectations after uploading audio:

```ts
const statusPanel = await screen.findByTestId("speak.status_panel");
expect(within(statusPanel).getByTestId("speak.submit")).toBeEnabled();
expect(within(statusPanel).getByTestId("speak.optional_context")).toBeInTheDocument();
expect(within(statusPanel).getByText("Add optional context")).toBeVisible();
expect(within(statusPanel).getByTestId("speak.optional_context")).not.toHaveAttribute("open", "");
expect(within(statusPanel).getByTestId("speak.label")).toBeInTheDocument();
expect(within(statusPanel).getByTestId("speak.notes")).toBeInTheDocument();
```

Add a separate idle/no-audio expectation:

```ts
const statusPanel = screen.getByTestId("speak.status_panel");
expect(within(statusPanel).queryByTestId("speak.optional_context")).not.toBeInTheDocument();
expect(within(statusPanel).getByTestId("speak.label")).toBeVisible();
expect(within(statusPanel).getByTestId("speak.notes")).toBeVisible();
```

Run:

```zsh
npm --prefix frontend test -- src/routes/tests/SpeakRoute.test.tsx
```

Expected before implementation: FAIL because `speak.optional_context` does not exist.

- [x] **Step 2: Pass an explicit display mode into `AssessmentStatusPanel`.**

In `frontend/src/components/speak/AssessmentStatusPanel.tsx`, add a prop:

```ts
contextMode: "inline" | "optional";
```

In `frontend/src/routes/SpeakRoute.tsx`, derive it without touching submit eligibility:

```ts
const assessmentContextMode =
  hasAttachment && lifecycleState !== "queued" && lifecycleState !== "running"
    ? "optional"
    : "inline";
```

Pass `contextMode={assessmentContextMode}` to `AssessmentStatusPanel`.

- [x] **Step 3: Render label/notes inline before audio and in a native disclosure after audio.**

In `AssessmentStatusPanel.tsx`, extract the current label/notes controls into a local render helper:

```tsx
const contextFields = (
  <>
    <label htmlFor="speak-label" style={fieldStyle}>
      <span>{translate("speak.label")}</span>
      <input
        id="speak-label"
        type="text"
        value={labelValue}
        onChange={(event) => onLabelChange(event.currentTarget.value)}
        style={controlStyle}
        data-testid="speak.label"
        data-semantic-id="speak.label"
      />
    </label>
    <label htmlFor="speak-notes" style={fieldStyle}>
      <span>{translate("speak.notes")}</span>
      <textarea
        id="speak-notes"
        value={notesValue}
        onChange={(event) => onNotesChange(event.currentTarget.value)}
        style={{
          ...controlStyle,
          minHeight: "132px",
          resize: "vertical",
        }}
        data-testid="speak.notes"
        data-semantic-id="speak.notes"
      />
    </label>
  </>
);
```

Render it as:

```tsx
{contextMode === "optional" ? (
  <details
    data-testid="speak.optional_context"
    data-semantic-id="speak.optional_context"
    style={{ display: "grid", gap: "0.75rem" }}
  >
    <summary style={{ cursor: "pointer", fontWeight: 700, color: "#10201c" }}>
      {translate("speak.optional_context_summary")}
    </summary>
    <div style={{ display: "grid", gap: "0.875rem", paddingTop: "0.75rem" }}>
      {contextFields}
    </div>
  </details>
) : (
  contextFields
)}
```

- [x] **Step 4: Add English copy only after the interaction is stable.**

In `locales/en.json`, add:

```json
"optional_context_summary": "Add optional context"
```

Keep existing `speak.label` and `speak.notes` keys unchanged.

- [x] **Step 5: Verify the focused Speak route test passes.**

Run:

```zsh
npm --prefix frontend test -- src/routes/tests/SpeakRoute.test.tsx
```

Expected: PASS.

### Task 4: Reduce Overlapping Speak Status Copy

**Files:**
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`
- Modify: `frontend/src/components/speak/AssessmentStatusPanel.tsx`
- Modify: `frontend/src/components/speak/RecorderPanel.tsx`
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `locales/en.json`

- [x] **Step 1: Decide the owner of pre-submit reassurance.**

Use this rule:

- `RecorderPanel` owns recording status and saved-take reassurance before submission.
- `AssessmentStatusPanel` owns submit eligibility, queued/running/cancelled/failed/completed assessment status.

That means `speak.recording_ready_checkpoint` remains visible after audio attachment, while `speak.handoff_hint` should not repeat the same "ready to review" message in the idle pre-submit state.

- [x] **Step 2: Add failing tests for one primary status plus one helper line.**

In `SpeakRoute.test.tsx`, after audio attachment and before submit:

```ts
const statusPanel = await screen.findByTestId("speak.status_panel");
expect(within(statusPanel).getByTestId("speak.handoff_hint")).toHaveTextContent(
  "Submit when you are ready for feedback.",
);
expect(screen.getByTestId("speak.recording_ready_checkpoint")).toBeVisible();
```

For queued/running state:

```ts
expect(within(statusPanel).getByTestId("speak.handoff_hint")).toHaveTextContent(
  "Your review is being prepared.",
);
expect(within(statusPanel).queryByText(/auto-refresh/i)).not.toBeInTheDocument();
```

Use localized text from the test translator rather than hardcoding in component code.

- [x] **Step 3: Collapse queued/running helper copy in `AssessmentStatusPanel`.**

Replace the current two queued/running paragraphs:

```tsx
{translate("speak.job_auto_refresh")}
{translate("speak.job_long_running")}
```

with one helper paragraph:

```tsx
{translate("speak.job_review_wait")}
```

Keep it visible only for `queued` or `running`.

- [x] **Step 4: Update English source copy.**

In `locales/en.json`, add or adjust only the affected `speak` keys:

```json
"job_review_wait": "This usually takes a short moment. You can leave this screen open while the review finishes."
```

Do not remove old keys until a follow-up locale cleanup confirms they are unused.

- [x] **Step 5: Verify focused Speak tests pass.**

Run:

```zsh
npm --prefix frontend test -- src/routes/tests/SpeakRoute.test.tsx src/components/speak/SpeakStatusRail.test.tsx
```

Expected: PASS.

### Task 5: Locale Fan-Out For Speak Copy

**Files:**
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`
- Modify: `locales/en.json`

- [x] **Step 1: Add matching keys across all locales.**

Add the same `speak.optional_context_summary` and `speak.job_review_wait` keys to German, Spanish, French, and Italian.

- [x] **Step 2: Keep placeholder parity exact.**

The new keys have no interpolation tokens. If implementation later adds variables, every locale must use identical token names.

- [x] **Step 3: Verify locale parity.**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
```

Expected: PASS.

## Phase 3: Verification And Evidence

### Task 6: Focused Visual And Regression Gate

**Files:**
- Modify: `frontend/tests/e2e/visualRefreshSmoke.spec.ts`
- Modify: `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`
- Modify: `docs/PLAN_STATUS.md`
- Modify: `docs/superpowers/plans/README.md`
- Add or modify: `docs/ux-audit-screenshots/2026-06-09/README.md`

- [x] **Step 1: Run focused unit coverage.**

Run:

```zsh
npm --prefix frontend test -- src/routes/tests/SpeakRoute.test.tsx src/components/speak/SpeakStatusRail.test.tsx src/components/history/HistoryProgressStory.test.tsx src/routes/tests/HistoryRoute.test.tsx
```

Expected: PASS.

- [x] **Step 2: Run the full frontend test suite.**

Run:

```zsh
npm --prefix frontend test
```

Expected: PASS. If the known jsdom line `Not implemented: navigation to another Document` appears without failing tests, record it as non-blocking.

- [x] **Step 3: Run typecheck.**

Run:

```zsh
npm --prefix frontend run typecheck
```

Expected: PASS.

- [x] **Step 4: Run i18n parity.**

Run:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
```

Expected: PASS.

- [x] **Step 5: Run the focused Playwright visual smoke.**

Run from `frontend/`:

```zsh
env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-09 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

Expected: PASS, with refreshed screenshots including `visual-refresh-smoke-speak-ready-mobile.png` and `visual-refresh-smoke-history-narrow-mobile.png`.

- [x] **Step 6: Run diff hygiene.**

Run:

```zsh
git diff --check
```

Expected: no output.

- [x] **Step 7: Update docs only after verification.**

Update `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`, `docs/PLAN_STATUS.md`, `docs/superpowers/plans/README.md`, and `docs/ux-audit-screenshots/2026-06-09/README.md` with the exact command, date, result, and any remaining browser/tooling caveat.

## Execution Notes

- Keep each subtask to five files or fewer.
- Do not edit screens without PAL-reviewed direction; this document is the PAL-reviewed direction for History and Speak only.
- If a browser launch fails again, rerun the smallest relevant Playwright probe and record the exact command, date, and error before marking the visual gate blocked.
- Preserve existing automation semantics. Add semantic IDs only for new controls such as `speak.optional_context`.
- Prefer reducing visible surfaces over adding more explanatory copy.

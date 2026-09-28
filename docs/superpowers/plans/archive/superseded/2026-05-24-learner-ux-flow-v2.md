# Learner UX Flow — Design Critique & V2 Plan

> **Archived on 2026-06-01.** This critique is superseded as an execution plan by
> `docs/superpowers/plans/2026-05-24-learner-ux-flow-refinement.md`.
> Keep it as design rationale only; do not execute its task order directly.

> **Second opinion on the v1 plan.** This document critiques the current design from first principles
> and proposes a more ambitious, engagement-first wave of improvements. It was originally written as
> a complement to `2026-05-24-learner-ux-flow-refinement.md`; the active execution order now lives in
> that refinement plan.

---

## Design Critique: Vostavo (code review, 2026-05-24)

### Overall Impression

The app is coherent and technically sound, but it reads as a local-AI control panel that happens
to have a practice feature attached. The header is dominated by five locale buttons, the sidebar
lists all nine routes at equal visual weight, and the most important action — recording a spoken
response — is buried three navigations deep behind a technically-framed setup flow. Every screen
works, but none of them feel like practising a language.

---

### Usability

| Finding | Severity | Recommendation |
|---------|----------|----------------|
| Nine nav items at equal priority (Home, Runtime Setup, Session Setup, Speak, Review, History, Library, Guide, Settings) | 🔴 Critical | Group into Practice / Progress / Discover with Settings at foot. Demote Runtime Setup entirely. |
| Locale switcher in the global header competes with the app title on every screen | 🔴 Critical | Move to Settings (v1 does this — do it first). |
| Home "Startup checks" section is always fully expanded with technical labels (ffmpeg, Whisper model, Inference runtime) | 🔴 Critical | Collapse to a single system-status chip when all checks pass. Show detail only on failure. |
| "Session Setup" form has no flow continuity into Speak — the learner must navigate manually | 🟡 Moderate | Add an in-page step progress indicator (Step 1 of 3 → 2 of 3 → 3 of 3) spanning Setup → Speak → Review. |
| Speak screen shows `Provider · model · Whisper` metadata during the high-focus recording moment | 🟡 Moderate | Collapse provider metadata into a one-line disclosure toggle. Put the prompt and the Record button front and centre. |
| Review empty state is a dead end ("Submit a speaking attempt before opening the review screen") | 🟡 Moderate | Replace with a CTA that takes the learner directly into the practice flow with one tap. |
| Home "Other surfaces" section with links that duplicate the sidebar | 🟢 Minor | Remove after navigation is grouped; the sidebar becomes sufficient. |
| "Speaker ID" is opaque to learners who just want to practise | 🟢 Minor | Auto-fill from the last session; show it as an editable profile field, not a required form input. |

---

### Visual Hierarchy

- **What draws the eye first**: The five locale toggle buttons in the header — this is wrong. The
  app name or a practice CTA should anchor the first glance.
- **Reading flow**: Header locale → sidebar nine items (flat) → main content. The sidebar list
  gets longer than the main content on small screens.
- **Emphasis**: Diagnostics and practice CTAs receive identical card styling and identical font
  weight. There is no clear primary action.

---

### Consistency

| Element | Issue | Recommendation |
|---------|-------|----------------|
| Tone | Some labels are learner-friendly ("Start new session") and others are deeply technical ("Inference runtime", "Base URL", "raw model IDs") on the same screen | Establish two tiers: learner copy for surface labels, technical copy inside advanced toggles only |
| Empty states | Review, History, Library all have different empty-state patterns (guard message / plain sentence / empty list) | Standardise: headline + one-sentence explanation + single primary CTA |
| Action buttons | `primaryAction` / `secondaryAction` CSS classes exist but are applied inconsistently across routes | Codify: one primary (filled) action per section, secondary actions as text/outline |
| Progress language | Review shows a `progress_delta` section but the History trend charts show different labels for the same concepts | Use one canonical term per metric across both screens |

---

### Accessibility

- **Color contrast**: `#33514b` text on `rgba(255,255,255,0.94)` passes AA. Status colours
  (`#166534` / `#9a6700` / `#b42318`) pass AA at the sizes used. No issues found.
- **Touch targets**: Action row buttons use `0.875rem` padding — borderline for 44 × 44 px
  mobile targets. Should be increased to `1rem` vertical.
- **Locale buttons**: `aria-pressed` is correct. However five buttons in the header with
  no `aria-label` on the group means screen readers announce them without context.
- **Step continuity**: There is no `aria-current="step"` equivalent anywhere in the
  Setup → Speak → Review flow.

---

### What Works Well

- The semantic ID system (`SEMANTIC_IDS`, `semanticAttributes`) is excellent — it keeps tests
  stable as copy changes. Do not touch this.
- Route guard behaviour is solid: learners cannot skip to Speak without setup, cannot open Review
  without a submission. These guards should survive every nav refactor.
- The Review screen's coach summary structure — strengths, priorities, next focus, next exercise —
  is genuinely valuable feedback design. It is under-used: that coach voice should start on the
  Home screen and continue through the entire flow.
- The `progress_delta` section in Review (score delta, new priorities, resolved priorities) is
  unique and compelling. It needs to be surfaced more prominently, not buried after long coaching
  text.
- Multi-locale support is well-structured. Adding engagement copy to all five locales is
  realistic.

---

## What V1 Gets Right (and Where It Stops Short)

V1 correctly diagnoses every structural problem and sequences the fixes well. The ordering
(locale move → Home → navigation → empty states → setup → session → speak → library) is sound.

V1 does **not** address:

1. **Motivation over time** — streaks, personal bests, returning-user recognition.
2. **Flow continuity** — the three practice screens feel like three separate destinations;
   learners navigate between them rather than being carried through them.
3. **Coach voice as a design system** — v1 calls for "coach-style microcopy" but does not
   define what that sounds like or where it appears.
4. **Contextual Home** — a returning user with a high score from yesterday should see a
   different Home than a first-time user. V1 simplifies the Home but does not make it adaptive.
5. **Next-step engine** — the Review screen already produces `answer_next_exercise` data.
   That recommendation should drive what the learner sees when they return to Home, not just
   appear as text at the bottom of a review card.
6. **Speak as a performance space** — the screen that matters most gets the least design
   attention in v1. It needs visual focus, not just copy refinement.

---

## V2 Product Direction

Vostavo should feel like a practice companion that knows you, carries you through each session,
and tells you — in plain, encouraging language — what you just achieved and what to do next.

The four principles that should govern every V2 decision:

1. **One obvious next action at every step.** The learner should never need to look at
   the sidebar to know what to do.
2. **The coach voice is always present.** From the Home greeting to the Review summary,
   the app speaks in one consistent encouraging register.
3. **Progress is visible and rewarding.** History, streaks, personal bests, and resolved
   priorities are surfaces of celebration, not data dumps.
4. **Technical complexity is a door, not a wall.** Setup, provider config, and diagnostics
   stay accessible but never compete with the practice path.

---

## Non-Goals For This Wave

- Do not change runtime detection, Whisper caching, or provider contracts.
- Do not redesign color tokens or the card/button design system.
- Do not add external service dependencies (e.g., remote sync, analytics).
- Do not remove the semantic ID system.
- Do not change route paths or route guard logic.

---

## V2 Task List

### Task A: Define The Coach Voice

**Purpose:** Establish a consistent encouraging tone so that all subsequent copy tasks use
the same register. This is a copy-design task, not a code task. Its output feeds every
other V2 task.

**Deliverable:** A short style guide added to `docs/copy/coach-voice.md`.

Rules:
- Address the learner directly ("You've completed 3 sessions this week.").
- Use action verbs for progress ("You improved your fluency score by 0.4.").
- Make recommendations concrete, not vague ("Try a picture-description task next." not
  "Keep practising.").
- Never use technical terms in primary copy. Reserve them for advanced toggles and
  Settings.
- Empty states always end with a CTA, never a full stop.

---

### Task B: Contextual Home Screen

**Goal:** The Home screen adapts to the learner's state rather than always showing the
same three cards.

**Four Home states to implement:**

| State | Trigger | Primary element | Secondary element |
|-------|---------|----------------|-------------------|
| **First run** | `!runtimeReadiness.ready` | Setup prompt card | System status chip |
| **Ready, no session today** | Runtime ready, `!hasSetupDraft(draft)` | "Start a session" CTA + coach greeting | Last session summary chip |
| **Session in progress** | Runtime ready, `hasSetupDraft(draft)` | "Continue your session" CTA (language + theme) | Start fresh link |
| **Just reviewed** | `hasReviewState(review)` | Review result summary + next-exercise recommendation | Start new session |

The diagnostics section becomes a collapsible "System status" chip: one line showing a
green dot when all checks pass, expandable on click to the full diagnostic list. It is
never the first thing the learner sees.

The "Other surfaces" section is removed. The regrouped sidebar (Task C) makes it
redundant.

**Files:**
- Modify: `frontend/src/routes/HomeRoute.tsx`
- Modify: `frontend/src/routes/HomeRoute.module.css`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `locales/en.json` (add `home.greeting_ready`, `home.greeting_in_progress`,
  `home.greeting_just_reviewed`, `home.system_status_ok`, `home.session_context`)
- Follow-up: remaining four locale files

**Checklist:**
- [ ] Implement the four state conditions in `HomeRoute.tsx`.
- [ ] Build the collapsible system-status chip; default to collapsed when all ok.
- [ ] Display session context in the "in progress" state (learning language, theme title,
  CEFR level) from `useAppStore` draft.
- [ ] Display next-exercise recommendation in the "just reviewed" state from
  `useAppStore` review data (`answer_next_exercise`).
- [ ] Remove the "Other surfaces" card.
- [ ] Preserve all `SEMANTIC_IDS.home.*` attributes.
- [ ] Run `npm --prefix frontend test -- HomeSetupRoutes.test.tsx`.

---

### Task C: Grouped Navigation

**Goal:** The sidebar communicates hierarchy, not a flat alphabetical list. Three visual
groups replace the nine undifferentiated links.

**Proposed groups:**

| Group label | Routes |
|-------------|--------|
| **Practice** | Home · Session Setup · Speak · Review |
| **Progress** | History |
| **Discover** | Library · Scoring Guide |
| *(footer)* | Settings |

Runtime Setup is removed from primary navigation entirely once runtime is configured.
It remains reachable from: Home warning state, Settings, and direct deep links.

The group labels are locale-keyed (`nav.group_practice`, `nav.group_progress`,
`nav.group_discover`) and rendered as non-interactive section headings inside the nav.

**Files:**
- Modify: `frontend/src/components/shell/AppShell.tsx`
- Modify: `frontend/src/components/shell/AppShell.module.css`
- Modify: `frontend/src/App.tsx` (extend `AppRouteDefinition` with optional `navGroup`
  and `isPrimaryNav` flags)
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `locales/en.json` (add `nav.group_practice`, `nav.group_progress`,
  `nav.group_discover`)
- Follow-up: remaining four locale files

**Checklist:**
- [x] Extend `AppRouteDefinition` with grouped-navigation metadata.
- [x] Assign each route a group in `routeDefinitions` in `App.tsx`.
- [x] Mark Runtime Setup as hidden from configured navigation; show it in
  the `'practice'` group only when runtime is missing.
- [x] Render group headings in `AppShell` sidebar as non-interactive labels
  above each group's `NavLink` set.
- [x] Style group headings as uppercase micro-labels with appropriate spacing.
- [x] Preserve `aria-label` on the `<nav>` element.
- [x] Run `npm --prefix frontend test -- HomeSetupRoutes.test.tsx LibraryGuideRoutes.test.tsx`.
- [x] PAL review before merging.

---

### Task D: Practice Flow Step Indicator

**Goal:** Session Setup, Speak, and Review feel like steps in one continuous session,
not three separate pages.

A compact step indicator (three labelled steps, current step highlighted) sits at the
top of each practice route's `<main>` area. It does not replace the page title; it sits
above it as a flow context strip.

Step labels: "1 · Choose a topic", "2 · Speak", "3 · Review".

The indicator is read-only — it shows where the learner is, it is not a navigation
control. Clicking a completed step is intentionally disabled to preserve route guards.

**Files:**
- Add: `frontend/src/components/shell/PracticeStepBar.tsx`
- Add: `frontend/src/components/shell/PracticeStepBar.module.css`
- Modify: `frontend/src/routes/SessionSetupRoute.tsx`
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `frontend/src/routes/ReviewRoute.tsx`
- Modify: `locales/en.json` (add `practice_flow.step_setup`, `practice_flow.step_speak`,
  `practice_flow.step_review`, `practice_flow.step_aria_current`)
- Follow-up: remaining four locale files

**Checklist:**
- [ ] Create `PracticeStepBar` component accepting `currentStep: 1 | 2 | 3`.
- [ ] Use `aria-current="step"` on the active step element.
- [ ] Render the bar at the top of `SessionSetupRoute`, `SpeakRoute`, and `ReviewRoute`.
- [ ] Add a Vitest unit test for `PracticeStepBar` rendering all three states.
- [ ] Run `npm --prefix frontend test -- SessionSetupRoute.test.tsx SpeakRoute.test.tsx ReviewRoute.test.tsx`.

---

### Task E: Speak As A Performance Space

**Goal:** The Speak screen becomes a focused, distraction-free environment. Prompt and
recording action are dominant; everything else is subordinate.

**Specific changes:**

- Move the provider/model/Whisper metadata line into a disclosure toggle labelled
  "Assessment details" — hidden by default.
- Make the prompt card visually the largest element on the screen (larger font, generous
  padding, subtle background distinction).
- Replace the "Assessment" section header with the coach voice: "You're ready to record."
  / "Recording in progress." / "Your recording is ready." depending on recorder state.
- Keep the submit button disabled explanation visible but styled as helper text below the
  button, not as a warning banner.
- Add a lightweight waveform level bar during recording (CSS-only, no external library)
  driven by the existing `MediaRecorder` audio data if the browser supports it; degrade
  gracefully to a simple "Recording... {seconds}s" text if not.

**Files:**
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `frontend/src/components/speak/RecorderPanel.tsx`
- Modify: `frontend/src/components/speak/AssessmentStatusPanel.tsx`
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`
- Modify: `locales/en.json` (add `speak.ready_coach`, `speak.recording_coach`,
  `speak.saved_coach`, `speak.assessment_details_toggle`)
- Follow-up: remaining four locale files

**Checklist:**
- [ ] Wrap provider metadata in a `<details>/<summary>` toggle; closed by default.
- [ ] Increase prompt card font size to at least `1.25rem`; add `1.75rem` top/bottom padding.
- [ ] Replace static "Assessment" heading with a state-driven coach message.
- [ ] Move submit-disabled explanation to a `<p role="status">` below the button.
- [ ] Add waveform bar with `aria-label="Audio level"` and graceful degradation.
- [ ] Run `npm --prefix frontend test -- SpeakRoute.test.tsx`.
- [ ] PAL review before merging.

---

### Task F: Review As A Celebration + Next-Step Surface

**Goal:** The Review screen opens with the result and the coach's top recommendation
front and centre, and closes with a clear invitation to the next session.

**Specific changes:**

- Move the `progress_delta` section to immediately after the band/score row, not below
  coaching text. This is where the learner feels progress.
- Make the `answer_next_exercise` recommendation a prominent CTA button: "Try this next:
  {exercise}" → navigates to Session Setup with the recommended theme pre-selected if
  possible, otherwise just opens Session Setup.
- Style the overall score with a larger numeral and a congratulatory coach line for
  scores above the learner's previous attempt.
- Replace the current "Try again / Change task / Open history" link row at the bottom
  with a single three-button action strip: "Try again" (primary) · "New topic" (secondary)
  · "View history" (text link).

**Files:**
- Modify: `frontend/src/routes/ReviewRoute.tsx`
- Modify: `frontend/src/components/review/ReviewSummary.tsx`
- Modify: `frontend/src/routes/tests/ReviewRoute.test.tsx`
- Modify: `locales/en.json` (add `review.next_exercise_cta`, `review.improvement_coach`,
  `review.action_strip_label`)
- Follow-up: remaining four locale files

**Checklist:**
- [ ] Reorder sections: score/band → progress delta → coach summary → coaching tabs →
  action strip.
- [ ] Implement `next_exercise_cta` button; pass recommended theme to Session Setup via
  router `state` if the theme string matches a known theme key.
- [ ] Add `improvement_coach` copy for positive-delta sessions.
- [ ] Implement three-button action strip; preserve `SEMANTIC_IDS.review.*` attributes.
- [ ] Run `npm --prefix frontend test -- ReviewRoute.test.tsx`.

---

### Task G: History As A Streak And Personal-Best Surface

**Goal:** History opens with a motivational summary (streak, personal best, total
sessions) before the detailed tables and trend charts.

**New "learner card" at the top of `HistoryRoute`:**

```
[  3-day streak  ]  [  Personal best: 4.2  ]  [  12 sessions total  ]
```

Each chip is derived from the existing `diagnosticsQuery` / history data — no new API
endpoints required. The streak is computed client-side: count consecutive days with at
least one saved run up to today.

If the learner has fewer than 2 sessions, the card shows a warm empty state:
"Your progress will appear here after your first session." with a "Start now" CTA.

**Files:**
- Add: `frontend/src/components/history/LearnerProgressCard.tsx`
- Add: `frontend/src/components/history/LearnerProgressCard.module.css`
- Modify: `frontend/src/routes/HistoryRoute.tsx`
- Modify: `frontend/src/routes/tests/HistoryRoute.test.tsx`
- Modify: `locales/en.json` (add `history.streak_days`, `history.personal_best`,
  `history.total_sessions`, `history.empty_coach`)
- Follow-up: remaining four locale files

**Checklist:**
- [ ] Create `LearnerProgressCard` component accepting `streak`, `personalBest`,
  `totalSessions` props.
- [ ] Compute streak and personal best from history rows in `HistoryRoute`.
- [ ] Render card at the top of the route, above trend charts.
- [ ] Render warm empty state when `totalSessions < 2`.
- [ ] Run `npm --prefix frontend test -- HistoryRoute.test.tsx`.

---

### Task H: Learner Profile Reduces Session Setup Friction

**Goal:** Session Setup feels like confirming today's goal, not filling out a form.

Auto-fill every field from the previous session stored in `useAppStore`. Show the current
values as a compact "Last session" summary at the top: "Last time: Spanish · B2 · Travel
narrative". The learner taps "Keep these settings" or edits individual fields.

This does not change any required fields or validation — it only pre-populates defaults.

**Files:**
- Modify: `frontend/src/routes/SessionSetupRoute.tsx`
- Modify: `frontend/src/routes/tests/SessionSetupRoute.test.tsx`
- Modify: `locales/en.json` (add `setup.last_session_summary`, `setup.keep_settings`,
  `setup.change_settings`)
- Follow-up: remaining four locale files

**Checklist:**
- [ ] Read last-used speaker ID, language, level, and theme from `useAppStore` draft or
  the last history row.
- [ ] Pre-populate form fields with those values on mount.
- [ ] Render the "Last session" summary chip if values exist.
- [ ] Run `npm --prefix frontend test -- SessionSetupRoute.test.tsx`.

---

### Task I: Standardise Empty States

**Goal:** Every empty state (Review, History, Library) follows one pattern:
headline → one-sentence explanation → single primary CTA.

| Screen | Headline | Explanation | CTA |
|--------|----------|-------------|-----|
| Review (no attempt) | "Nothing to review yet." | "Complete a speaking session and your results will appear here." | "Start a session" → `/session-setup` |
| History (no runs) | "Your practice history starts here." | "Each completed session adds a row to this page." | "Start your first session" → `/session-setup` |
| Library (no themes) | "No themes are saved yet." | "Add a custom theme or use one of the shipped samples below." | *(samples section acts as CTA)* |

**Files:**
- Modify: `frontend/src/routes/ReviewRoute.tsx`
- Modify: `frontend/src/routes/HistoryRoute.tsx`
- Modify: `frontend/src/routes/LibraryRoute.tsx`
- Modify: `frontend/src/routes/tests/ReviewRoute.test.tsx`
- Modify: `frontend/src/routes/tests/HistoryRoute.test.tsx`
- Modify: `frontend/src/routes/tests/LibraryGuideRoutes.test.tsx`
- Modify: `locales/en.json` (add `review.empty_headline`, `history.empty_headline`,
  `library.empty_headline` and corresponding `_body` keys)
- Follow-up: remaining four locale files

**Checklist:**
- [x] Replace Review and History empty states with the standardised pattern.
- [ ] Replace Library empty state with the standardised pattern.
- [x] Ensure Review and History CTAs preserve route guards by navigating to Session Setup, not directly to Speak.
- [x] Run `npm --prefix frontend test -- ReviewRoute.test.tsx HistoryRoute.test.tsx`.
- [ ] Run `npm --prefix frontend test -- LibraryGuideRoutes.test.tsx` after the Library slice.

---

### Task J: E2E Coverage And Screenshot Regression

**Files:**
- Modify: `frontend/tests/e2e/runtimeSetupLive.spec.ts`
- Modify: `frontend/tests/e2e/reviewHistoryFlow.spec.ts`
- Modify: `frontend/playwright.config.ts`
- Modify: `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`

**Checklist:**
- [ ] Add coverage for: Home four states, grouped nav, Practice Step Bar on all three
  practice routes.
- [ ] Add coverage for: Review celebration + next-step CTA, History learner card.
- [ ] Save screenshots for each new state under `docs/ux-audit-screenshots/2026-05-24-v2/`.
- [ ] Run `./scripts/run_e2e.sh`.

---

## Original Recommended Order (Archived)

This was the proposed order before the 2026-06-01 refinement superseded this
document for execution:

1. **Task A** — Coach voice guide. No code; defines the tone for all copy in Tasks B–I.
2. **Task C** — Grouped navigation. Structural; everything else depends on a cleaner nav.
3. **Task B** — Contextual Home. Highest learner impact; requires v1 Home simplification first.
4. **Task D** — Practice step bar. Low risk; adds orientation without changing any logic.
5. **Task E** — Speak as performance space. PAL review required; highest risk screen.
6. **Task F** — Review celebration + next step. Depends on step bar and coach voice.
7. **Task H** — Session Setup profile pre-fill. Low risk; pure UX convenience.
8. **Task G** — History learner card. Self-contained; can run in parallel with F or H.
9. **Task I** — Standardise empty states. Can run in parallel with any of the above.
10. **Task J** — E2E + screenshot regression. Always last.

---

## Verification Matrix

- Frontend unit tests: `npm --prefix frontend test`
- Frontend typecheck: `npm --prefix frontend run typecheck`
- Browser lane: `./scripts/run_e2e.sh`
- PAL review required before: Task C (navigation semantics), Task E (Speak screen).
- Backend smoke when setup changes: `./scripts/run_tests.sh tests/test_app_backend_config.py tests/test_runtime_status.py`

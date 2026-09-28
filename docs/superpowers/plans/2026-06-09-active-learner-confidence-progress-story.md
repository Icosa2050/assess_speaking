# Active Learner Confidence And Progress Story Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the saved-take moment in Speak feel more reassuring, then turn History's existing progress story into a clearer next-practice prompt without changing backend contracts.

**Architecture:** Keep Review unchanged in this slice because it already has the strongest next-step card, next focus, next exercise, progress delta, and retry/history actions. Add only derived, localized guidance to Speak and History using existing state and history rows.

**Tech Stack:** React 19, TypeScript, Vite, Vitest, Testing Library, existing i18n JSON, existing semantic IDs and CSS patterns.

---

## Status

Implemented and verified on 2026-07-22.

- Speak shows the saved-take checkpoint only for an attached recording before
  assessment begins.
- History derives its next-practice cue from the active learner/language story
  and hides it when there is no comparable prior attempt.
- Locale key and placeholder parity pass across all supported locales.

## PAL Review

PAL reviewed this plan before screen edits. Accepted constraints:

- Do not add another Review card; it would duplicate the existing next-step hierarchy.
- Speak's saved-take checkpoint should show only before submission starts, not during queued/running/failed/cancelled assessment states.
- Keep Speak copy complementary: the recorder owns "take saved/listen if useful"; the assessment panel owns the submit/review handoff.
- History's next-practice cue must use only existing filtered history row data. Do not claim backend `next_focus` or `next_exercise` data on the History list.
- Do not render a generic next-practice cue for a first-ever/no-comparison story.
- Keep the History cue informational, not a CTA button.

## Non-Goals

- No backend, API, assessment, upload, polling, or history persistence changes.
- No new npm dependencies.
- No Review layout/card changes in this slice.
- No gamification mechanics, streaks, badges, XP, avatars, or confetti.
- No hardcoded learner-facing strings.

## File-Based Tasks

### Task 1: Speak Saved-Take Checkpoint

**Files:**
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `frontend/src/components/speak/RecorderPanel.tsx`
- Modify: `locales/en.json`

- [x] Add failing route expectations for `speak.recording_ready_checkpoint` after recording and upload attachment.
- [x] Prove the checkpoint is not shown in idle, assessing, failed, or cancelled states.
- [x] Pass an explicit `showReadyCheckpoint` prop from `SpeakRoute` using saved attachment plus idle pre-submit lifecycle state.
- [x] Render a compact localized checkpoint inside `RecorderPanel`, under the status/preview area, with semantic ID `speak.recording_ready_checkpoint`.
- [x] Keep the checkpoint non-interactive and avoid duplicating the assessment panel's submit/review handoff copy.

### Task 2: History Next-Practice Story Cue

**Files:**
- Modify: `frontend/src/components/history/HistoryProgressStory.test.tsx`
- Modify: `frontend/src/components/history/HistoryProgressStory.tsx`
- Modify: `frontend/src/routes/tests/HistoryRoute.test.tsx`
- Modify: `locales/en.json`

- [x] Add failing component tests for a comparable two-attempt story that renders `history-progress-story-next-practice`.
- [x] Add a failing component test that hides the next-practice cue for a single-attempt story.
- [x] Add a failing route expectation proving the cue uses the current speaker/language filtered story.
- [x] Extend the story model with a next-practice message derived from latest theme plus latest observed focus when available.
- [x] Render a compact informational cue with semantic ID `history-progress-story-next-practice`.

### Task 3: Locale Fan-Out

**Files:**
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`
- Modify: `docs/superpowers/plans/README.md`

- [x] Add matching keys and interpolation tokens across German, Spanish, French, and Italian.
- [x] Update this plan index once verification passes.
- [x] Verify locale key parity and placeholder parity through the existing backend i18n test.

## Verification Commands

Run from repo root with zsh:

```zsh
npm --prefix frontend test -- src/routes/tests/SpeakRoute.test.tsx src/components/history/HistoryProgressStory.test.tsx src/routes/tests/HistoryRoute.test.tsx
npm --prefix frontend test
npm --prefix frontend run typecheck
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
git diff --check
```

If browser screenshot smoke is attempted and fails, record the exact command, date, and error before claiming a blocker.

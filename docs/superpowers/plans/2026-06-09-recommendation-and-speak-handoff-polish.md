# Recommendation And Speak Handoff Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the recommended setup path feel learner-aware and make the Speak submit/assessment handoff feel more reassuring without new backend contracts.

**Architecture:** Reuse existing frontend state, `/v1/history`, `queryKeys.history`, and localized copy. Session Setup may personalize language/theme from recent history for the entered `speakerId`, but it must keep B1/90 seconds because history rows do not carry prior CEFR or duration. Speak gets one inline handoff hint inside the existing assessment status area instead of another card.

**Tech Stack:** React 19, React Query, Zustand app store, existing i18n JSON, Vitest, Testing Library, Playwright smoke spec touch only where needed.

---

## Status

Implemented on 2026-06-09.

- Session Setup now uses recent same-learner history, when available, to make the recommended starter feel less generic while keeping B1/90 seconds explicit.
- Speak now shows an inline handoff hint inside the existing assessment panel for idle, attached, assessing, failed, and cancelled states.
- Verified passing: focused Session Setup and Speak route tests, full frontend tests, frontend typecheck, backend i18n parity, and `git diff --check`.

## PAL Review

PAL reviewed the slice before screen edits. Accepted refinements:

- Do not claim full personalization from history because `HistoryRow` has no CEFR or duration.
- Match history only by the setup nickname/`speakerId` field, with normalization and a recency window.
- Use separate copy for exact theme continuity versus language-only continuity.
- During pending/error/empty history states, keep the generic recommendation hint.
- Avoid a third Speak state card; render the handoff hint inline with `AssessmentStatusPanel`.
- Keep failed and cancelled handoff copy distinct because one is system-initiated and one is learner-initiated.

## File-Bounded Tasks

### Task 1: Session Setup History-Aware Recommendation

**Files:**
- Modify: `frontend/src/routes/tests/SessionSetupRoute.test.tsx`
- Modify: `frontend/src/routes/SessionSetupRoute.tsx`
- Modify: `frontend/src/components/setup/ThemeForm.tsx`
- Modify: `locales/en.json`

- [x] Add a failing route test where `apiClient.getHistory()` returns a recent row for the entered learner name and the recommended CTA selects that row's language and B1 theme.
- [x] Add a failing route test where history is stale or missing and the existing generic Italian B1/90s starter remains unchanged.
- [x] Add `getHistory` to the Session Setup route mock and default it to `{ items: [] }`.
- [x] Query history with `queryKeys.history`, but render generic copy while loading, on error, or when no recent matching row exists.
- [x] Match history rows by normalized `speaker_id` and `speakerId`, only use rows from the last 60 days, and only use languages present in the current theme library.
- [x] If the recent row's theme exists in B1 for that language, recommend that exact B1 theme; otherwise recommend the first B1 theme for that language.
- [x] Render a localized `setup.recommendation_hint` semantic element near `setup.recommended_start`.

### Task 2: Speak Inline Handoff Hint

**Files:**
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `frontend/src/components/speak/AssessmentStatusPanel.tsx`
- Modify: `locales/en.json`

- [x] Add failing route expectations for `speak.handoff_hint` in idle, attached, assessing, failed, and cancelled states.
- [x] Derive one localized handoff hint from existing `hasAttachment`, `isSubmitting`, and assessment lifecycle state.
- [x] Pass the hint into `AssessmentStatusPanel` and render it inside the existing status text region with semantic ID `speak.handoff_hint`.
- [x] Keep the hint non-interactive and avoid duplicating the status rail step labels.

### Task 3: Locale Fan-Out And Evidence Docs

**Files:**
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`
- Modify: `docs/superpowers/plans/README.md`

- [x] Add translated keys for the new setup recommendation hints and Speak handoff hints.
- [x] Preserve interpolation tokens across all locales.
- [x] Update the plan index to mark this slice implemented after verification.

## Verification Commands

Run from repo root with zsh:

```zsh
npm --prefix frontend test -- src/routes/tests/SessionSetupRoute.test.tsx src/routes/tests/SpeakRoute.test.tsx
npm --prefix frontend test
npm --prefix frontend run typecheck
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
git diff --check
```

Focused visual smoke remains blocked in this sandbox until Chromium/Chrome can launch; do not claim screenshot refresh for this slice unless the Playwright command actually reaches app assertions.

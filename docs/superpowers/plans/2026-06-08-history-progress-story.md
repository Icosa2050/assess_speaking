# History Progress Story Plan

Date: 2026-06-08

## Goal

Turn History from a report browser into a learner-facing progress story without
changing scoring, backend history data, Review behavior, navigation, or runtime
setup.

This is the first taste/product-design refinement after the completed visual
foundation pass. The target is the weakest surface identified in
`docs/ux-audit-screenshots/2026-06-05/visual-polish-follow-up-assessment.md`.

## Status

Implemented on 2026-06-08 with the conservative product-tone choices:

PAL reviewed the proposed slice on 2026-06-08. The implementation shape is sound,
but two copy/product claims are not safe to decide silently:

- whether the latest top priority should be framed as a directive "next focus"
  or as a softer observation from the latest attempt;
- whether a disappeared priority should be called "resolved" or the more precise
  "no longer flagged".

The implemented choices are:

- use "Noticed last time" instead of a directive "next focus";
- use "No longer flagged" instead of "resolved";
- keep opened-report collapse out of this slice.

## Inspiration Used

- Babbel's product-design writeup is the best fit for tone: progress and
  activity scaffolds should build confidence and reinforce motivation without
  becoming clutter.
  <https://www.babbel.com/en/magazine/flex-in-real-life>
- Duolingo's path research supports clearer guidance and a visible next action,
  while its self-efficacy findings are a useful guardrail for History: the page
  should make improvement feel credible.
  <https://duolingo-papers.s3.amazonaws.com/reports/Duolingo_whitepaper_language_read_listen_write_speak_2024.pdf>
- The HCI gamification-misuse paper is a negative constraint: do not add points,
  badges, streaks, leaderboards, or reward language that distracts from learning.
  <https://arxiv.org/abs/2203.16175>

## Design Direction

Add a compact `HistoryProgressStory` section near the top of History, after the
language filter and before aggregate metric cards.

The story section should use only existing filtered history rows:

- latest score and previous comparable score;
- score delta, with a neutral "steady" state for small changes;
- latest WPM and previous comparable WPM;
- WPM delta, with a neutral "similar pace" state for small changes;
- latest top priority, framed according to the user-confirmed tone;
- disappeared priority, only when the latest and previous attempts share the same
  task family/theme and the user-confirmed label is available;
- latest language, theme, and timestamp.

The section should feel like an adult coach:

- quiet score/delta anchor, not a celebration badge;
- compact evidence chips for pace, observed focus, and no-longer-flagged focus;
- accessible text equivalents for all delta states;
- no new dependency and no new charting library;
- no gamified rewards, streaks, XP, confetti, or leaderboards.

## Derivation Rules

The implementation must not rely on backend row order.

1. Start with the already speaker/language-filtered `HistoryViewRecord[]`.
2. Sort a copy by parsed timestamp descending, with original index as a stable
   fallback for equal or invalid timestamps.
3. `latest` is the first sorted row.
4. `previous` is the next sorted row in the same filtered set.
5. Score delta is meaningful only when both values are finite and
   `Math.abs(delta) >= 0.05`; otherwise show a neutral steady state.
6. WPM delta is meaningful only when both values are finite and
   `Math.abs(delta) >= 5`; otherwise show a neutral similar-pace state.
7. A disappeared priority may be shown only when `latest` and `previous` share
   the same `taskFamily` and normalized `theme`, and a previous top priority is
   absent from the latest top priorities.
8. One-attempt histories still render the story section with latest attempt
   context and neutral/no-comparison states.

## File-Based Tasks

### Task 1: Progress Story Component And Derivation

Files:

- Add `frontend/src/components/history/HistoryProgressStory.tsx`
- Add `frontend/src/components/history/HistoryProgressStory.test.tsx`
- Modify `frontend/src/components/ui/Icon.tsx` only if an existing icon is
  insufficient

TDD steps:

- Write failing tests for explicit timestamp sorting, one-attempt fallback,
  score steady threshold, WPM steady threshold, same-theme disappeared priority,
  and theme-mismatch suppression.
- Implement a pure derivation helper co-located with the component.
- Render an accessible `section` with `data-testid` and `data-semantic-id`
  values:
  - `history-progress-story`
  - `history-progress-story-score`
  - `history-progress-story-pace`
  - `history-progress-story-observed-focus`
  - `history-progress-story-no-longer-flagged`
- Use existing tokens and inline style patterns already used by History.

Acceptance:

- The component does not mutate input rows.
- Delta text is not color-only.
- One-attempt and no-comparison states do not remove the whole story section.

### Task 2: Localized Copy

Files:

- Modify `locales/en.json`
- Modify `locales/de.json`
- Modify `locales/es.json`
- Modify `locales/fr.json`
- Modify `locales/it.json`

TDD steps:

- Add only keys needed by `HistoryProgressStory`.
- Keep placeholder names identical across all five locales.
- Run locale parity and placeholder verification through the existing backend
  i18n test.

Acceptance:

- No hardcoded learner-facing strings are added to React components.
- Existing History and Review keys are reused only where the phrase structure is
  already correct; do not compose awkward sentence fragments to avoid adding a
  key.

### Task 3: History Route Integration

Files:

- Modify `frontend/src/routes/HistoryRoute.tsx`
- Modify `frontend/src/routes/tests/HistoryRoute.test.tsx`

TDD steps:

- Add a failing route test proving the story uses the current speaker/language
  filtered set.
- Add a failing route test proving the story recomputes when the language filter
  changes.
- Integrate `HistoryProgressStory` above metric cards while preserving all
  existing History semantic IDs and attempt/detail behavior.

Acceptance:

- Existing History route tests still pass.
- The story does not alter selected attempt behavior.
- The story does not trigger extra history-detail fetches.

### Task 4: Visual Smoke And Evidence

Files:

- Modify `frontend/tests/e2e/visualRefreshSmoke.spec.ts`
- Modify `docs/ux-audit-screenshots/2026-06-05/README.md`

Steps:

- Assert `history-progress-story` is visible after navigating from Review to
  History.
- Assert the story contains the filtered latest score context and the chosen
  focus/no-longer-flagged copy.
- Keep the 390px mobile overflow check.
- Refresh desktop and mobile History screenshots.

Acceptance:

- Mobile 390x844 first viewport shows the story and at least the start of the
  aggregate metric area.
- No visible serif typography returns.
- Screenshot evidence is documented.

## Non-Goals

- No changes to backend scoring, history persistence, or API contracts.
- No changes to `HistoryDetailPanel` or `ReviewSummary` in this slice.
- No collapse/restructure of the opened report yet.
- No global token/theme pass.
- No new npm packages.
- No gamification mechanics.
- No change to History filter semantics beyond explicitly sorting the story
  derivation input copy.

## Verification

Run, in order:

```zsh
npm --prefix frontend test -- HistoryProgressStory
npm --prefix frontend test -- HistoryRoute
npm --prefix frontend run typecheck
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py -q
cd frontend
env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-05 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
cd ..
git diff --check
```

If browser smoke fails, record the exact command, date, and error before
claiming a blocker.

## Decision Gate

Resolved on 2026-06-08:

1. Latest priority is framed as the softer "noticed last time" observation.
2. Disappeared priorities are labeled "no longer flagged".
3. This slice only adds the progress story; opened-report collapse remains a
   later slice.

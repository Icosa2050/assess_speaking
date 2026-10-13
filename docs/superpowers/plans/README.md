# Superpowers Plan Index

Last updated: 2026-10-06

This directory now keeps only current execution plans at the top level. Completed
or superseded task plans live under `archive/` so they do not look like the next
thing to implement.

See `docs/PLAN_STATUS.md` for the wider repository plan map.

## Current

| Plan | Status | Notes |
|---|---|---|
| [2026-10-06-cloud-gap-remediation.md](2026-10-06-cloud-gap-remediation.md) | Implemented offline; internal DMG accepted | Three production fixes, account/UI recovery, real process fixtures and corrected ASR CI discovery implemented. 1,103 backend, 14 cloud process, 12 real-ASR, 182 frontend and 26 browser cases pass; fresh internal DMG accepted. Remote/live/native/public-release gates remain separate. |
| [2026-10-04-cloud-provider-access.md](2026-10-04-cloud-provider-access.md) | Implemented baseline; October 6 repair slice accepted | Baseline plus reviewed October 6 repairs are implemented; use the cloud-gap implementation evidence for current acceptance and remaining live/quality/release gates. |
| `2026-06-09-learner-confidence-simplification.md` | Implemented simplification slice | PAL-refined slice: History duplicate priority reporting removed, History narrow-mobile story hardened, Speak optional context disclosure added, and overlapping assessment wait copy consolidated. |
| `2026-06-09-active-learner-confidence-progress-story.md` | Implemented confidence slice | PAL-reviewed slice: saved-take reassurance in Speak plus an honest History next-practice cue derived from filtered progress rows. Review stays unchanged to avoid duplicating the existing next-step card. |
| `2026-06-09-recommendation-and-speak-handoff-polish.md` | Implemented product-polish slice | PAL-reviewed slice: recent same-learner history informs the Session Setup recommended starter when safe, and Speak now has an inline assessment handoff hint without adding another card or backend contract. |
| `2026-06-09-session-setup-newbie-wizard.md` | Implemented Session Setup slice | PAL-reviewed Session Setup slice: two-step beginner setup frame, recommended starter path, advanced custom topic disclosure, runtime handoff callout, locale fan-out, focused coverage, and recovered mobile screenshot evidence. |
| `2026-06-09-visual-smoke-speak-confidence.md` | Implemented visual/testing slice | PAL-reviewed slice: hardened focused visual-smoke coverage with failed-gate Review and all-key-route 390px overflow checks, plus a bounded Speak confidence rail and copy pass. |
| `2026-06-08-history-progress-story.md` | Implemented visual slice | PAL-reviewed History polish: compact progress story, conservative "noticed last time" / "no longer flagged" copy, locale parity, route/unit coverage, and refreshed focused smoke screenshots. |
| `2026-06-05-runtime-setup-onboarding-readiness.md` | Implemented visual slice | Focused PAL/Stitch-informed setup onboarding pass: split readiness anchor, progress ring, localized microcopy, mobile overflow fix, and 2026-06-05 screenshot evidence. |
| `2026-05-24-learner-ux-flow-refinement.md` | Implemented UX contract | The 2026-06-01 learner-flow refinement is implemented through the code and focused browser-flow slices. Full Playwright execution remains an environment-sensitive stabilization gate tracked in `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`. |
| `2026-05-16-streamlit-retirement.md` | Completed contract | Kept at the top level because other roadmap docs still link to it as the Streamlit removal contract. It is no longer an open execution checklist. |

## Archived Completed

| Plan | Reason |
|---|---|
| `archive/completed/2026-06-05-enticing-learner-visual-refresh.md` | Visual foundation landed: global typography/tokens, primitives, Home/Speak/Review/History visuals, locale verification, focused Playwright smoke, and screenshot evidence. Follow-up taste assessment lives in `docs/ux-audit-screenshots/2026-06-05/visual-polish-follow-up-assessment.md`. |
| `archive/completed/2026-05-10-coderabbit-core-remediation.md` | Core remediation landed; the original checklist was not backfilled after the later `app_core`/Streamlit migration. |
| `archive/completed/2026-05-17-baseline-metric-semantics.md` | Baseline metric semantics landed and focused baseline tests pass. |
| `archive/completed/2026-05-19-runtime-setup-streamline.md` | Runtime setup streamlining landed and focused route tests pass. |

## Archived Superseded

| Plan | Reason |
|---|---|
| `archive/superseded/2026-05-24-learner-ux-flow-v2.md` | Archived as design rationale only; superseded for execution by the 2026-06-01 refinement in `2026-05-24-learner-ux-flow-refinement.md`. |
| `archive/superseded/2026-05-19-streamlit-removal-completion-concept.md` | Superseded by the completed `2026-05-16-streamlit-retirement.md` contract and the current `app_core` implementation. |

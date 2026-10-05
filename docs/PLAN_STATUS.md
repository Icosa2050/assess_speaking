# Plan Status

Last updated: 2026-10-04

This file distinguishes living roadmap documents from completed task plans. The
rule of thumb is:

1. `docs/superpowers/plans/` top level is for current executable plans.
2. `docs/superpowers/plans/archive/completed/` is historical evidence.
3. `docs/superpowers/plans/archive/superseded/` is stale planning that should
   not be executed.
4. Top-level `docs/*PLAN*.md` files are usually strategy or roadmap documents,
   so they are marked here instead of moved unless they are clearly obsolete.

## Implemented Or Completed

| Plan | Classification | Evidence / next handling |
|---|---|---|
| `docs/superpowers/plans/2026-05-16-streamlit-retirement.md` | Completed contract | All checkboxes are closed. Kept in place because several docs link to it. Current guard tests passed: `tests/test_streamlit_removal_contract.py`, `tests/test_app_core_imports.py`, and `tests/test_run_app.py`. |
| `docs/superpowers/plans/archive/completed/2026-05-17-baseline-metric-semantics.md` | Completed | Archived. Focused `ParsingAndBaselineTests` passed. |
| `docs/superpowers/plans/archive/completed/2026-05-19-runtime-setup-streamline.md` | Completed | Archived. Focused `HomeSetupRoutes` and `SettingsRoute` tests passed. |
| `docs/superpowers/plans/archive/completed/2026-06-05-enticing-learner-visual-refresh.md` | Completed visual foundation | Archived. Landed global typography/tokens, zero-dependency visual primitives, Home/Speak/Review/History visual artifacts, Runtime Setup onboarding readiness, locale parity verification, focused Playwright visual smoke, and desktop/mobile evidence under `docs/ux-audit-screenshots/2026-06-05/`. Next routing: use `docs/ux-audit-screenshots/2026-06-05/visual-polish-follow-up-assessment.md` for taste/product-design refinement. |
| `docs/superpowers/plans/archive/completed/2026-05-10-coderabbit-core-remediation.md` | Completed, stale checklist | Archived. The original checkboxes were never backfilled, but focused backend/runtime remediation tests passed after the later `app_core` migration. |
| `docs/superpowers/plans/2026-06-09-active-learner-confidence-progress-story.md` | Implemented confidence slice | Saved-take reassurance and the filtered History next-practice cue are implemented. Focused Speak/History tests and locale parity pass; Review intentionally remains unchanged. |
| `docs/IMPLEMENTATION_PLAN.md` | Implemented baseline plus living roadmap | Assessment core, local FastAPI baseline, React/Tauri direction, and Streamlit retirement are implemented. Keep as the high-level product roadmap. |
| `docs/LOCAL_BACKEND_ARCHITECTURE.md` | Implemented baseline plus living architecture | Local backend baseline and `app_core`/React/Tauri direction are current. Keep as architecture reference. |

## Partially Implemented / Still Useful

| Plan | Classification | Open handling |
|---|---|---|
| `docs/superpowers/plans/2026-06-09-learner-confidence-simplification.md` | Implemented simplification slice | PAL-refined slice completed on 2026-06-09. Evidence: focused Speak/History tests, full frontend tests, frontend typecheck, backend i18n parity, focused visual-smoke command with 390px route checks, attached-audio Speak mobile coverage, 360px History overflow coverage, refreshed screenshots under `docs/ux-audit-screenshots/2026-06-09/`, and `git diff --check` passed. |
| `docs/superpowers/plans/2026-06-09-recommendation-and-speak-handoff-polish.md` | Implemented product-polish slice | PAL-reviewed slice completed on 2026-06-09. Evidence: focused Session Setup/Speak route tests, full frontend tests, frontend typecheck, backend i18n parity, and `git diff --check` passed. |
| `docs/superpowers/plans/2026-06-09-session-setup-newbie-wizard.md` | Implemented Session Setup slice | PAL-reviewed two-step beginner setup frame, recommended starter path, advanced custom-topic disclosure, runtime handoff callout, locale fan-out, focused route coverage, full frontend tests, typecheck, backend i18n parity, localhost backend/frontend HTTP smoke, and mobile evidence at `docs/ux-audit-screenshots/2026-06-09/visual-refresh-smoke-session-setup-mobile.png`. |
| `docs/superpowers/plans/2026-06-09-visual-smoke-speak-confidence.md` | Implemented visual/testing slice | PAL-reviewed slice completed on 2026-06-09. Evidence: focused visual-smoke command passed with failed-gate Review coverage, all-key-route 390px screenshot/overflow checks, Speak confidence rail tests, full frontend tests, typecheck, and backend i18n parity. |
| `docs/superpowers/plans/2026-06-08-history-progress-story.md` | Implemented visual slice | PAL-reviewed History polish completed with conservative "noticed last time" / "no longer flagged" copy, localized story strings, focused route/component tests, opened-report disclosure, and refreshed visual smoke screenshots. |
| `docs/superpowers/plans/2026-06-05-runtime-setup-onboarding-readiness.md` | Implemented visual slice | PAL/Stitch-informed Runtime Setup onboarding pass completed on 2026-06-05. Evidence: focused setup tests, full frontend tests, typecheck, backend i18n test, localhost backend/frontend smoke, and desktop/mobile screenshots under `docs/ux-audit-screenshots/2026-06-05/`. |
| `docs/superpowers/plans/2026-05-24-learner-ux-flow-refinement.md` | Implemented learner UX contract | Home/shell cleanup, Settings language ownership, grouped navigation, Setup Guide readiness, Session Setup goal framing, Speak hierarchy, Review/History next-step states, Library/Guide framing, focused browser-flow coverage, and screenshot evidence are implemented. Full Playwright execution remains a separate environment-sensitive gate. |
| `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md` | Implemented focused flows; runner recheck blocked | Coverage includes review/history, live runtime setup, real-audio replacement, Home-to-Session-Setup-to-Speak, empty states, visual smoke, Library/Guide, and Review change-task behavior. Eleven tests in nine files collect on 2026-07-22, but Chromium is denied macOS Mach bootstrap access before app assertions in the current sandbox; keep the full-run gate open until it can run outside that restriction. |
| `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md` | Partially implemented | Runtime setup simplification landed; broader Speak, Review, History, Settings UX work remains a living lane. |
| `docs/SUPPORT_MAINTENANCE_PLAN.md` | Partially implemented / policy reference | Support bundles, cleanup, and diagnostics exist in code, but the plan still needs an `app_core`/React wording refresh before it can be treated as closed. |
| `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md` | Partially implemented / stale file map | Much of the functionality exists, but file references still name `app_shell` and Streamlit-era surfaces. Do not execute literally until rebased. |
| `docs/SAVED_CONNECTIONS_PRODUCTION_CREDENTIAL_PLAN.md` | Mostly implemented policy | Saved-secret behavior is implemented enough for current runtime flows; keep as credential policy until the doc is rebased from `app_shell` to `app_core`/React wording. |
| `docs/REPO_CLEANUP_PLAN.md` | Active cleanup roadmap | Root/package cleanup remains open. Do not archive. |
| `docs/CODERABBIT_REVIEW_REMEDIATION_PLAN.md` | Broad audit artifact | The core remediation plan is archived as completed, but this broader CodeRabbit coverage doc still records review scopes and should be closed only after a deliberate audit decision. |

## Living Product Roadmaps

| Plan | Classification | Notes |
|---|---|---|
| `docs/superpowers/plans/2026-10-04-cloud-provider-access.md` | Implemented; live authorization pending | ChatGPT browser sign-in and xAI Grok API connection; existing OpenRouter integration retained. Local regression and English/Italian connection journeys pass. |
| `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md` | Approved future roadmap | Local React/Tauri baseline is current; hosted persistence/auth is not implemented yet. |
| `docs/superpowers/specs/2026-07-22-signed-macos-dmg-design.md` | Approved packaging design, not yet executed | Developer ID-signed, notarized, stapled arm64 DMG with a PyInstaller onedir helper, packaged-PyAV ffmpeg removal, per-launch loopback auth token, and an inside-out signing pipeline. First entry under the new `docs/superpowers/specs/` directory for design specs that precede an execution plan. |
| `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md` | Supporting UX roadmap | Keep active, but execute only with PAL review for screen changes. |
| `docs/PLANNING_ALIGNMENT_META_PLAN.md` | Historical alignment map | Useful for lineage, but Streamlit-retirement sequencing is now stale. Prefer this status file plus current roadmap docs for execution decisions. |

## Deferred Or Exploratory

| Plan | Classification | Notes |
|---|---|---|
| `docs/PUBLIC_INFERENCE_ARCHITECTURE_PLAN.md` | Exploratory / deferred | Separate beta service idea; not part of current desktop baseline. |
| `docs/MOBILE_COMPANION_STRATEGY.md` | Deferred | Backend contract is in place, mobile implementation is not active. |
| `docs/MULTILINGUAL_CEFR_ASSESSMENT_PLAN.md` | Research-backed roadmap | Keep as language/CEFR expansion reference, not closed. |
| `docs/SPOKEN_CORPUS_CATEGORIZATION_PLAN.md` | Detailed proposal | Data/corpus lane remains separate and not closed. |
| `docs/SYNTHETIC_BENCHMARK_AUTOMATION.md` | Starting-point plan | Benchmark automation remains useful future work. |
| `docs/SHARED_FRONTEND_PARITY_INVENTORY.md` | Historical parity inventory | Keep as reference for Streamlit-to-React parity, not as an executable current plan. |
| `docs/RECORDER_UX_WIREFRAME.md` | Design artifact | Not an implementation plan by itself. |
| `docs/COACHING_BACKLOG.md` | Backlog with implemented milestones | Keep as backlog/history rather than archive wholesale. |

## Superseded

| Plan | Classification | Replacement |
|---|---|---|
| `docs/superpowers/plans/archive/superseded/2026-05-24-learner-ux-flow-v2.md` | Superseded design critique / backlog | Archived as rationale only. Execute `docs/superpowers/plans/2026-05-24-learner-ux-flow-refinement.md` instead. |
| `docs/superpowers/plans/archive/superseded/2026-05-19-streamlit-removal-completion-concept.md` | Superseded | Replaced by the completed `docs/superpowers/plans/2026-05-16-streamlit-retirement.md` contract and current `app_core` implementation. |

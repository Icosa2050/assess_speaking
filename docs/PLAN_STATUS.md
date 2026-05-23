# Plan Status

Last updated: 2026-05-20

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
| `docs/superpowers/plans/archive/completed/2026-05-10-coderabbit-core-remediation.md` | Completed, stale checklist | Archived. The original checkboxes were never backfilled, but focused backend/runtime remediation tests passed after the later `app_core` migration. |
| `docs/IMPLEMENTATION_PLAN.md` | Implemented baseline plus living roadmap | Assessment core, local FastAPI baseline, React/Tauri direction, and Streamlit retirement are implemented. Keep as the high-level product roadmap. |
| `docs/LOCAL_BACKEND_ARCHITECTURE.md` | Implemented baseline plus living architecture | Local backend baseline and `app_core`/React/Tauri direction are current. Keep as architecture reference. |

## Partially Implemented / Still Useful

| Plan | Classification | Open handling |
|---|---|---|
| `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md` | Partially implemented | The review/history, live runtime setup, and real-audio replacement specs exist. Broader browser coverage and a permitted full Playwright run remain open. |
| `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md` | Partially implemented | Runtime setup simplification landed; broader Speak, Review, History, Settings UX work remains a living lane. |
| `docs/SUPPORT_MAINTENANCE_PLAN.md` | Partially implemented / policy reference | Support bundles, cleanup, and diagnostics exist in code, but the plan still needs an `app_core`/React wording refresh before it can be treated as closed. |
| `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md` | Partially implemented / stale file map | Much of the functionality exists, but file references still name `app_shell` and Streamlit-era surfaces. Do not execute literally until rebased. |
| `docs/SAVED_CONNECTIONS_PRODUCTION_CREDENTIAL_PLAN.md` | Mostly implemented policy | Saved-secret behavior is implemented enough for current runtime flows; keep as credential policy until the doc is rebased from `app_shell` to `app_core`/React wording. |
| `docs/REPO_CLEANUP_PLAN.md` | Active cleanup roadmap | Root/package cleanup remains open. Do not archive. |
| `docs/CODERABBIT_REVIEW_REMEDIATION_PLAN.md` | Broad audit artifact | The core remediation plan is archived as completed, but this broader CodeRabbit coverage doc still records review scopes and should be closed only after a deliberate audit decision. |

## Living Product Roadmaps

| Plan | Classification | Notes |
|---|---|---|
| `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md` | Approved future roadmap | Local React/Tauri baseline is current; hosted persistence/auth is not implemented yet. |
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
| `docs/superpowers/plans/archive/superseded/2026-05-19-streamlit-removal-completion-concept.md` | Superseded | Replaced by the completed `docs/superpowers/plans/2026-05-16-streamlit-retirement.md` contract and current `app_core` implementation. |

# Current App Surface And Deprecation Map

Last updated: 2026-03-22

## Summary

The repository has moved beyond the old "single long dashboard" shape.

The current product-facing surface is now the multipage Streamlit shell:

- `streamlit_app.py`
- `pages/00_Setup.py`
- `pages/01_Session_Setup.py`
- `pages/02_Speak.py`
- `pages/03_Review.py`
- `pages/04_History.py`
- `pages/05_Library.py`
- `pages/06_Settings.py`
- `pages/07_Scoring_Guide.py`

The shell is backed by:

- shared state in `app_shell/state.py`
- navigation and bootstrapping in `app_shell/page_helpers.py`
- localization in `app_shell/i18n.py` and `locales/*.json`
- visual styling in `app_shell/visual_system.py`
- review rendering in `app_shell/review_components.py`
- runtime/provider setup in `app_shell/services.py`, `app_shell/runtime_resolver.py`, `app_shell/runtime_providers.py`, and `app_shell/secret_store.py`

The Stitch redesign is visible in both the current shell styling and the checked-in reference artifacts under `output/stitch/`.

## Current Repo Shape

### 1. Assessment Core

These files remain the core execution path and are not legacy:

- `assess_speaking.py`
- `asr.py`
- `audio_features.py`
- `metrics.py`
- `scoring.py`
- `llm_client.py`
- `schemas.py`
- `settings.py`

This layer still matters even as the app shell evolves, because both the shell and compatibility surfaces depend on it.

### 2. Primary Product UI

The primary UI path is the app shell:

- entry: `streamlit_app.py`
- shared shell code: `app_shell/`
- page implementations: `pages/`
- locale strings: `locales/`

This is the source of truth for future UX, layout, and navigation work.

### 3. UX Design References

The redesign direction is documented and illustrated in:

- `output/stitch/`
- `docs/DASHBOARD_UX_WIREFRAME.md`
- `docs/RECORDER_UX_WIREFRAME.md`
- `docs/MULTIPAGE_APP_SHELL_PLAN.md`

These files are now reference material for the shell rather than separate design exercises.

### 4. Compatibility And Legacy Surfaces

The old monolithic dashboard still exists:

- `scripts/interactive_dashboard.py`
- `tests/e2e/test_streamlit_e2e.py`
- `tests/e2e/test_streamlit_real_assessment_e2e.py`

This surface still works and still has test coverage, but it should be treated as a compatibility path rather than the primary UX.

### 5. Supporting Surfaces

These are still active, but they are not the same thing as the new shell:

- `scripts/progress_dashboard.py`
- `service/`
- CLI and automation scripts under `scripts/`

They should not be archived just because the old dashboard is eventually archived.

## Source Of Truth Going Forward

For product work, layout changes, flow changes, or shell behavior changes, treat these as canonical in this order:

1. `streamlit_app.py`
2. `pages/`
3. `app_shell/`
4. `locales/`
5. `output/stitch/` plus the UX wireframes

Treat these as compatibility or infrastructure, not the main UX source:

- `scripts/interactive_dashboard.py`
- older dashboard-specific E2E tests

## What Changed In Practice

The repo is no longer organized around one big Streamlit tools page.

The effective product model is now:

1. runtime setup
2. session setup
3. speak
4. review
5. history and library as supporting screens
6. settings as an advanced/runtime screen

This is a different mental model from the older dashboard, where setup, recording, history, prompts, progress, and settings lived together in one screen.

## Deprecation Candidates

### Candidate A: `scripts/interactive_dashboard.py`

Status:
- keep for now

Why it still exists:
- compatibility for the earlier all-in-one workflow
- existing tests still depend on it
- some product areas still exist there that the shell may not yet fully replace

Deprecation posture:
- no new feature work unless it is a bug fix, migration aid, or parity blocker
- add explicit legacy/deprecation messaging
- update README so it is not presented as the main app

Archive trigger:
- the multipage shell covers the remaining must-have workflows
- shell E2E coverage is accepted as the primary browser path
- team agrees the monolithic dashboard is no longer a supported entry point

### Candidate B: Legacy Dashboard E2E Coverage

Files:
- `tests/e2e/test_streamlit_e2e.py`
- `tests/e2e/test_streamlit_real_assessment_e2e.py`

Status:
- keep for now

Why they still exist:
- they protect the still-shipped compatibility dashboard

Deprecation posture:
- do not expand these tests for new product behavior
- keep them stable until the shell owns the equivalent user journeys

Archive trigger:
- replace with app-shell-first browser coverage
- confirm no release path still depends on the monolithic dashboard

## Soft-Deprecation Strategy

Use a non-breaking, staged deprecation path:

### Stage 1: Naming

- README states the multipage shell is the primary app surface
- README states the old interactive dashboard is a legacy compatibility surface

### Stage 2: Runtime Signaling

- the old dashboard shows a visible legacy/deprecation notice
- the notice points users toward `streamlit_app.py`

### Stage 3: Freeze

- no new UX work goes into the old dashboard unless needed for parity, migration, or critical fixes
- new navigation and runtime work lands only in the app shell

### Stage 4: Archive Readiness Review

Before archiving the old dashboard, confirm:

1. runtime setup is fully handled in the shell
2. setup -> speak -> review is stable in the shell
3. history and library workflows are accepted in the shell
4. settings/runtime configuration is accepted in the shell
5. README and onboarding scripts point to the shell first
6. remaining old-dashboard-only workflows are either migrated or intentionally dropped

## Recommended Next Step

Treat the app shell as the product and the old dashboard as compatibility.

That means:

- future UX work should start in `pages/` and `app_shell/`
- deprecation is documented now, not later
- archive work can happen cleanly once shell parity is signed off

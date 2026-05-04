# Playwright Flow Expansion Plan

Last updated: 2026-04-21
Status: Accepted supporting browser automation plan for the local desktop
baseline

## Summary

This plan defines the next Playwright browser flows worth adding for the
current Streamlit app shell.

This document is intentionally separate from the main implementation plan so it
can be revised after major app-shell phases stabilize.

This plan is subordinate to:
1. `docs/PLANNING_ALIGNMENT_META_PLAN.md`
2. `docs/IMPLEMENTATION_PLAN.md`
3. `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md`
4. `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md`

## Goal

Expand browser-level regression coverage where Streamlit `AppTest` logic
coverage is not enough.

The goal is:
1. catch browser-only regressions in navigation, layout, and widget behavior
2. validate the learner-facing flows that matter most after each stable phase
3. keep Playwright as the primary web E2E tool for the current desktop shell

## Non-Goals

This plan does not:
1. replace Playwright with Maestro for the current web shell
2. create a second governing product or execution plan
3. force browser-flow work in parallel with unstable screen refactors
4. define mobile or companion-app automation
5. require Task Master changes while the current implementation phase is still
   being actively reshaped

## Current Baseline

The repo already has a good Playwright foundation:
1. seeded browser happy path from Home to Review
2. browser upload flow and browse-files coverage
3. two-attempt flow into History
4. real-audio progression flow
5. narrow-screen runtime-setup flow
6. retained trace, video, and screenshot artifacts on failure

The repo also has broad Streamlit `AppTest` coverage for page logic, but that
does not replace browser-level coverage for rendered widgets and navigation.

## Current Browser Gaps

The main gaps worth closing are:
1. home-shell navigation beyond the current start path
2. review actions beyond the currently covered buttons
3. history interaction, filtering, and reopen behavior in a real browser
4. settings and support workflows in a real browser
5. library and guided sample flows in a real browser
6. responsive browser coverage outside runtime setup
7. the real recording path behind `st.audio_input`

## Stability Rule

Because the app is still being transformed heavily, this plan must be treated
as a re-baselinable follow-on plan.

Rules:
1. do not treat the flow list below as immutable
2. re-check this plan after any phase that materially rewrites navigation,
   localized labels, or page information architecture
3. if `pages/00_Setup.py`, `pages/02_Speak.py`, `pages/03_Review.py`,
   `pages/04_History.py`, or `pages/06_Settings.py` change substantially,
   update the flow scope before adding more Playwright coverage
4. if PAL review materially changes screen structure, revisit the affected
   Playwright tasks before implementation

Meta-plan alignment:
1. `Setup`, `Speak`, and `Settings` browser flows should wait until the active
   credential and support passes are stable enough to avoid churn
2. `Review` and `History` browser flows should be rechecked after the current
   desktop-baseline phase settles and again if the UX follow-on placeholders
   become active
3. new Playwright work should not compete with active shared-file work in
   `tests/test_app_shell_pages.py`, `app_shell/page_helpers.py`, or the PAL-
   reviewed screen files during the current phase

## Recommended Flow Order

### Subtask 1: Home and global navigation smoke

Files:
1. `tests/e2e/test_app_shell_navigation_e2e.py` `[CREATE]`
2. `tests/e2e/conftest.py`
3. `streamlit_app.py`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. browser coverage for `Home` actions beyond only `Start new`
2. explicit coverage for `Resume`, `History`, `Library`, `Guide`, and
   `Settings` when present
3. validation that gated routes stay gated when runtime setup is incomplete
4. no dependence on brittle CSS-only selectors for primary navigation

Timing:
1. start this only after the current home and shell navigation surfaces stop
   moving materially

### Subtask 2: Review action flow

Files:
1. `tests/e2e/test_app_shell_review_actions_e2e.py` `[CREATE]`
2. `pages/03_Review.py`
3. `app_shell/review_components.py`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. browser coverage for `Try again`, `View history`, and `New setup`
2. browser validation for scoring-guide or coaching affordances that remain in
   the product after the current phase
3. confirmation that learner-facing warnings stay visible in the rendered page
4. stable flow coverage for the main post-assessment branching paths

Timing:
1. recheck scope after any review-screen hierarchy cleanup before implementing

### Subtask 3: History interaction and reopen flow

Files:
1. `tests/e2e/test_app_shell_history_e2e.py` `[CREATE]`
2. `pages/04_History.py`
3. `tests/e2e/conftest.py`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. browser coverage for detail selector changes
2. browser coverage for speaker or language scope changes if those controls
   remain in the shipped surface
3. browser coverage for reopening or comparing attempts from History
4. stronger regression protection around rendered history details, not just
   logical defaults

Timing:
1. pair this with the stable history information architecture, not with an
   actively changing history refactor

### Subtask 4: Settings support and maintenance flow

Files:
1. `tests/e2e/test_app_shell_settings_support_e2e.py` `[CREATE]`
2. `pages/06_Settings.py`
3. `app_shell/page_helpers.py`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. browser coverage for the localized `Troubleshooting & Support` section once
   it lands
2. browser validation for warning placement, storage summary visibility, and
   cleanup affordances
3. browser coverage for saved-connection state that the learner actually sees
4. confidence that Settings remains usable after support-surface additions

Timing:
1. wait until the current Settings support pass is implemented and stable
2. do not open this while another thread is actively reshaping the same
   Settings surface

### Subtask 5: Library and guided sample flow

Files:
1. `tests/e2e/test_app_shell_library_e2e.py` `[CREATE]`
2. `pages/05_Library.py`
3. `pages/01_Session_Setup.py`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. browser coverage for selecting a sample or guided prompt path from Library
2. browser validation that the chosen sample or theme reaches the next learner
   step correctly
3. rendered-browser confidence for library cards, buttons, and empty states
4. regression protection for the route from exploration into one practice flow

Timing:
1. implement this only if Library remains part of the active learner journey
   after the current phase settles

### Subtask 6: Responsive browser smoke sweep

Files:
1. `tests/e2e/test_app_shell_responsive_e2e.py` `[CREATE]`
2. `pages/00_Setup.py`
3. `pages/02_Speak.py`
4. `pages/06_Settings.py`

Deliverables:
1. small-screen visibility checks for the core learner journey outside runtime
   setup
2. confirmation that primary CTA and status copy stay in-viewport
3. regression protection against collapsed or unusable layouts on narrow widths
4. one minimal but stable responsive smoke layer rather than per-widget pixel
   assertions

Timing:
1. do this after the affected screens have stable copy and section ordering

### Deferred Spike: Real recording flow

Files:
1. `tests/e2e/test_app_shell_recording_e2e.py` `[CREATE]`
2. `pages/02_Speak.py`
3. `tests/e2e/conftest.py`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. decide whether browser automation can cover `st.audio_input` reliably in the
   local shell
2. if yes, add one stable happy-path recording flow
3. if not, document why upload plus lower-level tests remain the practical
   coverage split for now
4. do not let this spike block the higher-value navigation and settings flows

Timing:
1. defer until the current `Speak` surface and any recording-state UX changes
   are stable

## Task Master Rule

This plan should not rewrite the active Task Master queue while the current
phase is being implemented in another thread.

Rule:
1. do not add or reorder active implementation subtasks just to fit this plan
2. when the current phase stabilizes, promote only the then-relevant Playwright
   items as follow-on work
3. if the current phase changes labels, routing, or screen order, re-baseline
   this doc before creating or updating automation tasks

## Future Re-Evaluation

If the repo later reaches a real companion-app stage:
1. revisit Maestro under `docs/MOBILE_COMPANION_STRATEGY.md`
2. keep that decision separate from the current Playwright plan for the
   Streamlit shell

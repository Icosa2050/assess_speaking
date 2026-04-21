# Local Desktop UX Refactoring Plan

Last updated: 2026-04-19
Status: Accepted supporting plan registered under the local desktop planning
hierarchy

## Summary

This plan adds one missing layer to the current desktop-baseline work:
1. a focused UX refactoring lane for the shipped local desktop experience
2. clear scope boundaries so UX cleanup does not drift into product redesign
3. file-based execution slices that fit the repo's collision rules

This plan is intentionally subordinate to the governing planning docs for the
current desktop baseline:
1. `docs/PLANNING_ALIGNMENT_META_PLAN.md`
2. `docs/IMPLEMENTATION_PLAN.md`
3. `docs/LOCAL_BACKEND_ARCHITECTURE.md`
4. `docs/SAVED_CONNECTIONS_PRODUCTION_CREDENTIAL_PLAN.md`

This document should be treated as:
1. a supporting local-desktop UX plan
2. non-authoritative on repo-wide execution order
3. safe to refine later without redefining product direction

## Position In Planning Hierarchy

This document is a supporting plan for the desktop baseline lane.

It is not part of the desktop baseline lane itself. It is subordinate to:
1. `docs/PLANNING_ALIGNMENT_META_PLAN.md` for execution order and collision
   rules
2. `docs/IMPLEMENTATION_PLAN.md` for product direction
3. `docs/SAVED_CONNECTIONS_PRODUCTION_CREDENTIAL_PLAN.md` for credential rules
4. `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md` for the support-surface
   sequencing that touches `Settings` and diagnostics

For overlapping files, this plan may only execute after the corresponding
meta-plan serial pass completes.

## Goal

Improve the current local desktop learner flow without changing the approved
product direction.

The goal is:
1. less ambiguity in setup and runtime readiness
2. clearer recording and assessment state transitions
3. more legible review and history surfaces
4. safer settings behavior around credentials and runtime choices
5. better learner-facing error handling

## Non-Goals

This plan does not:
1. define the future shared React UI
2. replace Streamlit with Tauri or web delivery
3. redesign the product brand or visual system from scratch
4. introduce hosted-only UX assumptions into the current local product
5. bypass PAL review for screen changes

## Locked Constraints

This UX plan must respect the current locked decisions:
1. the local desktop baseline remains the current priority
2. the local FastAPI backend contract stays the canonical product contract
3. public inference remains exploratory and separate
4. hosted auth and hosted product migration remain approved but deferred
5. app-shell runtime credentials come only from saved connections plus secure
   `secret_ref` storage
6. localization is required; no new hardcoded learner-facing strings
7. Maestro semantic labels must remain in scope for UX-affecting work
8. any screen changes in `pages/00_Setup.py`, `pages/02_Speak.py`, or
   `pages/06_Settings.py` must go through PAL review before implementation

## UX Principles

The local desktop UX should follow these principles:
1. backend-contract first, not frontend-only behavior
2. primary flow first, advanced controls second
3. learner-facing state always visible during long-running work
4. visible next step after each major action
5. errors explained in user language, not just logs or traces
6. local-first defaults preserved even when cloud features exist later

## Current UX Priorities

### Priority 1: Setup clarity

The learner should be able to tell:
1. whether Whisper is ready
2. whether a provider connection is ready
3. which connection is active
4. what is blocked and how to unblock it

### Priority 2: Speak flow confidence

The learner should be able to tell:
1. what prompt or task they are responding to
2. whether recording is live
3. whether audio was saved successfully
4. what the next action is after recording stops
5. whether assessment is queued, running, or failed

### Priority 3: Review readability

The learner should be able to tell:
1. overall result first
2. top priorities second
3. transcript and evidence next
4. warnings and degraded states without hunting

### Priority 4: History usefulness

The learner should be able to tell:
1. what changed between sessions
2. which session is worth reopening
3. whether a report is incomplete, degraded, or fully scored

### Priority 5: Settings safety

The learner should be able to tell:
1. what is saved locally
2. what credentials exist securely
3. when a change affects the active runtime path
4. what remains optional versus required

## UX Scope By Screen

Primary local-baseline screens for this plan:
1. `pages/00_Setup.py`
2. `pages/02_Speak.py`
3. `pages/03_Review.py`
4. `pages/04_History.py`
5. `pages/06_Settings.py`

Supporting non-screen files may be touched only when they help one of the
screens above:
1. `app_shell/services.py`
2. `app_shell/diagnostics.py`
3. `app_shell/page_helpers.py`
4. `app_shell/review_components.py`
5. related locale and test files

## Dependencies On The Governing Plans

To stay aligned with the governing plans and avoid churn:
1. do not rewrite UX around hosted auth yet
2. do not redesign around React routing yet
3. do not assume support endpoints exist until their backend plan lands
4. treat credential UX work as dependent on the saved-connections credential
   plan
5. treat collision-heavy files as serialized work, not parallel work

This means:
1. `Setup` and `Settings` work should wait until credential-core rules are
   stable enough to implement once
2. `Speak`, `Review`, and `History` can move earlier if they stay on the
   existing backend contract

## PAL Requirement

Before implementing screen changes in these files:
1. `pages/00_Setup.py`
2. `pages/02_Speak.py`
3. `pages/06_Settings.py`

We should:
1. review the specific UX approach with PAL
2. agree on the learner-facing state model
3. confirm that copy changes stay localized and packaging-safe

## File-Based Execution Plan

### Subtask 1: Recorder and speak-state cleanup

Files:
1. `pages/02_Speak.py`
2. `app_shell/page_helpers.py`
3. `locales/en.json`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. one clearer state model for idle, recording, saved, assessing, done, and
   failed
2. obvious primary CTA and next-step copy
3. learner-facing error copy for recorder and assessment failures
4. advanced controls visually secondary to the main recording path

Notes:
1. PAL review required before implementation
2. PAL review for this subtask is part of `PAL Gate A` in
   `docs/PLANNING_ALIGNMENT_META_PLAN.md`; do not schedule a separate PAL
   session for the same `Speak` surface
3. use `docs/RECORDER_UX_WIREFRAME.md` as an input only after it has been
   classified by the meta-plan triage step; if it is deferred or marked
   out-of-scope, derive the state model from this UX plan's priorities instead

### Subtask 2: Review screen hierarchy cleanup

Files:
1. `pages/03_Review.py`
2. `app_shell/review_components.py`
3. `locales/en.json`
4. `tests/test_app_shell_review_components.py`

Deliverables:
1. summary-first layout
2. explicit placement for warnings and degraded states
3. clearer hierarchy between score, priorities, transcript, and evidence
4. no hidden dependence on terminal logs for learner understanding

### Subtask 3: History triage and reopen flow

Files:
1. `pages/04_History.py`
2. `app_shell/page_helpers.py`
3. `locales/en.json`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. easier scan of recent sessions
2. clearer degraded-versus-complete session indicators
3. stronger reopen and compare affordances
4. less noise in row-level metadata

Notes:
1. the `pages/04_History.py` slice can move earlier on the current backend
   contract
2. any `app_shell/page_helpers.py` or `tests/test_app_shell_pages.py` work must
   follow the serial chain in `docs/PLANNING_ALIGNMENT_META_PLAN.md`

### Subtask 4: Runtime setup readiness messaging

Files:
1. `pages/00_Setup.py`
2. `app_shell/diagnostics.py`
3. `locales/en.json`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. clearer readiness states for Whisper and provider setup
2. cleaner blocked-state explanations
3. no misleading fallback messaging around credentials
4. simpler next actions when diagnostics fail

Notes:
1. PAL review required before implementation
2. should follow the credential-core decisions from the reviewed plans
3. this subtask must follow `PAL Gate A` in
   `docs/PLANNING_ALIGNMENT_META_PLAN.md`
4. `app_shell/diagnostics.py` is in the serial chain
   `Subtask 5 -> Subtask 8 -> Subtask 10b` of the meta-plan; UX changes to that
   file belong only after `Subtask 10b` completes

### Subtask 5: Settings safety and terminology pass

Files:
1. `pages/06_Settings.py`
2. `app_shell/services.py`
3. `locales/en.json`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. safer wording around saved keys and active connections
2. clearer distinction between local runtime settings and optional cloud-linked
   features
3. no ambiguous form state when provider or connection changes
4. fewer destructive or surprising settings interactions

Notes:
1. PAL review required before implementation
2. should happen after the credential-core work is stable enough to avoid
   redoing the form logic
3. this subtask must follow the final `Settings` pass in
   `docs/PLANNING_ALIGNMENT_META_PLAN.md`
4. `pages/06_Settings.py` belongs after meta-plan `Subtask 11`
5. `app_shell/services.py` belongs after meta-plan `Subtask 8`

### Subtask 6: Locale parity pass

Files:
1. `locales/de.json`
2. `locales/it.json`
3. `locales/fr.json`
4. `locales/es.json`

Deliverables:
1. keep non-English strings aligned with the approved English UX copy
2. no screen ships with new English-only learner text

## Sequencing Guidance

Recommended order:
1. review this plan after the meta-plan review settles
2. start with `Review` and `History` if they can move without credential churn
3. do `Setup` and `Settings` after credential rules are locked enough to avoid
   rework
4. keep `Speak` changes coordinated with PAL and recorder-state behavior

## Success Criteria

This plan is working when:
1. a first-time learner can tell what to do next on `Setup` and `Speak`
2. runtime and credential failures are understandable without reading logs
3. `Review` shows score, priorities, and degraded-state warnings in a stable
   order
4. `History` is useful for deciding what to reopen
5. screen-level UX cleanup does not fight the approved backend, packaging, or
   hosted plans

## Follow-On Rule

Now that the hierarchy review has registered this document:
1. keep it linked from `docs/IMPLEMENTATION_PLAN.md`
2. keep it referenced by `docs/PLANNING_ALIGNMENT_META_PLAN.md` as a supporting
   UX lane rather than a governing product plan
3. keep this plan local-desktop scoped unless the canonical product plans are
   explicitly revised

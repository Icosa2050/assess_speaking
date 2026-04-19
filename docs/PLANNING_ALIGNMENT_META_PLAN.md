# Planning Alignment Meta-Plan

Last updated: 2026-04-18
Status: Proposed reconciliation plan for the current desktop, support, hosted,
and exploratory planning docs

## Summary

This document aligns the recently added planning docs into one execution order.

It does not replace the existing plans. It defines:
1. which docs are canonical for near-term product work
2. which discrepancies must be resolved in docs before implementation drifts
3. which shared files must be changed serially instead of in parallel
4. which decision gates must be carried into implementation and raised to the
   user at the right time
5. the approved file-based execution order for the next implementation passes

Current planning lanes:
1. desktop baseline lane
   - `docs/IMPLEMENTATION_PLAN.md`
   - `docs/LOCAL_BACKEND_ARCHITECTURE.md`
   - `docs/SAVED_CONNECTIONS_PRODUCTION_CREDENTIAL_PLAN.md`
   - `docs/SUPPORT_MAINTENANCE_PLAN.md`
   - `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md`
2. deferred companion lane
   - `docs/MOBILE_COMPANION_STRATEGY.md`
3. separate exploratory service lane
   - `docs/PUBLIC_INFERENCE_ARCHITECTURE_PLAN.md`
4. approved product migration lane
   - `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md`
5. supporting UX lane
   - `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md`
6. supporting docs that must be triaged before implementation drift grows:
   - `docs/REPO_CLEANUP_PLAN.md`
   - `docs/RECORDER_UX_WIREFRAME.md`
   - `docs/MULTILINGUAL_CEFR_ASSESSMENT_PLAN.md`
   - `docs/COACHING_BACKLOG.md`
   - `docs/SPOKEN_CORPUS_CATEGORIZATION_PLAN.md`
   - `docs/SYNTHETIC_BENCHMARK_AUTOMATION.md`
   - `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md`

## Locked Decisions

1. The current priority is the local desktop baseline, not hosted auth, not the
   React or Tauri migration, and not the public inference beta.
2. The shared product API remains the canonical local and future hosted product
   contract:
   - `/v1/health`
   - `/v1/diagnostics`
   - `/v1/runtime`
   - `/v1/uploads`
   - `/v1/assessments`
   - `/v1/history`
   - `/v1/samples`
3. Local desktop support work may add local-only support endpoints:
   - `/v1/maintenance/storage`
   - `/v1/maintenance/cleanup`
   - `/v1/support-bundles`
   - `/v1/support-bundles/{bundle_id}`
4. Public inference remains a separate beta service and must not reshape the
   canonical product API.
5. Mobile remains a later companion that consumes backend contracts and must not
   depend on desktop UI exports.
6. App-shell runtime credentials come only from saved connections plus secure
   `secret_ref` storage.
7. Environment variables remain supported for CLI and standalone integration
   test flows only, never for desktop app-shell runtime behavior.
8. Any screen changes in `pages/00_Setup.py`, `pages/02_Speak.py`, or
   `pages/06_Settings.py` must go through PAL review before implementation.
9. macOS and Windows are first-class packaging targets for this pass. Linux is
   a no-regression target, not a first-class packaging deliverable.
10. `RuntimeMetadata` must be defined in one authoritative location:
    `app_shell/bootstrap.py`. Other files import or consume that shape instead
    of redefining it.
11. PAL review is a blocking gate before the credential screen pass and before
    the Settings support pass.

## Discrepancies To Resolve In Docs

1. `docs/LOCAL_BACKEND_ARCHITECTURE.md` says job metadata belongs in `jobs/`,
   but its later data-layout section still says backend job metadata may live in
   `reports_dir`. Resolve this by making `jobs/` the only canonical owner of job
   metadata.
2. Runtime metadata is currently described in multiple places with different
   intent. Unify it into one shared schema instead of reintroducing separate
   local, support, and hosted variants.
3. Packaging language drifts between documents. Standardize on:
   - macOS and Windows first-class
   - Linux supported on a no-regression basis
4. `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md` is the only approved product migration
   path for hosted work. `docs/PUBLIC_INFERENCE_ARCHITECTURE_PLAN.md` must stay
   clearly marked as exploratory and non-blocking.
5. Support-bundle language must stay local-desktop scoped until hosted
   persistence and user ownership actually exist.
6. `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md` Task 1 must be aligned
   with the unified runtime metadata schema, not just `runtime_mode` plus
   `packaging_safe`.
7. `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md` Task 6 must acknowledge
   that `app_shell/services.py` work is split earlier in this meta-plan and
   that the later Settings pass centers on `page_helpers.py`.
8. `docs/PUBLIC_INFERENCE_ARCHITECTURE_PLAN.md` must explicitly state that its
   `/v1/*` routes live on a separate service boundary and do not redefine the
   canonical product backend namespace.

## Shared Runtime Metadata Contract

Define one runtime metadata shape and reuse it everywhere:
1. `deployment_mode`
   - `local`
   - `hosted`
2. `launch_mode`
   - `repo`
   - `packaged`
3. `packaging_safe`
   - `true`
   - `false`
4. `auth_mode`
   - `guest`
   - `optional`
   - `required`

Owner:
1. the authoritative `RuntimeMetadata` definition lives in
   `app_shell/bootstrap.py`
2. lifecycle, launcher, diagnostics, and support-bundle code consume that
   exported shape instead of redefining local variants

Rules:
1. local desktop mode stays localhost-bound and auth-free in guest mode
2. hosted mode may require auth and different persistence implementations
3. support and cleanup features must work for both `repo` and future
   `packaged` launch modes

## Collision Rules

Do not run parallel implementation across these shared files:
1. `app_shell/services.py`
2. `app_shell/diagnostics.py`
3. `pages/06_Settings.py`
4. `tests/test_app_shell_pages.py`
5. `app_shell/page_helpers.py`
6. `locales/en.json`
7. `tests/test_app_shell_i18n.py`
8. `tests/test_app_shell_diagnostics.py`
9. `tests/test_app_shell_services.py`
10. `app_backend/contracts.py`
11. `app_backend/app.py`
12. `app_backend/config.py`
13. `app_backend/jobs.py`
14. `app_shell/backend_client.py`

Required sequence for collision-heavy areas:
1. credential core first
2. support and maintenance backend APIs second
3. support bundle and diagnostics refinement third
4. shared Settings UI pass fourth
5. hosted deployment seams only after the desktop baseline above is stable

Explicit serial passes:
1. `app_shell/services.py` is touched in Subtask 5, then Subtask 8
2. `app_shell/diagnostics.py` is touched in Subtask 5, then Subtask 8, then
   Subtask 10b
3. `pages/06_Settings.py` is touched in Subtask 6, then Subtask 11
4. `tests/test_app_shell_pages.py` is touched in Subtask 6, then Subtask 11
5. `app_shell/page_helpers.py` is touched in Subtask 10b, then Subtask 11, and
   only after that in any later local-desktop UX refinement pass
6. `locales/en.json` is touched in Subtask 12 and only after that in any
   follow-on `Review` or `History` UX cleanup
7. `tests/test_app_shell_i18n.py` is touched in Subtask 12 and only after that
   in any follow-on `Review` or `History` UX cleanup
8. `tests/test_app_shell_diagnostics.py` is touched in Subtask 5, then
   Subtask 10b
9. `tests/test_app_shell_services.py` is touched in Subtask 5, then Subtask 8
10. `app_backend/contracts.py` is touched in Subtask 7, then Subtask 14
11. `app_backend/app.py` is touched in Subtask 7, then Subtask 14
12. `app_backend/config.py` is touched in Subtask 9, then Subtask 14
13. `app_backend/jobs.py` is touched in Subtask 9, then Subtask 14
14. `app_shell/backend_client.py` is touched in Subtask 7 and only after that
    in any later hosted or client follow-on that reopens the same support
    contract surface

## Task Graph Tightening

Task Master should reflect the execution constraints above instead of carrying
them only as prose notes:
1. Subtasks 0A and 0B stay the only ready work until doc triage is complete,
   and both are high-priority blockers for the rest of the queue
2. Subtask 7 depends on Subtask 5, not Subtask 6, because it stays on backend
   and client files rather than PAL-reviewed screens
3. Subtask 8 depends on both Subtask 5 and Subtask 7 and also waits on its
   explicit support-bundle redaction decision gate
4. Subtask 9 waits on Subtask 8 plus an explicit cleanup-policy decision gate
5. Subtask 11 waits on Subtask 6, Subtask 10b, and an explicit Settings PAL
   strategy gate instead of carrying that decision only in its notes
6. follow-on `Review` and `History` UX cleanup remains visible as placeholder
   work after the desktop baseline, but it does not block hosted-seam work

The explicit blocker gates are:
1. runtime metadata decision gate before Subtask 3
2. `client_snapshot` decision gate before Subtask 5
3. `PAL Gate A` before Subtask 6
4. support-bundle redaction decision gate before Subtask 8
5. cleanup-policy decision gate before Subtask 9
6. Settings PAL strategy gate before Subtask 11

## File-Based Execution Plan

### Subtask 0A: Triage supporting docs set A

Files:
1. `docs/REPO_CLEANUP_PLAN.md`
2. `docs/RECORDER_UX_WIREFRAME.md`
3. `docs/MULTILINGUAL_CEFR_ASSESSMENT_PLAN.md`
4. `docs/COACHING_BACKLOG.md`

Deliverables:
1. classify each doc as blocking, deferred, or out-of-scope for the current
   desktop baseline
2. identify any collisions with cleanup, recorder UX, localization, or review
   surfaces
3. note which docs must be reflected in later implementation tasks

### Subtask 0B: Triage supporting docs set B

Files:
1. `docs/SPOKEN_CORPUS_CATEGORIZATION_PLAN.md`
2. `docs/SYNTHETIC_BENCHMARK_AUTOMATION.md`
3. `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md`

Deliverables:
1. classify each doc as blocking, deferred, or out-of-scope
2. identify any collisions with `app_backend/contracts.py`, `scripts/`, or
   later hosted work
3. classify the local desktop UX refactoring doc as a supporting desktop UX
   plan that remains subordinate to the desktop baseline lane and current
   collision rules

### Subtask 1: Reconcile docs core

Files:
1. `docs/IMPLEMENTATION_PLAN.md`
2. `docs/LOCAL_BACKEND_ARCHITECTURE.md`
3. `docs/SUPPORT_MAINTENANCE_PLAN.md`
4. `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md`

Deliverables:
1. align app-data ownership rules
2. align runtime metadata wording
3. align packaging-target wording
4. classify support endpoints as local-desktop extensions
5. align `SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md` Task 6 with the later
   `page_helpers.py` Settings pass

### Subtask 2: Reconcile docs boundaries

Files:
1. `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md`
2. `docs/PUBLIC_INFERENCE_ARCHITECTURE_PLAN.md`
3. `docs/MOBILE_COMPANION_STRATEGY.md`
4. `docs/SAVED_CONNECTIONS_PRODUCTION_CREDENTIAL_PLAN.md`

Deliverables:
1. keep the hosted product plan as the only approved product migration path
2. keep the public inference plan separate from the canonical product API
3. keep mobile explicitly non-blocking
4. keep credential rules compatible with later optional desktop sign-in
5. add the Linux no-regression qualifier where product text still implies
   first-class packaging parity

### Decision Gate 3A: Runtime metadata ownership and shape

Files:
1. `docs/PLANNING_ALIGNMENT_META_PLAN.md`
2. `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md`
3. `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md`
4. `app_shell/bootstrap.py`

Deliverables:
1. decide who owns `auth_mode`
2. decide whether `auth_mode` is mutable at runtime
3. decide whether `packaging_safe` is derived or explicitly stored

Task Master rule:
1. this gate is an explicit blocker before Subtask 3 begins

### Subtask 3: Add unified runtime metadata

Files:
1. `app_shell/bootstrap.py`
2. `app_backend/lifecycle.py`
3. `scripts/run_app.py`
4. `tests/test_run_app.py`

Deliverables:
1. one shared runtime metadata schema
2. explicit `deployment_mode`, `launch_mode`, `packaging_safe`, and
   `auth_mode`
3. no change to current local desktop behavior

Decision gate before implementation:
1. ask the user who owns `auth_mode` and whether it is mutable at runtime
2. ask the user whether `packaging_safe` is a derived diagnostic property or an
   independently asserted runtime field

### Subtask 4: Lock the secure credential core

Files:
1. `app_shell/secret_store.py`
2. `app_shell/runtime_resolver.py`
3. `tests/test_secret_store.py`
4. `tests/test_runtime_resolver.py`

Deliverables:
1. app-shell credential resolution ignores environment variables
2. secure-store migration remains supported
3. missing secure secrets produce a recoverable state
4. explicitly retire environment-fallback secret resolution in
   `app_shell/secret_store.py`

### Decision Gate 5A: `client_snapshot` scope

Files:
1. `docs/PLANNING_ALIGNMENT_META_PLAN.md`
2. `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md`
3. `app_shell/services.py`
4. `app_shell/diagnostics.py`

Deliverables:
1. decide whether `client_snapshot` includes credential-state summary fields
2. keep shell-side snapshot behavior aligned with the support-bundle plan

Task Master rule:
1. this gate is an explicit blocker before Subtask 5 begins

### Subtask 5: Update app-shell credential surfaces

Files:
1. `app_shell/services.py`
2. `app_shell/diagnostics.py`
3. `tests/test_app_shell_diagnostics.py`
4. `tests/test_app_shell_services.py`

Deliverables:
1. diagnostics report saved-credential state only
2. env-fallback messaging is removed from app-shell runtime behavior
3. `credentials missing` is explicit and localized

Decision gate before implementation:
1. ask the user whether `client_snapshot` should include a credential-state
   summary or only non-credential runtime context

### PAL Gate A: Review the credential-related screen approach

Applies to:
1. Subtask 6
2. Subtask 11 if the user prefers one combined Settings review

Deliverables:
1. PAL-reviewed screen plan for `pages/00_Setup.py`, `pages/02_Speak.py`, and
   the first pass on `pages/06_Settings.py`
2. explicit note on whether the Settings credential and support passes share
   one PAL review or two separate reviews
3. the Local Desktop UX Refactoring Plan's PAL requirement for `Setup` and
   `Speak` is satisfied through this gate rather than a separate PAL session

### Subtask 6: Update credential-related screens after PAL review

Files:
1. `pages/00_Setup.py`
2. `pages/06_Settings.py`
3. `pages/02_Speak.py`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. forms no longer prefill secrets
2. blank edit preserves the current saved secret
3. explicit clear-key flow exists
4. Speak warnings reference saved credentials only

Implementation rule:
1. a blank submission means no write operation is performed against secure
   storage for that connection secret

### Subtask 7: Add support and maintenance APIs

Files:
1. `app_backend/contracts.py`
2. `app_backend/app.py`
3. `app_shell/backend_client.py`
4. `tests/test_app_backend_api.py`

Deliverables:
1. storage summary endpoint
2. cleanup endpoint
3. support-bundle create and download endpoints
4. validation and success-path API coverage
5. `app_shell/backend_client.py` is updated with explicit new support methods,
   not just ad hoc call sites

### Subtask 8: Build support bundles and shell redaction

Files:
1. `app_backend/support_bundle.py` `[CREATE]`
2. `app_shell/services.py`
3. `app_shell/diagnostics.py`
4. `tests/test_app_shell_services.py`

Deliverables:
1. default bundle excludes `reports`, `recordings`, and `uploads`
2. secret values and `secret_ref` values are redacted
3. sanitized client and runtime context comes from the shell
4. exclusion rules are named explicitly against the canonical app-data
   directories, not left implicit

Decision gate before implementation:
1. ask the user whether `secret_ref` should be fully redacted or replaced with
   a presence-indicating mask in support bundles

Task Master rule:
1. model the `secret_ref` choice as an explicit blocker task before Subtask 8
   can begin

### Subtask 9: Add cleanup and retention rules

Files:
1. `app_backend/maintenance.py` `[CREATE]`
2. `app_backend/config.py`
3. `app_backend/jobs.py`
4. `tests/test_app_backend_config.py`

Deliverables:
1. safe purge for `tmp/`
2. expiry for support bundles
3. pruning for stale completed, failed, and cancelled jobs
4. no auto-delete for `reports`, `recordings`, or `uploads`

Decision gate before implementation:
1. ask the user for the retention window for completed, failed, and cancelled
   job metadata
2. ask the user whether `all_safe` means the full union of `tmp`, `jobs`, and
   `logs`, or a more conservative subset

Task Master rule:
1. model the retention-window and `all_safe` questions as an explicit blocker
   task before Subtask 9 can begin

### Subtask 10: Wire cleanup into backend startup

Files:
1. `app_backend/lifecycle.py`
2. `scripts/run_backend.py`
3. `tests/test_run_backend.py`
4. `tests/test_app_backend_lifecycle.py`

Deliverables:
1. cleanup runs during backend startup
2. active backend log is preserved
3. repo-local overrides still work
4. this pass depends on Subtask 3 completing first because both touch
   `app_backend/lifecycle.py`

### Subtask 10b: Add maintenance warnings in diagnostics

Files:
1. `app_shell/diagnostics.py`
2. `app_shell/page_helpers.py`
3. `tests/test_app_shell_diagnostics.py`
4. `tests/test_app_shell_page_helpers.py`

Deliverables:
1. diagnostics warn about oversized or stale `tmp/`
2. diagnostics warn about stale or excessive `jobs/`
3. diagnostics warn when logs exceed the expected rotation footprint
4. warnings direct the user to Settings instead of a separate workflow

### PAL Gate B: Review the Settings support surface

Applies to:
1. Subtask 11 if the user prefers a second dedicated Settings review

Deliverables:
1. PAL-reviewed plan for the support section layout, warning placement, and
   action affordances in `pages/06_Settings.py`
2. if the local desktop UX refactoring plan later adds a Settings terminology
   pass, it must follow this gate rather than opening a conflicting Settings
   review track
3. default to one shared PAL review session unless Subtask 10b materially
   changes the Settings information architecture enough to justify a second
   dedicated review

### Subtask 11: Add the Settings support surface after PAL review

Files:
1. `pages/06_Settings.py`
2. `app_shell/page_helpers.py`
3. `tests/test_app_shell_pages.py`
4. `tests/test_app_shell_page_helpers.py`

Deliverables:
1. localized `Troubleshooting & Support` section
2. storage usage summary
3. cleanup actions wired to backend APIs
4. maintenance warnings route the user to Settings

### Subtask 12: Finish localization batch

Files:
1. `locales/en.json`
2. `locales/it.json`
3. `locales/de.json`
4. `tests/test_app_shell_i18n.py`

Deliverables:
1. credential and support strings localized
2. no hardcoded strings in touched UI flows

### Subtask 13: Close locale and docs follow-up

Files:
1. `locales/fr.json`
2. `locales/es.json`
3. `README.md`
4. `docs/LOCAL_BACKEND_ARCHITECTURE.md`

Deliverables:
1. locale parity for the affected flows
2. README coverage for privacy-safe bundle defaults
3. final doc alignment for cleanup and runtime behavior

### Subtask 14: Add hosted deployment seams only after Subtasks 1-13

Files:
1. `app_backend/contracts.py`
2. `app_backend/config.py`
3. `app_backend/app.py`
4. `app_backend/jobs.py`

Deliverables:
1. storage and job interfaces split by runtime mode
2. hosted code stops depending on local paths and `history.csv`
3. no change to stable local desktop API behavior

### Follow-On Placeholder A: Review hierarchy UX cleanup

Files:
1. `pages/03_Review.py`
2. `app_shell/review_components.py`
3. `locales/en.json`
4. `tests/test_app_shell_review_components.py`

Deliverables:
1. summary-first review layout
2. clearer placement for warnings and degraded states
3. stronger hierarchy between score, priorities, transcript, and evidence
4. execution stays subordinate to
   `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md` and does not create a parallel
   governing lane

Task Master rule:
1. keep this visible after Subtask 13 without adding it to the current
   desktop-baseline critical path

### Follow-On Placeholder B: History triage and reopen UX cleanup

Files:
1. `pages/04_History.py`
2. `app_shell/page_helpers.py`
3. `locales/en.json`
4. `tests/test_app_shell_pages.py`

Deliverables:
1. easier scan of recent sessions
2. clearer complete-versus-degraded session indicators
3. stronger reopen and compare affordances
4. execution stays subordinate to
   `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md` and follows the `page_helpers`
   serial chain after the Settings passes

Task Master rule:
1. keep this visible after Subtask 11 and Subtask 13 without adding it to the
   current desktop-baseline critical path

## Verification

1. Launching the desktop app with provider env vars set does not make remote
   connections usable unless a saved secure secret exists.
2. A saved remote connection still works after restart when env vars are unset.
3. Clearing a saved key moves the connection into a recoverable
   `credentials missing` state.
4. Support bundles exclude `reports`, `recordings`, and `uploads` by default.
5. Support bundles redact both secret values and `secret_ref` values.
6. Cleanup removes transient data and stale metadata only.
7. Cleanup never auto-deletes user-owned report, recording, or upload content.
8. Local guest mode remains auth-free and localhost-bound.
9. Hosted Phase 1 seam work does not change the stable local `/v1/*` product
   contract.
10. Public inference work remains out of the canonical product API surface.

## Assumptions And Defaults

1. `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md` stays approved but deferred until the
   desktop baseline above is stabilized.
2. `docs/PUBLIC_INFERENCE_ARCHITECTURE_PLAN.md` stays exploratory and separate.
3. `docs/MOBILE_COMPANION_STRATEGY.md` stays deferred and non-blocking.
4. Any shared-file work called out in the collision rules is serialized.
5. This meta-plan should be updated again when the repo begins the React or
   Tauri migration for real.

## Decision Gates To Raise During Implementation

Do not silently decide these during implementation. Mark them and ask the user
when the corresponding subtask becomes active:
1. Subtask 3
   - who owns `auth_mode`
   - whether `auth_mode` can change at runtime
   - whether `packaging_safe` is derived or explicitly stored
2. Subtask 5
   - whether `client_snapshot` includes credential-state summary fields
3. Subtask 8
   - whether `secret_ref` is fully redacted or replaced with a
     presence-indicating mask
4. Subtask 9
   - the retention window for stale completed, failed, and cancelled jobs
   - the exact scope of `all_safe`
5. PAL gate strategy
   - whether Subtask 6 and Subtask 11 share one PAL review session or use two
     separate PAL reviews

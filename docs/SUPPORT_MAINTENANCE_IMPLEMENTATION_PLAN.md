# Support, Cleanup, And Packaging-Safe Maintenance Implementation Plan

Last updated: 2026-04-10
Status: Proposed

## Summary

This is the file-based implementation plan for the support and maintenance work
approved in `docs/SUPPORT_MAINTENANCE_PLAN.md`.

It keeps:
1. the current unified app-data root
2. backend-owned cleanup and export behavior
3. Settings as the user-facing support surface
4. packaging-safe behavior for both repo launch and future packaged apps on
   macOS and Windows

## Phase 1: Packaging-Safe Runtime Contract

### Task 1. Add explicit runtime-mode metadata

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/app_shell/bootstrap.py`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/app_backend/lifecycle.py`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/scripts/run_app.py`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_run_app.py`

Done when:
1. launcher payload exposes `runtime_mode`
2. launcher payload exposes `packaging_safe = true`
3. repo launch behavior remains unchanged
4. future packaged mode can be simulated without writable repo paths

## Phase 2: Backend Support And Maintenance APIs

### Task 2. Add backend maintenance and bundle contracts

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/app_backend/contracts.py`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/app_backend/app.py`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/app_shell/backend_client.py`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_backend_api.py`

Done when:
1. storage summary endpoint exists
2. cleanup endpoint exists
3. support-bundle create/download endpoints exist
4. API tests cover success and validation errors

### Task 3. Build support-bundle generation and redaction

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/app_backend/support_bundle.py`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/app_shell/services.py`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/app_shell/diagnostics.py`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_shell_services.py`

Done when:
1. bundle manifest contains platform, version, runtime mode, and redaction flags
2. default bundle excludes `reports`, `recordings`, and `uploads`
3. secret values and secret refs are redacted
4. sanitized client/runtime snapshot is passed from the shell instead of the
   backend reading Streamlit state directly

## Phase 3: Cleanup And Retention

### Task 4. Add cleanup engine and retention rules

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/app_backend/maintenance.py`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/app_backend/config.py`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/app_backend/jobs.py`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_backend_config.py`

Done when:
1. `tmp/` can be safely purged
2. support bundles older than 24 hours are removed
3. stale completed/failed/cancelled job metadata older than 30 days is pruned
4. cleanup never deletes `reports`, `recordings`, or `uploads`

### Task 5. Wire startup cleanup into backend launch

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/app_backend/lifecycle.py`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/scripts/run_backend.py`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_run_backend.py`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_backend_lifecycle.py`

Done when:
1. safe cleanup runs before ordinary backend work
2. repo-local override behavior remains unchanged
3. active backend log is preserved during cleanup
4. startup cleanup tests cover normal and edge cases

## Phase 4: UI And Diagnostics

### Task 6. Design and implement Settings support surface

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/pages/06_Settings.py`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/app_shell/services.py`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_shell_pages.py`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_shell_services.py`

Blocking rule:
1. discuss the Settings approach with PAL before implementing the screen change

Done when:
1. localized `Troubleshooting & Support` section exists
2. storage summary is shown
3. cleanup actions are wired through backend APIs
4. support-bundle generation is available from Settings

### Task 7. Extend diagnostics with maintenance warnings

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/app_shell/diagnostics.py`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/app_shell/page_helpers.py`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_shell_diagnostics.py`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_shell_page_helpers.py`

Done when:
1. diagnostics warn about oversized or stale `tmp/`
2. diagnostics warn about stale or excessive `jobs/`
3. diagnostics warn when logs exceed the expected rotation footprint
4. warnings direct the user to Settings instead of a separate workflow

## Phase 5: Localization And Documentation

### Task 8. Localize new support and cleanup copy

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/locales/en.json`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/locales/it.json`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/locales/de.json`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/tests/test_app_shell_i18n.py`

Done when:
1. new support strings are localized
2. no hardcoded support or cleanup strings remain in touched UI code

### Task 9. Finish locale batch and update docs

Files:
1. `/Users/bernhard/Development/assess_speaking-codex-v6/locales/fr.json`
2. `/Users/bernhard/Development/assess_speaking-codex-v6/locales/es.json`
3. `/Users/bernhard/Development/assess_speaking-codex-v6/README.md`
4. `/Users/bernhard/Development/assess_speaking-codex-v6/docs/LOCAL_BACKEND_ARCHITECTURE.md`

Done when:
1. locale batch is complete
2. README documents support-bundle privacy defaults
3. architecture docs document cleanup rules and packaging-safe runtime behavior

## Test Plan

### Packaging-safe behavior

1. repo launch remains green
2. simulated packaged mode does not require writable repo paths
3. support/export does not depend on `PROJECT_ROOT`

### Support bundle

1. default bundle excludes `reports`, `recordings`, and `uploads`
2. explicit opt-in includes only requested user-content groups
3. bundle contains diagnostics, runtime metadata, storage summary, and redaction
   metadata
4. secret values and secret refs are redacted

### Cleanup

1. startup cleanup purges `tmp/`
2. support bundles expire after 24 hours
3. old completed/failed/cancelled jobs are pruned after 30 days
4. active backend log remains intact

### UI

1. Settings support section is localized
2. maintenance warnings appear only when thresholds are crossed
3. diagnostics link the user back to Settings

### Final verification

1. `./scripts/run_tests.sh -q`
2. `./scripts/python.sh scripts/check_quality.py`

## Defaults

1. macOS and Windows are the first-class packaging targets
2. Linux should remain no-regression only
3. the unified app-data root stays canonical
4. the actual desktop packager is still deferred
5. Settings is the only new support UI surface in this pass

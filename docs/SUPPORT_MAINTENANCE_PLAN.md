# Support, Cleanup, And Packaging-Safe Maintenance Plan

Last updated: 2026-04-21
Status: Proposed

## Summary

This plan defines the next layer of desktop support and maintenance for
Vostavo.

The goal is to add:
1. a dedicated support/export bundle feature
2. deeper cleanup and retention policies for transient backend artifacts
3. a localized support surface in Settings
4. packaging-safe behavior for macOS and Windows first-class delivery, with
   Linux kept on a no-regression basis

The plan explicitly does **not** change the current app-data architecture to
separate OS-level state or log roots.

## Locked Decisions

### Keep the unified app-data root

Vostavo should continue using one per-user app-data root that already contains:

```text
<app-data-root>/
  backend_state.json
  jobs/
  logs/
  reports/
  recordings/
  uploads/
  tmp/
```

Reasons:
1. easier support-bundle generation
2. simpler local debugging
3. lower migration complexity
4. no user-facing value from splitting logs/state into separate OS locations

“Separate OS-level state/log roots” was evaluated and rejected for this pass.

### Make support and cleanup backend-owned

The backend should own:
1. support bundle generation
2. storage summaries
3. safe cleanup actions
4. retention enforcement

The app shell should only:
1. request those actions through the backend contract
2. supply sanitized client/runtime context
3. render results to the user

### Keep the UI entrypoint in Settings

The primary user-facing surface should be a localized
`Troubleshooting & Support` section in Settings.

Diagnostics should stay lightweight and only:
1. show warnings when maintenance attention is needed
2. direct the user to Settings

Do not add a new top-level support page in this pass.

### Make the feature packaging-safe now

The packager choice is still deferred, but these features must work both:
1. from repo launch
2. from a future packaged desktop app

That means:
1. no support feature may require writable repo paths
2. no cleanup logic may assume the checkout is the runtime location
3. launcher/backend runtime metadata must be explicit enough to support both
   repo and packaged operation later
4. support code should reuse the shared `RuntimeMetadata` shape with
   `deployment_mode`, `launch_mode`, `packaging_safe`, and `auth_mode` rather
   than inventing a support-only variant

## Public Backend Additions

These endpoints are local-desktop support extensions layered on top of the
canonical shared product API. They do not redefine the stable `/v1/*` product
surface used by the current desktop baseline and future hosted product work.

Add the following endpoints:

### `GET /v1/maintenance/storage`

Returns byte usage for:
1. `jobs`
2. `logs`
3. `reports`
4. `recordings`
5. `uploads`
6. `tmp`
7. cache roots

### `POST /v1/maintenance/cleanup`

Request:
1. `target = "tmp" | "jobs" | "logs" | "all_safe"`
2. `dry_run = bool`

Response:
1. deleted file count
2. freed bytes
3. warnings

### `POST /v1/support-bundles`

Request:
1. `include_reports = false`
2. `include_recordings = false`
3. `include_uploads = false`
4. `client_snapshot`
5. `client_diagnostics`

Response:
1. `bundle_id`
2. `filename`
3. `size_bytes`
4. `expires_at`

### `GET /v1/support-bundles/{bundle_id}`

Downloads the generated zip bundle.

## Support Bundle Rules

### Include by default

1. `backend_state.json`
2. backend diagnostics snapshot
3. sanitized app-shell/runtime snapshot
4. current backend logs
5. recent job metadata
6. storage summary
7. platform, version, and runtime metadata including `deployment_mode`,
   `launch_mode`, `packaging_safe`, and `auth_mode`

### Exclude by default

1. `reports/`
2. `recordings/`
3. `uploads/`

These paths may be included later only through explicit opt-in.

### Bundle staging

Generated bundles should live temporarily under:

```text
<app-data-root>/tmp/support-bundles/
```

## Cleanup And Retention Policy

### `tmp/`

1. purge on backend startup
2. allow manual cleanup from Settings

### `tmp/support-bundles/`

1. delete bundles older than 24 hours

### `jobs/`

1. keep startup recovery for stale queued/running jobs
2. delete completed, failed, and cancelled job JSON older than 30 days

### `logs/`

1. keep the current rotating `backend.log` strategy
2. manual cleanup may remove rotated backups and stale non-canonical log files
3. never delete the active backend log as part of ordinary cleanup

### `reports/`, `recordings/`, `uploads/`

1. never auto-delete
2. treat them as user-owned content

## UI Direction

Add a localized `Troubleshooting & Support` section at the bottom of Settings.

It should include:
1. storage usage summary
2. `Generate support bundle`
3. `Clear temporary files`
4. `Prune old job metadata`
5. `Clear rotated logs`

Diagnostics may add maintenance warnings when:
1. `tmp/` is stale or oversized
2. `jobs/` contains old or excessive metadata files
3. `logs/` exceed the expected rotation footprint

Those diagnostics should point to Settings instead of creating a separate
support workflow.

Any Settings implementation work must be reviewed with PAL before changes are
made to the screen.

## Packaging Notes

This plan is packaging-contract first with macOS and Windows as the first-class
targets for this pass and Linux kept on a no-regression basis.

Requirements for later delivery:
1. support/export must work whether the app is launched from the repo or from a
   packaged app bundle / installer
2. backend startup and maintenance code must not depend on writable install
   directories
3. runtime mode should remain explicit enough to distinguish repo launch from
   packaged launch

The actual packager decision remains out of scope for this document.

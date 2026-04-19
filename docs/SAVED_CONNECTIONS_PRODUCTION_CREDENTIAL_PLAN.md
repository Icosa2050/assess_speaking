# Saved Connections Production Credential Plan

Last updated: 2026-04-15
Status: Proposed implementation plan for the desktop app shell

## Goal

Make the desktop app shell production-safe by removing all environment-variable
API key usage from the app-shell flow and relying only on saved connections plus
secure `secret_ref` storage.

This plan applies to the app shell only:
1. saved connections in `workspace_prefs.json`
2. secure secret persistence through OS storage
3. runtime resolution for setup, diagnostics, speak, and settings

This plan does not change the lower-level CLI and standalone integration test
flows that intentionally still use environment variables outside the app shell.

## Confirmed Product Decisions

1. The app shell must ignore environment API keys for runtime behavior.
2. Raw API keys must never be written to `workspace_prefs.json`.
3. The active connection plus its `secret_ref` is the only credential source for
   app-shell runtime requests.
4. Editing an existing connection with a blank API key field keeps the stored
   secret unchanged.
5. Deleting a stored secret must require an explicit learner-visible action.
6. Changing a connection's provider in the form invalidates the current saved
   secret for that form session and requires a new key when the new provider
   needs auth.
7. Remote auth-required providers must fail closed when persistent secure
   storage is unavailable.
8. Legacy secure-store migration from `Speaking Studio` to `Vostavo` stays in
   place, but only within secure storage.

PAL review notes worth carrying into implementation:
1. environment keys may be detected for messaging, but never consumed
2. a dangling `secret_ref` must become a recoverable `credentials missing`
   state, not a crash or silent fallback
3. secure-storage failures need actionable platform-specific user messaging

## Implementation Tracks

### Track 1: Credential Core

Files:
1. `app_shell/secret_store.py`
2. `app_shell/runtime_resolver.py`
3. `app_shell/services.py`
4. `app_shell/diagnostics.py`

Changes:
1. Remove env-based secret lookup and env-based bootstrap hydration from the
   app-shell credential path.
2. Keep secure-store reads, writes, deletes, and legacy secure-store migration.
3. Replace env/session fallback statuses with secure-storage-only status
   reporting for app-shell runtime usage.
4. Treat a missing secret behind an existing `secret_ref` as `credentials
   missing` and let diagnostics surface that state cleanly.
5. Keep local providers usable without auth when they already support that
   today, but never source optional tokens from env.
6. Make remote auth-required providers unusable until a saved secure secret is
   available.

### Track 2: Setup And Settings UX

Files:
1. `pages/00_Setup.py`
2. `pages/06_Settings.py`
3. `pages/02_Speak.py`

Changes:
1. Stop pre-filling password fields with resolved runtime values.
2. Show non-sensitive state instead, for example `saved securely` or
   `credentials missing`.
3. Saving with a blank key on an existing connection must preserve the current
   secret.
4. Add an explicit `clear saved key` action with confirmation semantics.
5. When the provider choice changes in the form, clear the form's implicit
   secret association and require a new key if the selected provider needs one.
6. Update setup and settings save behavior so remote providers cannot be saved
   as active when persistent secure storage is unavailable.
7. Update Speak warnings so they refer only to saved connection credentials,
   never to shell environment variables.

Screen changes in this track should keep following the repo rule of discussing
the UX approach with PAL before implementation.

### Track 3: Localized Copy

Files:
1. `locales/en.json`
2. `locales/de.json`
3. `locales/it.json`
4. `locales/fr.json`

Changes:
1. Replace environment-fallback wording with saved-connection wording.
2. Add strings for:
   - saved securely
   - credentials missing
   - enter a new key to replace the saved key
   - clear saved key
   - secure storage required
   - environment keys are ignored in the desktop app
3. Update diagnostics, setup, settings, and speak wording so secure storage and
   missing-credential states are understandable without exposing raw secrets.

### Track 4: Remaining Locale And Core Tests

Files:
1. `locales/es.json`
2. `tests/test_secret_store.py`
3. `tests/test_runtime_resolver.py`
4. `tests/test_app_shell_diagnostics.py`

Changes:
1. Keep Spanish in sync with the other locale updates.
2. Replace env-fallback tests with saved-secret-only tests.
3. Add tests for:
   - env keys being ignored by app-shell secret resolution
   - legacy secure-store migration still working
   - dangling `secret_ref` reporting as missing credentials
   - remote-provider diagnostics failing cleanly without a stored secret

### Track 5: Page Tests

Files:
1. `tests/test_app_shell_pages.py`

Changes:
1. Replace env-based success cases with saved-secret cases.
2. Add coverage for:
   - blank edit preserves existing secret
   - explicit clear removes the saved secret
   - provider change invalidates secret reuse in the form
   - setup/settings copy no longer references env fallback
   - Speak warning uses saved-connection language only

## Runtime Rules After The Change

1. `workspace_prefs.json` stores connection metadata plus `secret_ref`, never
   raw secrets.
2. App-shell runtime loads credentials only from secure storage by `secret_ref`.
3. If `secret_ref` resolves to nothing, the connection remains visible but is
   treated as missing credentials.
4. If environment variables are present, the desktop app may warn that they are
   unsupported, but must not use them.
5. If secure storage is unavailable:
   - remote auth-required providers cannot be saved or used
   - local auth-optional providers may continue without a token
   - the UI must explain why saving the remote connection is blocked
6. Deleting a connection must also delete its stored secure secret.

## Verification

Implementation is complete when all of the following hold:
1. Launching the app with `OPENROUTER_API_KEY` or similar vars set does not make
   a remote connection usable unless that connection has a stored secret.
2. An existing saved OpenRouter connection still works after restart with shell
   env vars unset.
3. Editing a saved connection without entering a new key does not wipe the
   stored secret.
4. Clearing a saved key immediately moves the connection into a missing
   credentials state.
5. A missing secure-store entry behind `secret_ref` produces a learner-facing
   recovery path instead of a crash.
6. Setup, Settings, Speak, and diagnostics no longer mention env fallback as a
   supported credential source.

Recommended manual checks after implementation:
1. macOS with Keychain available and shell env vars unset
2. Windows with Credential Manager available and shell env vars unset
3. one remote saved connection with a stored key
4. one local Ollama connection without a key

## Open Decisions Already Settled For Implementation

1. Do not add automatic migration from shell env vars into secure storage.
2. Do not silently ignore env vars; if we surface them at all, it is warning
   only and never runtime fallback.
3. Do not keep session-only secrets for remote production connections.
4. Do not collapse the missing-secret state into a generic connection failure;
   it should be diagnosable and recoverable in the UI.

# Runtime Setup Rollout

Last updated: 2026-03-22  
Status: foundation shipped, connection-based setup flow still in progress

## Purpose

This document is the execution plan for moving the app from a single flat runtime configuration to a safer, more explicit runtime setup flow with:

- embedded `faster-whisper` for transcription
- explicit Whisper model installation
- local inference options for Ollama and LM Studio
- remote inference options for OpenRouter and other OpenAI-compatible providers
- secure local secret persistence through OS-backed storage when available

The goal is not to redesign the whole product. The goal is to make runtime setup reliable, understandable, testable, and expandable.

## Scope

Included in this rollout:

- Whisper setup and model download UX
- provider selection and basic provider testing
- secure secret persistence
- migration away from plain-text provider keys in app prefs
- connection-oriented runtime configuration
- mocked tests for local and remote provider wiring

Explicitly out of scope for this rollout:

- replacing `faster-whisper`
- remote ASR backends
- hosted auth
- billing, usage quotas, or account management
- native macOS or iOS packaging
- provider-specific model browsers beyond a basic technical test path

## Why This Direction

The repo already has the right building blocks:

- `asr.py` uses embedded `faster-whisper`
- `llm_client.py` already speaks to OpenRouter, Ollama, LM Studio, and generic OpenAI-compatible endpoints
- `app_shell/services.py` already moved part of the secret flow out of saved prefs
- `pages/06_Settings.py` already acts as a basic runtime setup surface

The missing pieces are mainly product shape and persistence discipline:

- one runtime config instead of reusable connections
- provider secrets not fully modeled as secrets
- no dedicated setup flow for first run
- no clean migration boundary between legacy prefs and the new runtime model

## Guiding Decisions

1. Keep `faster-whisper`.
   Transcription stays embedded and local-first. Ollama and LM Studio are inference providers, not ASR replacements.

2. Make Whisper installation explicit.
   Lazy-loading stays as a fallback, but setup should make model choice and model download visible.

3. Move from flat runtime fields to provider connections.
   Provider choice should be a saved connection, not a loose collection of fields.

4. Keep Whisper global.
   Whisper configuration belongs to the app runtime, not to any individual provider connection.

5. Use OS secure storage for secrets when available.
   Persistent secrets belong in Keychain or an equivalent backend via `keyring`, not in `dashboard_prefs.json`.

6. Support degraded modes honestly.
   If secure storage is unavailable, warn and use env or session-only behavior. Do not pretend plain files are secure.

7. Standardize LLM requests around an OpenAI-compatible runtime shape.
   The effective request contract should be:
   - `base_url`
   - `model`
   - `api_key`
   - `extra_headers`

8. Do not block save on health checks.
   Connection tests should help users, not trap them in setup.

## Current Baseline

### Already shipped

- `asr.py`
  - embedded `faster-whisper`
  - model cache detection
  - explicit download helper
  - lazy-load fallback
- `llm_client.py`
  - provider support for:
    - OpenRouter
    - Ollama
    - LM Studio
    - generic OpenAI-compatible endpoints
- `app_shell/secret_store.py`
  - secure-storage abstraction entry point
  - `keyring`-backed persistence when supported
  - fallback status reporting
- `app_shell/services.py`
  - secret migration out of plain-text prefs
  - runtime connection test helper
  - Whisper status and download helpers
- `pages/06_Settings.py`
  - provider/model/base URL/key entry
  - Whisper download action
  - provider test action

### Still missing

- a real multi-connection model
- migration logic isolated from service glue
- a runtime resolver that produces one normalized config for the rest of the app
- a dedicated first-run setup page
- stronger test coverage for migration and fallback modes
- cleanup of legacy flat runtime fields once the new path is proven

## Target Architecture

### Runtime layers

1. Global runtime preferences
   - UI locale
   - selected Whisper model
   - optional Whisper cache location
   - active provider connection id

2. Provider connections
   - local or remote inference endpoints
   - default model
   - auth mode
   - provider-specific metadata

3. Secret store
   - persistent OS-backed storage when available
   - env or session-only fallback when not

4. Runtime resolver
   - converts a saved connection plus its secret into:
     - `base_url`
     - `model`
     - `api_key`
     - `extra_headers`

5. Setup UI
   - chooses Whisper model
   - installs Whisper model
   - creates or edits provider connections
   - tests the chosen runtime

### Provider connection model

```python
@dataclass
class ProviderConnection:
    connection_id: str
    provider_kind: Literal["ollama", "lmstudio", "openrouter", "openai_compatible"]
    label: str
    base_url: str
    default_model: str
    auth_mode: Literal["none", "bearer"]
    secret_ref: str
    is_default: bool
    is_local: bool
    provider_metadata: dict[str, Any]
    last_test_status: str
    last_tested_at: str
```

### Metadata expectations

- OpenRouter
  - `http_referer`
  - `app_title`
- Ollama
  - local vs remote UI hinting
  - optional cloud labeling
- LM Studio
  - optional local-token hinting
- Generic OpenAI-compatible
  - no required metadata in the first pass

## Implementation Plan

### Phase 0: Foundation and Secret Discipline

Status: partially shipped

### Outcome

The app can already:

- detect Whisper model cache state
- trigger Whisper download explicitly
- save runtime secrets outside plain-text prefs
- test a selected provider configuration

### Remaining work in this phase

- replace ad hoc helper usage with one consistent `SecretStore` interface everywhere
- cover failure and fallback modes with dedicated tests
- optionally honor a global Whisper cache directory

### Primary files

- `app_shell/secret_store.py`
- `app_shell/services.py`
- `asr.py`
- `tests/test_secret_store.py`
- `tests/test_app_shell_services.py`
- `tests/test_asr.py`

### Acceptance criteria

- no provider API key is written back into `dashboard_prefs.json`
- legacy plain-text OpenRouter secrets are migrated out on load
- secure-storage status is available for UI copy
- missing secure storage does not silently degrade into file persistence

### Phase 1: Connection Model and Migration Scaffolding

Status: planned

### Outcome

The app can hold multiple saved provider connections while still reading old flat prefs during migration.

### Work

- add `ProviderConnection` and `active_connection_id`
- keep old fields readable during migration:
  - `provider`
  - `model`
  - `llm_base_url`
  - `llm_api_key`
  - `openrouter_api_key`
- isolate migration logic into `app_shell/migrations.py`
- make migration idempotent
- enforce exactly one default connection

### Primary files

- `app_shell/state.py`
- `app_shell/provider_types.py`
- `app_shell/migrations.py`
- `app_shell/runtime_providers.py`
- `tests/test_app_shell_state.py`
- `tests/test_runtime_migrations.py`

### Acceptance criteria

- if no connections exist and legacy runtime fields do, one connection is created from the legacy runtime
- repeating bootstrap does not duplicate connections
- connection defaults are normalized to one default
- legacy fields remain readable until the new flow is stable

### Phase 2: Runtime Resolver and Client Cleanup

Status: partially shipped

### Outcome

All runtime inference calls flow through one normalized configuration regardless of provider.

### Work

- add `app_shell/runtime_resolver.py`
- move provider-specific header derivation behind the resolver
- standardize resolution into:
  - `base_url`
  - `model`
  - `api_key`
  - `extra_headers`
- keep CLI and env aliases as compatibility inputs during migration

### Primary files

- `app_shell/runtime_resolver.py`
- `llm_client.py`
- `app_shell/services.py`
- `assess_speaking.py`
- `tests/test_llm_client.py`
- `tests/test_app_shell_services.py`
- `tests/test_assess_speaking.py`

### Acceptance criteria

- OpenRouter headers are derived centrally
- LM Studio, Ollama, and generic compatible endpoints use the same runtime contract
- missing secret conditions surface a useful message rather than corrupting saved state
- legacy CLI usage still works

### Phase 3: Dedicated First-Run Setup Flow

Status: planned

### Outcome

First run should land on a guided setup page instead of asking the user to discover everything in Settings.

### Sections

1. Whisper
   - choose a model
   - show cache state
   - download or install explicitly
   - explain size and speed tradeoffs briefly

2. Inference provider
   - Ollama local
   - Ollama remote or cloud-compatible
   - LM Studio local
   - OpenRouter
   - generic OpenAI-compatible

3. Connection details
   - label
   - base URL
   - model
   - optional bearer token or API key

4. Test connection
   - fast health or model-list check
   - optional short inference smoke test
   - save remains available even if the test fails

### Navigation rules

- first run routes to setup if no provider connection exists
- setup is considered complete once the first connection is saved
- Settings remains the long-term management surface after setup

### Primary files

- `pages/00_Setup.py`
- `pages/06_Settings.py`
- `app_shell/page_helpers.py`
- `streamlit_app.py`
- `locales/en.json`
- `locales/de.json`
- `locales/it.json`
- `tests/test_app_shell_pages.py`

### Acceptance criteria

- first-run routing works
- the user can download a Whisper model without leaving setup
- the user can save either a local or remote provider configuration
- translation coverage is complete for the new UI copy

### Phase 4: Provider-Specific Polish

Status: planned

### Outcome

The setup experience should feel intentional for the providers we actually support.

### Provider rules

- Ollama local
  - default URL: `http://localhost:11434/v1`
  - no auth by default
  - test with health or model list, then optional short completion

- Ollama remote or cloud-compatible
  - supports bearer auth
  - UI copy should distinguish it from local Ollama

- LM Studio local
  - default URL: `http://localhost:1234/v1`
  - token support should be allowed but can remain optional in the first polished pass
  - test with model list, then optional short completion

- OpenRouter
  - secret in `SecretStore`
  - free-text model input in the first pass
  - resolver adds required headers

- Generic OpenAI-compatible
  - power-user path
  - base URL + model + optional bearer key

### Primary files

- `app_shell/runtime_providers.py`
- `app_shell/services.py`
- `pages/00_Setup.py`
- `pages/06_Settings.py`
- `tests/test_app_shell_services.py`

### Acceptance criteria

- each supported provider has a usable default path in setup
- local and remote Ollama are distinguishable in the UI
- LM Studio works without forcing auth
- OpenRouter power users can enter a key once and keep it out of plain-text prefs

### Phase 5: Cleanup, Docs, and Rollout Hardening

Status: planned

### Outcome

The new setup path is test-covered, documented, and ready to replace the legacy flat runtime path.

### Work

- finish migration coverage
- remove dead secret-writing paths
- document the setup flow in `README.md`
- add manual smoke instructions for local users
- delay legacy-field removal until the new flow has proven stable

### Primary files

- `README.md`
- `docs/RUNTIME_SETUP_ROLLOUT.md`
- `tests/test_secret_store.py`
- `tests/test_runtime_migrations.py`
- `tests/test_app_shell_pages.py`
- `tests/test_app_shell_services.py`
- `tests/test_llm_client.py`
- `tests/e2e/test_streamlit_e2e.py`

### Acceptance criteria

- README setup instructions match the actual UI
- migration tests pass against legacy prefs fixtures
- the app is usable with:
  - local Whisper + local Ollama
  - local Whisper + LM Studio
  - local Whisper + OpenRouter

## Security and Persistence Rules

1. Persistent secrets must use OS secure storage when available.
2. The app must never silently write provider secrets back into `dashboard_prefs.json`.
3. If secure storage is unavailable:
   - warn clearly
   - prefer existing env vars when present
   - otherwise use session-only behavior
4. Migration must be idempotent.
5. Connection tests must never be treated as proof of safety, only as technical reachability checks.

## Test Matrix

### Unit and service coverage

- secure storage success path
- secure storage unavailable path
- env and session fallback behavior
- Whisper cache detection
- Whisper download state transitions
- connection migration from legacy prefs
- exactly-one-default connection invariant
- runtime resolver output for each provider type

### Mocked provider coverage

- Ollama local
- Ollama remote with bearer auth
- LM Studio without token
- LM Studio with token
- OpenRouter with required headers
- generic OpenAI-compatible base URL

### UI coverage

- first-run setup routing
- setup save flow
- secure-storage warning copy
- Whisper download action
- provider technical test success and failure messages
- translation parity for setup strings

### Manual smoke

- local Ollama instance running
- local LM Studio server running
- Whisper `tiny` model download on a clean machine

## File Ownership Map

Existing files with ongoing ownership in this rollout:

- `asr.py`
- `assess_speaking.py`
- `llm_client.py`
- `settings.py`
- `app_shell/state.py`
- `app_shell/runtime_providers.py`
- `app_shell/secret_store.py`
- `app_shell/services.py`
- `pages/06_Settings.py`

Expected new files:

- `app_shell/provider_types.py`
- `app_shell/migrations.py`
- `app_shell/runtime_resolver.py`
- `pages/00_Setup.py`
- `tests/test_secret_store.py`
- `tests/test_runtime_migrations.py`

Likely supporting touch points:

- `app_shell/page_helpers.py`
- `streamlit_app.py`
- `locales/en.json`
- `locales/de.json`
- `locales/it.json`
- `README.md`

## Packaging and Hosted Readiness

This rollout is intentionally local-first, but it should not trap the app in a local-only design.

Packaging implications:

- the setup flow should become the primary onboarding surface for a packaged desktop build
- secure secret storage should work well on macOS through Keychain via `keyring`
- other platforms may degrade, but the app should surface that honestly
- Whisper installation and provider testing should work without requiring a hosted backend

Hosted implications:

- provider connections should remain separate from future account identity
- the long-term hosted model should be:
  - `account_id`
  - `learner_profile_id`
  - `provider_connection_id`
  - `attempt_id`
- nothing in this rollout should assume `speaker_id` is the same thing as a future authenticated user
- future hosted auth can be added later without redesigning the provider and secret model

Important boundary:

- macOS secure storage is directly supported by this rollout
- iOS secure storage is not part of this rollout because the app is not an iOS-native product today

## Risks and Guardrails

- Risk: migrating secrets twice through overlapping helper paths.
  Guardrail: centralize migration in `app_shell/migrations.py`.

- Risk: legacy prefs resurrect plain-text secret persistence.
  Guardrail: tests should assert secrets do not reappear in saved prefs.

- Risk: setup becomes a hard blocker because provider tests fail.
  Guardrail: saving remains possible even if connection tests fail.

- Risk: LM Studio token support complicates the first pass.
  Guardrail: treat LM Studio auth as optional and defer advanced token UX if needed.

- Risk: macOS behavior is good but other platforms are inconsistent.
  Guardrail: expose storage status explicitly and test degraded modes.

## Deferred Follow-Ups

Keep these out of the core rollout:

- iOS-native secure storage implementation
- hosted auth or account sync
- remote ASR backends
- provider billing or quota management
- provider auto-discovery beyond simple local defaults
- model catalog or pull UX for LM Studio and Ollama
- usage analytics or per-connection cost tracking

## Definition of Done

This rollout is done when:

- a first-run user can choose and install a Whisper model
- a first-run user can save a local or remote inference connection
- secrets are persisted outside plain-text prefs on supported systems
- degraded secret-storage modes are clearly surfaced
- legacy prefs migrate cleanly into the new model
- the app still supports existing CLI and env-driven behavior
- the targeted unit, service, UI, and mocked-provider tests pass

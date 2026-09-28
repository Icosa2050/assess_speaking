# Shared Frontend Parity Inventory

Last updated: 2026-05-17
Status: Legacy parity inventory for the shared React + Tauri learner flow;
superseded for execution by the active Streamlit retirement plan

## Purpose

This document records the Streamlit-era local-desktop learner flow that the
React port used as its parity baseline. It deliberately separates three
concerns:

1. route and transition parity from the legacy Streamlit app
2. localization parity from the existing locale namespaces
3. a new semantic-id contract for React, Playwright, and Maestro

The local FastAPI contract remains canonical. Visible learner copy must continue
to come from the existing locale files. Semantic ids are the cross-framework
contract and must not be inferred from translated text.

Current execution note:
1. React/Tauri is now the primary local desktop UI lane
2. Streamlit files in this document are parity references and deletion gates,
   not targets for new product UX
3. use `docs/superpowers/plans/2026-05-16-streamlit-retirement.md` for the
   active task order

## Phase-2 scope

### Core learner-flow parity

| Shared route | Current source | Entry and guard rules | Primary next targets |
| --- | --- | --- | --- |
| `/` | `streamlit_app.py` | Always reachable. If runtime setup is incomplete, show runtime-setup branch instead of start/resume branch. | Runtime Setup, Session Setup, Speak, Review, History, Settings, secondary screens |
| `/runtime-setup` | `pages/00_Setup.py` | Reachable from Home and Settings. Used whenever runtime readiness is missing or the user wants to edit local runtime settings. | Home, Session Setup after save, Settings return path |
| `/session-setup` | `pages/01_Session_Setup.py` | Start-new from Home. Continue goes to Speak when an active connection exists, otherwise Runtime Setup. | Speak, Runtime Setup |
| `/speak` | `pages/02_Speak.py` | Requires saved session setup. If runtime connection is missing, guard to Runtime Setup. | Review, Runtime Setup, Session Setup guard |
| `/review` | `pages/03_Review.py` | Requires review payload or an in-flight assessment. Missing review guards back to Speak. | Speak, Session Setup, History |
| `/history` | `pages/04_History.py` | Reachable from Home or Review. Speaker and language scope are optional filters, not route requirements. | Review detail browsing, future reopen actions |

### Adjacent phase-2 support parity

| Shared route | Current source | Why it stays in scope |
| --- | --- | --- |
| `/settings` | `pages/06_Settings.py` | Owns saved connections, UI locale, runtime support actions, cleanup, and support-bundle workflows. It is not part of the main learner loop, but it is phase-2 parity work rather than a later product lane. |

### Follow-on parity after the main learner flow

| Shared route | Current source | Phase note |
| --- | --- | --- |
| `/library` | `pages/05_Library.py` | Secondary surface tracked after the main learner flow is stable. |
| `/guide` | `pages/07_Scoring_Guide.py` | Secondary reference surface tracked after the main learner flow is stable. |

## Route and transition parity

### Home

- Source: `streamlit_app.py`
- Localized intro and shell summary are always present.
- Runtime-ready branch:
  - `home.start_new` clears current attempt state and navigates to Session Setup.
  - `home.resume` is disabled until setup exists; it resumes to Review if a review
    already exists, otherwise Speak.
- Runtime-blocked branch:
  - `home.runtime_setup_button` navigates to Runtime Setup.
- Secondary actions remain reachable from Home:
  - `nav.history`
  - `nav.library`
  - `nav.guide`
  - `nav.settings`
- Diagnostics checklist remains visible on the primary screen.
- The debug expander is a development aid; if omitted from the first React pass,
  that gap must be explicit in the port task.

### Runtime Setup

- Source: `pages/00_Setup.py`
- Learner-visible sections stay explicit:
  - Whisper readiness
  - Provider choice
  - Connection fields
  - Action area
- Local-first behavior is preserved:
  - model download
  - local model detection
  - connection test
  - connection save
  - back-home navigation
- Saved-secret states remain explicit:
  - secret present
  - secret missing
  - clear-secret confirm/cancel/undo
- The React port must preserve secure local credential semantics and avoid any
  hosted-auth or env-secret fallback language.

### Session Setup

- Source: `pages/01_Session_Setup.py`
- Required learner inputs:
  - speaker id
  - learning language
  - CEFR level
  - theme or custom theme
  - duration
- Preview panel parity matters because it explains the selected task before the
  learner enters Speak.
- Continue behavior is guarded by runtime readiness:
  - active connection present -> Speak
  - no active connection -> Runtime Setup

### Speak

- Source: `pages/02_Speak.py`
- Guard rules are strict:
  - missing setup -> redirect to Session Setup
  - missing runtime connection -> redirect to Runtime Setup
- Core parity states:
  - idle
  - audio attached
  - assessing
  - completed
  - failed
- The main React flow must preserve both upload and recorder entry paths, plus
  explicit job-status feedback for queued, running, completed, failed, and
  canceled work.

### Review

- Source: `pages/03_Review.py`
- Review is allowed to self-heal by polling if an assessment is still in flight.
- Guard outcomes:
  - still assessing -> go back to Speak
  - missing review -> go to Speak
- Action parity:
  - `review.try_again` clears only the current attempt and returns to Speak
  - `review.new_setup` clears setup and returns to Session Setup
  - `review.view_history` opens History

### History

- Source: `pages/04_History.py`
- History remains useful without a speaker scope, but it prefers the current
  speaker and language when available.
- Parity includes:
  - language filter
  - attempts jump buttons
  - details select control
  - attempts table
  - detail report rendering through the existing review summary/report panels
- The React port must preserve degraded-vs-complete clarity rather than reducing
  History to a plain list view.

### Settings

- Source: `pages/06_Settings.py`
- Settings remains phase-2 parity because it owns:
  - saved connections
  - active/default connection selection
  - UI locale changes
  - model and base URL edits
  - whisper model download
  - connection testing and save
  - storage summary
  - cleanup preview and run
  - support-bundle generation
- The React port keeps `/settings` as one route, but splits the screen into four
  sibling sections rather than one long mixed form or a tab set:
  - Saved Connections
  - Runtime Defaults
  - Maintenance
  - Support Bundle
- Route-level return behavior is explicit in React:
  - `/settings` accepts an optional origin token from Home, Runtime Setup,
    Session Setup, Speak, Review, or History
  - one shared return control owns navigation back to that origin
  - missing or unknown origins fall back to Home
- Secret-preservation behavior must stay aligned with Runtime Setup:
  - saved secret present
  - saved secret missing
  - clear-secret confirmation pending
  - cleared-with-undo available before save
- Phase-2 may add local-desktop runtime-management endpoints for parity:
  - load runtime settings
  - save runtime settings
  - test a connection or discover local models
  - set default or delete a saved connection
  - inspect or download whisper models
- Hosted sign-in, hosted identity, and hosted-only credential flows remain out
  of scope.
- Additional out-of-scope fences for phase 2:
  - no env-secret fallback UI or helper copy in the React screen
  - no generic debug or log-browser panel
  - no connection import/export workflow

## Localization parity

The React port must keep these locale namespaces canonical and read them from
the repo locale JSON files instead of duplicating strings in TypeScript:

- `nav.*`
- `home.*`
- `runtime_setup.*`
- `setup.*`
- `speak.*`
- `review.*`
- `history.*`
- `settings.*`
- `task_family.*`

Additional parity rules:

1. Keep visible headings, button labels, helper text, and guard copy localized.
2. Preserve the current route-level titles and section headers used by the
   Playwright tests.
3. Treat browser-native file-picker text such as `Browse files` as incidental,
   not as the automation contract.
4. Keep accessibility assertions role-first; use semantic ids for stable
   automation targeting rather than replacing accessible names.

## Semantic-id registry

This registry is new. It is not a direct port of Streamlit widget keys.

Rules:

1. Every interactive learner-flow element used by React tests or Maestro must
   expose a stable `data-testid`.
2. Maestro semantic labels should use the same id string when a second hook is
   needed.
3. Use locale-key-shaped ids so the namespace stays readable across frameworks.
4. Dynamic rows keep a base semantic id plus stable data attributes such as
   `data-connection-id` or `data-report-id`.

| Surface | Current Streamlit key | Shared semantic id |
| --- | --- | --- |
| Home runtime setup CTA | `home_runtime_setup` | `home.runtime_setup_button` |
| Home start new | `home_start_new` | `home.start_new` |
| Home resume | `home_resume` | `home.resume` |
| Home open history | `home_history` | `home.open_history` |
| Home open library | `home_library` | `home.open_library` |
| Home open guide | `home_guide` | `home.open_guide` |
| Home open settings | `home_settings` | `home.open_settings` |
| Runtime setup download model | `runtime_setup_download_model` | `runtime_setup.download_model` |
| Runtime setup detect local models | `runtime_setup_detect_local_models` | `runtime_setup.detect_local_models` |
| Runtime setup test connection | `runtime_setup_test_connection` | `runtime_setup.test_connection` |
| Runtime setup save connection | `runtime_setup_save_connection` | `runtime_setup.save_connection` |
| Runtime setup back home | `runtime_setup_back_home` | `runtime_setup.back_home` |
| Runtime setup clear secret | `runtime_setup_clear_saved_key` | `runtime_setup.clear_saved_key` |
| Runtime setup confirm clear secret | `runtime_setup_clear_saved_key_confirm` | `runtime_setup.clear_saved_key_confirm` |
| Runtime setup cancel clear secret | `runtime_setup_clear_saved_key_cancel` | `runtime_setup.clear_saved_key_cancel` |
| Session setup speaker id | `setup_speaker_id` | `setup.speaker_id` |
| Session setup language | `setup_learning_language` | `setup.learning_language` |
| Session setup CEFR | `setup_cefr` | `setup.cefr` |
| Session setup theme | `setup_theme_select` | `setup.theme` |
| Session setup custom theme | `setup_custom_theme` | `setup.custom_theme` |
| Session setup save custom theme | `setup_save_custom_theme` | `setup.save_custom_theme` |
| Session setup duration | `setup_duration` | `setup.duration` |
| Session setup continue | `setup_continue` | `setup.continue` |
| Speak audio recorder input | `speak_audio_input` | `speak.audio_input` |
| Speak upload input | `speak_upload` | `speak.upload_input` |
| Speak remove recording | `speak_remove_recording` | `speak.remove_recording` |
| Speak label | `speak_label` | `speak.label` |
| Speak notes | `speak_notes` | `speak.notes` |
| Speak cancel assessment | `speak_cancel_assessment` | `speak.cancel_assessment` |
| Speak submit | `speak_submit` | `speak.submit` |
| Review try again | `review_try_again` | `review.try_again` |
| Review new setup | `review_new_setup` | `review.new_setup` |
| Review view history | `review_view_history` | `review.view_history` |
| History language filter | `history_learning_language` | `history.learning_language` |
| History detail select | `history_detail_report` | `history.details_select` |
| History jump action | `history_jump_{idx}` | `history.jump` |
| Settings open setup | `settings_open_setup` | `settings.open_setup` |
| Settings connection selector | `settings_connection_id` | `settings.connection_id` |
| Settings UI locale | `settings_ui_locale` | `settings.ui_locale` |
| Settings provider | `settings_provider` | `settings.provider` |
| Settings connection label | `settings_connection_label` | `settings.connection_label` |
| Settings model | `settings_model` | `settings.model` |
| Settings base URL | `settings_base_url` | `settings.base_url` |
| Settings API key | `settings_api_key` | `settings.api_key` |
| Settings whisper model | `settings_whisper_model` | `settings.whisper_model` |
| Settings clear saved key | `settings_clear_saved_key` | `settings.clear_saved_key` |
| Settings confirm clear saved key | `settings_clear_saved_key_confirm` | `settings.clear_saved_key_confirm` |
| Settings cancel clear saved key | `settings_clear_saved_key_cancel` | `settings.clear_saved_key_cancel` |
| Settings undo clear saved key | `settings_clear_saved_key_undo` | `settings.clear_saved_key_undo` |
| Settings test connection | `settings_test_connection` | `settings.test_connection` |
| Settings save | `settings_save` | `settings.save` |
| Settings cleanup preview | `settings_support_cleanup_preview` | `settings.support_cleanup_preview` |
| Settings cleanup run | `settings_support_cleanup_run` | `settings.support_cleanup_run` |
| Settings cleanup run confirm | `n/a` | `settings.support_cleanup_run_confirm` |
| Settings cleanup run cancel | `n/a` | `settings.support_cleanup_run_cancel` |
| Settings support bundle | `settings_support_create_bundle` | `settings.support_create_bundle` |
| Settings return | `settings_back` | `settings.return` |

## Automation notes

Current end-to-end coverage is mostly text- and role-based:

- `tests/e2e/test_app_shell_e2e.py`
- `tests/e2e/test_app_shell_runtime_setup_e2e.py`
- `tests/e2e/test_app_shell_real_history_e2e.py`

That remains useful as an accessibility check, but React parity work should
shift primary automation targeting to semantic ids for:

- major route actions
- form controls that recur across locales
- dynamic row actions in History and Settings
- long-running job-state controls in Speak

## Implementation notes for the first React pass

1. Scaffold first, screens second.
2. Read locale JSON directly from the repo so visible copy stays aligned with
   Streamlit during the migration.
3. Keep the local guest flow whole before optimizing for hosted or packaged
   variants.
4. Freeze Settings data seams before Task 51:
   - `runtime.settings` is local server state from the runtime-management API
   - `runtime` stays read-only server state from `/v1/runtime`
   - `maintenance.storage` and `maintenance.cleanup` stay separate server-state
     operations
   - `support-bundle` stays command-style local UI state rather than cached
     route data
5. Saving a connection invalidates `runtime.settings` and `runtime`. Cleanup
   invalidates `maintenance.storage`. Support-bundle creation only updates the
   last-run route state.
6. Treat this document as the shared reference for Tasks 44 through 58 rather
   than re-inferring parity screen by screen.

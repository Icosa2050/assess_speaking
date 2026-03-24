# Multipage App Shell Plan

This document replaces the "one long dashboard" direction with a testable app shell.

## Why The Dashboard Gets Replaced

The current Streamlit dashboard mixes too many jobs on one screen:

- session setup
- prompt delivery
- recording
- review
- progress history
- theme library management
- provider and model settings

The replacement principle is simple: one primary job per screen.

## Screen Map

### 1. Home

Purpose:
- start a new session
- resume a draft
- jump to history, library, or settings

Primary action:
- `Start new session`

### 2. Session Setup

Purpose:
- choose learning language
- choose CEFR level
- choose theme
- choose target duration

Primary action:
- `Continue to speaking`

### 3. Speak

Purpose:
- show one prompt
- record or upload one response
- submit one attempt

Primary action:
- `Submit for review`

### 4. Review

Purpose:
- show transcript
- show score
- show coaching summary
- retry same prompt or change task

Primary actions:
- `Try again`
- `Change task`

### 5. History

Purpose:
- browse previous attempts
- filter and inspect results

### 6. Library

Purpose:
- manage languages
- manage themes
- preview prompt collections

### 7. Settings

Purpose:
- change UI locale
- choose provider
- choose model
- hold advanced options outside the practice flow

## Transition Model

```text
Home -> Session Setup -> Speak -> Review
Home -> History
Home -> Library
Home -> Settings
Review -> Speak     (retry same prompt)
Review -> Setup     (change task)
Settings -> return_to_previous_page
```

Guard rules:

- `Speak` requires a valid setup snapshot.
- `Review` requires a submitted attempt.
- management screens stay reachable, but they do not interrupt the core speaking flow.

## State Model

The shell keeps one structured object in `st.session_state`, not many unrelated widget keys.

### Preferences

- `ui_locale`
- `provider`
- `model`

### Draft session

- `session_id`
- `learning_language`
- `learning_language_label`
- `cefr_level`
- `theme_id`
- `theme_label`
- `duration_sec`
- `prompt_id`
- `prompt_text`

### Recording state

- `status`
- `audio_path`
- `duration_sec`

### Review state

- `report_id`
- `transcript`
- `score_overall`
- `band`
- `summary`

### Navigation state

- `current_page`
- `return_to`

## Language Rules

The app keeps two language concepts only:

- `ui_locale`: interface language, warnings, helper text, labels, and coaching wrapper
- `learning_language`: language the learner is expected to speak

Rules:

- do not introduce a third user-facing language setting
- do not let UI locale silently change the learning language
- every user-facing string must go through locale files

## Testing Strategy

Every screen must be testable in isolation.

### Pure tests

- state defaults
- state transitions
- locale key completeness
- translation lookup

### Page tests

Each page gets:

- smoke render
- guard behavior
- locale render
- one happy-path assertion

### End-to-end tests

Minimum flow coverage:

- new session -> setup -> speak -> review
- resume draft
- German UI with Italian learning language
- save theme in Library, then select it in Setup
- settings change and return to previous page

## Backend Migration Seams

The shell should not bind page code directly to local files forever.

Introduce interfaces:

- `SessionRepository`
- `ThemeRepository`
- `HistoryRepository`
- `AssessmentService`

Suggested migration path:

1. keep the experimental shell local and session-state backed
2. add SQLite for draft sessions and persisted shell data
3. move to FastAPI plus Postgres when multi-user or cross-device resume matters

SQLite is good enough for local single-user drafts and history. It is not the long-term backend for shared sessions or queued assessments.

## Why The Shell Uses `pages/`

This scaffold uses Streamlit's file-based multipage layout so each page can be tested directly with `streamlit.testing.v1.AppTest`.

## Current Scope

This scaffold is intentionally narrow:

- clean screen boundaries
- structured session state
- locale files
- page guards
- test hooks

It does not yet replace the assessment pipeline. That migration happens after the shell is stable.

# Timed oral practice — implementation and verification

## Delivered workflow

The new **Timed speaking rehearsal** screen offers original English and Italian exercises for B1/B2/C1 goals. A full session has five minutes of shared preparation and three speaking parts capped at three, three and four minutes. A learner may finish a phase early. This is an app-authored solo training sequence; official CILS/telc timing profiles, an examiner and a responsive conversation partner remain separate work.

Completed parts and their manifest are saved atomically in native IndexedDB before advancing. Reloading preserves those parts. The current recording remains in memory until saved; the screen explicitly explains that leaving during recording discards that unsaved part. Each part has a bounded recorder, so the whole session is not buffered as one growing recording. Storage errors leave the current file available for retry.

Audio can be played/downloaded locally before requesting AI feedback. Analysis uploads each saved part through the existing bounded multipart path and runs one assessment at a time. Saved job IDs allow polling to resume after navigation. Pausing the wait leaves the backend job running for later retrieval. Browser locks prevent overlapping analysis in the same origin, and IndexedDB revision checks prevent stale tabs overwriting a newer manifest. Accepted results also appear in the existing backend History.

The combined review shows each part's target duration, actual recording duration, speaking-time WPM, existing practice score, coaching and next focus. Generic fallback and uncertain transcription are labelled. A targeted retry retains the chosen part's exact prompt, goal, duration, model configuration and saved parent session. It gets one minute of preparation; compatible before/after comparison uses existing History provenance. Narrative and opinion parts keep distinct task families so their scores do not share a progress series merely because both use three minutes.

Rehearsal removal deletes browser copies after a second explicit click; backend History remains. English and Italian have localized UI and authored prompts. New rehearsal strings use English fallback copies in the three other existing UI locales. There are no new dependencies.

## Review

Claude CLI reviewed selected source and authored tests only. Host review verified and fixed concurrent/stale session writes, expired upload retry, malformed completed-job recovery, unbounded waiting without an escape, microphone leakage on MediaRecorder construction failure, recorder-error partial files being mistaken for successful recordings, and Ogg extension handling. The same source was reviewed with OCR delegate rules; coverage accompanies this report.

Native IndexedDB transaction failure, saved state and two-tab behavior are browser-tested. An ambiguous network outcome during assessment creation still depends on the existing API contract: there is no new server-wide request-idempotency protocol in this slice. Browser storage is scoped to its origin/profile and can be cleared by the browser/user; analysed recordings retain their normal backend History lifecycle.

## Verification scope

The bilingual journey uses real MediaRecorder, IndexedDB, uploads, jobs and saved History, with explicitly labelled fixture inference. It checks preparation deadlines, timed stops, early completion, quota/disk failure recovery, reload, existing-job resumption, tab conflicts, playback/seek, range responses, a linked retry, and a narrow mobile viewport.

Deadline tests advance the browser clock after recording short real media. They do **not** provide a fifteen-minute recorded-audio soak, physical-device coverage or evidence of live AI feedback quality. B1/B2/C1 prompt/timing contracts are tested for all six combinations; the new browser journeys use B2 in both languages.

Final test results and commit details are recorded in the task response and the updated PR. Representative screenshots are retained under `frontend/output/playwright/journeys/`; host inspection confirmed readable English/Italian part cards, playback controls and targeted retry actions.

### Verification commands (2026-10-04)

- `npm --prefix frontend test`: 149 passed (including recorder cleanup and all six goal/language timing contracts).
- `npm --prefix frontend run typecheck` and `npm --prefix frontend run build`: passed.
- `node frontend/node_modules/playwright/cli.js test --config frontend/playwright.journeys.config.ts timedRehearsal.spec.ts`: 2 passed, 39.8 seconds on the final rehearsal implementation. Includes synchronous quota rejection and asynchronous transaction rollback, stale-tab conflict, competing analysis lock, expired upload, pause/reload/resume, local playback/seek and saved backend audio range requests.
- All 12 existing shared journeys passed after the recorder changes, including English/Italian B1/B2/C1 loops, failure recovery and WebKit in both languages. The combined run also exposed a two-tab test setup race; waiting for the second tab to finish opening before mutating the first fixed the test. The affected two rehearsal journeys were rerun separately and both passed; the final replay also verifies distinct task families and exact retry provenance.
- `git diff --check` and `.venv/bin/python scripts/check_quality.py`: passed.

Test-authoring fixes: Playwright worker launch options belong at file scope; the shared playback helper now accepts an explicit minimum duration while preserving its ten-second live default; short rehearsal fixtures specify one second. Tests wait for the saved-part acknowledgement before reload. Failed authoring runs were excluded from the pass totals.

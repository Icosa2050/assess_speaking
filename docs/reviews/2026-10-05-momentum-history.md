# Momentum history — implementation and wording pass

Date: 2026-10-05

## Approved direction

The user selected the Momentum mockup: graphical progress, a flexible weekly practice rhythm, specific reasons to repeat an exercise, recording playback, and accessible saved reviews. English and Italian are the current practice languages. All new interface strings are translated into the existing five UI locales (English, Italian, German, French, Spanish); this does not add German assessment support.

The approved headline remains “Little by little. Speak with more confidence.” B1/B2/C1 remain chosen practice goals.

## Implemented

- A real-data chart with connecting lines, a fixed 0–5 score axis, selectable recording metrics, and a factual score change from the first scored comparable attempt. The existing compatibility checks still isolate learners, languages, goals, task families, durations and analysis settings. Topics can differ; the UI says so.
- Weekly practice-day ring, weekday markers, adjustable one-to-seven-day goal, and local calendar boundaries. Multiple recordings on one day count once. Other learners/languages, dry runs, invalid dates and future dates do not add personal activity.
- Goals persist locally per learner/language, with an in-memory fallback if browser storage is unavailable. They are not synchronized between browsers/devices.
- A next-attempt action restores the saved prompt; actual recording controls provide playback. No mockup waveforms, example scores or invented milestones are shipped.
- Single-recording wording, factual activity/retry counts and positive signals independent of score gains. Negative/zero score changes remain visible as neutral facts.
- One saved-review list with explicit learner/language filters, full feedback, recoverable empty-filter state, and compact mobile navigation.
- History waits for refreshed results before automatically choosing the latest report. Saved UI language is restored when History is opened directly.

## Independent reviews

- PAL Qwen reviewed the English/Italian motivational wording after implementation. Accepted idiomatic Italian “Ogni passo conta”, singular day labels, simpler factual delta wording and retry terminology. Kept the user-approved aspirational headline instead of replacing it with a fragment.
- Claude CLI reviewed selected implementation files plus English/Italian strings. Addressed first-recording copy, safe legacy priorities, singular goals, calendar rollover, rounded “+0” changes, non-score metric/delta ambiguity, and inaccessible labels. Disabled links to attempts without a saved report while retaining their measured comparison data.
- OCR delegate preview/rule resolution: 16 reviewable implementation/locale files, all 16 reviewed by the host; 0 skipped, 100% coverage. Six changed/new test files were also reviewed separately (OCR excludes tests by default). Pre-existing unrelated untracked files were excluded and untouched.

Review input contained selected source/interface wording only, no credentials or learner reports.

## Verification

- Frontend: 162 tests passed; TypeScript check passed.
- Backend: 17 focused history/progress/i18n tests passed (21 deselected).
- Bilingual browser journeys: 10 passed across English/Italian B1, B2 and C1, including upload, microphone retry, saved history, playback, changed goals/languages and error recovery. These use the journey fixture inference backend, not new live model calls.
- Final isolated browser checks: 3 passed — empty Review/History, two-attempt review/history flow, and visual/responsive smoke down to 320/360px where covered by the existing suites.
- Live app: backend `/v1/health` and frontend `/history` returned 200. Existing saved ChatGPT sample reviews remained available. Italian headline and one-day labels verified after reload.

The isolated visual run used Vite port 4191 because another checkout occupied the normal smoke-test port. The temporary config was removed afterward; test source changes remain reproducible under the standard config with its intended servers.

## Remaining scope

The separately recorded older raw/weighted overall-delta calculation, free-text priority-resolution comparison, and intermittent submission transport error are not claimed fixed here. This change adds a local practice goal, not account/profile management or cross-device sync. Broader slogan/translation refinements can continue from this implemented page.

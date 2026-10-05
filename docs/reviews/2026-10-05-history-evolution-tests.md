# Upload, microphone, and History evolution

Scope: English and Italian, each at the learner's B1, B2, and C1 goal.

## Coverage added

Each of the six journeys now saves an upload, a microphone retry, and another upload retry. At one, two, and three saved attempts it checks:

- The learner/language filtered list has the correct count and newest-first order.
- The selected report is the newest saved attempt, including after reload.
- Every plotted score, duration, speaking-rate, and pause-time point matches persisted API data.
- The four before/after/change rows match the correct retry parent and localized numerical values.
- A single attempt has no connecting line or fabricated comparison.
- Selecting the first attempt removes later points from that historical view; selecting the latest restores all three.
- Earlier and current recordings play; seeking and HTTP byte ranges work.
- Three attempts on one day still earn one practice day, and the chosen weekly goal survives reload.
- Desktop and 360-pixel mobile screenshots capture the presentation; the page has no horizontal overflow.

Separate English/Italian component tests advance the clock from Sunday to Monday. They check a fresh week without navigation, preservation of the weekly goal, new-day credit, and a ring capped at 100% when the goal is exceeded.

Existing journeys continue to cover upload errors, assessment failure, cancellation, recovery, separation by goal/language, and missing audio.

## Presentation corrections

1. Same-day charts show localized times including seconds, so short repeat attempts have distinguishable endpoints. Multi-day charts retain date labels.
2. The saved-review hint no longer incorrectly says progress is below the list. Updated all five interface translations.
3. Review cards and the selected-review digest use locale-aware score formatting (e.g. Italian `3,5`).

4. History API timestamps now include an explicit UTC offset. Previously the API returned naive server-local times; a UTC browser interpreted a Berlin recording as two hours in the future, giving it no weekly credit. Existing naive report timestamps are resolved using the backend's local timezone; already-aware timestamps preserve their instant. Storage is unchanged, avoiding mixed naive/aware sorting in older report tools. Reports moved from another server timezone remain ambiguous because the old format did not save that information.

## Execution and limits

These are isolated Chromium journeys with real upload, MediaRecorder, audio storage, playback, and API persistence. Chromium supplies a synthetic microphone stream; ASR and LLM inference use deterministic fixtures. This verifies application behavior, not a physical microphone or live model quality. Fixture transcripts should not be interpreted as recognition of the synthetic microphone audio.

The initial browser run had one failure during a translation-triggered development reload while assessment was running. A clean browser run passed all 10 tests. The subsequent explicit-UTC test exposed the timestamp bug described above; both language journeys failed on missing weekly credit before the API correction. The final full run passed all 10 tests in 3.5 minutes, verifying that correction with an explicit UTC browser against the local backend. Claude CLI reviewed the selected diff. Its timezone/midnight test finding was addressed with explicit UTC browser time, UTC label expectations, distinct week-day counting, and a stricter accessible chart count assertion. Conditional concerns about the locale parameter and calendar timer were checked against the existing implementations; both are already present.

Unit tests: 164 passed; TypeScript check passed. Focused backend history/i18n/progress tests: 17 passed, 21 deselected. Full backend API suite after the timestamp correction: 26 passed. The preferred sibling Python environment is absent; the working repository `.venv` was used.

Reproduce:

```sh
npm --prefix frontend test
npm --prefix frontend run typecheck
cd frontend
npx playwright test --config playwright.journeys.config.ts bilingualPractice.spec.ts
```

The browser configuration starts fresh backend/frontend servers at ports 8814/4177 and uses an isolated temporary data directory. It does not modify learner history in another running instance.

Runtime smoke during the isolated journeys: frontend `/history` and backend `/v1/health` both returned HTTP 200. The older manually launched ports 4189/8819 were no longer running; no learner data was changed to restart them.

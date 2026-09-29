# English and Italian practice journeys — 2026-09-29

## Scope

English and Italian, each with B1, B2 and C1 as the chosen practice goal. The implemented slice is the repeat-practice loop: save the prompt/goal, practise, get feedback, retry, compare measurements, listen, and return to history. Full multipart exam simulations, dialogue partners, revised task banks and resumable long uploads remain in the canonical plan.

Environment: macOS 27.0.1, Node 24.11.1, locked Playwright 1.59.1/Chromium, Python 3.12 project `.venv`, Ollama 0.34.4 with `qwen3.5:4b`, cached faster-whisper `large-v3`. The prescribed sibling venv was absent; the current repository venv was bootstrapped with `scripts/setup_env.sh`. No npm dependencies or lockfiles changed.

## Verification

| Layer | Result | What it establishes |
| --- | --- | --- |
| Full backend/script suite | 567 passed, 5 opt-in tests skipped | API, jobs, persistence, reports, metrics, configuration, scripts |
| Frontend unit/component suite | 126 passed | Setup, state, review/history, locale handling, controls |
| TypeScript and production build | Passed | Includes the new test/config directories; existing ~715 kB bundle warning remains |
| Default Chromium browser suite | 8 passed, 5 opt-in tests skipped | Existing navigation, setup, settings/support, review/history, responsive visual smoke |
| New isolated bilingual journeys | 10 passed | Six language/goal loops, two failure/cancellation recoveries, two goal/language-isolation and missing-audio journeys |
| Real sample ASR integration | 3 tests passed; all 6 audio cases processed | Nonempty transcripts, expected detected language, language-specific metrics, plausible nonzero duration/word counts |
| Initial live Ollama matrix | 6 passed, 8 real assessments | English/Italian B1/B2/C1; B1 includes retry; real hybrid rubric/coaching, saved report reload and audio playback |
| Final live Ollama suite | 7 passed in 8.0 minutes, 8 real assessments | Model discovery/connection, English/Italian B1/B2/C1, B1 retries, explicit provenance, feedback-language smoke and coaching v2 |

Each Playwright configuration starts localhost backend and Vite servers and waits for `/v1/health` and the frontend URL. The new configurations use fresh app-data directories, separate ports, one worker and zero retries. They do not modify the normal app's settings or learner history. Temporary reports are retained for diagnosis; run directories are inherited by workers instead of being recreated for every worker. Screenshots are under `frontend/output/playwright/`.

### Coverage matrix

All six English/Italian × B1/B2/C1 fixture journeys cover:

1. Choose learner, language and goal; upload a bundled WAV through real multipart HTTP.
2. Submit through the real API/background worker; open review.
3. Retry with the exact saved prompt/goal and the actual report UUID as parent.
4. Capture Chromium's synthetic microphone through real MediaRecorder; decode and measure the resulting recording.
5. Save/reload history; check two comparable points and the before/after table.
6. Check numeric elapsed WPM against persisted word count and actual duration; check the short retry's duration gate.
7. Play both WAV and browser-recorded audio; seek and check time; request a byte range and verify HTTP 206.
8. Reload the page, inspect the saved session, check narrow mobile overflow, retry from history, and persist a third linked attempt.

Each language also covers an empty upload, one failed upload followed by resubmission, an assessment error, cancellation after work has started, recovery, and absence of a late cancelled report after the fixture's work window has elapsed. Separate journeys switch goals/languages for the same speaker, verify that unrelated attempts do not enter comparisons, and show useful recovery when saved audio returns 404.

Fixture tests replace ASR/rubric/coaching only. Audio decoding, deterministic metrics/checks, report construction, job processes, multipart uploads, history persistence and playback are real. Fixture reports carry `fixture_inference: true`, a fixture scorer version and a warning. They are created only by the test launcher, which production never imports.

The live suite uses the normal backend launcher, no inference mocks, the six bundled WAVs and the installed Ollama model. It rejects deterministic fallback, fixture provenance, missing rubric/coaching, or empty feedback. Feedback-language markers provide a lightweight wrong-language smoke; they are not a formal language classifier. Generated reports remain available for human inspection. No claim about a universal model-quality guarantee follows from these cases.

## Problems found and fixed

- **History language initialization race:** cached history from the previous language could initialize “all languages” before the new report arrived. Wait for the initial refresh; preserve the learner's later explicit filter choice. Covered by a deferred-query regression and both language-switch journeys.
- **Ollama timeout:** Qwen's default thinking mode exceeded 180 seconds, yielding deterministic fallback. Ollama now requests JSON, no thinking output and a 4096-token cap. The default-local path uses the same timeout and schema-validation/retry logic as other providers. Successful earlier runs needed roughly 13–22 seconds for rubric generation and 9–14 seconds for coaching; ASR time is separate and varies with load.
- **Failure classification:** exhausted rubric-schema validation uses `LLMSchemaError` and retains `llm_invalid_schema`; network/timeouts retain `llm_unavailable`. The compatibility helper retains its existing error key. A shared inference-profile constant is saved only when Ollama actually produced a rubric, so progress comparisons account for changed inference settings.
- **Misused assessor confidence:** a real Italian response described the rubric's confidence as learner confidence. Coaching v2 removes this field from its input and asks for a timed spoken retry with one observable focus, rather than a writing-only exercise.
- **Chart scale:** overall performance uses a fixed 0–5 axis, so a cohort's best score does not always look like the maximum possible result.
- **Test drift:** wrapper tests still mocked the removed Codex launcher; visual tests expected the removed mixed-condition progress story; an opt-in runtime check expected obsolete model-discovery text. Updated to the locked project CLI and current UI. A settings test now waits for its saved-key button.
- **Test strength:** real numeric comparisons, seek position, cancellation after the completion window, inference provenance, language checks and goal separation replace weak existence-only assertions. Explicit ASR opt-in no longer silently skips a broken runtime.

## Independent review

Claude CLI performed two read-only reviews with tools disabled: new journeys/history and then the Ollama client changes. Selected code/diffs only were shared under the user's standing approval. Raw responses are in [journey review](2026-09-29-claude-journey-review.md) and [Ollama review](2026-09-29-claude-ollama-review.md). Subsequent fixes were checked locally and tested; the raw responses describe their reviewed snapshots.

Accepted review points: stronger numeric/cancellation/seek assertions, explicit live provenance/language checks, fixture provenance, fixed retry mode, shell-safe config quoting, one temporary workspace per invocation, memoized history translation, distinct schema errors, compatible helper error key and inference metadata only for actual rubric inference.

Checked hypotheses: Whisper uses its existing model cache rather than the isolated app cache; the worker uses `multiprocessing` spawn and calls the module-level assessment function; retries compare to the exact parent; ordinary practice comparisons intentionally group by task family/goal and analysis conditions; provider/model are already part of the comparison key; missing-media checks use a fresh unplayed recording and one selected audio element. The existing schema validator already used `RubricResult.from_dict`; provider normalization and the runner's `json` import are present. The app currently stores normalized learning-language codes; legacy malformed codes are kept outside retry eligibility rather than guessed. These were not treated as confirmed regressions.

OCR delegate resolved file selection and rules locally. The host agent reviewed code; OCR did not send files to an LLM endpoint. See the [final coverage checklist](2026-09-29-bilingual-review-coverage.json). Its explicit skips are unrelated pre-existing user-owned files, preserved outside this commit. Tests excluded by OCR's default selection were additionally read and executed.

## Reproduction

From the repository root, with the locked frontend dependencies and project `.venv` installed:

```bash
.venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
NODE_ENV=development node frontend/node_modules/playwright/cli.js test --config frontend/playwright.config.ts
NODE_ENV=development node frontend/node_modules/playwright/cli.js test --config frontend/playwright.journeys.config.ts
RUN_AUDIO_INTEGRATION=1 WHISPER_MODEL=large-v3 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_sample_integration.py -q -rs
NODE_ENV=development node frontend/node_modules/playwright/cli.js test --config frontend/playwright.ollama.config.ts
```

The live suite requires a running local Ollama service with `qwen3.5:4b` and a cached Whisper large-v3 model. Its isolated server uses a 180-second per-request budget; production keeps the configured `LLM_TIMEOUT_SEC`/CLI budget. Browser launches and socket-binding tests ran outside the restrictive command sandbox. No Chrome installation or privacy-settings change was needed.

The initial live probe hit `apiRequestContext.get: read ECONNRESET` on 2026-09-29 while polling `127.0.0.1:8816`; it now uses Playwright's bounded `maxRetries: 2` for that GET. It does not retry HTTP errors or hide a missing server. The unrelated LM Studio crash was triggered by `lms ls --json` launching its GUI from the restricted shell, before inference. That probe was abandoned; no desktop-runtime launch is part of these tests.

## Research used

The existing Playwright framework was suitable; its coverage needed extending. The implementation follows official documentation for [parameterized tests](https://playwright.dev/docs/test-parameterize), [managed web servers](https://playwright.dev/docs/test-webserver), [API verification](https://playwright.dev/docs/api-testing), [selective mocking](https://playwright.dev/docs/mock), and [bounded request retries](https://playwright.dev/docs/api/class-apirequestcontext#api-request-context-get-option-max-retries). No extra test framework or npm dependency was added.

Ollama's [OpenAI compatibility documentation](https://docs.ollama.com/api/openai-compatibility) documents JSON output, output limits and `reasoning_effort`; [thinking controls](https://docs.ollama.com/capabilities/thinking) describe model-specific support. The installed model's `/api/show` advertised boolean thinking with `true` as its default. The test model is [Qwen 3.5 4B](https://ollama.com/library/qwen3.5:4b).

## Remaining limits

The backend skips two OpenRouter integration tests, two older real-audio assessment tests and the opt-in sample-ASR test in its default run; sample ASR was run separately. The default browser suite skips optional LM Studio/oMLX and legacy live tests unless explicitly enabled. Ollama received its own isolated live coverage. These runs cover local Chromium and supplied synthetic/sample speech, not every browser/device or uncontrolled learner recording. Large uploads, low-disk recovery and the full fifteen-minute exam simulator are still planned work; these tests do not establish that those future features exist.

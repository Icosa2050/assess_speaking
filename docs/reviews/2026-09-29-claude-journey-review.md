# Claude CLI raw review

Read-only review of selected source files/diffs. Includes unconfirmed hypotheses; see the bilingual journey report for validated dispositions.

I didn't run anything or use tools for this review. Line numbers come from counting the pasted text. For `HistoryRoute.tsx` I point to the code snippet instead, because the file is too long to count reliably. Findings marked **(H)** depend on code that wasn't included: `runner.py`, `jobs.py`, `services.py`, `run_backend.py` and `PracticeProgress.tsx`.

## High: the live Ollama test may not prove real rubric and coaching

**1. `frontend/tests/live/ollamaBilingual.spec.ts:72-85`: the check for real inference relies on a denylist.**
- It passes when `scores.mode === "hybrid"`, the rubric is 1–5 integers, coaching fields are non-empty and there is no `coaching_unavailable` warning.
- **(H)** If the runner has any other fallback, the test won't catch it. Examples: a deterministic rubric or coaching template, a warning named something else (e.g. `coaching_fallback`, `llm_parse_error`, `llm_timeout`), or a coaching fallback that still fills three priorities. All would pass.
- Suggested fix:
  - Assert that `report.warnings` fits an allowlist, or is empty apart from known benign warnings.
  - Assert `report.input.provider === "ollama"`, `report.input.llm_model === model` and `report.input.dry_run === false`.
  - Assert `scoring_model_version !== "journey-fixture-v1"` and that `warnings` does not include `journey_fixture_inference`. These are cheap guards that the fixture launcher isn't serving the test.
  - If the report records where the rubric or coaching came from (raw LLM response, latency, token counts), assert on that. Otherwise, check afterwards that Ollama `/api/ps` shows `model` loaded.

**2. `frontend/tests/live/ollamaBilingual.spec.ts:83-84`: coaching language is never checked.**
- `feedback_language: language` is sent, but only non-emptiness is asserted. A 4B model answering in English on the `it` cases would pass, so the "bilingual" claim isn't tested there.
- A lightweight fix would be a language-ID or stopword check on `coach_summary` and `next_focus`.

**3. `frontend/playwright.ollama.config.ts:22` together with `ollamaBilingual.spec.ts:26-27` (H, possibly a hard blocker).**
- The backend is started with `--cache-dir <fresh mkdtemp>/cache`.
- If `--cache-dir` controls where Whisper/HF models are stored, `whisper-models/large-v3` will always report `cached: false` in a brand-new temp directory. `HF_HUB_OFFLINE=1` (line 24) would then stop any download.
- If it's only an app cache, this is fine. Please confirm what `run_backend.py` does with `--cache-dir`.

**4. `ollamaBilingual.spec.ts:62` against config line 14: tight timeouts.**
- The B1 case allows up to 2 × 420 s of polling inside a 900 s test timeout.
- With `LLM_TIMEOUT_SEC=180` and possibly several LLM calls (rubric plus coaching, maybe retries), plus large-v3 on CPU, this is likely to be flaky. When it fails you'll get a timeout rather than a clear failure message.

## High: gaps in the journey fixture

**5. `tests/e2e/journey_backend.py:21-22` together with `bilingualPractice.spec.ts:141-150`: the cancellation test can't catch a job that keeps running.**
- The `cancelled` status is probably set by the API straight away. **(H)** If jobs run in threads, or cancellation is cooperative, `time.sleep(30)` can't be interrupted. The job would finish about 30 s later and could still write a history row.
- The check at line 150 (`toHaveLength(1)`) runs well before those 30 s are up. So "cancel actually stops work and persists nothing" is not proven.
- Also, if there is a single job slot, the "library" job at line 147 waits behind the sleeping job. That adds up to 30 s under the `review` 45 s timeout and the 90 s test timeout.
- Suggested fix: make the fixture block on a cancel signal or event and record whether it was interrupted. Assert on that, or re-check history after the sleep window has passed.

**6. `tests/e2e/journey_backend.py:38-46`: the fixture only patches some fields, so derived values go stale (H).**
- It sets `duration_sec` and the transcript after the dry run. Anything the original run computed from the dry-run transcript and duration keeps its old value: `wpm`, word count, `duration_pass`, `min_words_pass`, `final_score`, and the history index row.
- The comparison table assertions (`bilingualPractice.spec.ts:92,109`) only count rows (`tbody tr` = 4). They never check values, so a comparison showing inconsistent or wrong numbers passes.
- The 2.2 s recording compared against a 90 s target is a good case to check `duration_pass === false` and the displayed duration.

**7. `tests/e2e/journey_backend.py:45`: the fixture overwrites `report["input"]["dry_run"]` with `False`.**
- This makes the output look like real inference, and only the `journey_fixture_inference` warning shows otherwise.
- **(H)** If `meta.practice.dry_run` is derived separately from the request's `dry_run=True`, `comparisonKey` would return null. The journey assertions only pass because the two happen to line up.
- Please document which field `comparisonKey` actually reads. The provenance should stay honest, e.g. a dedicated `fixture: true` flag, instead of flipping `dry_run`.

**8. `tests/e2e/journey_backend.py:17`: the patch depends on how the function is called (H).**
- The fixture signature `(audio, whisper_model, llm_model, **kwargs)` breaks if any caller passes extra positional arguments.
- Patching `assess_speaking.run_assessment` only works if callers look it up on the module at call time, or import it after line 49. Also, "spawned workers re-import this module" is only true for `multiprocessing` spawn with this file as `__main__`; it isn't true for `subprocess` or `python -m` workers.
- The fixed-transcript assertion would fail loudly if the patch were missed, so this isn't a silent pass.
- **(H)** Also check whether `dry_run` still loads Whisper `small`. The journeys config doesn't set `HF_HUB_OFFLINE`, so it could reach the network.

## Medium: weak or misleading test assertions

**9. `bilingualPractice.spec.ts:96-97`: the seek assertion proves almost nothing.**
- `seeking === false` holds whenever a seek completes, including when the whole WAV is already buffered or the seek is ignored. It never checks `currentTime ≈ 1`.
- The Range check at lines 102-104 uses the API request context, not the browser's media pipeline.
- Suggested fix: assert `Math.abs(currentTime - 1) < 0.1` after the `seeked` event.

**10. `bilingualPractice.spec.ts:105-109`: the reload check depends on test order.**
- After reload the speaker scope is empty, so the page shows every speaker's history. The assertions only work because this test's session happens to be the newest row.
- `toContainText(goal)` is weak: text like "B1" can appear anywhere in the page.
- The same pattern appears in `ollamaBilingual.spec.ts:94-97`.

**11. `frontend/playwright.journeys.config.ts:13`: retries would break the counts.**
- There's no `retries: 0`, and all tests share one backend and data directory for the whole run.
- If retries are turned on (CI defaults, or `--retries`), the second attempt starts with existing history. That breaks `svg circle` = 2 (line 91), `toHaveLength(1)` (line 150) and the isolation counts.
- Suggested fix: pin `retries: 0` as the ollama config does, or make speaker IDs unique per run with `testInfo.retry` and a nonce.

**12. `bilingualPractice.spec.ts:182-185`: the missing-audio test has weak spots (H).**
- `el.load()` may be served from Chromium's media or memory cache and never hit the route.
- `.locator("audio").evaluate` fails in strict mode if more than one audio element is present.
- `getByRole("status")` could match an unrelated status element, which makes the assertion vacuous.
- Suggested fix: scope to a dedicated test ID.

**13. `bilingualPractice.spec.ts:121`: the third attempt doesn't check `target_duration_sec: 90` or `feedback_language`.** Lines 83 and 121 don't check `feedback_language` either. So a retry draft that loses the duration or feedback language would pass.

## Low: product code

**14. `practiceProgress.ts:42` against `HistoryRoute.tsx` (the `languageCode` normalization and the `practiceRows` filter).**
- The route trims and lowercases `learning_language`. `retryDraft` and `comparisonKey` use the raw value.
- A row stored as `"IT"` or `" it"` shows up under the filter, but `retryDraft` returns null. `onRetry` then does nothing silently. **(H)** That depends on `PracticeProgress` showing the retry button without checking `retryDraft` itself.
- The speaker filter has the same mismatch: `speakerScope` is trimmed, but `row.speaker_id` is compared untrimmed.

**15. `practiceProgress.ts:68`: comparisons can mix different prompts.**
- When an attempt isn't an explicit retry, `previous.at(-1)` compares against the latest attempt with the same key, even if the prompt text was different.
- If that's intended, it isn't tested. If it isn't intended, it contradicts the "never silently substitute" comment at line 63.

**16. `HistoryRoute.tsx` (`const translate = createTranslator(locale)` and the `scopedRecords` `useMemo` deps): the memoisation doesn't work.**
- `translate` is a new function on every render, so the memos (`scopedRecords` → `availableLanguages` → `detailRecords`) recompute every time and both effects re-run every render.
- There's no infinite loop because React skips state updates that set the same value. This is a performance issue, not a correctness one.
- Suggested fix: `useMemo(() => createTranslator(locale), [locale])`.

**17. Both configs, line 8: `mkdtempSync` runs every time the config is loaded.**
- That happens in the main process and in each worker. The temp directories are never deleted and pile up across runs, each holding recordings and reports.
- Separately, `journeys.config.ts:27` quotes paths with `"…"` (no escaping for `$` or `` ` ``), while the ollama config uses `quote()`. Both assume `<repo>/.venv` exists, but `AGENTS.md` points to `assess_speaking-codex-v6/.venv`. **(H)** It fails loudly if that directory is missing.

I left out CEFR certification benchmarks, as you asked.

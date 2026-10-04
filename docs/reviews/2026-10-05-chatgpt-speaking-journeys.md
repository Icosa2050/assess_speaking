# ChatGPT speaking journeys — October 5, 2026

Scope: live browser journeys through the isolated Vostavo frontend (4189) and backend (8819), using the user's saved ChatGPT authorization, GPT-6-Astra, cached Whisper large-v3, and repository sample speech. No new account credentials or private learner recordings were supplied.

## Completed checks

| Journey | Result |
| --- | --- |
| Reload Settings with saved ChatGPT account | Account and selected model survived; no new sign-in required |
| Submit without audio | Correctly disabled |
| English sample upload, ASR, rubric, coaching, history | Backend completed; initial UI transport failure recovered through History |
| English retry from History | Original prompt retained, new audio required, linked to original report; automatic Review navigation succeeded |
| Italian UI, setup, upload, ASR, rubric, coaching | Completed and automatically opened Italian feedback |
| Italian history and measurement selector | Report opened; words-per-minute chart available; no English attempt used as Italian baseline |
| Reload History | Report persisted; interface locale regressed to English |

English input: `samples/cefr/en/B1/travel_story.wav` (17.09 seconds, 56 transcribed words). Retry used exactly the same bytes, SHA-256 `d752ab6b6e54c3aefcb06a20ce53fe491a3db2b399a14ff270286cb47a233be1`; duration, word count, pace and other measured metrics matched. Italian input: `samples/cefr/it/B1/travel_story.wav` (16.05 seconds, 51 words). All three results used hybrid assessment with actual model coaching, rather than a dry run or deterministic fallback. Final scores were 2.5; short duration and missing explicit overseas destination triggered task checks. These are short workflow samples, not complete oral exams.

Saved sessions:

- English: `074330aa-9170-436c-b49b-08d20642eea8`
- English retry: `a06fd5da-f1bc-43d9-be07-5d3d20fd6990`
- Italian: `f9c683a1-f3c6-4af4-bcb7-18ad381fd61c`

Programmatic assertions verified completed status, provider, expected/detected language, feedback language, hybrid mode, nonempty coaching, retry parent ID, identical English audio/metrics and a separate Italian baseline. The requests submitted by the browser were used; no extra duplicate assessments were submitted through the API.

## Findings requiring fixes

1. **Score comparison mixes two measurements.** `assess_speaking.py` computes `score_delta.overall` using current `scores.llm` minus the CSV's previous `overall`. The latter is the raw rubric overall. The retry showed **+0.20** although its weighted score decreased from 4.4 to 4.2, its raw overall stayed 4, and its final score stayed 2.5. Compare the same field on both sides and add a regression with deliberately different raw and weighted scores.
2. **Paraphrased coaching appears resolved.** The same function derives `new_priorities` and `resolved_priorities` by exact string membership. Rewording the same task-coverage advice marked all old wording resolved and new wording newly introduced. Text changes alone are not evidence that a practice problem was resolved. Stable issue identities/evidence or direct prior/current presentation are needed.
3. **Saved interface language is not hydrated on History reload.** Italian was saved through Settings, and the backend still returns `ui_locale: it`. Reloading `/history` displayed English UI while preserving the Italian report. Hydrate the saved interface locale at app startup; add a non-Settings deep-link reload test.
4. **One submission lost its response but kept running.** The first English submission displayed `Assessment failed: Failed to fetch`; the accepted backend job nevertheless completed and was recovered through History. Both localhost services were healthy. Retry and Italian submission then tracked correctly, so the transport cause remains unconfirmed. Test loss of the create response after backend acceptance, and support recovering the accepted job without blindly creating a duplicate.

Minor display issue: Italian feedback includes the untranslated category `Underdeveloped detail`. The review also has narrow-panel overflow visible in earlier Settings screenshots. These are separate from successful authentication/inference.

## Automated regression results

- `.venv/bin/python -m pytest -q tests/test_cloud_accounts.py tests/test_app_backend_api.py tests/test_ollama_cloud_connection.py tests/test_provider_connection_checks.py`: **103 passed**.
- `npm --prefix frontend test`: **155 passed**.
- `npm --prefix frontend run typecheck`: passed.
- Local backend `/v1/health` and frontend `/history`: HTTP 200 after the journeys.

The in-app browser upload chooser successfully attached the first file, but a following optional-label lookup stalled because uploading collapsed that field. Subsequent UI operations used explicit short deadlines and refreshed state between dependent actions. This automation stall was not counted as an application assessment failure.

No production fixes were made in this test pass. Existing test success does not cover the four defects above. Full microphone/device coverage, long oral exams, forced OAuth revocation and statistically meaningful feedback-quality testing remain outside this run.

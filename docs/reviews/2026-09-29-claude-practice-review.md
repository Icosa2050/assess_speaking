# Claude CLI raw review

This is the initial external review, including unconfirmed hypotheses. See the validated review report for dispositions and fixes. No tools were enabled for Claude.

I read only the code you pasted and didn't run anything. A few findings depend on files that weren't included (`sessionDraft.ts`, `SpeakRoute.tsx`, `load_history_records`, `load_report_payload`, the `jobs.py` upload cleanup), and I've marked those as needing a check.

## High

**1. Discarding a recording before submitting drops the retry link.** `frontend/src/lib/state/appStore.ts` — `clearAttempt`, the `keepSetup` branch
`retryOfSessionId` is always rebuilt from `state.review.payload.report.session_id`. Say the learner clicks Retry in History, which sets `retryOfSessionId = X`, then discards a take and records again. The review payload is empty, so `retryOfSessionId` becomes `""`. The next attempt is saved without its parent and shows up as a normal attempt. The same thing happens on a second clear, because `review` was already reset by the first one.
Fix: fall back to the existing value, e.g. `reportSessionId || state.draft.retryOfSessionId || ""`.

**2. `history_rows` can crash the whole `/v1/history` response on CSV data.** `app_core/services.py` — `history_rows`, the `elapsed_wpm` line
If `load_history_records` returns raw CSV values, both parts break:
- `duration > 0` raises a TypeError when `duration` is a string like `"45.2"`.
- `words * 60` repeats the string instead of multiplying.

There's a second risk even if the math works. The runner writes `""` into the CSV when a metric is missing (`assessment["metrics"].get("duration_sec", "")`). If those blanks, or a value like `"123.0"`, reach the new `HistoryRow.duration_sec: float | None` / `word_count: int | None` fields, Pydantic validation fails. That fails the whole list, legacy rows included, which breaks "legacy records remain readable."
Fix: convert both values with a safe float helper before doing math, and pass `None` instead of `""`.

**3. `load_report_payload` may be called with an empty path for legacy rows.** `app_core/services.py` — `history_rows`
`getattr(record, "report_path", "")` can be `""` or `None` for old rows. Unless `load_report_payload` catches every exception, `Path("")` resolves to `.`, and opening it raises `IsADirectoryError`. Guard with `if report_path` before loading. Separately, every report JSON (including the full transcript) is now read on every `/v1/history` call. That is slow as history grows, but it isn't a correctness bug.

## Medium

**4. Retry replaces the whole draft instead of merging into it.** `appStore.ts` `applySetup`, called from `HistoryRoute.tsx` `onRetry`
`applySetup` runs `buildSessionDraft(updates)` from scratch. `retryDraft` only fills in speaker, language, goal, duration, task, theme and prompt. Every other draft field goes back to its default: feedback language, language profile key, any provider or model choices kept in the draft. The retry can then run with different settings, and `comparisonKey` will correctly refuse to compare it with its parent. Merge with `state.draft`, or have `retryDraft` include those fields.

**5. `themeId` is set to the prompt id.** `practiceProgress.ts` `retryDraft`
`themeId: context.prompt_id || ...` turns a theme id into something like `travel-b2`, because `applySetup` builds prompt ids as `${themeId}-${level}`. Any code that looks up the theme or prompt catalog by `themeId` (SpeakRoute, setup) will miss. When `prompt_id` is `""`, `promptId` stays `""` because `??` only replaces null or undefined. Keep the original theme id in `practice` and restore that instead.

**6. The goal is saved exactly as sent.** `assessment_runtime/runner.py` `_build_meta` → `practice.goal`
`request.target_cefr` is stored without normalising. A lowercase or blank value such as `"b2"` puts the attempt in a separate comparison group and fails the `CEFR_LEVELS.includes` check, so the Retry button disappears. Uppercase and validate it against B1/B2/C1 in the contract or in the runner.

**7. The replay audio may already be deleted.** `app_backend/app.py` `history_audio`
It serves `meta.audio_path`, which is the resolved upload path passed to the job. If `jobs.py` or the upload handler deletes or moves uploads after the assessment finishes, every replay returns 404. I can't see that code, so please confirm those files are kept.

**8. A retry whose parent can't be compared shows the wrong message.** `PracticeProgress.tsx` comparison section
When the parent was run under different conditions (for example, the model changed), `previous` is `null`. The panel then shows the "retry comparison" heading with the `practice.first` text, which tells the learner this is their first attempt. Show a separate message saying the parent can't be compared.

## Low

**9. `clearAttempt` handles `sessionId` the opposite way you'd expect.** `appStore.ts`, `clearAttempt(keepSetup: false)`
Starting over with a new setup keeps `state.draft.sessionId`, while `keepSetup: true` clears it. If the draft's `sessionId` ends up as the report's `session_id`, two history rows get the same id. That causes duplicate React keys, a wrong result from `comparisonAttempt`'s `session_id !==` exclusion, and the wrong audio in replay. Worth checking whether this was intended.

**10. The days count includes a blank "day".** `PracticeProgress.tsx`
`days` counts `row.timestamp.slice(0, 10)` for every row, so a row with an empty or invalid timestamp adds a `""` day. Count only timestamps that parse.

**11. Rows with empty session ids all get pulled in.** `HistoryRoute.tsx` `practiceRows`
If any visible legacy row has an empty `session_id`, `visibleIds` contains `""`, and every raw row with an empty id is included. PracticeProgress can then render duplicate `key` values.

I found no problems in the audio endpoint's path check (resolve, then `is_relative_to` the allowed folders, then the file-type allow-list), in the `dry_run` exclusion, or in the rule that a retry is compared only to its parent with the same prompt. I also saw no changes to how existing scores are calculated.

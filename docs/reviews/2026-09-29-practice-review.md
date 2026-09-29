# Practice-loop review — 2026-09-29

## Result

Claude CLI and OCR delegate reviews completed. Five confirmed medium-severity findings were fixed. No confirmed high/critical findings remain in the reviewed changes.

Claude reviewed the exact 54,918-byte bundle approved by the user: six source files and four backend diffs. Its review ran successfully with tools disabled and session persistence disabled. Subsequent fixes were reviewed locally and regression-tested, rather than represented as independently re-reviewed by Claude.

OCR v1.12.10 selected files and resolved rules locally; the host agent performed the review. No repository source was sent to an OCR LLM endpoint.

## Validated findings

| Reviewer | Severity | Finding and resolution | State |
| --- | --- | --- | --- |
| OCR delegate | medium | Comparison metadata used the report schema version rather than the actual scoring version, and omitted fallback scoring mode and resolved analysis settings. Hybrid and deterministic-only runs, or changed rubric/pause settings, could be compared as matching. Persist the actual scorer, mode and analysis signature and include them in the comparison key. | Fixed |
| OCR delegate | medium | History remained fresh in React Query for 30 seconds after a completed attempt. Returning quickly after a retry could show old results. Invalidate history when a saved review becomes available; the browser fixture now exposes only completed submissions so the test detects stale caching. | Fixed |
| Claude, validated locally | medium | Joining scoped history rows back to all raw rows by session ID allowed blank legacy IDs to bring in rows from other speakers or languages. Apply the speaker/language filters directly to raw rows and do not select a blank session ID. | Fixed |
| Claude, validated locally | medium | An explicit retry with a missing or incompatible parent displayed the first-attempt message. Show a distinct explanation for an unavailable or differently analysed parent. | Fixed |
| Claude, validated locally | medium | The assessment API accepts lowercase goal strings and baseline evaluation uppercases them, but saved practice metadata did not. This split comparison groups and disabled retry for b2. Normalize the persisted goal to uppercase. | Fixed |

## Coverage

- Workspace files seen by OCR: 67.
- `total_files`: 34 reviewable entries, identified by `(path, status)`.
- `reviewed_files`: 34; `skipped_files`: 0; `coverage_rate`: 100% of OCR-selected entries.
- Automatic exclusions: 33, mostly tests, Markdown and binary reference materials. These are listed separately in the coverage JSON; 100% does not mean every workspace file was selected.
- Changed regression tests and the review-cache fix in ReviewRoute were also inspected outside the initial selection.
- JSON rules focus on key spelling; material manifests were not revalidated against complete exam publications during this code review.

Commands: `ocr delegate preview --format json`, followed by `ocr delegate rule --format json` with every selected path, then `git diff HEAD -- <path>` for tracked files and direct reads for new files.

## Claude findings checked against surrounding code

Discarding audio calls `clearRecording`, not `clearAttempt`, so the proposed retry-link loss on discard is not present. The CSV loader already parses optional numeric fields; the proposed string-arithmetic failure is not present. The report loader guards blank paths and catches filesystem errors. Provider/feedback settings are not fields of SessionDraft; setup restores themes using their title rather than themeId. The worker retains uploaded audio. Actual report session IDs are generated independently of draft IDs. Invalid activity timestamps had already been fixed after the reviewed bundle was prepared. These hypotheses were not reported as confirmed defects.

The accepted Claude findings concern blank-ID filtering, unavailable-parent wording and lowercase goal persistence. OCR additionally identified insufficient comparison metadata and stale history caching.

## Verification

- `npm --prefix frontend test`: 125 passed.
- `npm --prefix frontend run typecheck`: passed.
- `.venv/bin/python -m pytest tests/test_assessment_runner.py tests/test_app_backend_api.py tests/test_app_backend_jobs.py tests/test_app_backend_contracts.py tests/test_app_core_services.py tests/test_app_core_i18n.py tests/test_playwright_task_setup.py -q`: 104 passed. After the final goal-normalization edit, all six runner tests passed again.
- From `frontend`: `NODE_ENV=development node node_modules/playwright/cli.js test -c playwright.config.ts tests/e2e/reviewHistoryFlow.spec.ts tests/e2e/smokeLocalGuest.spec.ts --reporter=line`: 3 passed, against localhost backend/Vite with the backend health readiness check.
- `git diff --check`: passed.

Browser assessment responses are deterministic fixtures; these checks do not establish live model feedback quality. Existing reports lacking complete comparison metadata remain readable outside the chart.

## Approval record

The user's standing authorization for Claude CLI code reviews is recorded in repository AGENTS.md. No repeated per-bundle approval is required by that instruction. Credentials, unrelated personal data, recordings and learner reports are outside the review-input permission.

A supplemental CodeRabbit invocation required by the generic repository review rule was rejected by automatic approval review because this request explicitly authorized Claude and OCR, not the additional CodeRabbit export. CodeRabbit did not run. This did not block either requested review.

## Artifacts

- [Coverage checklist and structured findings](2026-09-29-practice-review-coverage.json)
- [Resolved OCR rules](2026-09-29-ocr-rules.json)
- [Raw Claude response](2026-09-29-claude-practice-review.md)

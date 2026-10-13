# PR #16 implementation review

Implementation reviewed: `7edfabc7bac62b78145d27b4e122b9a016f6f8e9`, against `34c01e1`.
PR: https://github.com/Icosa2050/assess_speaking/pull/16

The user authorized sending selected implementation source to CodeRabbit. Five committed-diff CLI reviews completed, covering backend, Python client/settings, assessment runtime, frontend/native support and scripts. Their raw JSONL outputs are local temporary evidence, not published learner data. The GitHub CodeRabbit check itself skipped this 205-file PR because its account allowance is 100 files; that successful check is not a completed review. The CLI reviews used smaller directory scopes. Claude CLI is unavailable in this environment, so this delivery does not claim a new Claude review. Earlier implementation reviews retain their existing records.

## Verified findings and changes

- GitHub Codex: legacy Python submissions omitted the required sharing fingerprint. Resolve a wholly local route before upload; external audio/analysis/fallback routes require the caller to display and explicitly confirm the fingerprint. Preserve existing confirmation unchanged so stale consent remains rejected by the backend. [Finding](https://github.com/Icosa2050/assess_speaking/pull/16#discussion_r4224968946).
- GitHub Codex: a failed worker start left a queued job that swallowed retries with the same request identity. Write acceptance metadata after worker construction and discard it on start failure; close credential pipes and preserve uploaded audio. [Finding](https://github.com/Icosa2050/assess_speaking/pull/16#discussion_r4224968947).
- CodeRabbit backend: restrict journal disk-recovery responses to journal routes; unrelated provider/runtime errors retain their normal handling.
- CodeRabbit client: tolerate malformed/null history reports, release the sign-in lock before browser handoff, and preserve permission/contention errors from work inside an acquired journal lease.
- CodeRabbit runtime: malformed/non-object Groq success responses produce the safe retained-recording error instead of a raw parser exception.
- CodeRabbit scripts: process groups disappearing during timeout cleanup still yield exit code 124.
- CodeRabbit frontend: intentional LLM skips and ineligible reports do not offer ineffective resume. Retain resume for `content_unverified` reports caused by an actual unavailable/invalid analysis; an indiscriminate eligibility exclusion would break that recovery path.
- CodeRabbit frontend: apply the one-second encoder margin to the duration delivered to submission when browser decoding is unavailable; decoded duration remains authoritative.
- CodeRabbit frontend: show archive confirmation only after the archive action is requested.
- CodeRabbit native: prune only app-owned regular support draft attachments older than seven days on the next draft handoff. Preserve newer files, unrelated files and symlinks. Mail-imported drafts and learner-saved downloads remain independent.
- Full-suite triage: Python temporary names can include underscores. Accept those suffixes within the same bounded temporary-root/prefix and symlink checks, and exercise an underscore-containing fixture deterministically.

## Disposition of the spending finding

CodeRabbit reported permanently pending costs without a recovery path. Keeping unknown costs reserved is deliberate budget protection. `CloudSettingsPanel` already lists unresolved requests with a provider-cost confirmation and reconciliation action; `/v1/runtime/cloud/spending/{request_id}/reconcile` persists that settlement. Unit, UI and restart integration cases exercise it (`test_cloud_plan.py`, `CloudSettingsPanel.test.tsx`, `test_cloud_recovery_integration.py`). Do not silently release an unknown paid cost at a month boundary. No change required for this finding.

## Verification of review fixes

- Backend baseline: 1,249 passed, 19 opt-in skips, three existing warnings; bounded to 300 seconds, completed in 11 seconds. Localhost callback/integration fixtures ran outside the network-restricted sandbox. An earlier run exposed the temporary-suffix flake; it was corrected before this passing baseline.
- Frontend: 249 tests across 35 files passed; typecheck passed. Existing resume and duration tests now cover the corrected edge cases after recording finalization.
- Native: six Rust tests passed, including expiry, symlink safety, private atomic save and attachment survival after download cleanup.
- Quality checks and whitespace checks passed.

GitHub checks and rebuilt DMG evidence are recorded separately with their exact commit/artifact identities; these local results do not claim public signing, physical-microphone crackling diagnosis, live-provider acceptance or a human coaching study.

## OCR follow-up against current source

The supplied OCR review targets `7edfabc`; `b93b953` is the subsequent review-fix commit. Verify against that current source rather than treating all four findings as still open.

- **High, invisible recovery:** still valid. `JournalStatus` now retains `recovery_error`; the global notice detects it even with a null transaction. Settings displays the backend's repair guidance and disables unavailable backup/restore actions. A damaged purge journal remains intact and mutations remain blocked; the UI does not pretend that this state was automatically repaired.
- **Medium, support draft retention:** `b93b953` added seven-day pruning at the next handoff. This follow-up also prunes on app startup, covering copies from failed or abandoned handoffs without requiring another support request. Refuse linked draft directories and keep newer attachments, unrelated files, symlinks and Mail's own drafts untouched. Cleanup failure does not prevent app startup.
- **Medium, default Python submission:** already fixed in `b93b953`, with local-route acceptance before upload and explicit confirmation for external destinations. A later live CodeRabbit re-review of `jobs.py` and `journal_routes.py` completed with zero findings.
- **Medium, initial archived rehearsals:** still valid. Load browser archives alongside backend status on mount, preserve either successful result if the other request fails, and report initial-load errors. Immediate archived Undo is now exercised through real IndexedDB and the Settings UI.

All four new frontend regressions failed before the fixes and passed afterward. A Settings unit-test mock was updated to isolate unrelated journal/network reads instead of allowing a real fetch to race its settings-error assertion. CodeRabbit re-reviewed the six changed implementation/test files with zero findings; later test-only additions received local inspection.

Follow-up verification: 1,250 backend tests passed (19 opt-in skips), 253 frontend tests passed across 36 files, typecheck and seven Rust tests passed, and all five mandatory localhost browser contract identities passed once without retries. Browser coverage includes error-only recovery notices, repair guidance, immediate archived Undo and the existing restore/sharing recovery contracts.

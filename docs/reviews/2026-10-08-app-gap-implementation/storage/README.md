# Production storage, support draft and Ubuntu acceptance

**Latest recipient update:** `info@frommherz-it.ch` is now the editable default. The DMG was rebuilt and its native Settings field verified; see [recipient update evidence](support-recipient-update/README.md). The full acceptance results below describe the preceding backend-identical candidate and retain their original artifact hashes.

This record supersedes the earlier engineering snapshot in the parent README. Work stayed in the existing checkout, preserving mixed prior changes; no branch, worktree, commit, push or deployment was created. The owner’s full permission authorized selected source and six repository synthetic WAVs for p50. No credentials, learner recordings or learner reports were transferred or supplied to Claude.

## Delivered behavior

The combined candidate includes nickname persistence, mandatory Setup microphone calibration and playback confirmation, shared device/processing selection, actual RMS/peak practice metering, explicit Home checks, and the decoded 30-second minimum before ASR/AI. Captured audio duration is decoded locally when supported; the backend remains authoritative. Eligibility withholds misleading passes/grades/praise and progress comparisons for insufficient, invalid or unverified speech. Sharing summaries and accepted immutable jobs use the backend route fingerprint.

Settings now provides a production backup of backend reports/audio and this profile’s IndexedDB rehearsal sessions/recordings, native Save/Open, selective restore preview and durable recovery. Existing identities are skipped rather than overwritten. Credentials, accounts and recovery jobs are excluded. Archive retains data for Undo; a separate fingerprinted confirmation removes unshared owned data, previous report revisions, completed caches and JSONL entries. Shared references, resumable work and independent exported backups are protected. The [storage contract](../../../plans/2026-10-08-storage-contract.md) describes limits, provenance and recovery boundaries.

Support packages use native Save with Cancel/Retry and default exclusion of learner attachments. An editable recipient and explicit action prepare a macOS Mail draft with an independent private ZIP attachment. The app reports draft preparation, never delivery. Native testing observed `support@example.invalid` and the synthetic ZIP attachment, then discarded the unsent draft. No email was sent. Private attachment copies and Mail/receiver copies have separate lifetimes from expiring temporary packages.

Evaluation selection/accounting and the blind protocol are implemented; Review duplicate text is removed and History’s digest retained. Independent human ratings and broader accessibility observations remain open.

## Current verification

| Check | Result | Retained evidence |
|---|---|---|
| Ordinary backend | 1,231 passed, 19 opt-in skips, three warnings | `backend-tests.log` |
| Storage/API failure and recovery regressions | 30 passed, included above | `tests/test_journal_backup.py`; source manifest |
| Frontend | 249 passed in 35 files | `frontend-tests.log` |
| Typecheck and fresh production build | Passed | `typecheck.log`, `build.log`, `build-report.json` |
| Rust native validation/copy tests | Five passed | `rust-tests.log` |
| Offline cloud process/recovery/codec | 15 passed; mandatory JUnit gates verified | `cloud-tests.log`, `cloud-results.xml` |
| Local sharing/storage browser contracts | Five passed; exact identity gate verified | `browser-contracts.log` |
| Ubuntu browser lanes | 18 default, 26 connections, 14 uninterrupted journeys, five contracts passed; exact gates verified | `ubuntu-accepted/` |
| Ubuntu final storage regressions | 30 passed | `ubuntu-accepted/ubuntu-storage-tests-accepted.log` |
| Quality / duplication / whitespace | Passed; zero clones within configured frontend scope | `quality.log`, `duplication.log`, `git diff --check` |
| Frozen DMG acceptance | Two fresh runs passed real offline EN/IT ASR, protected APIs/history/ranged audio, support export, learner export/second-root restore/archive/purge, restart/ownership and signature checks | `artifact-acceptance.json`, `artifact-acceptance-repeat.json` |
| Native populated rehearsal backup | Upgrade preserved profile; Save Cancel/Retry; exact audio bytes retained; native Open preview skipped existing rehearsal | `native-backup-final.json`, `native-backup-synthetic.zip` |
| Native support draft | Correct test recipient/ZIP observed; app showed draft prepared; draft discarded unsent | `native-support-draft.json` |

The internal Apple Silicon DMG is `.build-macos/Vostavo-0.1.0-arm64-internal-adhoc.dmg`, SHA-256 `c527c38109589f468acccd04dd8a8af25613338813a2c8651b7a069b5e774ac5`. It was rebuilt with a fresh frozen helper, locked existing dependencies and no new npm/Rust packages. Native Mail handoff was observed on artifact `bcd2ed954055558b0d37dc8883bb69ef34084fea4c26ca86e97859f9c5e9b3a1`; later storage/helper changes are separately verified on the current artifact, with the Mail command implementation unchanged. Earlier artifact/native evidence is retained under `*-before-cache-fix` and `artifact-acceptance-before-copy-fix.json`; those are not current-hash acceptance.

The native backup contains one synthetic 31-second mono WAV, 992,044 bytes, SHA-256 `e68c0b2f7005f9a9aa454e357efcb1393885fd7b382cda7bdd12bac078adbbdc`, preserved byte for byte after restore, app upgrade and re-export. The saved ZIP is mode 0600. Only owned disposable native processes were closed; the learner’s running app/state/privacy database were untouched.

## Ubuntu application runs

Application runs use the authorized disposable p50 workspace `/tmp/vostavo-app-acceptance-20261008-CAjgDH`, Ubuntu 24.04 x86_64 and official Playwright 1.59.1 Noble image, Node 24.14.1, Chromium 147 and WebKit 26.4. Containers have no external network, provider keys, SSH agent, user journal or account mounts. Locked npm installation uses `--ignore-scripts`; runtime roots/cache/keyring/ports and guard logs are isolated. Final journeys use one browser worker, four CPU/four GB limits and zero retries. Required CI lanes retain bounded startup/run/teardown and exact identity accounting; no GitHub workflow or commit was dispatched.

The initial 31-second and subsequent 35-second synthetic microphone captures sometimes decoded below the production 30-second minimum under two-CPU contention. The app correctly refused review. Tests now capture 45 seconds and assert the decoded WAV duration; there is no production calibration/duration bypass. One intermediate run retained a stale `<40` timer assertion and failed six bilingual tests at 45.06 seconds; that test bound is corrected to `<55`. Timed-rehearsal and both WebKit cases already passed in that intermediate run. Earlier failure logs and result/guard JSON are retained in `ubuntu-before-final/` and `ubuntu-stale-duration-bound/`, rather than represented as clean acceptance.

The final corrected uninterrupted journey run passed all 14 cases in 19.0 minutes. The post-cache-fix storage suite passed 30 tests in 2.12 seconds and all five browser contracts in 25.1 seconds. Default (18) and connection (26) results from the preceding run also pass exact gates: all 63 required browser identities have execution evidence, without retries or mandatory skips. The Chromium (12) and WebKit (2) gates are verified on explicitly derived subsets of the same 14-case result, not claimed as additional executions. Final results, guards and gates are in `ubuntu-accepted/`. `ubuntu-source-manifest.json` records the exact 431 selected files at journey launch. The final focused storage run adds its test source and updates `app_backend/journal.py` and the artifact test harness, with remote hashes matched to `ubuntu-storage-patch-manifest.json`; source boundaries are reported separately rather than claiming every run used one immutable snapshot.

## Claude reviews and practical limits

Three logged-in read-only Claude CLI source reviews were completed under standing approval; inputs, outputs and manifests are retained. See [review dispositions](claude-dispositions.md). Findings led to fixes for lock deadlocks, durable completion/recovery, path/provenance handling, shared-media purging, selective restore, Mail attachments and optional cache cleanup. The last P1 was reproduced, fixed and covered by two regressions. Claude did not execute tests or accept a DMG, and its last input preceded that fix.

One earlier artifact submitted an assessment with HTTP 409 and did not retain the response body. Its exact cause remains unknown. Terminal-worker teardown was separately reproduced and hardened with two regressions; new artifact checks retain response diagnostics. This is not a retrospective claim that the original failure’s cause was proven.

Physical microphone listening/TCC/device changes and the original crackling diagnosis remain hardware acceptance work. Public Developer ID signing/notarization, minimum-macOS/second-machine tests, real Keychain/browser-return/live-provider acceptance, independent human quality evaluation and broader VoiceOver/user observations are not established by synthetic/local tests. Optional corrected-transcript coaching remains a separate product decision. Integrity hashes are not archive authenticity signatures. The backend cannot enumerate unknown browser profiles, and permanent removal cannot delete independent exported/Mail/receiver copies. An externally conflicting recovery record preserves originals for repair rather than silently overwriting them.

## Reproduction and evidence boundaries

Use the parent record’s bounded backend/frontend/cloud/Rust commands with the existing repository `.venv` (the prescribed sibling environment is absent). Current native acceptance: `scripts/test_macos_delivery.py .build-macos/Vostavo-0.1.0-arm64-internal-adhoc.dmg --report /tmp/acceptance.json`. Ubuntu uses the official locked Playwright container and the recorded fixture configurations; the complete journey capture now asserts decoded duration. `source-manifest.json` binds selected final local files and `evidence-manifest.json` binds retained records. Neither attests a clean checkout or all historical runs to one source revision.

A local contract attempt overlapped the build’s `npm ci` and failed because the install removed Playwright’s runtime module; it was rerun after installation and passed. Dependency installation and browser tests must run sequentially. Restricted CLI login and loopback/mount failures were rerun with already-authorized execution permissions. Those are environment failures, not suppressed passing test results.

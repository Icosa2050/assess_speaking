# October 8 implementation and internal DMG evidence

**Current acceptance:** the later [production storage/support/Ubuntu record](storage/README.md) supersedes the historical implementation snapshot below. Backup/restore/archive/purge and attached support draft are now implemented; the p50 source transfer was subsequently authorized and application runs executed. Counts, artifact hashes and pending statements below describe the earlier candidate only.

## Earlier engineering milestone

The [revised action plan](../../plans/2026-10-06-app-gap-action-plan.md) incorporates Claude CLI's **feasible with amendments** verdict. The exact plan-only input/output and hashes are retained here. Claude did not inspect source, execute tests or accept the artifact. This record describes host verification in the existing dirty checkout; no branch, worktree, commit, push or deployment was created.

## Implemented behavior

- Recording requires the shared Setup calibration: detectable five-second sample, complete playback and explicit clear-sound confirmation. Changing device/processing, reload or capture failure requires retesting. Practice shows measured RMS rather than decorative bars, warns on sustained silence/repeated near-clipping and retains the take. Home displays completed checks out of three. The earlier nickname navigation fix is included in the combined candidate.
- Review requires at least 30 decoded seconds before ASR/AI, including uploads/CLI/resumes. Browser capture uses a 31-second margin for encoding. Uncertain language cannot earn verified-language praise. Versioned eligibility withholds grades/comparisons for insufficient, invalid or unverified content. JSON retains explicitly provisional observations; new CSV grade columns are blank for ineligible reports, and legacy dashboard/summary readers withhold grades/deltas without rewriting saved evidence. Missing linked reports cannot validate CSV grades. Saved praise/strengths and progress deltas are also withheld for ineligible reports while retaining the original payload. Evaluation does not count withheld observations as successful CEFR pairs.
- Sharing summaries derive from backend-owned canonical routes in five locales. Submissions/resumes require the displayed route fingerprint under the preferences lock. Route/request-ID conflicts return 409 before dispatch; accepted jobs retain immutable settings. Rehearsal stops new jobs after a route change, preserves completed reports/audio and resumes only remaining parts after the updated route is accepted.
- Support download uses an existing-dependency native Save dialog, a bundle-ID-only command, canonical/expiry validation and private atomic copying. Cancellation/failure retains the generated package for retry. Browser fallback says download started; desktop success means the copy completed. Attachments remain independently opt-in. Default completed-job diagnostics exclude learner payloads; seeded-secret/nested-log/ZIP-entry/home/temporary-path tests cover privacy.
- Browser lanes are disjoint and isolated, with mandatory identity accounting, zero retries, no mandatory skips, empty caches, memory keyring, cleared provider/proxy environment and browser/backend/worker network guards. The launcher owns graceful backend shutdown and fixture cleanup. CI defines separately bounded default, connections, Chromium journeys, WebKit journeys and contracts jobs; remote execution is not claimed.
- Evaluation tooling selects/accountably excludes corpus cases, retains failed outputs in the denominator and separates held-out proficiency labels from inference task goals. The [blind protocol](../../evaluation/2026-10-08-blind-quality-protocol.md) is prepared; no human validation or accuracy result is claimed.
- The [two-store storage contract](../../plans/2026-10-08-storage-contract.md) and eight-test disposable archive experiment establish a backup foundation. Live backend/IndexedDB export, restore and recoverable deletion are **not implemented product features**. Review's duplicated focus/exercise is removed; History's existing digest remains. The synthetic 1k/10k backend benchmark is retained, with render/memory/accessibility measurement still open.

## Verification

| Check | Observed result | Evidence |
|---|---|---|
| Ordinary backend suite | 1,199 passed; 19 opt-in skips; two warnings | `backend-tests.log` |
| Offline cloud process/recovery/codec | 15 passed; identity/JUnit gates pass | `cloud-tests.log`, `cloud-results.xml` |
| Export/legacy regression selection | 83 focused tests passed, including nine new export cases | `export-regressions.log` |
| Frontend | 245 tests across 34 files passed | `frontend-tests.log` |
| Typecheck / production build | Passed | `typecheck.log`, `internal-build.log` |
| Rust | Three tests passed | `rust-tests.log` |
| Quality / duplication | Passed; zero clones | `quality.log`, `duplication.log` |
| Default browser lane | 18 passed, no retries/skips | `browser/default/` |
| Connections browser lane | 26 passed, no retries/skips | `browser/connections/` |
| Rehearsal sharing contract | One passed, no retries/skips | `browser/contracts/` |
| Complete Chromium/WebKit journey lane | 14 passed in one uninterrupted 14.1-minute run, no retries/skips | `browser/journeys/` |
| Final legacy coaching browser regression | Two saved-review cases passed; eligible feedback retained | `browser/legacy-coaching-final/` |
| Rebuilt frozen runtime | Mounted/copied self-tests, real EN/IT ASR/workers, APIs/audio/history/restart/ownership/signature checks passed | `internal-artifact-acceptance.json` |
| Installed WKWebView support save | Real dialog → cancel → same-bundle retry → chosen destination → saved; actual ZIP integrity/default exclusions/redaction/0600 passed | `native-support-save.json`, `native-support-default.zip` |

One final-hash artifact invocation returned HTTP 409 on assessment submission. Its generic error log is retained in `internal-acceptance-409.log`; that first run did not retain the response body, so its cause is unresolved. The acceptance harness now writes safe fixture error/route diagnostics against the tested artifact instead of leaving an earlier candidate's success report in place. Two fresh isolated repetitions passed on the final hash, including both real ASR languages. The failure was not reproduced and is an open release investigation, not silently classified as a harmless fixture error.

The ordinary-suite sandbox invocation failed 13 loopback binding fixtures with `PermissionError: [Errno 1] Operation not permitted`. The identical bounded command passed with authorized local listeners enabled; the failure log is retained. The prescribed sibling virtualenv is absent; the existing repository `.venv` is Python 3.12.11. The two warnings are Starlette's deprecated BlockingPortal usage and an intentionally duplicated ZIP entry in an attack-refusal fixture.

The pre-change collection contained 60 identities: 58 mandatory cases plus two live oMLX cases. All were preserved; the live cases moved to their opt-in collection and one mandatory rehearsal contract was added. `browser-identity-preservation.json` records the original preservation proof and subsequent addition separately. Mandatory recording lanes use actual Setup calibration and ≥31-second recordings. The final narrow coaching/praise change followed the completed default/connection lanes and landed during the journey run; four new unit cases and two separately executed saved-review browser cases verify that change. All 59 mandatory identities passed; the two additional executions are fresh focused checks, not Playwright retries. The bulk runs are not claimed to be a single immutable-source execution. Repeated packaged TTS and generated PCM are functional fixtures, not labelled learner speech.

## Artifact and provenance

Internal Apple Silicon DMG: `.build-macos/Vostavo-0.1.0-arm64-internal-adhoc.dmg`
SHA-256: `5747d2bdd1587be4883d8b123c63b614ca03a99dea2a53ba416d4b618ca20c27`

The backend candidate used fresh pinned CPython 3.12.11/PyInstaller 6.16 helper packaging. The final frontend-only iteration reused that unchanged helper, with locked Rust, reproducible npm installation and embedded production frontend. `internal-build.log` records the fresh backend build and `internal-build-final-frontend.log` the final iteration. No npm/Rust dependencies were added. The artifact is ad-hoc signed; public Gatekeeper acceptance is pending. Real frozen ASR uses tiny weights copied into isolated offline cache, repository/Homebrew/global-cache reads denied, and 31-second repeated packaged English/Italian TTS. Actual ASR outputs are functional evidence only; word counts vary between runs. The broker self-test does not establish a complete packaged cloud journey.

`artifact-source-manifest.json` records the selected source snapshot observed during the final frontend-only build. `source-manifest.json` records the current acceptance snapshot, including the subsequent diagnostic-only acceptance-harness change and a test-file whitespace cleanup; production source still matches the artifact snapshot. Both include mixed pre-existing and current work and a tracked diff hash; neither is a build-controller attestation, clean commit or complete pre-change checkpoint. `review-input-manifest.json` binds exactly what Claude reviewed. `evidence-manifest.json` binds final retained logs/results/build/native records. Failed native-path-redaction evidence and earlier artifact acceptance are retained separately and are not acceptance for the final hash.

Native UI checks ran only the copied test app with disposable state and synthetic diagnostics. The test copy was closed and removed; its saved ZIP/evidence were retained. The user's running app, privacy database, recordings and credentials were not altered.

## Reproduction

Run from the repository root with the working virtualenv and dependencies already installed:

```sh
PYTHON_KEYRING_BACKEND=scripts.journey_keyring.MemoryKeyring VOSTAVO_HOME=/tmp/vostavo-app-gap-unit .venv/bin/python scripts/run_bounded_check.py 300 .venv/bin/python -m pytest -q --ignore=tests/test_cloud_pipeline_integration.py --ignore=tests/test_cloud_codec_integration.py --ignore=tests/test_cloud_recovery_integration.py
PYTHON_KEYRING_BACKEND=scripts.journey_keyring.MemoryKeyring VOSTAVO_HOME=/tmp/vostavo-app-gap-cloud .venv/bin/python scripts/run_bounded_check.py 180 .venv/bin/python -m pytest -q tests/test_cloud_pipeline_integration.py tests/test_cloud_recovery_integration.py tests/test_cloud_codec_integration.py --junitxml=/tmp/app-gap-cloud.xml
.venv/bin/python scripts/check_integration_junit.py cloud /tmp/app-gap-cloud.xml
.venv/bin/python scripts/check_integration_junit.py recovery /tmp/app-gap-cloud.xml
.venv/bin/python scripts/check_integration_junit.py codec /tmp/app-gap-cloud.xml
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run check:duplication
.venv/bin/python scripts/check_quality.py
cargo test --locked --manifest-path frontend/src-tauri/Cargo.toml
VOSTAVO_TEST_BACKEND_PORT=8934 VOSTAVO_TEST_FRONTEND_PORT=4234 npm --prefix frontend run test:e2e -- --config=playwright.config.ts
npm --prefix frontend run test:e2e -- --config=playwright.connections.config.ts
npm --prefix frontend run test:e2e -- --config=playwright.journeys.config.ts
npm --prefix frontend run test:e2e -- --config=playwright.contract.config.ts
.venv/bin/python scripts/check_browser_results.py default frontend/output/playwright/default/results.json
.venv/bin/python scripts/check_browser_results.py connections frontend/output/playwright/connections/results.json
.venv/bin/python scripts/check_browser_results.py journeys frontend/output/playwright/journeys/results.json
.venv/bin/python scripts/check_browser_results.py contracts frontend/output/playwright/contracts/results.json
UV_CACHE_DIR=/tmp/vostavo-uv-cache .venv/bin/python scripts/build_macos.py --mode internal
.venv/bin/python scripts/test_macos_delivery.py .build-macos/Vostavo-0.1.0-arm64-internal-adhoc.dmg --report /tmp/internal-artifact-acceptance.json
```

Commands needing local listeners/mounts ran with the app's approved execution permission. The journey lane has a 20-minute overall limit and zero retries. Preserve the corresponding browser results and guard logs before running another invocation in the same output category. The old 14-case run's raw guard log was overwritten by an earlier contract-output bug; its functional result remains in `browser-before-final-export-check`. The completed final 14-case run uses separate contract output and retains 83 Python process guard records. Default/connection/contract raw guard evidence is also retained separately from the final focused legacy-coaching invocation.

## Remaining work

- Physical WKWebView microphone/TCC, MP4 decoding, audible playback/no monitoring, Bluetooth/device changes and permission recovery. Actual crackling cause is still unknown; metering alone does not diagnose it.
- Public signing/notarization/stapling, upgrade/Keychain ownership, macOS 14 hardware, clean user/second Mac, cold-cache and capped live provider acceptance. The full frozen offline cloud harness remains open; broker-only smoke is insufficient.
- Investigate the one non-reproduced final-artifact sharing conflict. Preserve diagnostics and repeat on the release candidate; two subsequent isolated passes do not establish its cause.
- Observed Ubuntu app/remote CI runs. Automatic approval review rejected transferring repository source and synthetic WAVs to p50; no transfer, push or workflow dispatch occurred. The generic Ubuntu browser capability probe is retained separately.
- EN/IT specialist D0 adjudication, consenting learner D1 study, normalization/calibration of practice targets, and broader accessibility/user observations.
- E1/E2 production backup/restore/deletion and cross-store failure recovery. The archive prototype is not surfaced as a finished learner backup feature.
- H1 support sending: recipient and email-draft versus HTTPS-upload mechanism are pending the owner's answer. Download works independently. No delivery destination is invented and no package is transmitted automatically.
- G corrected-transcript coaching remains optional, pending its feature decision.

The engineering milestone is implemented; the multi-phase roadmap and public-release/human-validation gates are not complete.

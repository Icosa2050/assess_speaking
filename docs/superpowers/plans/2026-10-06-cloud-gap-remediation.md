# Repair cloud access and close the integration test gap

**Date:** October 6, 2026. **Status:** Reviewed by Claude CLI and PAL Qwen; amendments incorporated. Offline repair slice implemented; exact internal DMG accepted. Live/native/release gates remain open. **Delivery:** macOS internal DMG first; existing server/backend methods remain supported.

## Objective and evidence

Make OpenRouter browser login retain a usable credential, make split Groq transcription handle legitimate empty partitions, preserve authorization terminal states and restore real-ASR CI discovery. Then verify the cloud assessment path across its real local boundaries and rebuild the internal DMG with honest acceptance evidence.

The [test-gap audit](../../reviews/2026-10-06-cloud-test-gap-audit.md) reproduces three production defects and a workflow defect. Its [14 executable probes](../../reviews/2026-10-06-cloud-test-gap-probes.py) yield nine passes and five failures; two storage cases reproduce credential loss and two ASR cases reproduce the same partition defect. Existing 1,051 backend tests, 171 frontend tests and 25 browser journeys pass. Existing cloud browser routes are intercepted; those counts do not establish successful cloud integration.

The full existing backend suite covers 50% of cloud-broker branches, 69% of Groq ASR branches and 63.3% of OpenRouter authorization branches. These measurements exclude automatic child-process instrumentation. They guide scenario selection; a coverage target alone cannot close this work.

## Scope and constraints

- Work in the current checkout; preserve existing uncommitted/user-owned files. No branch/worktree change, commit or remote publication is part of this plan.
- Reuse installed HTTPX, PyAV, pytest, Playwright and credential storage. Add no dependencies or auth framework. Preserve deliberate blank-key deletion semantics outside the defective publisher.
- Keep keys and renewable credentials in the backend. Fixtures use fake credentials, isolated app data and an in-memory keyring; unavailable storage is exercised explicitly. Never use the user's saved accounts in automatic tests.
- Keep managed paid fallback opt-in, use a distinct Paid OpenRouter connection, reserve spending before dispatch and retain ambiguous charges. Do not promise exactly-once remote execution after an unknown provider outcome.
- Keep cloud transcription's human-review/preview caveat. Do not promote an ASR model, CEFR accuracy or learner-error retention from fixture tests.
- The user authorized implementation on October 6 after the two plan reviews. The offline application changes, fixture tests and internal DMG build are executed; live accounts/spending remain separate. Reviewer input contains only selected repository source, plan, audit and fixture tests.

## Execution order

The required repair slice is Steps 1–5, the core integration cases in Step 6, representative spending/crash oracles in Step 7, the repaired UI states and one real-backend browser flow in Step 8, and the feasible local/artifact gates in Step 9. An exhaustive crash/locale Cartesian matrix is not required. Native/live-account evidence stays separately labelled. Restore the CI command in Step 5 immediately after the red reproductions and P1 repair; it need not wait for cancellation/UI work. The numbered sections group responsibilities rather than mandate that dependency-free workflow edits wait until last.

### 1. Promote the failing reproductions into permanent regressions

**Files:** `tests/test_cloud_plan.py`, `tests/test_cloud_gap_regressions.py` (the existing account suite is preserved), and a new bounded `tests/test_cloud_pipeline_integration.py` for process integration. Keep the audit probe file as historical evidence.

Port the five failing assertions into the existing test lane without xfail, weakening their expected behavior, or mocking the defective boundary. Keep the small direct partition test in fast tests. Put real 24 MB splitting/20-minute synthetic encoding in a separately bounded integration test; stream its fixture instead of retaining a large recording in Git. Keep actual key publication tests fast, using real FastAPI routes and a real loopback callback with only remote key exchange mocked.

Run only the selected regression tests before fixes and save their expected red pytest/JUnit evidence. That intermediate run is not a required green gate. After the fixes, the same permanent tests must pass without xfail/skip in ordinary discovery. Existing nine passing audit probes are reusable where they cover an otherwise missing boundary; do not duplicate equivalent assertions merely to increase the test count. In particular retain actual broker ASR dispatch, managed paid dispatch/reconciliation, connection-probe policy enforcement, partial-chunk resume and listener expiry.

**Exit:** credential loss, overlap-only partitions and cancel-during-exchange fail for the documented reasons; all fixture credentials/data are isolated and callbacks/processes have bounded cleanup.

### 2. Fix OpenRouter credential publication first — P1

**Files:** `app_backend/cloud_routes.py`, `app_backend/app.py` connection-update route for regression coverage; `app_core/services.py` and `app_core/secret_store.py` only if a verified defect needs a shared change; runtime settings tests.

Change the OpenRouter publisher to delegate credential ownership to the existing `save_provider_connection(..., api_key=key, persist_draft=False)` path. Remove the publisher's separate pre-write/session fallback rather than changing global empty-key deletion behavior. The shared service already verifies persistent storage and retains a session-only fallback; this is one persistence owner, not a promise of exactly one platform keyring write. Keep the post-login model empty until the user chooses it explicitly.

Preserve a previously usable active analysis connection until the user explicitly selects/saves the new model/connection. Apply this only to incomplete OpenRouter login publication. If there is no usable prior connection, remain in setup with a clear model-selection requirement. The shared save service currently activates the new connection; if restoring the prior selection, use the existing default-selection helper, which synchronizes runtime fields before persistence. Do not simply restore its ID while leaving the new key in `prefs.llm_api_key`: that could overwrite the previous account's key. Assert the previous connection/key is unchanged. Keep the sequence under the existing preference lock.

Test the complete sequence: start → valid callback → actual connection persistence → usable saved/session key → model selection/update with an empty form key → brokered provider request. The provider request is fixture HTTP, with an assertion that the backend supplies the correct key. Exercise both working keyring and unavailable/write-failing keyring. Persistent read-back failure must use the existing session fallback rather than report usable durable storage.

The current update caller (`app_backend/app.py:294–329`) already preserves an existing key when provider identity is unchanged and `clear_saved_secret` is false. Verify that behavior through the API; do not introduce a second retention policy. Explicit clear/delete and provider-identity changes retain their established deletion semantics. `save_state_preferences` removes plaintext API-key fields before writing (`app_core/services.py:723–724`); assert disk files after publication/reload rather than assuming it.

Assert the status/UI differentiates persistent and session-only storage through existing metadata. The saved key remains usable across a reconstructed app instance with the same fake persistent keyring; explicitly cleared session-only memory requires reconnect. Disconnect/delete removes the corresponding key without deleting another account. Callback/API responses, preferences, reports, job payloads and fixture logs contain no credential. Saving the connection cannot select a paid model implicitly. Inject a preference-write failure after storing the new key; clean up only the new attempt's key/reference on failed publication and verify previous accounts survive. Never clean up a published newer grant using a stale failure.

**Exit:** both failed publication probes pass; explicit model selection and test/inference work with the stored reference; no global deletion behavior regresses.

### 3. Correct transcription partition semantics — P2

**Files:** `assessment_runtime/groq_asr.py`, `assessment_runtime/media.py` only if necessary, ASR tests.

Validate the provider's unfiltered word timing output separately from midpoint ownership. Define genuine missing timing as nonempty provider text with no raw entry containing valid timing and nonempty word text. Validate all raw timestamps, including discarded overlap words. A valid response whose words all belong to neighboring overlap may yield an empty central partition. Keep the existing half-open ownership rule: `keep_start <= midpoint < keep_end`; equality at a split belongs to the right partition. Multi-part reconstruction uses only owned words, preserving case, punctuation where supplied and real repetitions; no words or confidence are invented.

Regression matrix: empty audio response; true missing timestamps; silent central partition; left/right overlap-only speech; midpoint equality; actual repetition across a boundary; invalid/nonfinite/reversed timings; monotonic absolute output; both speech models. Exercise real PyAV FLAC size measurement and splitting with the unmodified 24,000,000-byte cap in one bounded test. Fast tests can lower the cap while retaining the actual codec and multipart path.

Cover partial success then 429/transport failure, cancellation before the next dispatch, retained input, temporary-file cleanup, low disk at preparation/write time and resume reusing only successful chunks. Quota errors must not automatically retry transcription or switch its provider. Changing audio/model/policy invalidates the appropriate cache.

Failures are not cached. The partition-only fix must retain successful chunks whose output is unchanged; verify before/after compatibility instead of automatically bumping `policy: 2`. Add a regression for language detection when the first partition is silent/overlap-only. Prefer the first partition with owned words and a detected label; keep no-speech/unknown language explicit and retain human review for disagreement rather than inventing certainty. If this changes an already accepted aggregate result, bump only the aggregate transcript-stage normalization identity. Bump chunk policy only if chunk output for previously accepted inputs actually changes. Model/audio/range identities remain required. Do not discard unrelated validated feedback stages.

**Exit:** direct and real-cap overlap probes pass; omission of real timestamps still fails; every outgoing fixture upload is under the actual cap and successful chunks are reused.

### 4. Preserve authorization terminal states — P3

**Files:** `app_core/openrouter_auth.py`; OpenRouter callback/state tests and focused UI tests.

Under the authorization lock, classify failed exchange/storage only if the attempt is still processing. Preserve cancelled or expired attempts when an in-flight exchange completes. Keep publication and cancellation ordering explicit: cancellation before publication prevents publication; an already published successful grant is not retroactively labelled cancelled.

Use events/barriers instead of sleep-based races. Test cancel/expiry while waiting and during exchange, exchange failure, malformed key, storage failure, unknown attempt, replacement attempt and replay. Assert no publication after cancellation/expiry, listener/timer shutdown and no secret in status/detail. Wrong host/path/state callbacks must not consume a valid attempt. Test consent-denial behavior against the implemented callback contract; change it only if the test demonstrates a separate failure.

Invoke `_expire` with a controlled clock/event during the paused exchange, not merely a status read. Replace a processing attempt via `start()` and verify the older callback returns rejection without changing its cancelled/expired state or consuming the replacement attempt. Successful publication before a later cancel remains connected; cancel/expiry before publication prevents it.

**Exit:** cancellation probe passes; expired stays expired; late responses cannot change the terminal state or publish a credential.

### 5. Restore the real-ASR CI lane — P2

**Files:** `.github/workflows/real-asr-selfhosted.yml`, `.github/workflows/ci.yml` only for the new required integration lane/time limits, appropriate existing workflow-test helpers.

Replace the sample module's unittest invocation with pytest. Preserve `RUN_AUDIO_INTEGRATION=1`, the runner's local model cache, pipefail and the 30-minute job limit. Give sample execution its own bounded step and `faulthandler_timeout` diagnostics. Write/upload JUnit results with the existing diagnostic artifacts.

Add a stdlib XML execution-result gate: the enabled sample lane must pass its reference check and all 11 actual-ASR cases (six bilingual/level samples, two M4A, two noise and silence; 12 total today), with no mandatory skips/failures/errors or collection-only success. Match mandatory JUnit case IDs/categories against a manifest-derived list; one always-running manifest check or unrelated extra tests cannot mask absent ASR cases. Assert pytest is installed in the runner interpreter. Missing models/audio must fail explicitly when opted in. Disabled live lanes remain opt-in in ordinary CI.

Run cloud fixture process integration on pull requests with a five-minute step limit and useful failure logs/JUnit. Give frontend-smoke/quality jobs explicit limits where absent; Playwright has a global timeout. Avoid CI credentials or new dependencies. Keep existing backend methods/tests and the separate local-ASR lane.

Record command, opt-in flags, cache/model identity and JUnit gate result. Run the exact workflow command locally where its prerequisites are available. If unavailable, record discovery only and leave execution open. Configuring CI is not remote CI execution: no push/dispatch is authorized here. Record the workflow URL/runner identity only after an actual authorized remote run; keep that gate pending until then. Required PR fixture integration must execute nonzero mandatory cases locally and be configured with the same result gate remotely.

### 6. Verify the real API → worker → broker → report boundary

**Files:** `tests/test_cloud_pipeline_integration.py`, existing backend test helpers, `app_backend/jobs.py`/`app_backend/cloud_runtime.py` only for defects the new tests expose.

Use real FastAPI upload/submission/status/history/resume routes, `JobManager`, spawned worker, private pipe, cloud runtime, PyAV preparation, response validation, checkpoints and CSV/report persistence. Inject fixture HTTP transport only where the backend talks to the provider. Do not replace `_complete`, `Process`, `_serve_cloud`, stage generation or assessment execution in these integration cases. Reject unexpected network destinations. Use bundled synthetic English/Italian samples or generated audio; no user recordings or saved connections.

Use a launcher/helper under `tests/helpers/`, excluded from the shipped package, that installs the fixture HTTPX transport before app startup. Provider HTTP is owned by the backend, so the spawned worker does not need a provider client or endpoint override. Keep the logical request URLs fixed to the real approved provider endpoints in the fixture assertions; MockTransport handles them without a TCP connection. Disable environment proxy mounts (`trust_env=False` in the fixture client factory) and scrub proxy/cloud-credential variables from test subprocess environments. Do not change production endpoints.

Add an inherited test-only socket/DNS guard (for example a `sitecustomize` module on the test launcher's PYTHONPATH) for the parent, backend-restart subprocess and spawned children. Allow only explicit loopback/Unix IPC destinations; any real non-loopback connect/resolution attempt fails. Do not confuse a captured fixed provider URL handled by MockTransport with an outbound network connection. Prove the guard is active in each process with a deliberately refused harmless external-connect probe. If a claimed boundary cannot be exercised through this seam, report it as missing rather than replace it with an in-process test and retain the stronger label.

Required scenarios and observable assertions:

| Scenario | Required evidence |
| --- | --- |
| Groq ASR + managed free OpenRouter | Actual multipart encoding and expected model/headers, zero-price/no-fallback payload, valid rubric/coaching, expected cloud provenance, human-review preview, report/history/audio persistence |
| Saved ChatGPT analysis + Groq ASR | Existing credential renewal/protocol fixtures with the new broker path; no worker key, current token resolved in backend, fixed endpoint, complete validated feedback |
| Free 429 → explicitly enabled managed paid OpenRouter | Exact dispatch order, distinct saved connection, ledger reservation exists before paid POST, reported cost reconciles, correct route/fallback provenance |
| Same quota with fallback disabled; auth/schema/unknown errors | No paid request; retained audio and useful existing safe failure/degraded-report contract |
| Coaching fails after rubric succeeds, then explicit resume | Same take/original date, no repeat ASR/rubric dispatch, fresh failed-stage dispatch, previous report retained and one history row |
| Lost accepted submission response | Same request ID creates one job and one set of provider dispatches through the actual submission API |

Use structurally and semantically valid fixture replies (including quote membership and supported score/language fields). Do not weaken production validation to make fixtures pass. Preview human review is required even when valid feedback is accepted. If the real guarded assessment intentionally degrades, assert that explicitly and provide a separate adequate-input case demonstrating accepted feedback.

Assert fallback is sticky for subsequent stages of the same runtime. On explicit resume, a new runtime checks free first; paid cache reuse must not be mistaken for a fresh paid POST or a new charge. Submission idempotency already exists (`AssessmentCreateRequest.request_id`, `JobManager.submit` matching before busy check); exercise that implementation rather than introduce a new feature.

Each core case has a maximum 60-second wall time, shorter fixture HTTP timeouts, bounded status polling and process/thread teardown on success/failure. Do not wait for production's 360/660-second IPC polls to expire in a test. CI's outer timeout is a backstop, not cleanup. Emit sanitized phase, PID, exit status, dispatch counters and cache/ledger identifiers on failure; never tokens or learner text. Keep real-cap encoding and the backend-restart/crash lane in separately bounded steps.

### 7. Verify cancellation, crashes and concurrent recovery at spending boundaries

**Files:** integration tests, job/cache/ledger modules only for demonstrated failures.

Use controlled events at reservation, before POST, after provider result/cache publication and before worker receipt/feedback acceptance. Kill or cancel the real worker at those boundaries. Use a real backend subprocess for at least one backend-restart case; application reconstruction alone does not prove process restart behavior.

First prove which existing fixture handler/checkpoint events can synchronize the boundary. Where needed, wrap the real operation with a test-only event/barrier that still calls the original implementation; do not replace its behavior or add a production control flag. Require representative worker-loss-after-publication, unknown-result-after-dispatch and backend-restart cases. An unsynchronized coarse kill cannot be labelled proof of an exact publication boundary. Each crash case is capped at 90 seconds with finally-block child/thread cleanup.

Assert published replies survive worker loss and are reused after restart; rejected invalid replies can regenerate without deleting a newer replacement. While a prior identical prompt remains in flight, an explicit recovery must wait/reuse rather than dispatch concurrently. Confirm across processes, not only threads, that the ledger and per-request locks serialize paid reservations/dispatch. Preserve one take/date/history row.

**Unknown-charge decision: retain the existing budget-only protection.** A pending reservation consumes the monthly allowance, but does not itself lock that prompt against a later user-initiated paid retry. There is no ledger-to-reply identity today; adding that gate is a separate feature and is not silently required by this repair. After crash/unknown outcome, assert the retained reservation and warning, no automatic quota fallback or transport replay, and that an explicit resume creates/reserves another estimate only if the remaining app/provider allowance permits it. With insufficient allowance it must fail before POST. With enough allowance, a sequential duplicate remote attempt remains possible and must be stated plainly.

Test user-confirmed reconciliation and budget availability after actual cost is recorded, using a fresh isolated ledger per case. Known pre-generation 4xx (excluding 408) already releases the estimate through `reconcile(..., 0, source=...)`; retain that implementation and its tests. 408/5xx/transport/HTTP-200 unknown outcomes stay reserved. Reconciliation already records source/time and rejects non-pending rows; this is verification work unless a test proves a defect. Do not assert that every recovery has zero duplicate remote work: no provider idempotency guarantee is established.

**Exit:** controlled worker/backend death cannot produce a silently free unknown request, stale reply crossing stages or a concurrent duplicate paid POST. The remaining unknown-outcome boundary is documented and tested.

### 8. Exercise frontend error/recovery behavior

**Files:** `CloudSettingsPanel.test.tsx`, route tests, `frontend/tests/e2e/cloudConnections.spec.ts` and the isolated connections config.

Add focused component tests for initial loading failure/retry, unsuccessful save retaining draft, failed login, cancel, expiry, browser-open failure with a working manual link, session-only connection explanation and failed reconciliation retaining the reservation. Include missing/deleted saved accounts and locale switches. Exercise all five UI locales at the component/copy level; retain bilingual browser flow coverage.

Limit five-locale checks to changed strings and repaired states. Exercise representative failure transitions in English/Italian rather than multiplying every unrelated state across every locale.

At least one browser scenario must use actual backend cloud settings persistence and validation and verify state after reload; intercept only remote provider behavior. Retain existing mocked UI contracts for speed but label their scope. Verify login's real published connection appears and has a usable key through safe metadata, not a stubbed connected response. A browser click alone does not establish macOS native handoff.

Playwright cannot intercept backend key exchange. Launch its backend with the Step 6 fixture/guard helper and replace `webbrowser.open` only inside that test process with a recorded no-op. Parse the actual authorization URL and deliver a direct loopback callback with the correct state and host. Do not click through to real OpenRouter authorization in an automatic test. Browser network guards permit the local app/fixture surfaces only. Fake keyring and temporary app data are mandatory; no native browser, user account or login Keychain is opened.

Do not redesign screens. Fix UI state only where new tests expose an unusable flow. Model selection stays explicit and fallback starts disabled.

**Exit:** success/failure states preserve correct configuration and recovery actions; real backend save/publication is observed and reload verified.

### 9. Run bounded final checks and rebuild the internal DMG

Execute focused regressions first, then the required full backend/frontend/type/build checks, isolated connections journeys and localhost health/Vite smokes. Use `/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python` when present; otherwise follow the repository bootstrap instructions and document a verified working-venv fallback. Keep unrelated untracked sample workflow work outside this change.

Suggested checks, with an external process timeout wrapper/CI step limit rather than confusing diagnostics with a timeout:

| Gate | Command / harness | Maximum wall time |
| --- | --- | --- |
| Focused cloud regressions + core fixture process integration | Python pytest selected new/existing cloud files, `-o faulthandler_timeout=60`, JUnit | 5 minutes; 60 seconds per core case |
| Full backend | Python pytest `--ignore=tests/test_sample_workflow.py`, same diagnostics | 5 minutes |
| Frontend unit/type/build | `npm --prefix frontend test`, `run typecheck`, `run build` | 3 minutes each |
| Browser connections | `npm --prefix frontend run test:e2e -- --config playwright.connections.config.ts --global-timeout=180000` | 4 minutes |
| Real 24 MB codec case | Dedicated integration case | 90 seconds |
| Crash/backend-restart lane | Dedicated fixture integration cases | 5 minutes; 90 seconds per case |
| Packaged acceptance | `scripts/test_macos_delivery.py` on the new exact DMG | 10 minutes; retain per-operation limits |

Keep test app data and keyring isolated. Measure targeted branch coverage again as supporting evidence, including child instrumentation if practical; report it separately from end-to-end assertions. Run quality/duplication/diff checks required by the repository. No dependency changes are expected.

Build to a new artifact directory, preserving the prior DMG. Rebuild the frozen helper because backend/auth/ASR code changed. Before claiming packaged cloud execution, verify a safe frozen-helper seam exists. Test-only launchers/PYTHONPATH guards from source tests are not presumed to work inside a frozen binary. Packaged tests must use isolated app data and a fake/disabled keyring; reject the test before saving a connection if it could reach the user's native Keychain.

Use the existing fixed self-test entrypoint only if it can safely exercise compiled-in cloud fixtures with an injected transport, no configurable alternate endpoints, no external keyring, no listener and a finite JSON result. Do not introduce a general production endpoint/auth bypass or assume arbitrary test code should be added to the shipped binary. If that seam does not exist and cannot be provided narrowly under existing self-test policy, keep packaged cloud traffic explicitly open. Rebuilding and passing local packaged ASR/settings still permits an internal test artifact, but does not close that cloud gate. Source process integration remains mandatory in either case.

Mount/copy/eject the exact new DMG, deny checkout/Homebrew/global-cache fallback as before, verify FLAC, loopback/session controls, real local English/Italian ASR, cloud fixtures where supported, worker lifecycle, reports/history/restart and signature after use. Record the matching artifact SHA-256 and minimum audited macOS version. Ad-hoc internal signing remains distinct from Developer ID/notarized release.

**Exit:** mandatory local fixtures/regressions are green; the exact rebuilt artifact passes all feasible packaged gates and limitations are stated in its evidence. Remote CI and packaged cloud traffic, when not actually executed, remain separate open items rather than successful checkbox equivalents.

## Separate live-account and release gates

After the offline slice is green, perform a small explicit live acceptance run with dedicated accounts/keys, synthetic or specifically consented audio and a predeclared provider/app spending cap. Current provider account/model/schema availability must be checked then. Account consent/credentials and a chosen paid-test cap are required inputs; the planning/review request does not authorize borrowing personal recordings or unbounded provider spending.

Verify real OpenRouter PKCE browser return, usable key/model selection, managed free feedback, Groq word/language output, explicitly enabled paid fallback, actual usage-cost reconciliation and persistent restart/disconnect. Verify the existing ChatGPT login with the new broker path. Record safe metadata/model/date and dispatch/cost, not credentials or learner reports in review exports. Live assertions compare safe booleans/hashes rather than printing raw credentials/provider bodies through pytest failure introspection.

Native browser launch, OS Keychain, microphone/TCC permission/denial and clean-machine Gatekeeper/notarization are separate macOS distribution gates. Do not reset the user's privacy database or replace their installed app during automatic tests.

Learner transcription fidelity/CEFR calibration requires a consented corpus with human references and a separately defined quality threshold. It remains a follow-up quality gate; completing these repairs does not remove the preview caveat.

## Completion checklist

- [x] Red evidence is recorded; the same permanent regressions pass after fixes without xfail/skip.
- [x] OpenRouter retains usable persistent/session keys through publication/model selection/inference/delete.
- [x] Real-cap split transcription accepts overlap-only central partitions; true missing timestamps fail safely.
- [x] Cancel/expiry terminal state and listener lifecycle remain correct after late exchange completion.
- [x] Sample-ASR workflow is corrected; exact local execution/JUnit evidence is recorded when prerequisites exist. Remote execution status is separate and requires an authorized actual run.
- [x] Real local API/spawned-worker/broker/report integration covers managed free and paid policy and ChatGPT/Groq combination.
- [x] Crash/concurrent recovery keeps cost, cache, take identity and provenance correct, with unknown outcomes explicit.
- [x] Frontend recovery tests and a real-backend cloud persistence/publication browser case pass.
- [x] Full required local checks and exact new internal-DMG acceptance pass; unexecuted remote CI, packaged cloud, live/native gates remain individually open and labelled.
- [x] Implementation records and plan indexes report final evidence and remaining gates accurately.

## Review record

Both reviewers returned **approve with amendments** for the preserved draft. Claude CLI reviewed the plan plus selected source; PAL used `qwen/qwen3.8-2.4t-a95b` to review the full prose plan after its file-input budget rejected the source bundle. Its review is planning feedback, not independent source verification. Required amendments are incorporated above; conditional assertions were checked against current source. See the [review dispositions and input manifest](../../reviews/2026-10-06-cloud-gap-plan-reviews.md) for exact scope, raw outputs and decisions. Neither reviewer reran implementation tests or reviewed an implemented repair. The required offline checklist is complete; unexecuted remote/live/native and full packaged-cloud gates are recorded separately in the implementation evidence.

## Execution evidence

See [implementation evidence](../../reviews/2026-10-06-cloud-gap-implementation.md) for final commands, counts, preserved red evidence, source/artifact hashes and remaining gates. The unchanged pre-execution amended plan is preserved in [this snapshot](../../reviews/2026-10-06-cloud-gap-plan-amended-snapshot.md).

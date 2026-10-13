# Cloud gap repairs — implementation evidence, October 6, 2026

The reviewed offline repair slice is implemented in the current checkout. The three reproduced production bugs and the real-ASR discovery defect are repaired; additional account deletion/reload and frontend recovery defects are covered. The exact rebuilt internal DMG passed acceptance. Live accounts and public/native distribution gates remain separate below.

## Changes

- OpenRouter publication delegates the returned key to the existing connection service. It retains persistent or session-only credentials and a blank model; a usable prior analysis account stays selected. API model updates preserve a blank form key, inference resolves the stored key in the backend, and deleting one account leaves the other key intact. Failed preference publication cleans only the newly allocated reference. Working, unavailable, write-failing and read-back-failing fake storage are covered; losing session memory requires reconnect. No plaintext key is persisted.
- Groq normalization distinguishes valid overlap-only words from omitted timestamps. Half-open ownership preserves repetitions, invalid timing still fails, and actual 24,000,000-byte splitting succeeds. Detection uses an owned speech partition; disagreement/empty speech stays unknown. Aggregate transcript identity is `groq_v3`; compatible chunk identity remains policy 2. Partial quota recovery, cancellation, cleanup and low disk have regressions.
- Late exchange failures preserve cancellation/expiry. Replacement attempts retain their listener/timer; completed listeners cancel their own timer. Invalid host/path/state cannot consume an attempt.
- Deleting the selected account intentionally requires another explicit selection. Hydration now tolerates this state instead of throwing while other saved accounts remain.
- The sample-ASR workflow runs pytest and gates the exact 12 mandatory IDs. Cloud core, recovery and codec lanes gate 8/5/1 mandatory IDs. Skipped, failed, missing or duplicate required cases cannot pass the gate. Jobs/steps and Playwright have explicit limits; the codec command has a 90-second process-group timeout. Server tests remain.
- Cloud settings now show loading/retry, session-only storage and missing accounts, and offer an explicit action to remove stale selections. Failed save/reconciliation preserves drafts and reservations. Browser launch failure retains a working manual link. Five locales and representative login terminal states are covered.

## Boundary evidence

`tests/test_cloud_pipeline_integration.py` uses a real backend subprocess, actual upload/submission/status/history/resume API, `JobManager`, spawned worker, private broker pipe, real audio preparation and assessment validation, checkpoints, reports and CSV. Only provider HTTP and browser launch are fixture seams under `tests/helpers/`; production endpoints are unchanged. Fake keyrings and isolated app data are mandatory. Fixture HTTP clients disable proxy inheritance. The inherited socket/DNS guard proves refusal in backend and worker PIDs and blocks external destinations.

Eight core cases cover managed free OpenRouter + Groq ASR, ChatGPT token renewal/Responses streaming + Groq ASR, explicit free-quota-to-paid fallback, disabled fallback, auth/schema/unknown failures, and failed coaching followed by explicit resume. Stable submission IDs create one actual job. Adequate-input cases accept validated rubric/coaching while retaining cloud-ASR human review. Resume keeps one take/date/history row, old reports, ASR/rubric checkpoints and a fresh failed-stage request.

Five recovery cases cover controlled worker loss after reply publication, backend-process restart at that boundary, a paid POST with an unknown outcome across restart, cross-process ledger budget serialization and identical-prompt serialization. Published replies are reused. Pending estimates survive unknown outcomes; insufficient budget prevents a POST. With enough allowance, explicit sequential retry may create another remote attempt while the old charge remains pending: this is the reviewed budget-only policy, not a remote exactly-once guarantee. User-confirmed reconciliation retains source/time. Cross-process identical requests dispatch/pay once while overlapping.

The separately bounded codec fixture streams twenty minutes of synthetic audio, invokes real PyAV FLAC measurement, and verifies both outgoing halves stay under the production cap with neighbor speech owned once.

## Final checks

| Gate | Result |
| --- | --- |
| Permanent pre-fix regressions | 5 expected failures, 9 passes; recorded before fixes, no xfail |
| Full backend, excluding unrelated untracked sample-workflow work and separately bounded cloud lanes | **1,103 passed, 19 skipped**, 11.68 seconds |
| Real source cloud process/codec/recovery | **14 passed**, 31.65 seconds uninstrumented; final instrumented run **14 passed**, 62.21 seconds |
| Corrected sample-ASR command, `RUN_AUDIO_INTEGRATION=1 WHISPER_MODEL=tiny HF_HUB_OFFLINE=1` | **12 passed**, 22.32 seconds; all required IDs passed, no skips |
| Desktop owner EOF / SIGTERM, `VOSTAVO_TEST_LISTENERS=1` | **2 passed**, 1.89 seconds |
| Frontend units | **182 passed / 30 files**, 6.16 seconds |
| Frontend typecheck/build | Passed; final frontend build included in fresh DMG pipeline |
| Connections browser journeys | **26 passed**, 39.9 seconds; one real backend persistence/PKCE callback case; other route-mocked contracts retain their stated scope |
| Quality / duplication / diff checks | Passed; zero backend duplication clones |
| Exact DMG acceptance | **ok:true**, seven checks; mounted/copied helper, offline cloud broker/ASR fixtures, real frozen local bilingual ASR, history/audio/restart and signature |

All commands ran on October 6, 2026 with external timeout wrappers for full checks, shorter fixture barriers/polling and teardown. The prescribed `assess_speaking-codex-v6/.venv` is absent; its earlier bootstrap attempt could not resolve downloads in the sandbox. The verified existing repository `.venv` (Python 3.12.11) was used. The original tracked account test suite is unchanged: an initial filename collision was corrected by moving the new cases to `test_cloud_gap_regressions.py` before the final 1,103-test run. The historical red JUnit still uses the initial file location; the assertion bodies were preserved in the new permanent file. Existing unrelated untracked work is excluded rather than modified.

The 19 ordinary-suite skips remain opt-in: four live legacy OpenRouter cases, eleven local-ASR cases, two real full-assessment cases and two desktop-owner cases. The eleven local-ASR and two owner cases were subsequently executed above. Fixture tests do not establish live provider account/model availability.

Supporting **parent-suite** coverage (separate from subprocess assertions):

| Module | Lines | Branches |
| --- | --- | --- |
| `app_backend/cloud_routes.py` | 85.1% | 80.0% |
| `app_backend/cloud_runtime.py` | 84.6% | 71.4% |
| `app_core/cloud_policy.py` | 90.5% | 86.3% |
| `app_core/openrouter_auth.py` | 91.4% | 83.0% |
| `app_core/openrouter_cloud.py` | 95.5% | 86.8% |
| `assessment_runtime/checkpoints.py` | 92.2% | 81.1% |
| `assessment_runtime/groq_asr.py` | 96.6% | 90.9% |

Broker branches rose from the audit's 50.0% to 71.4%; ASR branches from 69.0% to 90.9%. Optional child-process coverage recorded the successful free/paid/ChatGPT paths with both multiprocessing and thread tracing. Earlier multiprocessing-only tracing missed broker threads; that dataset is not presented as complete process coverage. Coverage is supporting evidence, not the acceptance oracle.

## Artifact

[Internal arm64 DMG](../../.build-macos/cloud-gap-2026-10-06/Vostavo-0.1.0-arm64-internal-adhoc.dmg), [acceptance report](../../.build-macos/cloud-gap-2026-10-06/acceptance-report.json), [build report](../../.build-macos/cloud-gap-2026-10-06/build-report.json), [verification manifest and retained JUnit/log hashes](../../.build-macos/cloud-gap-2026-10-06/verification-manifest.json).

SHA-256: `315766b10b31af04ea2cec56028b4945115087cfa5cd54fa6dbb520920ca0895`. Version 0.1.0, arm64, minimum audited macOS **14.0**. The prior cloud-plan DMG was preserved. The backend was rebuilt after the final production changes; the fixed existing `--self-test` entrypoint exercises compiled cloud ASR/broker adapters with immutable fixture inputs, no listener, account storage or configurable alternate endpoint. Mounted and isolated copied helper checks explicitly require `cloud_broker_fixture:true`. Packaged local ASR produced English 56 words/17.09 seconds and Italian 49 words/16.05 seconds.

This is an ad-hoc internal test artifact. Source cloud API/worker/report integration is verified; the frozen helper's full cloud worker/report journey remains a separate acceptance gate beyond its compiled broker smoke and existing local-worker journey.

## Remaining gates

- Remote CI execution: configured, not pushed/dispatched or observed. No workflow URL is claimed.
- Live OpenRouter PKCE, managed free access, Groq output, ChatGPT login/renewal and paid fallback/cost: requires chosen dedicated accounts, consented/synthetic audio and a declared spending cap. No live app-provider inference/spending was performed.
- Complete packaged cloud assessment with worker/report persistence: not executed; compiled fixed cloud broker/ASR smoke passed.
- Native browser handoff, OS Keychain and microphone/TCC permission/denial: interactive acceptance remains.
- Developer ID signing, notarization and clean-machine Gatekeeper: public release remains pending.
- Learner transcription fidelity and CEFR calibration: needs a consented corpus/reference quality gate; human-review preview stays.

The Claude CLI and PAL reviews were reviews of the plan, not of this final implementation or these test results. See the [review dispositions](2026-10-06-cloud-gap-plan-reviews.md) and immutable [amended plan snapshot](2026-10-06-cloud-gap-plan-amended-snapshot.md). No branch/worktree change, commit, push, account-message or public deployment was performed.

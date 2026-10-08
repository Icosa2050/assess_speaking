# Vostavo app gap action plan

Date: October 6, 2026
Updated: October 8, 2026 — local engineering, production storage and support draft implemented
Status: Internal DMG engineering milestone implemented; remaining acceptance and product phases tracked individually
Delivery focus: Apple Silicon macOS DMG

Repair the browser test lanes, make sharing statements match the route accepted by the backend, complete native release acceptance, and independently evaluate coaching quality. Verify support package downloads in the installed app and add an explicit send flow. Then address backup/deletion and Review/History usability. Preserve the current English/Italian practice loop, reports, recovery behavior, and server methods/tests.

Claude's verdict was **feasible with amendments**. This plan incorporates the review with the qualifications recorded in [review dispositions](../reviews/2026-10-06-app-gap-plan-review-dispositions.md). The [reviewed draft](../reviews/2026-10-06-app-gap-plan-reviewed-draft.md), [raw CLI response](../reviews/2026-10-06-app-gap-plan-claude-raw.md) and [run metadata](../reviews/2026-10-06-app-gap-plan-claude-run.json) preserve what was actually reviewed. Claude reviewed supplied text and source excerpts; it did not execute tests or inspect the DMG. The amended plan has been checked by the host against the repository, not independently re-reviewed by Claude.

The October 8 execution revision was also reviewed with Claude CLI. It is **feasible with amendments**; the exact [input and response](../reviews/2026-10-08-app-gap-implementation/claude-plan-review.md) are retained. That review inspected supplied plan text only and did not execute tests or inspect source/the DMG. The implementation status below is host verification, not Claude acceptance. The local H1 flow now uses an editable recipient and a native Mail draft with the ZIP attached. Three subsequent source reviews and their fixes are recorded in [storage review dispositions](../reviews/2026-10-08-app-gap-implementation/storage/claude-dispositions.md).

Work remains in the current checkout. Before each slice, inspect existing changes and preserve user-owned work. Estimates below exclude recruitment, human ratings, inference run time, account/certificate provisioning and waiting for external services. Roles describe responsibilities, not staffed assignments or delegated chats.

## October 8 execution revision

Implement the remaining engineering work in the current checkout in this order:

1. Finish recording safeguards: measure live practice input, warn on sustained silence/near-clipping, remove the decorative signal claim, and show the three required Home checks clearly. Retain mandatory Setup calibration and the 30-second review minimum.
2. A1/A2: preserve the collected browser identity union while separating fixture-only/live tests, share isolated launch configuration, assert network guards, verify actual result JSON, and add independently bounded CI jobs. Mandatory calibration must be exercised in updated recording journeys; no test-only production bypass.
3. B: resolve/display a backend-owned sharing route, bind it to ordinary and resumed submissions under the preferences lock, and stop rehearsal analysis when its displayed route changes. Preserve immutable runtime snapshots and accurately describe local audio/cloud text/fallback destinations in all locales.
4. V: implement a versioned eligibility contract for current/legacy reports, withholding grades/comparisons for insufficient, invalid or unverified content; track task completion separately. Preserve raw observations with explicit provisional status. Verify export/dashboard withholding, including legacy data, before claiming V’s engineering contract complete.
5. H0a: provide a real native ZIP save through existing Tauri/rfd capabilities, retain a truthful browser download fallback, and add cancellation/retry/content-redaction evidence. H1 uses a learner-entered recipient and an explicit native Mail draft action with the ZIP attached; sending remains the learner’s action in Mail.
6. D tooling: add explicit corpus selection and complete output accounting, separate held-out proficiency labels from task goals, and produce a reproducible blind-rating protocol. Authored fixtures do not become human labels; recruitment/adjudication are external work.
7. C (including H0b installed support save): rebuild an internal candidate after the code/checks settle, probe mounted/copied startup, backend/self-test and support save, and bind evidence to artifact hashes. Physical listening/TCC and public signing/live-account acceptance remain separate rows.
8. E/F: establish the two-store backup/maintenance contract before destructive UI; implement recoverable storage behavior and evidence-based Review/History improvements. G remains optional and requires its feature decision.

Implementable code and tooling proceed while independent external gates wait. Do not mark the overall roadmap complete on source tests alone. The owner’s subsequent full permission authorized the selected source and six repository synthetic WAVs for isolated p50 acceptance. No learner audio, credentials, push or deployment was included. The earlier automatic-review rejection is historical and resolved by that authorization.

The engineering sequence replaces the stale optional-microphone and first-batch ordering. Historical verification counts below are retained as dated evidence, not claims about the current candidate. The Claude review and [dispositions](../reviews/2026-10-08-app-gap-implementation/review-dispositions.md) preserve the reviewed snapshot and accepted amendments. A source-manifest hash records the acceptance snapshot; no checkpoint commit was requested or created. The pre-change browser identity collection is preserved, but a complete pre-change source hash was not captured and is not claimed.

## Assessment baseline

The October 6 assessment recorded 1,105 ordinary backend tests passing with 19 opt-in skips; 14 offline cloud process/codec/recovery tests; 12 real local Whisper checks; 182 frontend tests; passing typecheck/build; 26 connection browser tests; two desktop shutdown tests; and one Rust relocation test. All 14 journey identities passed across two runs: the first passed nine before a 210-second suite limit, and a targeted second run passed the remaining five. This is not a successful uninterrupted full-suite run. The default browser collection passed 33, failed one PKCE fixture case and skipped two live oMLX cases. Parent-process Python source coverage was 88.7% line and 77.3% branch; subprocess fixture execution was not measured by that coverage run.

The earlier cloud-remediation record reports 1,103 ordinary passes. The newer checkout also includes two sample-workflow tests; the retained assessment log reports 1,105. These figures describe different collection states. The [assessment evidence record](../reviews/2026-10-06-app-gap-assessment-evidence/README.md) retains logs, coverage summary and reproduction commands, with limitations on reconstructing the original invocations stated explicitly.

The failing PKCE persistence case submits a fixture code to the ordinary backend, which returns HTTP 400. It passes under the connection configuration's fixture transport. CI currently runs the default configuration, without separate connection or journey jobs. Fix collection without weakening the persistence test.

Sources: [CI](../../.github/workflows/ci.yml), [connection tests](../../frontend/tests/e2e/cloudConnections.spec.ts), [connection configuration](../../frontend/playwright.connections.config.ts), [journey configuration](../../frontend/playwright.journeys.config.ts), [cloud acceptance](../reviews/2026-10-06-cloud-gap-implementation.md), [macOS requirements](../delivery/macos.md).

## Priority, sequence and responsibility

| Priority / slice | Responsible role | Dependency | Engineering estimate | Deliverable |
|---|---|---|---|---|
| P0 Recording safeguards | Engineer | Mandatory Setup calibration | 1–2 days | Measured input, bounded warnings and explicit readiness |
| P0 A1: default/connection CI | Engineer | None | 1–1.5 days | Disjoint collection, isolated launchers, exact result gate |
| P1 A2: Chromium journeys | Engineer | None; reuse A1 gate | 1–2 days | Clean-environment 12-case lane with bounded startup/run/teardown |
| P1 A3: WebKit CI probe | Engineer | A1 result gate | 0.5–1 day probe | Two upload cases on a verified platform; explicit runner decision |
| P0 B: sharing contract | Engineer | None; verify with A | 3–5 days | Backend route identity and accurate messages in five locales |
| P0 V: assessment eligibility | Engineer | B; before E0/D1 | 3–5 days | Shared versioned eligibility; grade/progress/export guards |
| P1 C: native/cloud acceptance | Engineer and app owner | Local A1/A2 and B for candidate | 5–9 days plus external prerequisites | Artifact-bound acceptance matrix; unresolved rows remain open |
| P1 D: quality evaluation | Engineer and EN/IT specialists | Protocol starts now | 6–10 days plus recruitment/ratings | Authored-case review, blind learner pilot and defect backlog |
| P2 E: backup/deletion | Engineer and product owner | E0 storage contract | 15–25 days: E0 3–5, E1 5–8, E2 7–12 | Full two-store backup, staged restore and recoverable deletion |
| P2 F: Review/History usability | Engineer and representative users | A; coordinate with E | 6–10 days plus observations | Focused UX/accessibility improvements and history benchmarks |
| Optional G: transcript corrections | Engineer and product owner | D findings, E0 manifest, feature decision | 4–7 days | Separate feedback-only record; original spoken evidence immutable |
| P1 H0: support package download | Engineer | Existing support bundle; coordinate with C | 1–2 days | Verified native save, ZIP contents/redaction and recovery |
| P2 H1: send support package | Engineer and app owner | H0; editable recipient and native Mail | Original estimate 3–5 days; local draft implemented | Explicit attached draft, accurate handoff result and failure recovery |

Use the **October 8 execution revision** as the single canonical order. The table records effort ranges, not a conflicting execution order. Source work in H0a precedes C; installed H0b is measured on the rebuilt artifact. E production backup/restore/archive/purge and the F code slice are implemented; broader human/accessibility measurements remain open. G remains optional. External prerequisites do not stop independent local engineering.

## Reported bug fixes — historical verification

- [x] October 8: preserve the learner nickname across page navigation before setup submission, including edits and explicit clearing. Starting another practice retains the nickname while resetting the practice/report/recording state. Four regression cases failed before the fix; all 187 frontend tests, typecheck, production build and the dedicated localhost Playwright navigation case now pass. This fixes navigation within the current app session; a rebuilt installed DMG has not been verified for this change.


- [x] October 8: add **Setup Guide → Test microphone** with an explicit permission request, a live input meter, local signal detection, and clear denied/missing/busy/silent/timeout/unsupported feedback. Stop capture on success, cancellation, timeout and navigation; release late permission grants. Successful recordings in Speak/Rehearsal update shared microphone readiness. Home now counts speech recognition, AI and microphone checks rather than assigning a fixed 72% to arbitrary warnings; maintenance notices remain visible separately. Checks persist across navigation/new practices during this app session and invalidate on supported permission/device change events. Verified: 202 frontend tests, typecheck/build, 16 backend diagnostics/i18n tests, and four localhost Chromium regressions (three microphone cases plus nickname). The Vite/backend smoke used isolated app/cache directories; backend `/v1/health` passed the Playwright startup probe. Native OS permission prompting and a rebuilt installed DMG remain unverified.

  Browser audio evidence: a generic native Web Audio probe showed a stalled clock with the default output on this test machine (0.00533 s), and a working silent sink (1.01333 s, peak 0.109). The implementation feature-detects silent output, retains the default path for other browsers, and distinguishes a stalled engine from silence. The API is documented by [Chrome for Developers](https://developer.chrome.com/blog/audiocontext-setsinkid/). Chromium tests supply generated PCM audio; no learner recordings are used or uploaded. Desktop and 390 px screenshots were inspected, with no horizontal overflow; test output and screenshots are temporary local artifacts.

## Microphone flow follow-up — October 8 investigation

The navigation-menu Speak link uses the same RecorderPanel and app-wide store as Home. A new explicit browser regression confirms that entering Speak from navigation, recording and returning Home updates readiness without opening Runtime Setup. Both Home and drawer cases pass: five Chromium regressions total, 202 frontend unit tests and typecheck. The original fixed 72% score was unrelated to visiting Setup. The new source cannot produce 72% from its three equally weighted checks; a still-visible 72% would indicate an older frontend build, not the navigation path. The installed DMG has not been rebuilt.

Historical gaps identified before the current implementation: decorative recorder bars and readiness granted merely by a nonempty Blob. Both are now fixed. Live recording measures RMS/peak; saving practice audio never grants Setup calibration. Calibration remains session-scoped and must be repeated after reload/device/processing changes.

Relevant official product evidence, checked October 8:

| Product | Documented behavior | Application here |
|---|---|---|
| [Yoodli practice guide](https://support.yoodli.ai/en/articles/9550465-practice-with-yoodli) | Practice screen requests microphone permission; gear controls select the microphone; a volume meter confirms input before recording. | Put input checks and selection at the recording entry point. |
| [Speechling quickstart](https://speechling.com/help/quickstart?overwriteDefault=true) | Record directly in the exercise, see the voice waveform, stop, replay and optionally record again before submitting for coaching. | Measure real input and retain convenient local playback before analysis. |
| [ELSA Speech Analyzer published walkthrough](https://vn.elsaspeak.com/wp-content/themes/theme-sa-tour/index.html) | Home offers Start Recording or Upload Recording; guided practice reminds learners to enable their microphone and choose a quiet environment. | Keep direct practice access and distinct recording/upload readiness. |

This is evidence from official guides and a published walkthrough, not hands-on verification of authenticated competitor apps.

The owner subsequently required a microphone check in Setup after reporting crackling. This supersedes the optional-test recommendation above for live recording.

- [x] Require a five-second native MediaRecorder sample, at least one second of detectable input, full playback and explicit clear-sound confirmation before live recording. Speak through Home/drawer and Rehearsal use the same calibration gate. Uploads remain available without microphone access.
- [x] Show real RMS/peak metering during the sample; reject silence, very quiet input and repeated near-full-scale peaks. Retain local playback for manual crackling/distortion rejection. Level metering does not prove absence of crackling.
- [x] Select the input device after permission and reuse it with the same processing settings and native recording MIME selection for practice. Request automatic gain control off where supported; allow echo/noise reduction to be compared on/off. Explain system input-volume adjustment rather than offering a software slider that cannot repair input clipping.
- [x] Require retesting after device/processing changes, permission revocation, capture failure or app reload. Release streams/contexts on cancellation, navigation and late permission grants; bound permission and encoder waits. Discard the local sample on leaving Setup/retesting. Preserve rehearsal learner choices through the required Setup visit.
- [x] Add real RMS/peak input metering and bounded sustained-silence/clipping warnings during practice. Preserve the take and invalidate calibration on bad input. Normal pauses and single transient peaks do not trigger a failure.
- [x] Replace Home’s numeric percentage with the explicit completed-check count (`0/3` through `3/3`), retaining accessible readiness text and maintenance notices.
- [ ] Rebuild and test the installed macOS DMG with physical microphones, playback, device changes and permission recovery before claiming native acceptance or that the reported crackling is fixed. Its actual cause is not yet established.

The research and October 8 microphone implementation are outside the original Claude feasibility review. Source/browser verification does not replace installed WKWebView/TCC and hardware acceptance.

## Ubuntu and GitHub execution

The app owner has made the Ubuntu server available through `ssh p50` and permits GitHub as an alternative. On October 8, SSH, Ubuntu 24.04 x86_64 and Docker were verified. An isolated Ubuntu Playwright 1.59.1 container provides Node 24; Chromium and WebKit both passed headless launch and a generic synthetic file-input probe. See [Ubuntu probe evidence](../reviews/2026-10-08-ubuntu-browser-probe.md). These checks establish browser capability, not application journey acceptance.

Use p50 for the A2/A3 Ubuntu runs and backend/frontend verification, in temporary container workspaces with bounded resources and fresh app-data/cache roots. Keep server credentials and learner journals out of the fixture environment. Selected source and six repository synthetic CEFR WAVs were transferred under the owner’s full permission. Application acceptance ran in a network-disabled Playwright container, with fresh stores and no accounts or learner data. See the [current storage/Ubuntu evidence](../reviews/2026-10-08-app-gap-implementation/storage/README.md) for source snapshots, results and retained earlier failures.

GitHub Actions is an alternative for required workflow evidence, recorded against its actual commit. A p50 run of the current source snapshot does not establish that a GitHub workflow passed. Continue building/signing the DMG and verifying installed WKWebView, microphone and Keychain behavior on macOS. The Ubuntu environment addition is outside the October 6 Claude review.


## Assessment validity follow-up — October 8 short-speech report

The learner reported that saying “bla bla bla” earned WPM ≥80 and fillers ≤6 passes. Source diagnosis: `evaluate_baseline` excluded duration/minimum-word failures from its invalidating gates and treated unknown content validity as eligible. The five-word minimum is an LLM-call threshold, not a validated sufficiency standard. WPM can be calculated for three words; zero dictionary-matched fillers does not establish coherent speech. The displayed ≥80 is a B1 practice target, distinct from the observed rate.

- [x] Withhold baseline judgments when duration/minimum-word/language/topic/content checks fail, or content validity is unverified. Preserve raw observations and record invalidating/unverified gates. Guard the frontend when opening older saved reports with stale passes. Hide Review and History-detail grades when the minimum-word check fails or scoring was skipped for low word count, with localized instructions to record connected sentences. Add Expected/Actual column labels.
- [x] Follow-up learner decision: require **at least 30 seconds of recording duration** for review. Speak blocks known short recordings while retaining playback/removal; backend checks decoded duration before local/cloud transcription and AI review, including uploaded files, CLI and resumed jobs. A rejected attempt produces a localized retry message, no review and no History entry. This is an app policy chosen by the learner, not a validated CEFR cutoff. Longer silence/repetition still requires content checks.
- [x] Fix contradictory language praise: lack of a confident mismatch no longer implies a language pass. Unresolved results stay unknown, baseline judgments are withheld, and uncertain language uses conservative fallback coaching. Required-language praise requires positively verified language/content; unknown is also not diagnosed as wrong language. Language-specific tests now use recordings at the 30-second boundary, preserving those regressions alongside explicit short-input rejection tests.
- [x] Implement a versioned eligibility contract across current/legacy reports, API, journal, History, comparisons, rehearsal and exports: insufficient speech, content unverified, invalid content, and assessable. JSON retains original numeric observations with explicit provisional status; new CSV rows leave ineligible grade fields blank. Legacy dashboard/summary reads withhold grades and progress without rewriting saved evidence, including when the linked report is missing. Evaluation cannot count withheld grades as CEFR pair successes. Saved coaching praise/strengths and progress deltas are also withheld for ineligible reports, while original evidence remains inspectable.
- [x] Separate task completion from metric reliability: falling short of a long task's target duration does not invalidate otherwise verified observations. Retain local short-take playback and the 30 decoded-second review policy, with a one-second client encoding margin.
- [ ] Independently validate speech sufficiency by task/language through D. The 30-second policy and five-word LLM minimum are not CEFR requirements.
- [ ] Treat pace as practice guidance and filler counts as observations until calibrated. Show count plus a sample-normalized rate and examples; sparse samples receive no judgment. Zero recognized fillers is not a strength when content is insufficient/invalid. Avoid classifying repetitions with a single-word blacklist: quoted phrases, emphasis and ordinary learner hesitation can be legitimate.
- [ ] Extend D0's adversarial evaluation with three words, long repeated syllables, silence/ASR hallucinations, wrong language, legitimate short answers, quoted “bla bla” in a coherent answer, and meaningful hesitant speech. Verify normal practice guidance remains available. Make content validation failure/unavailability explicit, rather than silently granting passes.

The [Council of Europe qualitative spoken-language grid](https://www.coe.int/en/web/common-european-framework-reference-languages/table-3-cefr-3.3-common-reference-levels-qualitative-aspects-of-spoken-language-use) assesses range, accuracy, fluency, interaction and coherence; it does not define these app WPM/filler thresholds. [Yoodli's filler guidance](https://yoodli.ai/blog/yoodli-skill-reduce-um-filler-words) uses a percentage as a coaching goal. That supports normalizing the observation, not importing its target as a validated CEFR rule.

Verification for the immediate guard: 63 backend assessment/i18n tests, 208 frontend tests, typecheck/build, and two localhost Chromium cases (old three-word report versus eligible speech). Backend `/v1/health` passed the browser startup probe on port 8913. No learner recording was sent to a model or external service. Installed DMG not rebuilt; broader validity/normalization work above remains open.

## A Repair browser CI

Primary files: `.github/workflows/ci.yml`, the three `frontend/playwright*.config.ts` files, `frontend/tests/e2e/cloudConnections.spec.ts`, `tests/e2e/journey_backend.py`, and result helpers under `scripts/`.

### A1 Default and connection lanes

- [x] Record the pre-change union of collected test identities, including file, project and parameterized title. Move real PKCE persistence into a fixture-only spec/directory and give default/connection configurations disjoint collection rules. Preserve all 58 mandatory identities and move the two live oMLX identities to an opt-in collection. Add one rehearsal sharing contract, for 59 mandatory cases. Derive fixture backend URLs from configuration rather than a hardcoded port.
- [x] Explicitly set `retries: 0` for mandatory fixture lanes. Emit Playwright JSON and diagnostic artifacts. Reuse the existing integration verifier's mandatory-case accounting where practical, adding JSON parsing for browser status/retry metadata. Reject failed, flaky, skipped, missing, duplicate and collection-only results. Titles alone are insufficient identities. Add verifier tests for these failure modes; keep pytest JUnit gates working.
- [x] Give every invocation fresh app-data/cache/HF roots, memory keyring and isolated ports. Clear inherited provider keys and proxy variables; install the test-only external-network guard for backend and spawned workers and assert its log at teardown. Refuse collisions rather than reusing another server. Keep production endpoints and packaged code free of test seams.
- [x] Add the connection job with locked dependency setup and useful logs/traces uploaded on failure. Decouple ordinary backend results from browser success while keeping all required jobs in aggregate readiness. Preserve cloud process/codec/recovery and opt-in real-ASR lanes. Align CI Node with the repository Node 24 declaration, follow the npm dependency policy and use `npm ci --ignore-scripts` without new dependencies.

### A2 Chromium journey lane

- [x] Replace the journey launcher's hardcoded `.venv/bin/python` assumption with an explicitly selected interpreter matching CI setup. Install its actual media/system prerequisites. Reuse A1's environment isolation, memory keyring, network guard and fresh roots. Prove the 12 Chromium journeys run with an empty HF cache; a fixture lane must not need a model download or provider account.
- [x] Replace stale pre-calibration timings with five-second calibration, actual playback and decoded-duration checks. Ubuntu’s constrained encoding required 45-second synthetic captures; production still enforces 30 decoded seconds. The full journey budget is 30 minutes and CI bounds execution at 30.5 minutes within a 45-minute job. Retain actual case/suite timing and tune from observed remote runs; local execution is not remote CI evidence.

### A3 WebKit platform decision

Both mandatory WebKit upload cases pass locally on macOS in the same uninterrupted 14-case journey run. Both application WebKit upload cases also passed on Ubuntu in the isolated application runs. CI defines a separate required WebKit job. The p50 run establishes platform feasibility; it is not an observed GitHub workflow.

- [x] Run both WebKit application upload journeys on p50 with locked Playwright/system dependencies. Both passed with no retries/skips; required identities remain in CI. Source WebKit coverage does not replace installed WKWebView/TCC microphone acceptance in C.

Acceptance: locally executed required default/connection identities and all 14 journey identities pass without retries or mandatory skips, with a documented platform for WebKit. A1/A2/A3 platform feasibility is verified locally and on p50; a GitHub workflow run remains distinct evidence. A clean CI checkout has no provider credentials and cannot make external inference calls. Remote CI is proven only by an observed workflow for a known commit; local work does not wait for authorization to push or dispatch.

## B Bind sharing statements to accepted routing

Implemented: canonical v1 route fingerprints exclude volatile token/catalog data, include actual local/LAN/cloud/fallback endpoints, refuse unknown/automatic destinations, require receipts before new/resumed dispatch and preserve immutable cloud settings. Request-ID/fingerprint conflicts return 409. Resume preflight retains the original transcription model and current analysis preferences. The preferences lock assumes one backend process; multi-process server coordination remains future work. A dedicated contract browser case proves that a route change after the first part preserves its completed report/audio, creates no remaining jobs, and reload resumes only the remaining parts with the newly displayed route. The immutable first job retains its original model.

Primary files: `app_backend/{app,contracts,jobs,cloud_routes,cloud_runtime}.py`, `app_core/preference_lock.py`, frontend API types/client, setup components, Speak/Rehearsal/resume UI, `locales/{en,it,de,es,fr}.json`, and focused API/component/browser tests.

- [x] Define a backend-owned sharing-route response and fingerprint: ASR provider/connection/model; effective analysis provider/connection/model/endpoint; and every enabled fallback destination/model/mode. Exclude secrets. Display only sanitized hostnames for compatible endpoints, without userinfo, path or query. Explain that OpenRouter sends text onward to its selected model provider. Unknown destinations are visibly unavailable and block submission until resolved.
- [x] Add the fingerprint to assessment submission and both resume contracts. Validate it under the same preferences critical section as settings read and immutable worker/runtime snapshot creation; include it in request-ID idempotency identity. A mismatch returns HTTP 409 before any provider dispatch. Document server-client compatibility: missing identity cannot silently imply local sharing; update first-party/server clients and contract tests together. Keep lock ordering consistent with ChatGPT authorization and job submission to avoid deadlocks.
- [x] Preserve CloudRuntime's existing constructor snapshot of settings/connections. Ensure accepted jobs use the validated snapshot, including fallback, rather than recreating it after releasing the lock. Test settings changed before publication and during execution; neither may reroute an already accepted job. A resumed job is a new acceptance against its newly displayed route, while existing cache provenance remains valid.
- [x] Replace both unconditional messages: ChatGPT's “transcription stays local” and CloudSettingsPanel's “audio goes to Groq.” Use one derived summary through the established translation utilities and five locale files, not additional positional-array indices. Local Whisper and Groq must each show their actual audio destination; text/context recipients and enabled fallback must also be clear.
- [x] Refresh and show the summary immediately before ordinary submission, resume and rehearsal analysis. If routing changed, show it and require a fresh submit action. Revalidate each rehearsal part and stop the remaining loop on mismatch; a saved session's creation-time configuration is not consent for a later destination.
- [ ] Cover local/local, local/remote, Groq/ChatGPT, Groq/OpenRouter, compatible endpoints, enabled fallback, missing accounts, load failure, stale tabs, concurrent settings edits, idempotent replay and both resume routes. Assert that credentials/secret references never enter display text or API sharing metadata.

Acceptance: displayed destinations match the backend-accepted route, including resumes and each rehearsal part. Concurrent settings changes yield an explicit conflict, not an undisclosed destination. Focused concurrency tests and representative local browser journeys pass.

## C Complete native and cloud acceptance

- [x] Rebuild the internal ad-hoc DMG with a fresh frozen helper; pass read-only mounted/copied self-tests, isolated EN/IT real ASR workers, protected health, reports/history/ranged audio, restart/owner-exit, signature validation and installed WKWebView support save. Exact artifact evidence is linked in the implementation record.
- [x] Reproduce and harden terminal-worker teardown: maintenance waits briefly for terminal workers, while active work remains refused. Two regression cases cover this behavior. The original historical 409 lacked its response body, so its exact cause remains unknown; current artifact acceptance retains response diagnostics and is reported separately.
- [ ] Verify ad-hoc-to-Developer-ID upgrade identity, existing Keychain ownership/readback and hardened-runtime microphone entitlements on the signed candidate. Retain user state across upgrade; do not reset the main user’s TCC or credentials.

Primary files: `scripts/{test_macos_delivery,build_macos}.py`, `frontend/src-tauri/src/desktop.rs`, desktop process tests and `docs/delivery/macos.md`. Change production code only for reproduced defects.

- [ ] Define a matrix for local/cloud paths, EN/IT, microphone grant/denial, saved credentials after restart, browser return, empty cache, quit during work, second launch, playback and history. Bind outcomes to artifact SHA-256, macOS version/date and sanitized diagnostics. Check pinned/hashed build dependencies and their cache availability before scheduling a rebuild; a download failure is a distinct prerequisite. Verify macOS 14.x on an actual machine before claiming the declared minimum is tested.
- [ ] Time-box a 1–2 day feasibility probe for the full frozen cloud upload → worker → broker → validated report → restart/history journey using offline fixtures. Source `PYTHONPATH` injection is not evidence for a frozen helper. Decision criteria: no shipped endpoint override, no learner secrets, and no changes to the main user's hosts/trust store. Use disposable roots and an isolated user/VM for any environment routing. If no safe harness works, record the reason, leave offline frozen-cloud acceptance open and continue other rows. The existing compiled broker smoke covers only the broker.
- [ ] Run the installed webview in an isolated macOS user through physical microphone grant and denial recovery, actual Keychain save/read/restart/delete, browser authorization return, and cold-cache download followed by offline reopening. Ad-hoc-artifact results are provisional; repeat on the exact Developer ID candidate. Do not reset the main user's privacy database.
- [ ] Prepare dedicated provider accounts, a declared spending cap and consent-compatible synthetic inputs for live testing. Cover OpenRouter login/model selection, Groq timestamps, ChatGPT renewal, quotas, fallback and cost reconciliation. These live calls require the accounts/cap decision before execution; this plan starts no inference calls.
- [ ] Once signing credentials exist, build the Developer ID candidate and repeat native acceptance on its exact DMG. Verify notarization/stapling, quarantine-preserving browser download, clean-user/second-Mac installation and startup without development tools. A full live packaged-cloud journey can establish the live artifact gate if the offline harness is unavailable; record the offline gate separately rather than calling it passed.

Acceptance: separate records for internal offline artifact, native behavior, live provider paths and public signed release. Public cloud-support claims require full frozen-worker evidence on the release candidate for the advertised routes; compiled-broker-only evidence is insufficient. Missing accounts, signing credentials or hardware keep their rows pending without blocking independent local work.

## D Independently evaluate coaching quality

Engineering protocol: [blind quality protocol](../evaluation/2026-10-08-blind-quality-protocol.md). Corpus accounting and held-out-label separation are implemented and tested. Specialist sign-off, audio recruitment and adjudicated ratings remain pending.

Primary files: feedback-quality fixtures/README, `scripts/evaluate_feedback_{generation,review}.py`, calibration evaluation/manifests and versioned study protocol/report tooling.

- [ ] Split D into D0 authored-feedback review and D1 learner-audio pilot. English and Italian specialists independently rate the existing 20 v2 authored cases, then adjudicate disagreements. Keep v1 separate. D0 runs the chosen release configuration and records confidently false or meaning-changing corrections; make each confirmed defect a permanent regression.
- [x] Give the generation CLI explicit corpus/version selection and default to all selected IDs rather than six fixed v2 cases. Record requested/excluded IDs with reasons, reject missing IDs, and retain malformed, repaired, rejected and unavailable outputs in the denominator. Alternate-provider support remains Ollama/LM Studio; a cloud extension is separate work subject to the declared cap. Use recording hooks to verify expected-label fields never enter request construction, without rejecting natural-language overlap with legitimate transcript content.
- [x] Before D1, separate a fixed protocol task goal (or `None`) from human proficiency labels. Held-out `expected_cefr` is now separate from the fixed task goal; captured-runner evidence checks that only the task goal reaches inference. Add a deterministic test changing only the held-out label and requiring identical score/band/level fields, plus a captured-runner/request test proving labels are not supplied to inference.
- [ ] Target 60 clips, EN/IT × B1/B2/C1 × ten clips, from at least 20 consenting speakers with several speakers per cell. Recruit by placement/self-report and assign cells after two independent blind ratings and adjudication; report actual counts and shortfalls. Existing synthetic samples and unrated folder labels are not labelled learner data. Use opaque clip IDs and pseudonymous speakers, with independent transcripts. The current B1 floor must be stated if below-range clips appear.
- [ ] Keep study recordings/transcripts in a dedicated data/cache root outside the repository and personal journal, with consent, retention/deletion dates and compatible provider choices. Raters see feedback in randomized order with model/provider/prompt identity hidden; maintain an unblinding key separately. Never send study material as source-review input.
- [ ] Freeze source/prompt/model/ASR/scorer identities. Run one default configuration, three repeats of a fixed subset, and one supported alternate configuration. Pre-register decision tolerances, agreement statistic and speaker-aware uncertainty method with the specialists before viewing results. Report grammar false positives, missed seeded errors, wrong/meaning-changing corrections, withheld feedback, WER, repeatability and rater/model agreement by language/level. Keep audio proficiency and transcript-controlled coaching evaluations separate.
- [ ] Produce a reproducible report and ranked defect backlog. Use speaker-held-out tuning/validation for any later boundary changes. Label CEFR provisional and pronunciation an ASR intelligibility proxy; this small pilot cannot establish population calibration.

Acceptance: D0 has adjudicated authored-case results and no unresolved confidently false or meaning-inverting correction in the tested release configuration. D1 has blinded labels, complete outcome accounting and uncertainty, with incompleteness reported. D1 recruitment is not a DMG gate; stronger assessment claims require a larger held-out validation study.

## E Add backup, restore and recoverable deletion

Primary files: backend journal/history services/contracts/jobs, `app_data.py`, History/Settings UI, `frontend/src/lib/rehearsal/storage.ts`, and focused API/process/browser tests. Reuse stdlib archive/checksum tools, native IndexedDB and existing native capabilities.

### E0 Storage contract and maintenance

- [x] Map History/CSV/JSONL, reports/revisions, managed audio, retries, jobs/cache and rehearsal IndexedDB, including shared references and current-origin boundaries.
- [x] Implement a versioned SHA-256 inventory with stable identities, path sanitization/rebinding, provenance and explicit omissions. Restore skips existing identities and previews those skips. Account state, credentials, jobs and stage caches are excluded; restored reports remain non-resumable.
- [x] Coordinate every normal writer and CLI assessment with the backend lease/file locks and every rehearsal writer with the shared browser lock. Refuse live/resumable work and stale previews. Persist cross-store publication/recovery state and completion receipts; compensate definite second-store failures and recover lost responses without blindly overwriting data.

### E1 Backup and restore

- [x] Expose Settings backup, native Save/Open and browser fallback. Use bounded media transfers, expanded-archive limits, checksums, disk-space checks and cancellation before publication. Preview selected new identities, omissions and non-resumable limitations before commit.
- [x] Verify cross-root restore, populated-store selective skips, corrupt/traversal/duplicate archives, missing recordings, low disk, interruption, idempotent completion and startup recovery. Exact-artifact acceptance restores real EN/IT reports/audio; native WKWebView saves a populated synthetic rehearsal backup with matching audio bytes.

### E2 Archive, Undo and confirmed purge

- [x] Back up both browser rehearsal manifests/Blobs and backend attempts. Cover populated-v1 migration, rollback after IndexedDB abort, interrupted completion/reload and stale/concurrent browser writers.
- [x] Archive with Undo; separately preview and confirm permanent removal. Protect shared media and retry links, refuse active/resumable jobs, remove owned report revisions/completed caches/JSONL entries and upload sidecars. Rehearsal purge preserves its analysed backend attempts; independent exports remain independent copies.
- [x] Use same-directory tombstones and persisted recovery records for backend removal. Inject failures around moves/index publication, preserve unrelated data, and recover on restart. Optional empty-directory cleanup cannot strand a committed purge; two regressions cover shared-cache and injected cleanup failures.

Acceptance evidence is in the [production storage contract](2026-10-08-storage-contract.md) and [current verification record](../reviews/2026-10-08-app-gap-implementation/storage/README.md). Checksums establish archive integrity, not authenticity. An unknown browser profile cannot be enumerated by the backend; purge protects the references supplied by the current profile. External recovery conflicts are retained for repair rather than overwritten.

## F Improve Review and History usability

- [x] Remove duplicate next-focus/exercise text in Review while retaining the prominent practice card. Retain History’s existing compact digest/full-detail disclosure. No callable Stitch capability was available in this session; changes stayed within the current product structure.

The isolated synthetic service benchmark measured median warm-filesystem loads of 92.404 ms / 846,280-byte payload for 1,000 attempts and 1,010.736 ms / 8,482,780 bytes for 10,000 attempts (macOS arm64, ten logical CPUs, Python 3.12.11). It measures backend loading with the final legacy-export guards while a browser journey was running, not HTTP/render/filter time, memory or physical machine-model performance. Keep those measurements pending; the growing payload warrants a future bounded pagination experiment. See the retained benchmark JSON.

Primary files: Review/History routes/components, recorder controls, history endpoints and existing route/browser evidence.

- [x] Consolidate duplicated Review focus/exercise text and retain the prominent practice action with warnings/provenance accessible. No callable Stitch capability was available; the code slice stayed within the existing design.
- [x] Retain History's compact digest and explicit full-detail disclosure. Explain practice score, 1–5 band and provisional CEFR without implying a percentage is measured proficiency.
- [ ] Verify desktop, 390px/360px, 200% zoom, keyboard and VoiceOver, including installed WKWebView. Cover navigation/error focus, timer/status announcements, record/upload, playback and disclosures. Add regressions for demonstrated behavioral defects; selectors alone are not accessibility evidence.
- [ ] Observe three to five users new to the app, within supported practice levels, completing setup, an attempt and a retry. Record completion, intervention, time and confusion, and repair recurring blockers.
- [ ] Benchmark generated 1,000/10,000-attempt journals on a named machine. Measure payload, API/render/filter time and memory, especially repeated CSV/audio lookup, resume job scans and per-row comparison work. Set budgets from measurements; add indexing/pagination only for demonstrated problems, retaining legacy readability.

Acceptance: clear next action, compact nonduplicative detail, completing keyboard/VoiceOver flows, no narrow-screen overflow and documented performance evidence. User recruitment does not gate unrelated release work; reproduced severe native usability defects do.

## G Optional corrected-transcript coaching

Primary files: Review/API/checkpoints, frontend `practiceProgress.ts`, backend `assessment_runtime/comparison.py`, and E0's backup contract. Product owner decides whether D findings justify this feature.

- [ ] Preserve original audio, ASR text/timestamps, metrics and report. Store corrections as a separate version with parent/provenance and a forward-compatible backup record agreed in E0.
- [ ] Generate explicitly labelled feedback-only output. Do not reuse word timings for new fluency metrics, overwrite spoken scores, or mix edited-text coaching into score progress. Apply B's sharing acceptance, existing quota/budget rules and distinct checkpoint identity.
- [ ] Link versions in History and backup/delete handling. Test grounding, immutability, cancellation/provider failure, repeat submission, migration, and exclusion from comparisons in both frontend/backend with shared parity fixtures.

Acceptance: corrected recognition text supports separate coaching without changing recorded spoken evidence. G is not a DMG prerequisite.

## H Support package download and send

Primary files: `frontend/src/components/settings/SupportPanel.tsx`, frontend API client/types, `app_backend/support_bundle.py`, support endpoints/contracts, `tests/test_app_backend_support_bundle.py`, Settings tests, native save/share integration and five locale files. Reuse [support bundle policy](../SUPPORT_MAINTENANCE_PLAN.md) and existing generation/download APIs. Learner backup/restore in E remains a separate storage contract.

### H0a Source save and H0b installed acceptance

H0a implements the main-window-only native bundle-ID command, main-thread `rfd` dialog, private atomic file copy, expiry/source validation, truthful browser fallback and cancellation/retry/privacy tests. H0b passed on the exact rebuilt internal candidate, including cancellation, same-bundle retry, ZIP integrity, default exclusions, temporary-path redaction and private permissions. Public signed-artifact repetition and the remaining hardware/failure matrix below remain pending.

- [x] Exercise Settings → create support package → save ZIP in the installed internal WKWebView using synthetic diagnostics and disposable app-data roots. Verify the learner can choose a destination, the file exists and opens, its filename/contents are correct, and the success message reflects a completed save. If the current Blob/anchor download cannot do this in WKWebView, use existing native file-save capabilities and retain browser download behavior for the server UI.
- [x] Keep reports, recordings and uploads excluded by default, with independent explicit inclusion controls. Inspect the actual ZIP for credential/token/secret-reference redaction, private path handling and selected contents; use seeded fake secrets rather than real accounts. Check these defaults and opt-ins in the UI as well as the backend.
- [ ] Cover save cancellation, insufficient disk space, archive-generation failure, expired/missing bundle, lost backend connection and a large optional attachment. Preserve the chosen inclusion state, provide a retry and show success only after the operation succeeds. Verify bundle expiry/cleanup, keyboard/VoiceOver operation and all five locales.
- [ ] Add meaningful component/API/browser regressions and exact-artifact native evidence to C's acceptance record. Repeat download on the signed public candidate. Keep existing support/backend tests and update the older support documentation's stale file references where this slice touches them.

Acceptance: a synthetic support ZIP can be generated, saved to a learner-chosen location and opened from the installed DMG; cancellation/failure is reported accurately and content defaults/redaction are verified. Download is usable without a support delivery service.

### H1 Explicit native email draft

- [x] Use macOS Mail with `info@frommherz-it.ch` as the owner-selected default recipient, which remains editable. The package preview shows size and selected categories before the explicit draft action. All five locales describe handoff accurately; Download remains available. No receiver service or recipient is invented.
- [x] Attach a private independent ZIP copy, validate the owned source/recipient and check attachment creation. Declare Apple Events usage and entitlement; bound handoff with timeout and preserve the generated package on failure. A draft is reported as prepared, never delivered, and is never sent automatically.
- [x] Verify actual native Mail recipient/ZIP attachment with synthetic diagnostics and `support@example.invalid`, then discard the unsent draft. Rust/component tests cover independent-copy lifetime, recipient validation and failures. The observed handoff is bound to its tested artifact; later backend-only rebuilds are recorded separately.

Retention: temporary generated packages expire separately from private draft attachment copies. Draft copies remain in the app’s support-email-drafts directory until cleared; saved exports, Mail drafts and receiver copies have independent lifetimes. Actual email delivery, receiver policy, and public-signed-artifact repetition are outside this local handoff acceptance.

## Verification and gates by outcome

Use the existing working `.venv/bin/python` after a version/import probe: the sibling path named in AGENTS.md is absent on this machine. Any correction to that instruction is a separate small documentation change after inspecting user-owned state. Bootstrap a missing/stale environment with `scripts/setup_env.sh` rather than system Python.

For each code slice, run focused meaningful regressions plus `npm --prefix frontend test`, `npm --prefix frontend run typecheck` and relevant localhost browser/health smoke. Run backend baseline for API/provenance/storage changes. Before a combined candidate, run the full ordinary suite and relevant cloud/ASR/browser lanes with explicit limits and diagnostics. A single coverage percentage does not establish native or model quality.

| Outcome | Required gates | Not required for this outcome |
|---|---|---|
| Local implementation slice | Focused regressions and relevant frontend/backend baselines | Remote dispatch, signing, learner recruitment |
| Internal DMG candidate | Local A1/A2, B, V guards, recording safeguards, H0b native support save, rebuilt helper and artifact-bound mounted/copied acceptance/installed UI smoke. Physical WKWebView microphone/AAC/no-monitoring remains an explicit acceptance row | A3 remote-platform decision, D1, E–G, H1 delivery setup, public signing |
| Merge / remote CI claim | Observed required jobs on a known commit, retained reports and mandatory identities; explicit WebKit runner decision | Local passes alone are insufficient |
| Public local-mode DMG | Internal gates; V including progress/comparisons/exports; D0 defect review; applicable source browser regressions; Developer ID/notarization/stapling; exact signed-artifact microphone, Keychain, browser return, cold-cache, support ZIP download, quarantine and clean-machine acceptance | D1 recruitment, E, F observations, G, H1 delivery setup |
| Public cloud-support claim | Public gates plus complete packaged cloud journeys for advertised provider routes and capped live-account evidence | Broker-only smoke does not satisfy this gate |
| Stronger CEFR/assessment claims | Independent labels, blind outcomes and larger held-out validation | Authored cases, TTS samples and a small pilot alone are insufficient |
| Backup/deletion feature | E0+E1+E2 recovery/round-trip evidence | A backend-only export cannot claim full learner backup |
| Support draft feature | H0 content/redaction, editable recipient, explicit draft action and verified attached native handoff with recovery | Draft preparation is not delivery; sending remains in Mail |

A pending offline frozen-cloud harness remains visible even if capped live artifact evidence establishes the corresponding public cloud-support route. Keep other outcomes moving while external gates wait. Existing reports, server methods and bounded server tests remain valuable.

Defer official partner exam simulation, phoneme/prosody scoring, adaptive curriculum, added learning languages and hosted rollout. These are separate product investments.

## Current implementation milestone

Follow the October 8 canonical execution order. Accept source recording/A1/A2/B/V/H0a engineering with relevant tests; record H0b under C on the exact artifact. Production E0/E1/E2 backup/restore/archive/purge and H1 native attached draft are implemented and tested. D tooling/protocol and F service benchmarks remain engineering foundations. Human D0/D1 ratings, broader F observations/accessibility, G’s feature decision, actual email delivery and public release acceptance remain distinct follow-on work. Do not mark those features or the overall roadmap complete.

See the [October 8 implementation evidence](../reviews/2026-10-08-app-gap-implementation/README.md) for exact current test results, source/artifact hashes, native acceptance and limitations. Historical counts below describe earlier patches.

October 8 minimum-duration verification: 185 focused backend assessment/language/coaching/provenance/jobs/i18n tests and 211 frontend tests pass, plus typecheck/build. Seven Chromium cases pass in an isolated localhost Vite/backend run, including an actual worker rejecting a synthetic 12-second WAV without a History entry, Home/drawer microphone readiness, permission denial and old-report guards. Three focused minimum-duration/saved-review cases were rerun after the language guard, and both Home/drawer recording cases were rerun with explicit assertions that short takes disable review and show the 30-second instruction. Backend health passed on port 8914. Boundary tests cover 29/29.999 seconds rejected and 30 seconds accepted; no transcription/rubric/coaching calls or logs occur for rejected inputs. This dated patch did not rebuild the DMG. The combined candidate is now rebuilt and recorded separately; old stored reports are annotated on read, not regenerated.

October 8 required-calibration verification: 227 frontend tests, typecheck/build, 16 backend diagnostics/i18n tests and nine isolated Chromium cases passed. Backend health startup probe passed on port 8920. The setup/playback, Home/drawer, rehearsal and excessive-input cases pass alongside the minimum-duration and saved-review regressions. Headless playback uses a silent output only in the browser fixture; no physical microphone/audibility or installed DMG acceptance is claimed. See [calibration verification](../reviews/2026-10-08-microphone-calibration-verification.md).

# Live-test modernization — 2026-10-03

## Test ownership and migration

| Previous check | Current home and decision |
| --- | --- |
| OpenRouter rubric/coaching round trips | `tests/test_integration_openrouter.py`: two independently collected languages for each operation, explicit model/key, transcript-grounded rubric and duration-validated coaching |
| Direct real-audio report and score-ranking tests | `tests/test_real_audio_assessment.py`: both 66/82-second M4A fixtures, provider-neutral prerequisites, decoded duration, report contract; strict AI success and explicitly allowed guarded fallback are distinct |
| Sample transcription smoke | `tests/test_sample_integration.py`: six hash-bound source-text references, error-rate smoke thresholds, word timing, two longer M4As, seeded noise in both languages and silence |
| Real-audio browser score ranking | Removed `realAudioHistory.spec.ts`; shared `tests/live/ollamaBilingual.spec.ts` covers upload, recorded retry, history, playback/download and changed analysis conditions |
| Optional local runtime setup | Replaced by ordinary-CI `runtimeSetupRecovery.spec.ts`: URL/model discovery contract and a specific failed connection followed by successful retry, using intercepted provider responses |
| oMLX authentication/setup and assessment | Remains a separate optional adapter lane; assessment uses the tracked Italian travel WAV and matching theme instead of an absent output file. oMLX is not silently treated as covered by Ollama/LM Studio |
| Increasing scores as proof of progress | Removed. Deterministic provenance regression covers decrease/flat/increase, and existing tests exclude changed analysis conditions from comparison |

## Acceptance and prerequisites

Default tests skip resource-dependent cases. Once explicitly enabled, missing recordings,
models, credentials or services are failures. Local providers do not require cloud keys.
OpenRouter's small text-only checks require an explicit `OPENROUTER_MODEL` rather than an
implicit preview model. These tests do not submit the longer recordings to a cloud provider
unless that provider is explicitly selected by the person running them.

The shared journey runner requires **every requested provider** to be available. Choose
only the providers you intend to test with `--providers`; `--probe-only` is informational
and may report unavailable providers without failure. Execution requires a Playwright JSON
report with at least one executed test; missing/all-skipped reports fail. Failed tests cannot
be turned into success by this check.

Keep three outcomes separate:

1. Workflow completion: transport, report persistence, UI and recording recovery work.
2. AI output accepted: rubric/coaching passed the production validation contract.
3. Feedback quality reviewed: the advice and diagnosis are actually useful and correct.

For the direct audio test, strict mode requires accepted AI feedback. Set
`REAL_AUDIO_ALLOW_GUARDED_FALLBACK=1` to check deliberate transcript/schema rejection and
labelled deterministic fallback instead. Missing/unreachable providers fail in both modes.
The shared browser runner records its existing acceptance scope and per-attempt output
contract; these test outcomes do not assign a language level.

## Commands

Use the repository environment specified in AGENTS.md (the bootstrapped `.venv` fallback
when the sibling environment is absent). Run from the repository root:

```sh
# Fast prerequisite, fallback and runner regressions; no provider requests.
.venv/bin/python -m pytest -q tests/test_live_suite_contracts.py tests/test_live_journey_runner.py tests/test_feedback_provenance.py

# Offline cached Whisper, no LLM or microphone needed.
RUN_AUDIO_INTEGRATION=1 WHISPER_MODEL=tiny HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -v tests/test_sample_integration.py

# Longer recordings stay local with this configuration; start Ollama first.
RUN_REAL_AUDIO_ASSESSMENT=1 ASSESS_SPEAKING_REAL_PROVIDER=ollama ASSESS_SPEAKING_REAL_LLM_MODEL=qwen3.5:4b HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -v tests/test_real_audio_assessment.py

# Four small authored-text OpenRouter calls (repairs may add calls).
# Supply OPENROUTER_API_KEY through your environment, never in a committed file.
RUN_OPENROUTER_INTEGRATION=1 OPENROUTER_MODEL=your-model-id .venv/bin/python -m pytest -v tests/test_integration_openrouter.py

# Deterministic browser recovery; no AI services required.
NODE_ENV=development node frontend/node_modules/playwright/cli.js test -c frontend/playwright.config.ts frontend/tests/e2e/runtimeSetupRecovery.spec.ts

# Isolated real journeys, explicit installed providers.
.venv/bin/python scripts/run_live_journeys.py --providers ollama
```

`ASSESS_SPEAKING_REAL_BASE_URL`, `ASSESS_SPEAKING_REAL_WHISPER_MODEL`,
`ASSESS_SPEAKING_REAL_AUDIO_PATH` and `ASSESS_SPEAKING_REAL_LANGUAGE` override the direct
audio defaults. A custom recording replaces the bundled pair, so an English override does
not inadvertently assess an Italian second fixture as English.

## ASR scope

The reference texts come from `scripts/generate_sample.sh`, bound to the tracked audio by
SHA-256. Normalization ignores punctuation/case and source-text accent omissions. The 35%
clean and 40% noisy word-error ceilings are coarse transcription regression tripwires, not
quality certification. Controlled noise uses a fixed random seed and approximately 20 dB
SNR. Silence must not produce invented speech. Long M4As exercise the same ffmpeg decoding
step as production before Praat measurements. No human reference transcript is claimed for
those two files; they currently check decoding and timing only in the ASR lane.

Recorded retry/history checks already live in the shared browser suite. The separate oMLX
lane remains useful for its authenticated OpenAI-compatible endpoint; wider provider
unification can follow if that runtime becomes part of the supported launch configuration.
Historical plans describing the deleted browser tests remain historical records; use this
migration note and the README for current commands.

## Verification on 2026-10-03

- Current working-tree backend baseline: **857 passed, 17 skipped**. The explicit
  opt-in cases are four OpenRouter text checks, two longer-audio assessments and eleven
  ASR cases. Separating cases increases visibility; the higher skip count is not lost coverage.
- Frontend: **139 passed**, typecheck passed.
- Ordinary Chromium browser suite: **10 passed, 2 optional oMLX tests skipped**.
  The two setup-recovery checks and two Review/History checks also passed focused runs.
  The existing history mock needed `comparison_verified: true` to represent the new
  accepted-comparison contract. No application guard was weakened.
- Cached tiny-Whisper ASR lane: **12 passed** (one manifest check plus eleven live cases).
  Initial test-authoring errors used upstream timestamp names rather than the app's `t0/t1`
  fields and passed M4A directly to Praat; both were fixed to exercise the production contract.
- Local Ollama Qwen3.5:4b longer recordings: **two strict failures**, followed by **two
  passing explicitly guarded workflow checks**. In the guarded run, test1 attempted
  generation and rejected output with `llm_invalid_schema`; test2 correctly skipped
  generation with `transcript_uncertain`. Neither accepted AI feedback. JUnit properties
  distinguish generation attempted, output accepted, acceptance scope and unreviewed quality.
- The first direct smoke also caught stopped Ollama as a provider failure. Ollama was then
  started and its installed model verified before the final independent two-recording runs.
- OpenRouter live execution was blocked by automatic approval review pending explicit
  authorization for the four authored-text payloads. No cloud calls were made by this task.
- Repository quality and diff checks passed. No dependency or lockfile changes were made.

Browser verification used a temporary copy of the ordinary configuration with isolated
ports 4195/8895, separate temporary app data, and server reuse disabled; the generated
configuration was removed afterwards. Exact command:

```sh
NODE_ENV=development node frontend/node_modules/playwright/cli.js test -c frontend/playwright.legacy-audit.config.ts
```

This run also smoke-tested Vite and the backend health endpoint through Playwright's
web-server readiness checks. No Chrome installation or privacy setting change was required.
Local logs are `/tmp/legacy-tests-{backend-final,frontend,browser-verified,asr-verified}.log`;
strict and guarded audio results are `/tmp/legacy-tests-real-audio-final.log` and
`/tmp/legacy-tests-guarded.log`, with properties in `/tmp/legacy-tests-guarded.xml`.

Claude CLI reviewed selected test/runner sources only. Accepted findings tightened setup-only
false passes, evidence presence, local child credential isolation, cached/offline ASR,
per-recording collection, provider/model assertions, specific UI recovery and missing noisy
word timings. Claims that production forces ASR language or that no replacement retry/history
coverage existed were disproved by source inspection. The shared runner now also records
skipped/failed/flaky test counts and rejects those outcomes. Evidence inspection failures
are recorded as failures rather than leaving the run marked running. Existing source hashes
remain the authority for the tested working tree; HEAD alone does not identify it.

These edits coexist with previously uncommitted feedback-hardening modules and live-runner
work. They were not independently committed, because doing so would omit required modules
or sweep unrelated work into this change.

# Reproducible live bilingual workflow tests

The existing Playwright live suite exercises real ASR and assessment inference through
setup, Speak, Review and History. It supplies tracked English and Italian WAVs to
Chromium's microphone input, so the browser really records speech with MediaRecorder.
It also uploads files normally. No physical microphone, OpenAI API key, Codex CLI,
or API quality evaluator is required.

## Prerequisites

Follow `AGENTS.md` for the Python environment. If the prescribed sibling venv is
absent, use `scripts/setup_env.sh .venv` and the repository interpreter. Install
frontend dependencies from the existing lockfile with `npm --prefix frontend ci
--ignore-scripts`, and install the locked Playwright Chromium browser if needed.
The suite introduces no npm dependencies.

Ollama must already be serving its selected installed model. LM Studio is optional;
its OpenAI-compatible server must be running. OpenRouter is optional and uses only
`OPENROUTER_API_KEY` from the environment. The setup test bootstraps the credential
through the normal local backend with untraced Node fetch, then tests and saves the
connection through the UI. The test backend explicitly uses
`scripts.journey_keyring.MemoryKeyring`: secrets remain in that process, never in
the OS keychain or evidence. Native keychain persistence is outside this suite's scope.
No credential is entered into the browser. OpenRouter assessments send the bundled sample's transcript
to that provider and may incur its normal inference charges. This is assessment
inference, not an external evaluator of the test results.

Cache both Whisper models before the offline-ASR tests. Defaults are `large-v3` and
`tiny`; the latter intentionally changes the analysis conditions for the recorded
B1 retry. This does not imply that tiny is recommended for learner assessment.

## Run the existing suite

From the repository root:

```sh
.venv/bin/python scripts/run_live_journeys.py
```

The runner starts with Ollama, then runs LM Studio and OpenRouter when their selected
models and credentials are available. It reports unavailable providers explicitly;
a missing Ollama model or any executed test failure returns a nonzero exit status.
It starts a fresh normal backend and Vite server for each provider, refuses to reuse
existing servers, and retains separate app-data and cache directories. Defaults use
ports 8816 and 4179; choose different ports with `VOSTAVO_LIVE_BACKEND_PORT` and
`VOSTAVO_LIVE_FRONTEND_PORT` if those are occupied by your own services.

Probe availability without assessments, or repeat the complete matrix:

```sh
.venv/bin/python scripts/run_live_journeys.py --probe-only
.venv/bin/python scripts/run_live_journeys.py --repeat 2
.venv/bin/python scripts/run_live_journeys.py --providers ollama --grep 'setup discovers|en B1'
.venv/bin/python scripts/run_live_journeys.py --providers openrouter
```

Use `--output /absolute/path/to/a/new-directory` for a chosen evidence destination.
Explicit provider subsets support focused reruns; the selected providers always run
in Ollama → LM Studio → OpenRouter order. A `--grep` filter automatically includes
the connection setup test. The first failure stops that provider's suite, so a setup
failure cannot cascade into six misleading workflow failures.
The runner rejects an existing directory to preserve earlier runs. Model overrides
are `OLLAMA_E2E_MODEL`, `LMSTUDIO_E2E_MODEL`, `OPENROUTER_E2E_MODEL`, and the matching
`*_E2E_ALTERNATE_MODEL` variables. LM Studio defaults to the available 3B Qwen model
and changes to its 7B counterpart for the B1 retry when both are available. Otherwise
it retains the same LLM while changing Whisper. The output includes exact model names,
source hashes, dependency versions, sample hashes and selected settings.

OpenRouter defaults to `mistralai/mistral-small-3.2-24b-instruct`, verified on this account without
changing its zero-data-retention policy. A catalog entry and valid key do not guarantee
that the account's routing/privacy constraints allow a particular model. Inspect
[OpenRouter's zero-retention model filter](https://openrouter.ai/docs/api/api-reference/models/get-models)
and retain connection-test failures rather than loosening account privacy settings.

## Coverage and evidence

Each provider runs UI connection discovery/testing/saving and six language/goal
cases: English and Italian at B1, B2 and C1. B1 includes a second attempt recorded
from its WAV through the real browser microphone pipeline. The retry changes Whisper
and, when available, the LLM model. Tests verify request parameters, persisted report
and history provenance, exact retry parent/prompt, unchanged earlier reports, distinct
comparison cohorts, transcript preservation, real inference, playback, seeking, range
requests and byte-identical recording downloads. The default strict mode rejects fixture
inference, dry runs and generic validation/coaching fallback. Observed ASR uncertainty is
an expected safety outcome: tests require discarded criticism, retained diagnostics and a
visible manual-review notice. A passing workflow never proves grammatical correctness.

To verify UI recovery when a model fails output validation, explicitly opt into a separate
functional acceptance scope:

```sh
.venv/bin/python scripts/run_live_journeys.py \
  --providers ollama,lmstudio,openrouter --grep 'it B1' --allow-guarded-fallback
```

This mode accepts retained `llm_invalid_schema`/`coaching_unavailable` results only when
Review visibly labels fallback and saved reports, History, playback and downloads remain
consistent. Rejected rubrics must have no retained LLM score; no rejected output is counted
as successful model feedback. Transport/provider unavailability still fails. The manifest
and per-attempt output-contract evidence record this scope explicitly; strict mode remains
the default and overrides any inherited fallback environment flag.

Recording downloads are tested on Speak. Saved audio is retrieved from History and
compared to those downloads. The app has no general report-export control in this
journey; report JSON is retained as test evidence rather than described as a UI export.

Each provider directory contains:

- `app`: saved settings, uploads, recordings, reports, jobs and backend logs.
- `playwright`: screenshots at setup, settings, Speak, Review and History; downloaded
  and saved audio; submitted parameters; complete status/history JSON; evidence hashes
  and explicitly exported `browser-trace.zip` files with browser actions, network
  requests and snapshots for passing and failing cases, alongside Playwright test traces.
- `results.json`, `html` and `runner.log`: the Playwright results and reproduction log.
- `evidence-summary.json`: decoded recording measurements and report content for review.

`manifest.json` records availability, versions, samples, commands and provider outcomes.
Interrupted/probe runs remain available; they are not counted as successful workflows.

MediaRecorder WebM files initially have an unknown browser duration. The test verifies
that seeking to the end discovers the last timestamp, then seeks back to one second
and compares that duration to the backend's real decoding measurement. This preserves
a real playback/seek check instead of treating infinity as a valid measured duration.

## Review output quality separately

Passing workflows prove persistence, transport and real model execution. They do not
prove that a rubric or coaching claim is correct. Inspect the actual retained recordings,
transcripts, rubric evidence, coaching, warnings and screenshots. The evidence inspector
only measures audio; it never calls an LLM or declares a semantic quality pass:

```sh
.venv/bin/python scripts/inspect_journey_evidence.py /path/to/run \
  --output /path/to/run/evidence-summary.json
```

For each reviewed report, record whether grammar examples actually demonstrate the
claimed error, whether feedback uses the selected language, whether the exercise is
specific and achievable, and whether the UI makes failed duration/content gates clear.
Do not require a later attempt's score to be higher: changed analysis conditions and
stochastic inference can change the result without a change in learner skill. The
initial host review is documented under `docs/reviews`; subjective listening and physical
microphone/device testing remain separate from decoded-audio and transcript review.

The deterministic fixture journeys remain available through
`frontend/playwright.journeys.config.ts` for bilingual UI recovery/cancellation and
WebKit coverage. They complement this live suite and must not serve as a quality oracle.
Chromium's [file capture implementation](https://chromium.googlesource.com/chromium/src/+/9d5b1e82901614e80b8e7c9dc7cdb674987e4feb/media/audio/fake_audio_input_stream.h)
supports supplying a WAV in place of its default synthetic beeps.

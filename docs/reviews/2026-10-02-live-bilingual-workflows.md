# Live bilingual workflows — 2026-10-02

The existing Playwright live journey was extended to configure each provider through
the UI, upload six tracked English/Italian B1/B2/C1 recordings, record B1 retries
through Chromium MediaRecorder, change analysis settings, and verify Review, saved
History, playback, seeking and recording downloads. No OpenAI key, Codex CLI result
or API evaluator was used as a quality oracle. See
[reproduction instructions](../testing/live-bilingual-workflows.md).

Evidence for the complete matrix is retained at
`frontend/output/live-journeys/20261002-matrix-2`, including isolated app data,
reports, actual WAV/WebM recordings, screenshots, traces and a manifest. Earlier
pilot/interrupted runs are retained separately and excluded from final pass counts.

## Verified outcomes

| Provider | Models | Playwright tests | Real assessments | Evidence directory |
| --- | --- | --- | --- | --- |
| Ollama | Qwen 3.5 4B; large-v3 → tiny ASR | 7 passed | 8 | `20261002-matrix-2/01-ollama` |
| LM Studio | Qwen 2.5 3B → 7B; large-v3 → tiny ASR | 7 passed | 8 | `20261002-matrix-2/01-lmstudio` |
| OpenRouter | Mistral Small 3.2 24B; large-v3 → tiny ASR | 7 passed | 8 | `20261002-openrouter-mistral/01-openrouter` |

The evidence directories are relative to `frontend/output/live-journeys`. The
combined index is `20261002-results-summary.json`: **21 tests, 24 assessments and
24 decoded saved recordings** in passing runs, with no test retries or flakes.
Decoded duration differs from report duration by at most 0.004 seconds. Credentials
were checked against retained files and decompressed trace archives; no actual
OpenRouter key was found. Host notes are in each successful provider directory's
`host-quality-review.json`. Failed provider/model experiments remain separate.

Repository verification: backend **586 passed, 5 skipped**; frontend **131 passed**;
TypeScript, production Vite build and repository quality checks passed. Existing
fixture bilingual/WebKit journeys also passed **12 tests**. The live setup test
verifies `/v1/health` and actual localhost Vite behavior for every successful provider.
The missing prescribed sibling venv was bootstrapped using the repository's
`scripts/setup_env.sh .venv` fallback. No npm dependencies or lockfiles were changed.
The current `codex/local-practice-readiness` checkout was used and user-owned
untracked files were preserved.

## Scope of the initial quality review

The host reviewed actual ASR transcripts, rubric evidence and coaching, decoded
the saved recordings to check signal/duration, and inspected representative Review
and History screenshots. This is a text/evidence review, not subjective listening:
pronunciation, naturalness and physical microphone/device behavior remain unverified.
These short, supplied sample clips are 16–23 seconds, below the selected 90-second
task duration. Duration failures are expected and must stay visible. B1/B2/C1 are
sample/task labels, not independently certified proficiency scores.

## Findings from Ollama Qwen 3.5 4B

| Case | Initial host verdict | Evidence |
| --- | --- | --- |
| English B1 upload | Unsupported correction | Claims missing articles while quoting “spent most of the weekend walking near the beach”, which already contains them. |
| English B1 recorded/tiny | Unsupported correction | Calls “to the coast” a missing-article error and treats the valid past-tense phrase “came home” as a tense problem. |
| English B2 upload | Unsupported correction | Claims an article error in “a balanced system”; coaching asks for “the system”, changing meaning unnecessarily. |
| English C1 upload | Plausible but generic advice | No recurring grammar errors; recommends richer syntax and vocabulary although complex linking is already present. This is not proof of CEFR accuracy. |
| Italian B1 upload | Unsupported correction | Labels correct “Siamo andati”, “abbiamo visitato”, “abbiamo mangiato” as recurring past/auxiliary errors. One explanation admits no evident auxiliary error but coaching still asks for correction. |
| Italian B1 recorded/tiny | ASR contamination and unsupported correction | The same supplied speech becomes “Lo scorso hanno affatto”, “spiaccia”, “revamo”. Feedback attributes these to learner grammar and falsely criticizes otherwise correct past forms. |
| Italian B2 upload | Unsupported correction and contradiction | Treats valid “è un modello flessibile” as a mood error; says “modello” repeats although it occurs once. Says explicit linking is missing despite “Allo stesso tempo, però”. |
| Italian C1 upload | Harmful suggested correction | Asks to replace grammatical “Se vogliamo che … sostengano” with “garantiscono”, which is not an appropriate grammatical substitution in that clause and changes its meaning. |

The Italian tiny retry produces incomplete rubric comment strings, despite valid
JSON and completed coaching. The report records low confidence and failed language/
content checks; the screenshot displays a manual-review warning, but the prominent
coaching still assigns blame to the learner. Neither transcript agreement nor a
successful HTTP/job result establishes trustworthy advice.

The Italian reports also expose zero deterministic cohesion markers despite real
connectives in their transcripts. Some feedback amplifies that metric into a false
claim of “total lack of connectives”. Review language-specific marker coverage and
avoid turning detector counts into categorical linguistic claims.

Changing Whisper is persisted correctly and separates History chart cohorts. Review
still displays numerical changes against the linked parent; such a delta must not
be interpreted as learner improvement/decline when analysis conditions changed.

Before learner-facing release, address evidence-grounded grammar criticism, ASR
uncertainty, incomplete comment detection and the presentation of cross-setting
comparisons. Workflow passes alone do not justify release-quality assessment claims.

## LM Studio Qwen 2.5: model-change review

The English B1 upload with the 3B model also invents a tense error in “came home”,
and its example “I got back home” is a proposed correction, not a quote from the
transcript. Its coaching requests at least 30 seconds rather than the 90-second
task. The recorded retry persists the 7B model and tiny Whisper correctly. It stops
alleging a grammar error, but its exercise contradicts itself: “90-second speech”
followed by “Aim for a duration of 15 seconds.” These are material coaching defects.

The inspected English B1 History screenshot clearly reports one comparable attempt
and says the retry was analysed under different conditions; both original and retry
remain in the journal. Playback shows the actual retained 18-second recording. This
is useful provenance behavior even when the generated coaching is unreliable.

The English B2 3B report falsely asks for “saves” in “can … save commuting time”,
calls this a past-tense error, and treats valid “should” as the wrong auxiliary.
It also claims there are no linking words despite “However” and “In my view”.
The next exercise asks for 60 seconds rather than the configured 90 seconds.

English C1 alleges weak ordering using “yet also on simplified arguments”, which
does not occur in the actual transcript. Italian B1 upload avoids invented grammar
errors but gives generic connector advice and calls the short clip's length adequate.
The Italian B1 7B/tiny retry inherits ASR errors and invents an agreement problem with
“mio”, which does not occur in the transcript. It also changes “siamo andati” into
“si sono andati” in its evidence quotes. The saved model provenance is correct;
the linguistic evidence is not.

Italian B2 avoids invented grammar and offers plausible elaboration advice, although
the generated Italian is awkward. Italian C1 falsely diagnoses repeated “ma” and
“sostengano”, each of which occurs once. Coaching perpetuates that false diagnosis.

## OpenRouter setup triage

The original `qwen/qwen3-8b` catalog entry and key validation passed, but actual
inference failed with HTTP 404: its only endpoint was excluded by this account's
zero-data-retention policy. No account settings were changed. The app's connection
endpoint leaves `LLMClientError` uncaught, returning HTTP 500 without CORS headers;
the browser consequently says “Failed to fetch” rather than explaining the provider
restriction. This is a product error-reporting defect, retained for follow-up.

The smallest relevant probe was rerun on 2026-10-02:

```sh
.venv/bin/python scripts/run_live_journeys.py --providers openrouter \
  --output frontend/output/live-journeys/20261002-openrouter-retry-2
```

Its connection-test wait failed with `Timeout 30000ms exceeded while waiting for
event "response"`. The explicitly exported browser trace confirms the failed local
POST; a direct app `test_runtime_connection` probe exposes the underlying ZDR 404.
An earlier retry exposed a native-fetch `response.ok()` test-code mistake; that was
corrected to the boolean property and is not counted as an app defect.

Qwen `qwen/qwen-2.5-7b-instruct` passed that same real connection helper under the
unchanged privacy policy. The cloud matrix uses it with a process-only test keyring
and untraced credential bootstrap through the normal backend. No secret enters the
browser input or OS keychain; native keychain persistence is outside this test's scope.
See [OpenRouter's primary model-filter documentation](https://openrouter.ai/docs/api/api-reference/models/get-models).

The initial custom-browser teardown closed the browser before Playwright could
export its network trace. Screenshots, reports, downloaded media and test-step
traces were retained, but those earlier browser network traces are incomplete. The
fixture now explicitly exports `browser-trace.zip` before closing the browser.

Qwen 2.5 7B's ZDR-compatible setup succeeded, but the first real English assessment
returned invalid coaching (all five required coaching keys missing). The app emitted
`coaching_unavailable` and substituted deterministic advice. The suite intentionally
failed rather than accepting that fallback as a successful real-LLM workflow; evidence
is in `frontend/output/live-journeys/20261002-openrouter-zdr`.

## OpenRouter Mistral Small 3.2 review

The cloud rerun uses `mistralai/mistral-small-3.2-24b-instruct`, with the existing
zero-retention policy intact. Its English B1 upload and recorded/tiny retry avoid
invented grammar corrections. Both give specific, achievable 90-second exercises
with explicit transition-word and vocabulary targets. Connector advice remains
generic; this limited sample does not establish calibrated proficiency or spoken
delivery accuracy. The inspected English Review screenshot clearly preserves the
failed duration gate alongside the generated coaching.

English B2 avoids false grammar criticism and grounds its duration observations in
the actual metrics. Its next exercise is 30 seconds against a 90-second task;
this might be a scaffold, but the shorter target is unexplained. English C1 again
alleges missing cohesive/transitional phrases despite “because”, “yet” and “not only
… but also”. The general recommendation for more varied linking can be useful;
the categorical absence claim is unsupported.

Italian B1 upload also avoids fabricated grammar errors and gives a measurable
90-second connector exercise. However, it declares pronunciation correct despite
the LLM receiving text/metrics, not audio: this is an unsupported pronunciation
claim. The tiny retry again turns ASR artifacts into learner errors. It also asks
for a definite article before “intreno”; the supplied speech says grammatical “in
treno”, which does not require that article. A stronger LLM does not cure uncertain
transcription or justify pronunciation judgments from text.

Italian B2 repeats the unsupported pronunciation claim, although its grammar advice
does not invent an error and the 90-second exercise is concrete. Italian C1 avoids
fabricated grammar errors and retains correct evidence quotes, but the alleged
connector issues have empty example strings. Coaching remains generic despite
high confidence. These outputs do not warrant a blanket semantic-quality pass.

The final custom-browser trace export and local credential-store isolation were
verified by focused UI setup reruns for both local providers, retained in
`20261002-local-trace-checks-2`. Both passed and exported real network/snapshot traces.
An earlier cold-start probe exceeded the test's 20-second detection wait while the
app allows 30 seconds; the test now waits 35 seconds, preserving the application's
own timeout and error behavior. No product timeout was changed.

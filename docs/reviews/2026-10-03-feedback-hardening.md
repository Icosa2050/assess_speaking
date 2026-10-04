# Feedback hardening review, 2026-10-03

Plan: `docs/plans/2026-10-03-feedback-hardening.md`.

Three parallel agents reviewed and implemented feedback validation, metrics and UI/API
changes. The logged-in Claude CLI reviewed the plan and then the implementation using
selected source, diffs and authored test fixtures only. No recordings, learner reports,
API credentials or unrelated personal files were sent. Claude was an engineering reviewer,
not the feedback-quality oracle.

## Implemented safeguards

- New live English/Italian metric profiles use Unicode tokenization, phrase boundaries,
  longest nonoverlapping connector matches and phrase-level fillers. Explicit historical
  benchmark profiles and versions remain frozen; benchmark alignment pins its profile.
- New rubric generation validates nonblank comments and quotes against the transcript
  (NFC, case, whitespace and apostrophe normalization; no fuzzy or punctuation removal).
  Grammar/lexical findings need quoted examples; absence-type coherence findings can omit
  them. Invalid output gets one repair attempt, then the entire rubric is unavailable.
  Scores and History categories are not retained from rejected output.
- Coaching separates shorter preparation drills from an explicit configured-duration full
  retry. The authoritative duration check and actual recording/speaking times go to the
  prompt. Optional new fields leave historical reports readable.
- OpenRouter generation and its setup probe use strict output schemas and parameter-aware
  routing. Unsupported capabilities fail visibly without silently weakening the protocol.
- Native/chunked ASR retain optional word probabilities and nonempty-segment diagnostics.
  At least three observed words with a 25% low-confidence ratio (probability below 0.5),
  or a 25% weak observed-segment ratio (avg_logprob below -1 or no_speech_prob above 0.6),
  makes feedback provisional and skips text criticism. Missing confidence is unknown;
  high language-detection probability does not establish transcript correctness. Claimed
  issue examples also cannot overlap observed low-confidence ASR words; incomplete alignment
  fails conservatively. Repeated quote occurrences are all checked. This contextual gate
  retries the full rubric, then labels the result uncertain on exhaustion. Low-confidence
  spans and timings remain available in the saved diagnostics (policy v2).
- Review only displays verified comparisons matching History's provenance contract.
  Explicit retries must match their actual parent and prompt. Legacy reports remain
  readable without unverified deltas. Repeated same-file reports include a session suffix
  and exclusive creation, preventing accidental overwrite.
- Fallback/uncertain feedback is labelled above coaching. The Guide names the existing
  acoustic dimension as an ASR intelligibility proxy. Provider setup errors return shaped
  502 responses with CORS and credential redaction.

## Review findings and decisions

Claude correctly identified generation-only validation, actual-duration context, empty
silence segments, common duration notation, multiword fillers and capability-probe parity
as concerns. These were addressed. A focused second review of the ASR quote guard found
mid-word quote matches, globally applied alignment failures and lost uncertainty across
mixed retry failures. Those were repaired with bounded quote matches, per-quote alignment
coverage and retained uncertainty across failed attempts; focused regressions cover each. The suggested requested/effective scoring-mode mismatch
was disproved: practice metadata comes from the report's effective scores.mode. The report
schema keeps input and progress_delta dictionaries intact; an integrated save/reload test
verifies the marker survives. Both CLI and desktop use the shared runner. Unknown legacy
conditions are intentionally excluded, matching the History contract.

The required codebase-memory skill was read. Its index_status and list_projects tools both
returned `Transport closed` on 2026-10-03. Focused direct-source fallback was used, without
claims of exhaustive graph coverage.

## Verification and retained evidence

Final regression results: **711 backend tests passed, 5 skipped; 136 frontend tests passed**.
Frontend typecheck, production build and standard repository quality checks also passed. Extended quality scans also report unchanged broad-exception sites
outside the new guards; the standard required check passes. No npm dependencies or lockfiles
were changed. The missing prescribed sibling Python environment was already bootstrapped
using the repository's setup_env fallback in the previous task.

Initial evidence is retained in `frontend/output/live-journeys/20261003-feedback-hardening-2`.
English Ollama upload/retry and both OpenRouter bilingual journeys passed structurally.
The Ollama Italian browser attempt failed after concurrent locale/UI edits triggered repeated
Vite hot updates; the backend assessment completed and was saved. Browser trace console
records the updates, and its screenshot shows Review's empty state. The next matrix retained
hashes, reports, recordings, browser traces and screenshots. Further manual review discovered
that a low overall uncertainty ratio can hide a criticized low-confidence span. Its Ollama
Italian retry exercised the new span guard successfully, but its test process had already
loaded the old zero-LLM-time assertion before the guard was introduced. That retained failure
is followed by a fresh, frozen-source Italian upload/recorded-retry matrix across all three
providers (`20261003-feedback-hardening-guarded`). The earlier matrix
(`20261003-feedback-hardening-final`) is retained as mixed-source diagnostic evidence:
Ollama failed that stale assertion, LM Studio 3B failed strict quote validation, and
OpenRouter passed setup plus English/Italian upload and recorded-retry workflows. Its
setup probe used the strict rubric schema. The separate post-guard snapshot is diagnostic,
not a claim that the earlier matrix ran unchanged sources.

The new explicit `--allow-guarded-fallback` acceptance scope verifies functional handling
of rejected outputs, while the default strict quality-contract tests remain strict.
Network failures are not accepted. Green functional tests do not mean the model's feedback
passed validation or human semantic review.

Final frozen-source replay: **6 Playwright tests passed, 6 assessments retained**, across
Ollama Qwen3.5:4b, LM Studio Qwen2.5:3B → 7B, and OpenRouter Mistral Small 3.2:24B.
Each provider completed Italian upload and real MediaRecorder retry with large-v3 → tiny
ASR. All three retries discarded uncertain issue evidence and saved labelled deterministic
fallback with manual-review state. Setup, Review, History, recording seek/playback, range
retrieval and byte-identical downloads passed. Source hashes match the initial manifest.
The completed evidence scan found no actual OpenRouter key in files or decompressed trace
archives; counts are retained in `verification.json`. Decoded recordings last 16.053–17.160
seconds, have nonzero RMS, and agree with report duration within 0.003 seconds. These short
samples intentionally fail the configured 90-second task duration; they do not validate
longer learner performance. Earlier strict failures remain failed in their manifests.

The host manually found a remaining semantic failure in the initial Ollama English upload:
valid first-person plural travel narration was labelled a number-agreement error. A later
Ollama English run also claimed a missing subject in a quote explicitly containing “we”,
called “the beach” unnatural, and treated “I still remember” as an incorrect present tense.
These claims are unsupported. LM Studio and Mistral repeatedly overstated connector absence
in English text that contains “and”; suggested sequencing improvements are reasonable, but
claims of no linking words are not. These outputs passed quote membership. This is a model-quality failure, not a validation success proving accuracy.
The initial Mistral tiny-ASR retry also criticized transcription mistakes despite a low
overall weak-word ratio (4 of 50 words). Replaying its retained recording confirmed a low-
confidence verb in the quoted diagnosis; this motivated and directly exercises the span guard.
The new authored corpus is `tests/fixtures/feedback_quality/bilingual_v1.json`, outside the
CEFR-scoring fixture discovery directory, with clean/seeded English and Italian pairs.
It needs independent bilingual review before use as a semantic benchmark. The frozen-source
LM Studio 3B Italian upload also produced English rubric comments despite Italian feedback
settings, an unsupported “no mispronunciations” claim, and overstated unclear narrative
order. These remain semantic/language-quality failures; its coaching was Italian, and its
7B tiny-ASR retry was safely discarded. Host inspection observations are retained separately in
`20261003-feedback-hardening-guarded/human-review.json`; they are not automatic model grades or independent native-speaker validation. The
inspector’s `human_quality_verdict` remains `unreviewed` pending independent human review.

## Limits and next work

These guards do not establish grammatical truth, transcript accuracy, pronunciation or CEFR
validity. Confident ASR mistakes can still evade the diagnostic gate. There is no transcript
correction/review-resolution workflow yet; provisional results can be checked via playback
and retried with better audio or ASR settings. Genuine longer learner recordings and manual
transcripts are still needed. Compare two or three eligible inexpensive cloud models on the
same authored texts before any larger audio sweep. No OpenAI evaluator is required.

## Follow-up: inexpensive OpenRouter text comparison

On 2026-10-03, the host ran the eight authored bilingual cases twice through each of
Mistral Small 3.2:24B, Qwen3 30B A3B Instruct 2507, and Gemma 3:27B: **48 assessments**,
each generating a rubric followed by coaching. All three passed the production strict-schema
capability probe. The [current public catalog](https://openrouter.ai/api/v1/models) listed
structured-output support; actual account routing was tested without changing privacy settings.

Evidence: `frontend/output/live-journeys/20261003-cheap-text-comparison-2`, including the
reproduction script, source hashes, every request body/response, usage, repair attempts,
actual routed providers, and `host-review.json` with a judgement and reason for each case.
No headers, credentials, real recordings or learner reports are included. A first local probe
in `20261003-cheap-text-comparison` omitted the explicit credential argument required by the
client and made no HTTP requests; it remains diagnostic evidence, not an availability result.

Production prompts, schemas, temperature 0.2 and one validation repair were retained.
A common experiment note explicitly identifies authored text and unavailable audio. All audio
metrics are `unknown (no recording)`; only text word count is supplied. Full retry duration is
90 seconds, and the duration gate is unknown. Completion output is capped at 4096 tokens.
No reference corrections are included in model prompts. Model scores are ignored for quality.
This experiment compares model-plus-router deployments, not isolated model weights: providers
changed between some requests and repairs. There is no OpenAI evaluator or Codex CLI oracle.

| Model | Final rubric/coaching contracts | Clean texts with structured false grammar issues | Seeded cases with sound diagnoses | Repair requests | API-reported cost |
| --- | --- | --- | --- | --- | --- |
| Mistral Small 3.2 | 16/16 each | 0/8 | 5/8 | 15 | $0.008083 |
| Qwen3 30B A3B | 16/16 each | 0/8 | 4/8 | 0 | $0.008582 |
| Gemma 3 27B | 16/16 each | 0/8 | 5/8 | 5 | $0.009263 |

Total API-reported cost, including probes and repair attempts: **$0.025928** across 119
requests. All responses supplied cost metadata; this is observed API usage, not a billing
invoice. Final source hashes match, and an evidence scan found no actual OpenRouter key.

The semantic criterion is deliberately stricter than flagging an error: the explanation must
identify its grammatical mechanism. A correct replacement with a false explanation does not
pass. Category labels alone are not decisive. These are host-agent judgements on four seeded
error types repeated twice, not independent native-speaker validation or a statistically
reliable ranking. Zero clean structured false positives does not mean all comments were sound.

- Qwen correctly explained both English errors twice, missed Italian gender agreement twice,
  and either rejected the lexical verb `sostenere` for the wrong reason or missed the
  subjunctive error. It repeatedly invented rhythm, pauses and clear pronunciation despite
  having no recording, and propagated those claims into coaching.
- Gemma correctly explained five seeded responses, missed the Italian subjunctive twice,
  and reversed the modal correction once: it recommended `saves` after `can`. Its Italian
  gender finding sometimes contradicted the accuracy comment. One exercise was writing-only;
  other exercises used repetition counts without the requested duration. It also invented
  hesitation and speed observations.
- Mistral correctly handled gender and basic English agreement, but one modal explanation
  called `can` the subject. Its two subjunctive explanations wrongly invoked plural agreement
  or a condition instead of the dependency on `vogliamo che`; one offered the correct form
  while comments and coaching still called grammar correct and omitted that correction.
  Connector-absence claims also overlooked existing `because`. Its numerous repairs mostly
  supplied missing issue quotes, showing the grounding guard working but adding latency/cost.

**Decision:** keep the existing default; this pilot establishes no reliable replacement.
Cost is not the immediate constraint. Before another model sweep, separate text criticism
from acoustic assessment, check consistency between issue lists and summaries/coaching, and
expand native-speaker-reviewed minimal pairs around the failed Italian grammar and modal cases.
Treat optional vocabulary enrichment separately from errors. The current validation proves
shape and provenance, not grammatical truth; neither prompt instructions nor a larger model
can be assumed to fix that distinction.

To reproduce from the repository root, copy the retained `compare.py` into a new empty
evidence directory and run it with `.venv/bin/python` and the existing `OPENROUTER_API_KEY`
environment variable. The script writes beside itself; use a new directory to preserve prior
evidence. It uses three concurrent model workers and sequential cases within each model.
No application source or model default changed during this follow-up.

## Follow-up: acoustic claims and coaching consistency

Implemented generation-only English/Italian/German claim checks in
`assessment_runtime/feedback_claims.py`. Known pronunciation, accent, rhythm,
hesitation, pause and speed diagnoses are rejected rather than silently removed. A complete
statement that these cannot be assessed is allowed; a contradictory pronoun follow-up is
rejected. Known learner quotations are exempt, while arbitrary model quotation marks are not.
Common topic/orthography phrases have explicit exemptions. Unknown paraphrases can escape,
and unusual topical or correction wording can still be conservatively rejected: this is a
bounded heuristic, not proof that all acoustic claims are impossible.

Rubric comments cannot deny retained grammar findings. Coaching must reference a whole-word
quoted example from every retained grammar finding in its priorities; several findings can
share one priority. Error-free goals for a future attempt remain permitted. Qualified
accuracy statements must reference a retained example. The coaching client now requires an
explicit rubric keyword argument; the shared CLI/desktop runner and integration callers pass
it. Quote coverage proves mention, not correct advice: a model can still give a false diagnosis
or an incorrect correction. Historical report loading stays permissive. Prompt versions are
`rubric_multilingual_v3` and `coaching_multilingual_v4`, keeping changed outputs out of old
History comparison cohorts.

The standing-approved Claude CLI reviewed source and authored regression cases only. Its first
attempt failed with `The model's tool call could not be parsed (retry also failed)`. The smaller
retry completed and identified pronoun follow-ups, quoted evidence, ordinary topic words,
inflections, limitation variants, denial variants, exception handling and substring priority
coverage. Those informed the fixes. Nonempty issue quotes were already required by rubric
validation; that reported empty-list path is not reachable from an accepted generated rubric.
The graph service still returned `Transport closed`; focused direct-source inspection was used.

Final backend baseline: **750 passed, 5 skipped**. Frontend baseline: **136 passed** and
successful typecheck. Standard quality checks and diff checks pass. No npm dependency change.

The initial 48-case guard replay is retained as
`frontend/output/live-journeys/20261003-claim-guard-comparison`; it accepted rubric/coaching
for 13/16 Mistral, 10/16 Gemma, and 2/16 Qwen cases after repair. Its loaded first policy
remained fixed throughout that Python process; the initial source snapshot is retained.
An Italian `velocità di produzione` paraphrase still escaped and motivated the speed-pattern
refinement. It is diagnostic evidence for the first policy, not validation of final sources.
The intermediate 12-case confirmation and first real UI replay also remain available.
Final verification uses separately named `20261003-claim-guard-final-text` and
`20261003-claim-guard-final-ui` directories; validation failures remain failures in their
raw evidence. Passing guarded workflow checks demonstrate safe fallback handling, not model
quality or successful generated criticism.

Final retained text confirmation: 12 cases across three models. Live contracts accepted 3/4
Mistral, 2/4 Gemma and 0/4 Qwen. Local revalidation after adding the observed Italian
`velocità di emissione` synonym retains 3/4 Mistral, 1/4 Gemma and 0/4 Qwen; that newly rejected
Gemma response remains intact with a separate revalidation verdict. These are output-contract
counts, not accuracy grades. Gemma's remaining accepted modal response still recommended
`saves` after `can`, and Mistral's modal explanation remained misleading despite a correct
coaching correction. One Mistral preparation exercise was writing-only. Quote coverage and
consistency checks do not establish grammatical truth or complete exercise quality.
The final 12-case API-reported cost was $0.007110105, including probes and repairs.

The final real OpenRouter journey passed **2 Playwright tests and 2 assessments**, covering
setup, Italian file upload, real MediaRecorder retry with changed Whisper settings, Review,
History, seeking/playback, range requests and byte-identical downloads. Both reports used
clearly labelled deterministic fallback with manual-review state; rejected rubric scores
were absent. The saved screenshot was inspected. Recordings decoded to 16.053 and 17.160
seconds and agreed with report durations within 0.003 seconds. They intentionally failed
the 90-second duration target. Neither is a semantic model-quality success.

The UI process predates only the final `emissione` synonym and policy-version metadata
addition; its saved reports were checked against the final policy. Its original manifest
hashes are preserved. `final-policy-revalidation.json` and `final-policy-verification.json`
record the separate checks rather than claiming the earlier live calls used later sources.
Future live-run manifests now hash the new claim-policy module. New reports record
`feedback_claim_policy=bounded_text_claims_v1`, included in History's analysis signature;
backend integration tests verify persistence. The completed evidence scan found no actual
OpenRouter key in retained files or decompressed traces. No external evaluator was used.

# Style validation fixes and bounded OpenRouter comparison

Date: 2026-10-03. Branch: `codex/local-practice-readiness`. Changes remain uncommitted; existing user files and other concurrent changes are preserved. Model defaults and dependencies are unchanged.

## Fixes

The four confirmed OCR/Claude findings now have regression coverage:

- Optional explanations reject explicit correction/error claims in English, Italian and German, while allowing scoped reassurance such as “needs no correction”, “isn't an error” and “nicht falsch”. Negative reassurance and subsequent error clauses remain checked.
- Proposed target-language wording is treated as speech content. Acoustic vocabulary inside a replacement or quoted source is permitted; assessor explanations still undergo acoustic-claim checks.
- Coaching checks distinguish incidental praise from a reference to an optional item. Explicit partial replacements of the changed wording are checked; an unchanged shared fragment does not automatically collide with an unrelated grammar correction.
- The OpenRouter connection probe includes the required empty `style_suggestions` array.

The feedback policy is versioned `bounded_text_claims_v2`; saved provenance distinguishes earlier policy results. These checks are bounded heuristics, not proof of grammatical correctness or meaning preservation. Claude's separate observations about the pre-existing acoustic guard and goal checking remain follow-up work.

## Verification

- Full backend: **875 passed, 17 skipped** (`.venv/bin/python -m pytest -q`). Existing Starlette/Anyio deprecation warning remains.
- Frontend: **139 passed in 22 files**, typecheck passed.
- Quality script and `git diff --check` passed.
- Existing fixture-backed English/Italian B1 upload, recording retry, Review, History, playback and download journeys: **2 passed**. Backend/storage/audio are real; ASR and generated feedback are fixtures. They demonstrate workflow and display behavior, not model quality.
- Original review reproducers now accept the valid reassurance/content cases and reject the previously accepted correction bypasses. Retained evidence: `frontend/output/live-journeys/20261003-style-validation-fixed-ui/`.

The real Whisper/OpenRouter workflow is recorded separately below. A workflow pass with guarded fallback does not count as valid model feedback.

## Cheap OpenRouter comparison

Evidence: `frontend/output/live-journeys/20261003-style-validation-fixed-text/`. The retained harness extends the previous experiment and calls production generation code. It runs ten authored bilingual v2 cases once per model: clean/seeded English modal, Italian gender and person agreement, Italian informal volere-register variants, and optional-style cases. Expected labels are not supplied to the model. There is no audio in this comparison and no external quality evaluator.

| Model | Accepted rubric | Accepted coaching | Requests including probe/repair | API-reported cost USD |
| --- | ---: | ---: | ---: | ---: |
| Mistral Small 3.2 24B | 1/10 | 1/10 | 22 | 0.00478096 |
| Qwen3 30B A3B 2507 | 1/10 | 1/10 | 22 | 0.00806485 |
| Gemma3 27B | 1/10 | 0/10 | 23 | 0.00748010 |

All three probes succeeded. Costs include probes and repairs; total **$0.02032591**, with cost reported on all 67 calls. Routes varied: Mistral used DeepInfra/Parasail, Qwen used Nebius/DekaLLM, Gemma used Nebius. This single repeat cannot establish model/provider reliability.

Only **3/30 rubrics** and **2/30 coaching outputs** passed validation. None of the nine seeded-error attempts retained an accepted rubric. Rejection is not evidence of a correctly detected error, and cannot be scored as an accepted missed-error result. The three accepted rubrics had no explicit grammar false positives, an insufficient denominator for a useful accuracy claim. Failures were dominated by acoustic claims, overlapping optional/error/lexical evidence, and contradictory grammar commentary.

Host inspection of the actual retained outputs found semantic problems even in accepted reports:

- Mistral rewrites “È importante che noi partiamo domani” as “Dobbiamo partire domani”: importance becomes obligation.
- Gemma changes “should offer a balanced system” into “could consider offering a flexible system”: both recommendation strength and meaning change.
- Qwen's pronoun omission is a plausible optional alternative, but emphasis depends on context. Its coaching proposes replacing `biglietti` with itinerary/luggage, which are not equivalent, and assumes air travel without evidence. It also frames adding a connector to a grammatical clause as a correction. Its fluency comment contains an interruption claim that the bounded acoustic guard missed.

No cheaper model is ready to promote on this evidence. More API runs alone will not solve meaning preservation and false lexical corrections. The next useful work is targeted meaning-preserving examples/validation and independent bilingual review, followed by repeated comparisons with recorded routes and failure rates.

Only `style_validation.py` changed after the comparison process imported its source. `final-policy-revalidation.json` records the original manifest hashes and rechecks the three accepted rubrics/two coaching outputs locally against the final policy; all still pass. This is explicitly not regenerated live-model evidence. The exact credential scan passed and request headers are not retained. No recordings or learner reports were sent to Claude.

## Real workflow evidence

The first live run retained setup and English passes, but Italian failed its Review navigation assertion after a Vite server disconnect/reload reset in-memory state. The assessment itself completed with guarded fallback. A diagnostic rerun briefly failed because explicit tracing duplicated Playwright's automatic tracing; the harness uses the config to start tracing and saves the custom context before closing. Both failed runs remain retained for diagnosis.

The final null-sink rerun passed; its result and retained evidence are detailed below.

Further inspection of the last rejected seeded responses: all nine quoted the seeded problem, so these are unavailable outputs rather than simple missed detections. However, Mistral explains `can saves` as agreement with the supposed subject `time`, an incorrect rule; Qwen's Italian person-agreement explanation invents `partare` and incorrectly argues the subjunctive is unjustified after `è importante che`. Gemma gives the intended rule in these three final rejected responses. This is diagnostic raw-output evidence, not a retained-output accuracy rate; see `seeded-rejected-output-inspection.json`.

Playback diagnosis: two later runs stalled at exactly 0.032 seconds while the recording was fully buffered (`readyState=4`, 17.0925-second duration, not paused or ended). A focused localhost probe of the retained WAV reproduced this outside React, with and without fake microphone flags. Muting and disabling out-of-process audio did not help. Chromium's `--disable-audio-output` null sink advanced playback to 1.844607 seconds after two seconds. The live harness therefore uses that flag; it continues to check decoding duration, advancing playback time, seeking, range serving and downloaded bytes. It does not test the physical speaker. The config starts tracing, and the custom context saves its trace before closing.

Focused final regression rerun: **149 passed** across style, LLM-client and feedback-claim tests. Localhost backend `/v1/health` returned ready and Vite returned HTTP 200 during the isolated run.

Final real rerun completed: **3 tests passed in 3.8 minutes** (setup plus English and Italian B1 journeys), with **four assessments**. Command:

```sh
.venv/bin/python scripts/run_live_journeys.py --providers openrouter --allow-guarded-fallback --grep 'B1' --output frontend/output/live-journeys/20261003-style-validation-fixed-live-null-sink
```

Evidence: `frontend/output/live-journeys/20261003-style-validation-fixed-live-null-sink/`, including `host-evidence-review.json`, original reports, recordings, screenshots, browser traces and runner log. The launch source hashes match the final source exactly. Credential scanning of JSON evidence passed. The host inspected both languages' Review screenshots and retained reports, plus decoded audio measurements; no independent human listening/linguistic certification is claimed.

The journeys verified saved original reports remain unchanged after retries, `large-v3` to `tiny` settings are reflected in saved inputs/history, incompatible analysis cohorts are not presented as comparable progress, recordings decode to approximately 16–18 seconds, playback/seek succeeds against the null sink, range requests return 64 bytes with HTTP 206, and saved recording bytes match downloaded input bytes. All four reports contain `bounded_text_claims_v2`. All four reject the model's acoustic fluency claims and retain deterministic-only scores with visible manual-review/general-practice warnings. Thus **workflow passes, personalized feedback quality does not**.

The Italian tiny-ASR retry changes the clean source into fragments including `Lo scorso hanno ha fatto`, `intreno`, and `vamo`. No accepted grammar feedback is generated in this run, but this shows why transcription membership/confidence alone cannot certify a learner error. The fallback's generic advice also deserves future qualitative review: English connector detection returns zero despite visible `and` clauses, so “use more connectors” should not be read as an independently established deficiency. The prepared B1 samples are short relative to the 90-second target; these are reproducible functional journeys rather than validated level-calibration recordings.

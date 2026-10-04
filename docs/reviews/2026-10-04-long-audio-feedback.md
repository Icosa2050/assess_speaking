# Long-recording diagnosis and combined feedback review

Date: 2026-10-04. Branch: `codex/local-practice-readiness`.

## Experiment

Compared the two existing M4A regression recordings with cached Whisper `tiny` and `large-v3`, using local Ollama `qwen3.5:4b`, the same Italian task, 60-second target, 180-second request timeout and one validation retry. All recording content stayed local.

| Recording | ASR | Observed low-confidence words | Baseline outcome |
| --- | --- | --- | --- |
| test1 | tiny | 12/92 (13.0%) | Optional rewrite touched uncertain ASR evidence; rejected |
| test1 | large-v3 | 4/93 (4.3%) | Unsupported delivery claim; rejected |
| test2 | tiny | 30/113 (26.6%) | Global ASR uncertainty gate; generation skipped |
| test2 | large-v3 | 1/98 (1.0%) | Identical original/replacement style item; rejected |

These are model confidence observations, not transcription-accuracy scores. No independently transcribed reference was used for these recordings.

## Changes and second run

- Removed acoustic measurements from the text-feedback prompts. Defined the legacy fluency fields as continuity of ideas visible in text. Audio measurements remain in the report.
- Changed Ollama generation to schema-constrained output, retaining bounded output and disabled thinking. Official support: [Ollama structured outputs](https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx).
- Versioned both prompts, the inference profile and the claim policy so changed analysis conditions do not become comparable history.
- Tightened whole-word evidence validation; isolated provider credentials; rejected reasoning-only, refused and truncated answers; repaired malformed-history handling and confirmed guard false positives.

Replayed the exact retained ASR outputs through the revised production assessment pipeline with the same Ollama settings. This isolates generation changes; it is not a second independent ASR measurement. Outcomes remained guarded fallback in all four cases:

- test1/tiny: the same text was labelled both an error and optional style.
- test1/large-v3: a style quotation still overlapped uncertain ASR evidence.
- test2/tiny: the unchanged ASR uncertainty gate skipped generation.
- test2/large-v3: an issue quotation was not complete-word evidence from the transcript.

The old acoustic-claim failure was not observed in this single rerun. The changes do **not** establish reliable semantic feedback from this small model. Larger ASR alone did not resolve generation quality. No thresholds were relaxed, no fabricated evidence was accepted, and no claim of a live quality pass is made.

Local evidence (ignored): `frontend/output/live-journeys/20261004-long-audio-{baseline,after}`. Each run retains source hashes, ASR diagnostics, requests/responses and resulting reports. These learner records are excluded from external review and the commit.

## Claude and OCR review

Claude CLI received selected source only. Confirmed and fixed: provider-key cross-propagation in the low-level client, reasoning fallback, partial-word evidence, style-word containment false positives, Italian topic-word false positives, whole-paragraph goal exemption and malformed history types. Regression tests cover each.

Retained strict rejection of invalid optional style: filtering after scoring could leave a score based on a bad diagnosis. The prompt requests at most three clear grammar findings; coverage of retained findings is unchanged. OpenRouter schema compatibility remains an opt-in live gate; no speculative schema downgrade was introduced. Connection callers continue to pass their resolved credential explicitly.

OCR delegate uses local deterministic file selection/rules and host review, without an OCR model endpoint. Coverage is in the accompanying JSON. CodeRabbit was blocked by automatic approval review because its external destination had separate authorization requirements; Claude/local review continued. No CodeRabbit review is claimed.

## Verification

- Backend: 907 passed, 17 opt-in tests skipped (current local checkout).
- Frontend: 139 passed; TypeScript and production build passed.
- Default Chromium: 10 passed, two optional oMLX tests skipped; isolated backend health/Vite startup succeeded.
- Bilingual fixture journeys: 12 passed, including six English/Italian B1/B2/C1 loops, error/cancellation recovery and WebKit upload/replay in both languages.
- Focused guard/provider/history regressions: passed; repository quality checks and `git diff --check` passed.

The live M4A comparison above is a **quality limitation**, not a passing strict assessment test. Prior sample-ASR and provider evidence remains dated in the migration/testing reports. This work does not implement a full oral-exam session or certify proficiency.

Second Claude pass also identified non-answer typed content blocks, clause/exception scope and negated-correctness style claims; these were tightened with regressions. Runner assessment passes the transcript explicitly. Generic compatible/proxy endpoint configuration remains intentional; source-only helper review did not establish an unsolicited credential transfer through a production caller. The checks remain bounded, with topic/quotation ambiguity and semantic correctness outside their guarantees. The final guard refinements were tested with authored cases; the Ollama replay predates those additional tightenings.

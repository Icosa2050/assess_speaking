# Feedback hardening after bilingual live journeys

The retained 24 assessments exposed unsupported issue quotes, incomplete connector counts,
coaching duration contradictions, and Review comparisons across changed analysis settings.
This plan was discussed with three parallel agents (feedback, metrics, UI) and the logged-in
Claude CLI using source, diffs and authored fixtures only. Claude is an engineering reviewer, not a quality oracle.

## Implementation sequence

1. Improve the live English/Italian connector profiles and Unicode token matching. Preserve
   historical benchmark keys and versions; give changed live measurements new versions.
2. Validate newly generated feedback against the transcript, retry once, then discard an
   invalid rubric entirely. Allow no-error responses. Keep historical report readers permissive.
   Add nonblank coaching checks and a distinct configured-duration full-retry instruction;
   shorter scaffold exercises remain valid. Supply actual recording duration and duration gate.
3. Retain optional ASR confidence diagnostics. Observed weak confidence makes the report
   provisional and skips text criticism; missing diagnostics remain unknown, not reliable.
   Also reject an issue quote overlapping observed low-confidence ASR words, even below
   the whole-transcript threshold. Keep recording playback available for human review.
   Model size alone is not a gate.
4. Match Review progress to History's saved provenance: speaker, language, goal, task,
   duration, scoring mode/version, analysis signature, provider/model and ASR settings.
   An explicit retry must match its parent; old reports without provenance stay readable
   but do not receive verified comparisons.
5. Show fallback and transcript uncertainty beside coaching. Return structured provider
   failures during setup; normalize transport failures in the LLM client.

## Verification

Use authored bilingual regression cases and canned model responses for quote provenance,
empty/no-error responses, retry exhaustion, transcript uncertainty, duration contracts,
connector boundaries and frozen fixtures. Check cross-setting comparisons and legacy reads.
Run backend tests, frontend tests/typecheck and a localhost browser/health smoke, then retain
fresh English/Italian live-provider evidence. Do not require an OpenAI API key.

## Limits and follow-up

Quote membership proves provenance in ASR text, not grammatical correctness or what was
actually spoken. Confidence thresholds are conservative, uncalibrated heuristics; confident
ASR hallucinations remain possible. This does not validate CEFR levels or pronunciation.
There is no transcript-correction workflow in this change: a provisional result can be
reviewed using playback and retried with better audio or ASR settings.

After these checks, compare two or three eligible inexpensive OpenRouter models on the same
authored transcripts before running final audio journeys. Broader quality validation needs
consented, genuine longer learner recordings, manual transcripts and native-speaker review.
No broad model sweep or publication is part of this change.

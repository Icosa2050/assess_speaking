# Blind coaching and eligibility evaluation protocol

Status: engineering protocol ready; specialist agreement, recruitment and ratings pending. This document pre-registers the procedure, not observed accuracy. Freeze and sign the final tolerance sheet with the EN/IT raters before unblinding outcomes.

## D0 authored feedback

Use all 20 cases in `tests/fixtures/feedback_quality/bilingual_v2.json`. Keep v1 separate. `scripts/evaluate_feedback_generation.py --corpus … --provider ollama --model … --output …` defaults to every case. A subset requires repeated `--case` and an explicit `--exclude ID=REASON` for every omitted case. Failed, repaired, malformed and unavailable outputs stay in the denominator. Expected labels remain evaluation metadata and never enter generation requests.

Two EN/IT specialists independently rate grammatical correctness, semantic preservation, grounding in quoted evidence, helpfulness, confidently false corrections and meaning-changing corrections. Adjudicate disagreements without overwriting either original rating. Randomize outputs with a recorded seed; hide provider/model/prompt/configuration and store the unblinding key separately from rating packets. Confirmed defects become permanent regression cases. Schema validity alone cannot pass D0.

The minimum decoded duration is 30 seconds for audio assessment. Three words and legitimate answers under 30 seconds should be refused before ASR/LLM; score this as expected withholding, not a proficiency error. Include separate **31-second** audio conditions: sustained silence, repeated “bla”, meaningful speech quoting “bla”, coherent but shorter-than-task speech, uncertain-language speech, confident wrong-language speech, off-topic speech and genuine fluent responses. Authored/TTS fixtures establish functional behavior only; human labels require independent listening.

## D1 learner-audio pilot

Target 60 clips: EN/IT × B1/B2/C1 × ten, from at least 20 consenting speakers with multiple speakers in every cell. Recruit by placement/self-report, then assign cells after two independent blind audio ratings and adjudication. Preserve actual cell counts and recruitment shortfalls. Exclude decoded clips under 30 seconds with an explicit reason and include them in screening accounting. Folder names and synthetic voices are not proficiency labels.

Use opaque clip IDs and pseudonymous speaker IDs. Keep consent, provider-sharing choices, withdrawal procedure and retention/deletion dates in a dedicated study root outside the repository and personal journal. Store independent reference transcripts and pronunciation/intelligibility observations separately. Do not send recordings, transcripts or learner reports to source-review CLIs.

Freeze source/prompt/model/ASR/scorer identities and the eligibility-contract version. Inference receives a fixed protocol task goal, or `None`, independently of held-out proficiency ratings. `--task-goal` supports this distinction. A regression changes only the held-out label and checks identical runner inputs and model outputs. Do not tune a threshold on the rating set.

Run the default release configuration, three repeats on a preselected speaker-stratified subset and one supported alternate configuration. Keep per-clip failures and withheld outputs, every attempted call, costs and repair attempts. Account for screened, excluded, planned, completed, failed, withheld and rated clips; these are distinct denominators.

Report eligibility outcomes separately from task completion and raw observations. Insufficient, invalid and unverified content must not become a grade, a proficiency comparison or an improvement in History. Missing feedback is not counted as a correct correction. Report grammar false positives, seeded-error misses, unsupported/meaning-changing corrections, WER, repeatability and agreement by language/level. Report raw rater agreement and an ordinal agreement statistic agreed before unblinding; use speaker-cluster bootstrap uncertainty. Audio proficiency and transcript-controlled coaching have separate tables.

Release tolerances require specialist sign-off before the run. The minimum D0 gate is no unresolved confidently false or meaning-inverting correction in the tested configuration. D1 is a pilot with explicit uncertainty and limitations, not population calibration. The B1 floor, provisional CEFR and ASR intelligibility proxy must remain visible in the report. Later tuning and validation use speaker-held-out partitions.

Required artifacts: consent inventory (private), frozen input/configuration manifest, separate unblinding key (private), raw generation outcomes, both independent ratings, adjudication ledger, aggregate report and ranked regression backlog. No learner data is included in the repository protocol evidence.

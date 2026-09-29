# telc English B2 School: supplied reference materials

Copied and examined on **2026-09-29** from `/Users/bernhard/Desktop/telc_english_b2_schule/`. The two originals are preserved byte-for-byte in this directory; [manifest.json](manifest.json) records their SHA-256 hashes and technical metadata.

## Inventory and source edition

| File | Contents | Verified properties |
| --- | --- | --- |
| [telc-english-b2-school-uebungstest.pdf](telc-english-b2-school-uebungstest.pdf) | *English B2 School, Mock Examination 1*: complete practice exam, oral tasks, assessment criteria, examiner instructions, answer sheets, solutions and listening transcripts. | 48 A4 pages; extractable text; unencrypted; 2,122,099 bytes. |
| [telc-english-b2-school.mp3](telc-english-b2-school.mp3) | Companion listening-comprehension track. | 21:14.586; mono MP3; 44.1 kHz; nominal 128 kbit/s; 20,395,473 bytes. Full ffmpeg decode completed without reported errors. |

The PDF imprint states **first edition, 2018**, with ISBN 978-3-940728-92-0, book order number 5114-B00-010101 and CD order number 5114-CD0-010101. PDF creation/modification metadata also dates to June 2018. The March 2023 filesystem timestamp is not the publication edition. This is the **School variant**; do not silently label it generic telc English B2. Current provider specifications were not independently checked during this ingestion.

Inspection used text extraction, rendered-page review of the oral tasks/rubric/administration, full MP3 decode validation and local ASR samples at 00:00–00:40, 10:00–10:20 and 20:20–20:40. The audio opening explicitly identifies English B2 School, the mock examination and Listening Comprehension Part 1. The middle passage matches the boarding-school interview in listening Part 2; the late passage matches the school sports-hall announcement in Part 3. These are spot checks, not a full manual transcription. This material contains no scored learner speaking performances.

Publisher copyright notices are retained. These files are project references; no product redistribution permission is established here, and they have not been registered as application assets. Candidate/examiner instructions within the booklet describe the examination and do not direct repository operations.

## Page map

| Material | PDF viewer page(s), one-based | Printed page(s) |
| --- | --- | --- |
| Imprint and edition | 4 | Unnumbered |
| Overall structure | 7 | 5 |
| Oral format overview | 24 | 22 |
| Unscored social contacts | 25 | 23 |
| Part 1: presentation | 26 | 24 |
| Part 2: discussion | 27 | 25 |
| Part 3: collaborative task | 28 | 26 |
| Oral rubric | 36–37 | 34–35 |
| Scores and pass thresholds | 38–39 | 36–37 |
| Oral administration | 41–42 | 39–40 |
| M10 oral score sheet | 43 | Unnumbered form |
| Answer key | 44 | 42 |
| Listening transcripts | 45–46 | 43–44 |

## Oral structure in the supplied edition

Candidates prepare individually for **20 minutes**. The ordinary paired oral conversation lasts approximately **15 minutes**, followed by approximately five minutes for examiner assessment. Preparation is separate from that examination slot. Two licensed examiners assess each candidate independently and then agree on marks. The booklet also mentions individual/three-candidate arrangements, but does not establish exact timing for a three-candidate app simulation.

| Phase | Expected candidate behaviour | Timing / scoring |
| --- | --- | --- |
| Preparation | Read the task sheets and take personal notes. Dictionaries are permitted; electronic devices and discussion with other candidates are prohibited by the supplied instructions. | 20 minutes; notes may be consulted later but should not be read as a script. |
| Social contacts | Introduce yourself or establish a natural conversation with a known partner. | Unscored warm-up. |
| Part 1 — Presentation | Present a chosen topic, answer the partner's questions, listen to their presentation and ask relevant questions. | About **90 seconds per presentation**; about four minutes for the whole part including interaction. /25. |
| Part 2 — Discussion | Discuss the shared passage, support opinions with reasons/personal examples, respond to the partner and consider problems/solutions. | About five minutes. /25. |
| Part 3 — Task | Plan jointly, contribute proposals, react to alternatives and identify actions, responsibilities and possible problems. | About five minutes. /25. |

The moderator introduces transitions, protects speaking opportunities and intervenes briefly when conversation stalls or one person dominates. The primary exchange is between candidates. Agreement is not a mandatory outcome of joint planning. Approximate timings should remain approximate in the profile; the 90-second presentation is one stage within the larger Part 1 interaction.

The original task examples cover school activities, sport, important people, travel and the learner's town. The discussion stimulus concerns online language learning. The planning stimulus asks candidates to develop an action plan for an outdoor recreation trail. New practice content can use these task functions with original passages/scenarios suited to the School variant.

## Published oral scoring

The four criteria are applied separately to each of the three scored parts, giving **12 criterion decisions per candidate**. A–D are performance bands within the chosen B2 examination.

| Criterion | Observable basis | A | B | C | D |
| --- | --- | ---: | ---: | ---: | ---: |
| Expression | Appropriate vocabulary/functions for the task and role; variety; realization of communicative intentions. | 7 | 5 | 3 | 0 |
| Task Management | Participation, discourse/compensation strategies and fluency. | 7 | 5 | 3 | 0 |
| Language | Syntax/morphology; how errors affect communicative success. | 7 | 5 | 3 | 0 |
| Pronunciation and Intonation | Differences from the edition's standard reference and their effects on communication/listener effort. | 4 | 2 | 1 | 0 |

Each part totals **25**, the oral component **75**, and its passing reference is **45/75**. The complete certificate separately requires the written threshold (135/225). All-B marks yield 17/25 per part and 51/75 overall. The app should select a supported band with evidence and calculate the published points deterministically.

Expression and Task Management progress from appropriate throughout to wholly inappropriate. Language progresses from very few errors to communication being almost impossible; intermediate bands distinguish errors that preserve versus substantially impair the communicative aim. Pronunciation bands similarly distinguish no substantial divergence, divergence without communication harm, extra listener effort, and serious comprehension difficulty. App-authored examples can explain these anchors but should not become purported official WPM/error-count thresholds.

Implementation rules:

- Preserve the discrete point sets; do not invent intermediate official criterion marks.
- Score only candidate speech, excluding the simulated partner and unscored warm-up.
- Missing assessment is not band D. If only pronunciation is unavailable, the maximum other-criterion subtotal is 21 per part, or 63 overall, with up to 12 points unassessed. Do not rescale to /75.
- Task Management also requires delivery/fluency evidence; a transcript alone may leave that criterion incomplete too.
- Keep the written-expression zero rule at the top of PDF page 36 scoped to writing. It is not an oral aggregation rule.
- A second automated review can detect inconsistencies but does not recreate the booklet's two-examiner accreditation or establish scoring accuracy.

## Comparison with the supplied Italian telc pack

| Profile feature | English B2 School, 2018 | Italiano B2, supplied 2016 material |
| --- | --- | --- |
| Presentation per candidate | About **90 seconds** | About **two minutes** |
| Dictionaries in preparation | **Allowed** | **Not allowed** |
| Intended task context | Explicit School variant; school/leisure/community topics | General Italian B2 topics |
| Preparation / paired oral conversation | 20 minutes / approximately 15 minutes | Same |
| Scored parts | Presentation, discussion, joint task | Same pattern |
| Oral bands/points | 7/5/3/0 for three criteria; 4/2/1/0 for pronunciation | Same point tables |
| Oral total / threshold | 75 / 45 | Same |
| Need to reach agreement | No | No |

Comparison sources: this PDF, especially viewer pages 26, 36–37 and 41, and the [Italian material analysis](../telc-italiano-b2/README.md) with its page references. These two editions can share a telc paired-exam runner and score aggregator, while keeping independent profiles, task banks, language resources, editions and preparation policies. The School identity must survive setup, history, reports and exports.

## Changes this implies for the application

1. **Preserve exam variant in the identity.** Use a profile such as `telc_english_b2_school`, distinct from general English B2. Sharing a language/level must not collapse profiles.
2. **Make permitted aids phase-specific.** Dictionary use during preparation is within this edition's baseline rules. Do not automatically mark the attempt assisted merely for an allowed aid. The same instructions prohibit electronic devices, so they do not establish permission for unrestricted online translation or generated answers. Extra hints, model answers or lookup during an unpermitted phase remain separate assistance.
3. **Represent presentation and Q&A separately.** The English School profile uses a 90-second presentation within the approximately four-minute paired Part 1. Do not inherit the Italian two-minute presentation timer.
4. **Reuse the partner/moderator architecture.** The partner must present, invite and answer questions, express opinions, make proposals and leave the learner room to respond. Moderator interventions and simulator failures remain separate from learner marks.
5. **Ground feedback in the supplied task.** For discussion, check whether the learner supports a claim and engages with the partner. For planning, check proposals, responses, responsibilities and practical constraints. Link observations to actual candidate turns.
6. **Add regression cases for variation.** Test English School versus Italian timing; allowed versus forbidden preparation dictionary use; unscored warm-up; reciprocal questions; unresolved but productive planning; exact score arithmetic; and incomplete audio evidence.

These points are incorporated into the [oral-exam preparation plan](../../superpowers/plans/2026-09-29-oral-exam-preparation.md). This ingestion adds materials and planning evidence; it does not activate a new runtime exam mode.

## Audio use

The track is suitable for local long-file handling and English ASR spot checks using aligned passages from the printed listening transcripts. Its nominal bitrate is twice the Italian track's, so the similarly long files also exercise different byte sizes. A proper error-rate test would need a verified, aligned reference covering instructions, repeats and pauses.

It is listening stimulus material, with no candidate oral score attached. Keep it outside learner-speaking calibration fixtures and oral-score regression expectations. No listening-exam feature or automatic fixture registration was added.

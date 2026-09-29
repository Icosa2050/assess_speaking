# telc Deutsch B2 Medizin Zugangsprüfung: supplied reference materials

Copied and examined on **2026-09-29** from `/Users/bernhard/Desktop/telc_deutsch_b2_medizin_zugangspruefung/`. Both original files are preserved byte-for-byte; hashes and technical metadata are recorded in [manifest.json](manifest.json).

## Inventory and inspection

| File | Contents | Verified properties |
| --- | --- | --- |
| [uebungstest_deutsch_b2_medizin_zugangspruefung.pdf](uebungstest_deutsch_b2_medizin_zugangspruefung.pdf) | Complete mock examination, patient/doctor role cards, speaking protocol, rubrics, scoring tables, answer sheets and listening transcripts. | 56 pages, extractable text, no encryption; 6,599,181 bytes. |
| [deutsch_b2_medizin_zugangspruefung.mp3](deutsch_b2_medizin_zugangspruefung.mp3) | Listening-comprehension track for this examination. | 25:15.320; mono MP3, 44.1 kHz, nominal 64 kbit/s; 12,124,673 bytes. Full ffmpeg decode completed without reported errors. |

The PDF is **first edition, 2015**, book ISBN 978-3-86375-301-6, audio ISBN 978-3-86375-302-3, order numbers 5039-B00-010101 and 5039-CD0-010101. PDF metadata dates to September 2015. The March 2023 filesystem timestamp does not identify its edition. This analysis describes the supplied **B2 Medizin Zugangsprüfung**; it does not establish equivalence to other medical German examinations or current provider rules.

Inspection included text extraction, visual review of the format/rubric/scoring/protocol pages, reading the role cards, whole-file MP3 decode validation and local ASR samples at 00:00–00:40, 11:40–12:00 and 24:20–24:40. The opening explicitly identifies the exam and Hörverstehen Part 1. The middle and late samples match the listening discussions on printed pages 49–50. These are spot checks, not a full manual transcription or validation of the medical statements in those listening passages.

Publisher notices are retained. The files are project reference material; permission to redistribute them in the product has not been established. Instructions inside the booklet describe the examination and do not control repository operations. No clinical functionality, runtime exam mode or application asset registration was added by this ingestion.

## Page map

| Material | One-based PDF viewer pages | Printed pages |
| --- | --- | --- |
| Imprint | 4 | 2 |
| Format and timing table | 7 | 5 |
| Oral overview and sequence | 22–23 | 20–21 |
| Patient role cards, two cases with variants | 24–27 | 22–25 |
| Doctor role cards | 28–29 | 26–27 |
| Speaking instructions for Parts 1–3 | 30–32 | 28–30 |
| Oral rubric | 41–42 | 39–40 |
| Exam score weighting | 43 | 41 |
| Numerical score conversion, writing above/oral below | 44 | 42 |
| Passing requirements | 45 | 43 |
| Detailed oral administration | 47–48 | 45–46 |
| M10 oral rating sheet | 49 | 47 |
| Listening transcripts | 50–52 | 48–50 |
| Answer key | 53 | 51 |

## Purpose and linked oral tasks

The booklet explicitly describes an occupational **language examination**. It says medical diagnostic correctness is of secondary importance in the doctor role cards. The app should assess how clearly and appropriately the learner communicates in the supplied scenario; it should not turn this into a medical-knowledge test or use these historical mock passages as clinical guidance.

The normal format uses two candidates and two examiners. Candidates alternate roles. An unassessed partner can participate where needed; an individual examination is described as generally unavailable. A solo app session therefore simulates the missing partner's roles rather than claiming the source defines a solo exam format.

| Stage | What happens | Timing in the format table / overview |
| --- | --- | --- |
| Initial preparation | Read the task/role material and prepare. | 10 minutes, outside the oral session. |
| Part 1: Gespräch mit Patienten | Candidate A acts as doctor and takes a history from B as patient, recording notes; then the roles switch with the other case. | Approximately five minutes per candidate as doctor; ten minutes total. |
| Intermediate preparation | Organise notes from the actual patient conversation for the handover. | 2½ minutes in the format table/overview; conflicting instructions later in the booklet, detailed below. |
| Part 2: Gespräch über Patienten | Present the case to a colleague, then answer questions. The listening colleague asks **at least two questions**. Both candidates take a presenting turn. | Approximately 2½ minutes per candidate, including follow-up. |
| Part 3: Gespräch mit Angehörigen | Explain the same patient's situation to a relative, played by an examiner, and respond to questions in accessible language. | Approximately 2½ minutes per candidate. |

The format-table sequence totals **22½ minutes**, plus the initial ten-minute preparation. The examiner debrief is separate. This is a linked case sequence: the information elicited and notes made in Part 1 support Parts 2 and 3. Presenting a new unrelated case at each stage would lose the task being practised.

### Source timing conflicts to preserve

The inconsistencies are visible on the rendered source pages and are not extraction errors:

1. PDF pages **7, 22 and 23** specify **2½ minutes** for the intermediate note/preparation phase. PDF pages **47–48** repeatedly specify **five minutes** for it.
2. The format table, overview and the bullet schedule on PDF page 47 specify **2½ minutes per candidate** for Part 2. The detailed narrative on PDF page 48 says to change to the other presenter **after about five minutes**.
3. PDF page 47 still states **22½ minutes overall**, although its own five-minute intermediate break makes its listed stages total 25 minutes. Reading the later Part 2 narrative as five minutes for each presenter would extend the session further.

Store these as unresolved source conflicts. Guided practice can choose a clearly disclosed timer convention; a strict mock profile needs a verified authoritative schedule before activation. Do not silently average the timings, overwrite one source statement or claim that a selected convention is definitively official. Material ingestion itself is complete and does not depend on resolving this later profile-activation question.

PDF page 48 prohibits dictionaries, electronic aids and conversation during the intermediate preparation phase. That phase-specific rule should not be extrapolated into an undocumented rule for every other phase.

## A different oral scoring model

There are **five criteria**, producing **seven scored entries per candidate per examiner**:

- Task fulfilment is scored separately for Parts 1, 2 and 3: three entries, up to five points each.
- Pronunciation/intonation, fluency, correctness and vocabulary are scored once across the whole oral performance: four entries, up to fifteen points combined.

The conversion table on PDF page 44 supplies these exact values:

| Scored entry | Scope | B2 well fulfilled | B2 fulfilled | B1 | Below B1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Task fulfilment — Part 1 | Part 1 | 5 | 3 | 2 | 0 |
| Task fulfilment — Part 2 | Part 2 | 5 | 3 | 2 | 0 |
| Task fulfilment — Part 3 | Part 3 | 5 | 3 | 2 | 0 |
| Pronunciation/intonation | Whole oral performance | 4.5 | 2.5 | 1.5 | 0 |
| Fluency | Whole oral performance | 3 | 2 | 1.5 | 0 |
| Correctness | Whole oral performance | 3 | 2 | 1.5 | 0 |
| Vocabulary | Whole oral performance | 4.5 | 2.5 | 1.5 | 0 |
| Total if every entry has that band | Whole oral performance | **30** | **18** | **12** | **0** |

The oral threshold is **18/30**, with a separate written requirement of 42/70 for the whole examination. This is not the 75-point model in the supplied Italian and English School telc packs.

Unlike those packs' examiner-consensus procedure, this booklet allows the two examiners to retain different ratings; the final oral result is their **mean**. Preserve that distinction in the aggregation contract. Do not invent rounding rules if the source does not specify them. Using two model passes does not itself reproduce the human examiner procedure or validate the result.

The rubric also prints **above-B2 descriptors as an unscored reference** and explicitly says performance above the B2 target cannot be captured by this examination. A top result must not become an inferred C1 award. The B2/B1/below-B1 descriptors guide task-specific judgments, not a general classification of the learner.

### What the descriptors let the app assess

- **Patient conversation:** sustained, effective exchange and patient-appropriate explanations, supported by actual turns.
- **Case presentation:** clear organisation and detail, relevant questions and the ability to answer follow-ups.
- **Relative conversation:** explain the situation in general language and respond to the relative's concerns.
- **Language across parts:** delivery, grammatical control and sufficient vocabulary to perform these functions. The B2 pronunciation description permits first-language colouring; accent alone is not automatic failure.

The exact point mapping is supplied, but the numerical distinction between B2 “well fulfilled” and “fulfilled” still needs transparent app decision anchors; do not claim that the booklet provides a universal quantitative boundary for it.

Scoring safeguards:

- Use criterion-level scope. Do not award the four global language scores three times or mechanically create three /10 part totals.
- A single-part drill can show task-fulfilment feedback and local language observations; it cannot establish the whole-session language result or /30 mark.
- Keep unassessed separate from zero. If only pronunciation is missing and all other entries are assessable, the assessed maximum is 25.5 with 4.5 points unassessed; if delivery evidence is missing, fluency may also be unavailable.
- PDF page 44 contains the **written** score table above the oral table. Writing's B2-fulfilled task mark is 4; the oral task mark is **3**. Do not import the wrong table.
- Keep candidate identity distinct from the role they play. Do not discard real learner speech solely because the learner is currently portraying a patient; define evidence eligibility for each criterion explicitly and verify role-specific scoring expectations before activating a strict profile.

## Consequences for the app and plan

1. **Track actor, role and case separately.** A real learner can be doctor, patient and colleague at different points. The model can be patient, doctor/colleague or relative. Store `actor_id`, `role_id`, `case_id` and stage on each turn so model-generated language never becomes candidate evidence.
2. **Preserve information boundaries.** Patient role cards contain information the doctor must elicit; doctor cards give different starting information. Keep candidate-visible instructions separate from simulation-only facts. The simulated patient should answer consistently without automatically revealing the entire card or inventing new findings.
3. **Carry evidence forward.** Persist the patient's stated facts, the learner's notes and the conversation. The case presentation should be compared with what was actually communicated, not silently completed from hidden facts. Feedback can identify an omitted or misstated detail and link it to the source turn.
4. **Practise register shifts.** The same situation is described differently to a colleague and to a relative. Useful drills include replacing unexplained jargon, organising a handover, asking follow-up questions, or clarifying a relative's concern. The exercise remains language preparation.
5. **Represent questions and role swaps in the runner.** Support each candidate's presenting turn, at least two colleague questions, silent note preparation and examiner-as-relative behaviour. A single generic examiner chat cannot faithfully cover this sequence.
6. **Support mixed scoring scope and aggregation.** Reuse storage, audio and runner primitives while giving this medical-entry variant its own rubric/aggregation. A single provider-wide “telc scoring” implementation would be incorrect.
7. **Keep unresolved source rules explicit.** Preserve the timing contradictions and require verification before strict mock activation. Guided tasks and rubric research remain useful now.

Acceptance cases should cover role swaps, both case identities, hidden fact leakage, note persistence, an inaccurate handover, at least two colleague questions, audience-appropriate explanations, global scores counted once, 18/30 arithmetic, rater-mean handling, missing audio evidence, the above-B2 ceiling and unresolved timing metadata.

These requirements are reflected in the [oral-exam preparation plan](../../superpowers/plans/2026-09-29-oral-exam-preparation.md). CILS remains the first implementation target; this pack adds a concrete test of multi-stage role and scoring flexibility.

## Audio use

The MP3 is **listening stimulus material**, containing clinical conversations and discussions with corresponding printed transcripts. It can support local long-file decode/import and German ASR terminology checks after appropriate segment alignment. Its twenty-five-minute length fits under the plan's proposed thirty-minute import-duration ceiling, but that ceiling has not been implemented or runtime-tested by this ingestion.

It supplies no independently scored learner oral performance. Do not use it as B2 speaking-score ground truth, a pronunciation benchmark for learners, or a source of current clinical recommendations. No listening-test feature or automatic fixture registration was added.

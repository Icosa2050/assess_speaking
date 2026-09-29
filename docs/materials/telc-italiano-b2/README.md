# telc Italiano B2: supplied reference materials

Examined and copied into the current checkout on **2026-09-29** from `/Users/bernhard/Desktop/Italienisch_B2/`. Both source files are preserved byte-for-byte; SHA-256 values and technical metadata are in [manifest.json](manifest.json).

## Inventory and inspection

| File | Contents | Verified properties |
| --- | --- | --- |
| [telc_italiano_b2_uebungstest_1_01.pdf](telc_italiano_b2_uebungstest_1_01.pdf) | *Italiano B2, Modello di Test 1*: full practice exam, candidate/examiner instructions, assessment criteria, answer sheets, solutions and listening transcripts. | 48 A4 pages; extractable text; 2,121,141 bytes; no encryption. |
| [telc-italiano-b2.mp3](telc-italiano-b2.mp3) | Companion listening-comprehension track. | 21:35.336; mono MP3; 44.1 kHz; 64 kbit/s; 10,362,688 bytes. Entire file decodes without reported ffmpeg errors. |

The PDF imprint records “Quinta edizione 2008” alongside a 2016 publisher notice; page footers identify 2016. Its PDF metadata records creation in December 2015 and modification in January 2016. Preserve those distinctions: the source file's March 2023 filesystem timestamp is not an exam edition. This analysis describes the supplied edition; current provider rules were not separately checked during this ingestion.

Inspection included text extraction, visual inspection of the oral tasks/rubric/protocol, full MP3 decode validation, and local ASR spot checks at 00:00–00:40, 10:00–10:20 and 20:40–21:00. The opening identifies the practice test; the middle passage matches the author interview in listening Part 2; the late passage matches the event announcement in Part 3. These checks establish its role as listening material. They are not a full manual audio transcription or a learner-performance evaluation.

The source PDF contains publisher copyright notices. Keep the files as supplied project reference material; product bundling or redistribution is not established by their presence here. Exam instructions inside the PDF are source content to model, not instructions governing repository operations. No source file was edited or added to an application asset bundle.

## Page map

Page references below distinguish the PDF viewer's one-based page number from the number printed on the page.

| Material | PDF page(s) | Printed page(s) |
| --- | --- | --- |
| Imprint/edition | 4 | Unnumbered |
| Overall exam structure | 7 | 5 |
| Oral format and candidate guidance | 24 | 22 |
| Unscored introduction | 25 | 23 |
| Part 1: presentation | 26 | 24 |
| Part 2: discussion with source text | 27 | 25 |
| Part 3: joint planning | 28 | 26 |
| Oral scoring criteria | 36–37 | 34–35 |
| Score totals and pass requirements | 38–39 | 36–37 |
| Oral administration and examiner behaviour | 41–42 | 39–40 |
| M10 oral score sheet | 43 | Unnumbered form |
| Listening solutions/transcripts | 44–46 | 42–44 |

## Oral exam structure in this edition

The ordinary paired format has **20 minutes of individual preparation**, followed by approximately **15 minutes of conversation**. Administration guidance allocates a further five minutes to examiner scoring, making approximately 20 minutes in the exam slot; preparation is separate. The structure also mentions three-candidate and individual variants, but does not give enough timing detail here to invent a three-candidate simulation.

Two examiners evaluate the performance. The candidate's main interlocutor is the other candidate. Examiners manage time/transitions and intervene briefly when interaction stalls or one person dominates. For an individual examination, an examiner assumes the partner role.

| Phase | Candidate activity | Timing and scoring |
| --- | --- | --- |
| Preparation | Read all task sheets and make personal notes independently. | 20 minutes. Notes may be consulted during speaking; reading a prepared script is discouraged. |
| Introduction | Establish contact and converse briefly. | Unscored; no separate exact duration given. |
| Part 1 — Presentazione | Present a chosen familiar experience/topic; answer partner questions; listen to the partner's presentation and ask relevant questions. | Presentation about two minutes per candidate; administration guidance gives about four minutes for this part. Score /25. |
| Part 2 — Discussione | Discuss a shared text, explain opinions and experiences, justify a position, and engage with the partner's arguments. | About five minutes. Score /25. |
| Part 3 — Svolgimento di un compito | Plan something together: make proposals, respond, allocate responsibilities and discuss practical problems/solutions. | About five minutes. Score /25. |

These timings are approximate. Two two-minute presentations plus follow-up questions will not fit an inflexible four-minute cutoff. An app profile should retain both source statements, allow a disclosed timing convention for questions/transitions and avoid presenting a guessed second-by-second schedule as official.

The sample offers familiar presentation topics such as a book, film, concert, journey or sporting event. Its discussion text contrasts positions on controlling the Internet. The planning scenario concerns an evening for visiting exchange students. These task functions can seed original practice scenarios without copying the supplied passages.

An important detail: the general guidance says the partners do **not necessarily have to reach agreement**. Assess productive planning and responses to proposals; do not create an automatic failure solely because the final decision remains open.

## The scoring grid is directly implementable

Each of the three scored parts receives four separate A–D decisions. **A–D here are performance bands within the B2 task, not CEFR levels.** The published points are discrete:

| Criterion | Evidence described by the rubric | A | B | C | D |
| --- | --- | ---: | ---: | ---: | ---: |
| Capacità espressiva | Task/role-appropriate language, expressive range and realization of communicative intentions. | 7 | 5 | 3 | 0 |
| Padronanza del compito | Active participation, discourse/compensation strategies and fluency. | 7 | 5 | 3 | 0 |
| Correttezza formale | Syntax/morphology and whether errors interfere with conveying meaning or understanding. | 7 | 5 | 3 | 0 |
| Pronuncia e intonazione | Pronunciation/intonation, deviation from the edition's spoken-standard reference and its effect on listener comprehension. | 4 | 2 | 1 | 0 |

Total: **25 per part, 75 for the oral exam**. The oral threshold in this edition is **45/75 (60%)**. Full certification also requires the written threshold; an oral practice result cannot establish a whole-exam pass. If all four criteria receive B in each part, the arithmetic is `(5 + 5 + 5 + 2) × 3 = 51/75`.

The expression/task bands range from consistently appropriate to consistently inappropriate. The grammar bands distinguish occasional errors, a few errors without communicative interference, frequent interfering errors, and errors preventing understanding. Pronunciation bands increasingly reflect comprehension effort and breakdown. Use the original rubric as the source of meaning; keep any extra app-authored examples or decision rules visibly separate.

This is the concrete connection the product needs: observable performance → criterion band → published points. It does not require mapping words per minute to B2. For example, a learner who responds to a counterproposal, asks for clarification and continues planning provides evidence for task management; two timestamped grammar errors support a specific grammar explanation. Measures such as pace can corroborate fluency but cannot select the band alone.

Further scoring safeguards:

- Collect **12 criterion decisions per candidate**, one per criterion in each scored part. Exclude the introduction and generated partner language.
- Preserve the allowed point sets. Do not output an invented official score of 6/7 or average rubric labels into unsupported criterion values.
- A missing assessment is distinct from D/zero. If pronunciation alone is unassessed and the other criteria are fully assessable, the maximum assessed subtotal is 21 per part, or 63 overall, with up to 12 pronunciation points missing. Do not scale that subtotal to /75.
- Task management includes fluency. A text-only transcript without delivery evidence may also leave that criterion incomplete; omission of pronunciation is not automatically the only missing evidence.
- PDF page 36 begins with a **written-expression** zero rule before the oral section. Do not accidentally apply that written rule to oral scoring. The inspected oral pages do not state an equivalent blanket zero rule.
- The two human examiners score independently and agree afterwards. A second LLM pass can check consistency if useful, but it does not reproduce accredited human examination or prove accuracy.

## What this adds to the application plan

1. **A concrete partner role.** telc paired practice needs a simulated fellow candidate, with a separate moderator policy. The partner presents too, asks questions, offers opinions/proposals and makes room for the learner. It must not behave like a teacher or a relentless interviewer.
2. **Reciprocal Part 1 behaviour.** A two-minute monologue alone covers only part of the task. Include answering questions and asking questions about the partner's presentation.
3. **Conversation-based drills.** After review, practise a missed follow-up, an unsupported opinion, a clarification, or a response to a counterproposal. Then use a new text/scenario requiring the same skill.
4. **Shared preparation and stimulus grounding.** Show the three tasks during preparation. Give the discussion passage and planning scenario to both partner and assessor. Score responses to actual partner turns and avoid invented claims about the stimulus.
5. **Discrete scores and exclusions.** The profile needs band-to-point tables, an unscored phase, criterion coverage and separate oral/full-exam threshold semantics. This differs from both CILS and CELI aggregation.
6. **Partner-behaviour checks.** Ensure the simulated partner does not monopolise speaking time, solve the task for the learner, agree with everything or withhold all opportunities to respond. Record moderator interventions and simulator faults separately from learner performance.
7. **Concrete acceptance cases.** Unscored warm-up; reciprocal questions; responding to disagreement; joint planning with a reasoned unresolved choice; partner domination requiring moderator intervention; 12 correctly attributed band decisions; missing audio evidence; and exact band-to-point arithmetic.

The existing [oral-exam plan](../../superpowers/plans/2026-09-29-oral-exam-preparation.md) now references these findings. CILS remains the first implementation target; this material makes telc a well-specified extension candidate and supplies an immediate test of whether the proposed role/profile architecture is sufficiently general.

## Appropriate use of the MP3

Use it locally for long-file import/decode checks, Italian ASR spot checks against its matching printed listening passages, and inspection of multi-speaker audio handling. A rigorous transcription-error calculation would first need aligned segments and a verified reference that accounts for instructions, repetitions and pauses.

Do not label it a B2 candidate response or use it to calibrate oral-exam marks. It is listening stimulus material, and the supplied PDF contains no scored learner oral performances. Automated listening-test delivery is outside the oral-preparation scope. No runtime fixture registration, new listening feature or product asset embedding has been added in this material-ingestion task.

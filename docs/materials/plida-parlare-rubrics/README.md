# PLIDA oral assessment rubrics — B1, B2 and C1

Added and examined on 2026-09-29. These are unchanged copies of the three user-supplied PDFs from the Desktop. Each contains one landscape page; all five criterion columns, score bands and footnotes were checked against rendered pages. [manifest.json](manifest.json) records provenance and SHA-256 hashes.

## Inventory

| Source | Contents | Dating evidence |
| --- | --- | --- |
| [B1 rubric](PLIDA-B1-Criteri-valutazione-prove-Parlare.pdf) | Five criteria; five paired score bands; grammar reference and accent footnote | PDF metadata: 25 March 2021 |
| [B2 rubric](PLIDA-B2-Criteri-valutazione-prove-Parlare.pdf) | Five criteria; five paired score bands; interaction scope and grammar reference | PDF metadata: 29 March 2021 |
| [C1 rubric](PLIDA-C1-Criteri-valutazione-prove-Parlare.pdf) | Five criteria; five paired score bands; interaction scope | PDF metadata: 29 March 2021 |

Metadata dates do not establish publication editions or current exam rules. All findings below refer to page 1 of the respective supplied PDF. Copyright notices identify Società Dante Alighieri; these copies are project reference material. Product redistribution permission has not been established.

## What these sources establish

All three assess **Efficacia comunicativa**, **Interazione**, **Lessico**, **Grammatica** and **Pronuncia**. Each criterion has shared descriptors for **9–10, 7–8, 5–6, 3–4 and 1–2**. The sheets do not explain how to choose the individual point within each pair.

B2 and C1 explicitly restrict the interaction column to the interaction test. The B1 header has no equivalent scope note; its administration manual is needed to settle exact scoring scope. A monologue supplies no evidence for reciprocal interaction.

The B1 pronunciation footnote explicitly expects a strong foreign accent in every score band. Feedback must distinguish accent from errors that affect clarity or increase listener effort.

## How the demands change

This table paraphrases the rubrics, especially their stronger performance bands. It is a comparison of expected behaviours within each selected exam level, not a conversion between scores across levels.

| Criterion | B1 emphasis | B2 emphasis | C1 emphasis |
| --- | --- | --- | --- |
| Communicative effectiveness | Complete the task, organise key points clearly and use suitable connectors | Develop precise arguments with examples/details, structure the discourse and speak with manageable listener effort | Cover complex points precisely, qualify statements and integrate varied supporting details |
| Interaction | Maintain conversation, ask for clarification, confirm understanding and observe courtesy | Connect to the partner's contributions, add arguments, take/hold/yield the floor | Adapt flexibly, react to changes, engage the partner and control register |
| Vocabulary | Use topical vocabulary and compensate for gaps | Use specific terms, collocations and effective circumlocution | Express shades of meaning with precise vocabulary and reformulate without disrupting flow |
| Grammar | Control familiar structures; stronger performances show range with isolated errors in harder structures | Combine range with control; stronger performances make occasional, often corrected slips | Sustain high control over varied structures, including complex passages |
| Pronunciation | Remain understandable despite a potentially strong accent | Maintain clarity and use rhythm/intonation to emphasize important points | Maintain phonological control and use intonation to convey nuanced meaning |

The middle bands are useful coaching anchors too. For example, B2 5–6 interaction can contain relevant contributions that do not consistently connect to the partner's turn; B2 7–8 connects appropriately most of the time and proposes arguments. C1 5–6 can adapt and manage turns effectively while still failing to develop the discussion consistently. These are concrete differences an interactive exercise can elicit and review.

## App implementation implications

The following are proposed app behaviours derived from the source descriptions; they are not additional official PLIDA scoring rules.

1. **Select the exam and level before assessment.** Store level-specific criterion descriptors and their source page. A B1 score of 8 and a C1 score of 8 are not interchangeable measurements.
2. **Return a criterion band with evidence.** For each judgement, show candidate excerpts/audio timestamps, the relevant behaviour, missing evidence and one next exercise. Preserve paired bands until a sourced rule or explicitly app-authored scoring policy resolves individual points.
3. **Make task completion inspectable.** Tie each prompt requirement to the learner's response and show which points were developed, merely mentioned or omitted. Assess examples for relevance and support rather than rewarding their count alone.
4. **Capture genuine interaction.** Link candidate responses to actual partner turns; examine relevance, contribution, clarification, turn management and register. Use `not_applicable` for interaction in monologue practice and `insufficient_evidence` when an interactive attempt is incomplete. Keep generated partner speech out of candidate scoring.
5. **Use audio for pronunciation.** Attach audible examples of intelligibility problems, stress or intonation; transcription confidence cannot establish pronunciation quality. Avoid an accent-removal objective, especially given the explicit B1 footnote.
6. **Turn the B2/C1 differences into drills.** B2: give a claim, support it, address a partner objection and hand over the turn. C1: qualify a claim, reformulate a nuance, adapt to a changed premise and adjust register for a different interlocutor. These are original coaching exercises, not claimed official task formats.
7. **Track repeat-attempt improvement within the same profile.** Report better task coverage, more relevant responses, successful paraphrases and fewer evidenced communication problems. Counts and timings can describe the attempt; these PDFs provide no words-per-minute, pause-duration or error-rate cutoffs for awarding a CEFR level.

## Remaining specification gaps

These rubric sheets provide no task sequence, preparation/exam timing, prompt-choice rules, final aggregation, criterion weights, score conversion, pass threshold or zero-score rule. They also do not settle the per-task versus whole-session scope of all other criteria. Do not infer a /50 total, a /30 conversion or a passing boundary from the layout or colours.

The B1 sheet references the oral commissions manual, pp. 18 onward, for recurring structures; B2 references its manual, pp. 19 onward. Those manuals were not supplied. Exact expected structure inventories and complete administration/scoring rules remain source requirements before enabling a strict PLIDA mock profile. Practice and criterion-specific feedback can be developed from these descriptors while the missing rules are sourced.

See the [oral-exam preparation plan](../../superpowers/plans/2026-09-29-oral-exam-preparation.md). CILS remains the initial implementation priority.

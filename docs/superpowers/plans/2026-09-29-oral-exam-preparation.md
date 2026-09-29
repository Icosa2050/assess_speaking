# Speaking training ground: product and implementation plan

**Date:** 2026-09-29

**Status:** First practice-loop slice implemented locally on 2026-09-29; the remaining milestones below are planned. Revised around the user's training-first direction on 2026-09-29.

### Implementation checkpoint — 2026-09-29

Implemented:

- New saved reports retain the exact prompt, chosen goal, timing, analysis model settings and explicit retry parent. Existing CSV history remains compatible.
- Review → Try again links to the actual saved session. History → Practise this prompt again restores the original English/Italian prompt, goal and timing.
- Dated history plots and before/after comparisons separate learner, language, goal, task family, timing and recorded analysis settings. Explicit retries compare to their parent; older reports with missing context remain readable outside the chart.
- Whole-recording WPM, duration and detected pause time accompany the existing overall score. Missing measurements remain unavailable rather than becoming zero.
- Before/after recording playback uses a session-based endpoint restricted to app-managed recordings/uploads, with byte-range support for seeking. Missing recordings leave feedback intact.
- The former priority-string disappearance claim and mixed-condition trend summaries are removed from the active History screen.

Verification has been extended to the full backend/frontend suites, ten bilingual browser journeys using real API/jobs/storage with fixture inference, and real Whisper + Ollama assessments for all six English/Italian B1/B2/C1 combinations. The live suite also covers runtime setup and B1 retries. See [the current verification report](../../reviews/2026-09-29-bilingual-journeys.md) for exact results and limitations. Follow-up fixes address history language-filter initialization, a fixed 0–5 score chart axis, bounded Ollama JSON requests, and spoken coaching that excludes assessor confidence.

Remaining: the new four-dimension practice rating, full prompt-aware coaching, stable skill/focus tracking, assistance declarations, transcript comparison, revised exercise banks, weekly queue, resilient uploads and full exam/conversation modes. The chart deliberately labels the current score as existing overall performance; it is not the planned new rating. Claude CLI and OCR delegate reviews are complete; five confirmed findings were fixed. See `docs/reviews/2026-09-29-practice-review.md` for the findings, coverage and verification. Standing Claude review approval is recorded in AGENTS.md.


Local usability follow-up: complete macOS/browser launcher, bounded multipart uploads with dynamic disk limits and recovery, one assessment at a time, LM Studio live verification, and WebKit upload/replay journeys are implemented. See [local practice readiness](../../reviews/2026-09-29-local-practice-readiness.md). Long-upload resume after app restart and full exam/conversation modes remain planned.

**Canonical plan:** This replaces the earlier CEFR/upload and CILS-only plans. It also supersedes the product direction in `docs/MULTILINGUAL_CEFR_ASSESSMENT_PLAN.md`.

## 1. Product purpose and boundaries

The app is a **speaking training ground used repeatedly over weeks and months**. A learner chooses B1, B2 or C1 as a goal, practises oral tasks, gets a few useful observations, retries, and sees how their performance changes. Exam materials supply meaningful tasks and expectations. Human examiners make official proficiency decisions.

The central question is: **“What am I getting better at, and what should I practise next?”** B1/B2/C1 determine task challenge and coaching expectations. The learner can change their goal without passing an app placement test. An exam preference adds relevant task formats; it is optional for starting practice.

The product delivers five things, in this priority order:

1. A short, easy-to-start practice session with one achievable focus.
2. A saved attempt with measurable performance, useful AI coaching and a simple overall practice rating.
3. One-click repetition and before/after playback so the learner can hear improvement.
4. A personal progress view across attempts, recurring exercises and new topics.
5. Responsive conversation practice and optional exam rehearsals as the product grows.

**Initial release: English and Italian**, both with B1/B2/C1 goals, saved attempts, retry comparison and progress history. **German is the next language**, after the shared training loop works in both launch languages. Retain the German materials for that next phase. Give English and Italian equal access to every core training feature.

Use original Italian exercises informed by the supplied PLIDA descriptors and CILS tasks. Build original English exercises with language-appropriate coaching and examples; the supplied telc English B2 School material informs that specific optional exam variant. General English B1/C1 training does not need to wait for a named exam pack. Keep School task content explicitly labelled rather than applying it to every English learner. CILS B2/C1 are the first Italian exam formats; English telc B2 School practice is an English extension using the same runner. Full exam administration and score conversion are dependencies of optional exam modes; ordinary drills and progress tracking ship independently.

### Language rollout

| Stage | Languages and scope | Delivery requirement |
| --- | --- | --- |
| Now | English and Italian; B1/B2/C1 training goals | Original exercises, suitable ASR configuration, language-specific fillers/tokenisation, coaching examples, retry and progress views in both languages |
| Next | German; same training features | Add its language resources and check the shared loop; medical-entry role-play is an optional specialist pack |

Keep a separate active goal, practice queue and history filter per language. Language switching restores that language's last goal and focus. Activity totals can show all practice, with language labels; performance charts always show one language and comparable task conditions. Reuse the application and metric definitions across languages, with language-specific extraction and coaching resources.

Prioritise practice usefulness, continuity and motivation. Keep official evaluation with human examiners. Use a short explanation of the AI practice rating in the guide; the normal review screen leads with progress and the next action. Research into perfect level classification, expert panels and population benchmarks is outside the delivery path. Reading a stimulus and listening/responding to a partner remain part of oral practice.

## 2. Exam materials inform the exercises

### Verified starting formats

| Profile | Oral format and timing | What the app must support |
| --- | --- | --- |
| CILS DUE-B2 | Approximately 10-minute administration envelope. Published sample: 2–3-minute conversation, then approximately 1.5-minute monologue on a topic or image. | Personal-topic conversation; uninterrupted production; prompt/image choice; separate part results. |
| CILS TRE-C1 | Approximately 15-minute administration envelope. Published sample: role-play, 2–3 minutes preparation and 3–4 minutes conversation; monologue, 3 minutes preparation and 2–3 minutes speaking. | Practical objectives, negotiation and register; separate preparation phases; sustained opinion with reasoning. |
| CELI 3/B2 | Approximately 15-minute interview, with material supplied approximately 15 minutes beforehand: photo, text to summarise/discuss, and situational role-play. | Shared preparation before multiple tasks; source-text coverage; image-grounded conversation; role-play. |
| CELI 4/C1 | Approximately 18–20-minute interview, with material supplied approximately 20 minutes beforehand: two photos, a text and a graph/table. | Photo comparison; summary and discussion; accurate description of chart data; a longer session. |
| telc Italiano B2, supplied 2016 model | 20 minutes individual preparation; approximately 15 minutes paired conversation: unscored introduction, presentation with reciprocal questions, discussion, joint planning. | Separate simulated candidate and moderator roles; shared preparation; reciprocal questions; discrete A–D scoring per criterion/part. |
| telc English B2 School, supplied 2018 model | 20 minutes preparation; approximately 15 minutes paired conversation; 90-second presentations with reciprocal questions, discussion and joint planning. Dictionaries permitted during preparation in this edition. | Preserve School variant; separate presentation/Q&A timing; phase-specific permitted aids; shared telc paired-exam runner and discrete scoring. |
| telc Deutsch B2 Medizin Zugangsprüfung, supplied 2015 model | Format table: 10 minutes preparation and 22½-minute paired oral sequence with doctor/patient role swaps, case presentation and relative conversation. Later instructions conflict on intermediate preparation and Part 2 timing. | Linked cases and notes; actor/role separation; register shifts; criterion-level mixture of part/session scoring; unresolved timing gate before strict mock activation. |

CILS sources: [administration timing](https://cils.unistrasi.it/public/articoli/28/Files/Istruzioni%20somministrazione.pdf), [B2 oral sample](https://cils.unistrasi.it/public/articoli/198/Produzione%20orale%20Due%20B2.pdf), [C1 oral sample](https://cils.unistrasi.it/public/articoli/199/Produzione%20orale%20Tre%20C1.pdf). CELI sources: [CELI 3 specification](https://www.unistrapg.it/sites/default/files/docs/certificazioni/celi-3-descrizione-prova.pdf), [CELI 4 specification](https://www.unistrapg.it/sites/default/files/docs/certificazioni/celi-4-descrizione-prova.pdf).

The CILS task examples are the June 2017 materials currently linked by the [official preparation page](https://cils.unistrasi.it/1/89/188/Esempi-prove-di-esami.htm). Store that edition explicitly. Approximate administration envelopes are not continuous candidate speech. Preserve published preparation and task ranges; choose and disclose a simulation duration within each range. Do not force their sum to equal a rounded overall envelope or claim one universal 15-minute format.

The B2 sample specifies no separate preparation period; extra planning time is a guided-practice option. It permits a limited examiner intervention if a monologue stalls; C1 Part 2 explicitly requires no intervention. Preserve that difference. Use a fixed, neutral B2 continuation prompt when appropriate, recording that it occurred; never supply content. At the selected simulation limit, close the part. Record early finishes without filling the remaining time with coaching. The B2 conversation sheet lists four scenarios but refers to choosing between two: model an examiner-selected pair and candidate choice as an explicit simulation convention pending clarification of that edition. Version prompt-selection rules rather than guessing they are identical for every part.

The user-supplied [telc Italiano B2 materials and examination notes](../../materials/telc-italiano-b2/README.md) now provide a concrete partner-role case. Their published grid maps A/B/C/D to 7/5/3/0 for expression, task management and grammar, and 4/2/1/0 for pronunciation, separately for each of three scored parts: 25 per part, 75 overall, oral threshold 45. These are facts about the supplied edition, not a fresh verification of current telc rules. Its MP3 is listening stimulus material, not a scored learner oral performance. Keep it as reference/import material, outside the candidate-scoring examples.

The telc pack needs the learner to hear a partner presentation and ask questions, respond to arguments and plan jointly; an examiner interview alone cannot cover it. Model the moderator separately from the partner, leave the introduction unscored, and do not require eventual agreement as an automatic success condition. This is an extension candidate alongside the planned CELI work; CILS remains first. [Cambridge C1 Advanced](https://www.cambridgeenglish.org/exams-and-tests/qualifications/advanced/format/) also uses four parts and normally two candidates, including collaborative work. Advertise these partner-based profiles only after their simulation and rubrics are implemented.

The supplied [telc English B2 School pack and comparison](../../materials/telc-english-b2-school/README.md) confirm the same score tables and 45/75 oral threshold, but different presentation timing and preparation-aid rules: 90 seconds and dictionaries permitted, versus approximately two minutes and dictionaries prohibited in the Italian material. Keep `exam_variant`, edition, presentation substage timing and phase-specific `permitted_aids` in each profile. An aid allowed by that profile is ordinary exam practice; extra coaching is separately marked as assistance. The English School tasks and language resources require their own bank, even when the runner/aggregation is shared. These observations describe the supplied editions, whose current rules still need checking before profile activation.

The supplied [German B2 medical-entry pack](../../materials/telc-deutsch-b2-medizin-zugangspruefung/README.md) requires a different scoring implementation: three part-specific task-fulfilment scores plus four language scores over the whole session, total 30 with an oral threshold of 18. Its two human examiner totals are averaged, whereas the other supplied telc packs describe consensus. Store scope on each criterion rather than imposing one scope on an entire profile. Above-B2 descriptors in this pack are explicitly unscored; do not turn a top mark into C1. Its patient-to-colleague-to-relative sequence depends on the same case and candidate notes, with role swaps and at least two colleague questions. Keep `actor_id`, `role_id` and `case_id` separate, and preserve candidate-visible versus hidden role information. The material specifies language assessment; clinical correctness is not a new scoring dimension. Documented timing conflicts remain in its manifest and must be resolved before activating a strict mock; guided exercises can use an explicit simulation convention.

### CILS scoring reference for optional exam rehearsals

| Criterion | B2 part 1 /10 | B2 part 2 /10 | C1 part 1 /10 | C1 part 2 /10 |
| --- | ---: | ---: | ---: | ---: |
| Communicative effectiveness | 4 | 4 | 3 | 2 |
| Morphosyntactic correctness | 3 | 3 | 3 | 3 |
| Lexical adequacy/richness | 2 | 2 | 2 | 3 |
| Pronunciation/intonation | 1 | 1 | 2 | 2 |
| Total | 10 | 10 | 10 | 10 |

Sources: [B2 criteria](https://cils.unistrasi.it/public/articoli/198/criteri%20valutazione%20DUE-B2%20nuove.pdf), [C1 criteria](https://cils.unistrasi.it/public/articoli/199/criteri%20di%20valutazione%20Tre-C1.pdf). The [CILS guidelines](https://cils.unistrasi.it/public/articoli/73/linee_guida_cils.pdf) specify 11/20 as the oral-skill pass threshold; the full certificate requires passing all skills.

These published maxima and criteria do not supply a complete machine scoring algorithm. Store **official criteria/weights** separately from **app-authored scoring anchors and model judgments**. Where the provider publishes point-by-point descriptors, use those as the primary anchors and identify any app interpretation separately.

CELI illustrates a different scoring contract: the official [CELI 4 oral scale](https://www.unistrapg.it/sites/default/files/docs/certificazioni/competenze-punteggi-orale-CELI-4.pdf) supplies 1–5-point descriptors for lexical, sociolinguistic, grammatical and phonetic competence. For example, its sociolinguistic descriptors distinguish a candidate who can lead much of the conversation from one who participates appropriately but cannot lead it. This gives the app concrete behaviours to elicit and cite. The [CELI 3](https://www.unistrapg.it/sites/default/files/docs/certificazioni/celi-3-valutazione.pdf) and [CELI 4](https://www.unistrapg.it/sites/default/files/docs/certificazioni/celi-4-valutazione.pdf) scoring documents scale the oral raw total /20 by three to /60, with 33/60 as the oral threshold. Preserve raw and displayed marks and whole-oral aggregation; do not manufacture independent official marks for each CELI stimulus. The CELI 3 detailed descriptor file is linked from the [official index](https://cvcl.unistrapg.it/pagine/esami-celi-generici), but timed out during this research; retrieving and checking it remains a CELI 3 activation task.

## 3. Learner experience

### Entry and practice modes

The first setup becomes **language → goal B1/B2/C1 → start a short exercise**. Exam preference and exam date are optional. On return, open a home screen with **Continue practice**, the current focus, a small recent-progress summary and a choice of a five- or ten-minute session. Keep previous selections. Exam selection and part/mode controls belong in the exam-practice area.

| Mode | Experience | Result |
| --- | --- | --- |
| Focus exercise | One small objective, a short response, quick feedback and immediate retry. | One focus comparison and saved measurements. Default daily mode. |
| Guided part practice | Explain the task, show a response structure, allow hints, replay and restart. | Coaching by criterion, with assistance recorded. |
| Timed part practice | Exam-style instructions, preparation and time limit; no coaching while speaking. | Measurements, coaching and the overall AI practice rating; optional exam mark later. |
| Full oral mock | All required parts in sequence; examiner behaviour and timing follow the profile. Feedback arrives after the session. | Part-by-part review and a combined result if assessment coverage permits it. |

Begin with a short microphone/playback check and verify that the required ASR, examiner and voice services are available. Download/load task media before starting the clock. Explain whether audio/text will be processed locally or by the configured provider using existing provider settings.

The exam screen shows only the current instructions/stimulus, preparation or task clock, recording status and appropriate controls. Examiner dialogue is spoken in the exam language. Prompt replay, transcript display, hints or extended time count as assistance and remain visible in history. Accessibility accommodations should be explicit session settings; do not silently blend these attempts into strict mock trends.

### The daily training loop

1. **Choose one focus:** for example, support an opinion with a relevant example, answer an objection, or finish an explanation within two minutes. Suggest a focus from recent attempts and allow the learner to change it.
2. **Attempt:** record 60–180 seconds or complete a short exchange. Save the attempt before feedback processing.
3. **Review:** show one success, one improvement priority, three useful measurements and an overall AI practice rating. Expand for the full transcript, language corrections and exam criteria.
4. **Practise:** offer a brief model phrase or strategy and a focused 30–60-second rehearsal. Label demonstrations clearly and keep them out of the learner's measurements.
5. **Retry:** retain the same prompt and focus. Show before/after audio side by side and explain the observed change, including when it stayed similar or became harder.
6. **Transfer and revisit:** offer a fresh topic using the same skill, then bring the skill back in a later session. Completing the loop is progress even when the rating stays flat.

Keep **Try again** and **Try a new topic** available directly from the review, without repeating setup. A repeated prompt is useful rehearsal and earns practice credit. History labels it so the learner can also see how the skill transfers to new topics.

Example for a CILS C1 complaint role-play: “You explained the problem clearly, but when the examiner rejected a refund you repeated the complaint without proposing another remedy.” Link the exchange, model a concise alternative proposal, and offer a new negotiation scenario. For a monologue, help the learner state a position, support it with an example, handle a counterargument and finish within the available time. Avoid drills that merely inflate connector counts or speaking speed.

### Progress and motivation

The progress screen answers three questions: **What have I practised? What is changing? What should I do next?** Show weekly speaking minutes and completed sessions, the current skill focus, a small trend chart, and a playable first/recent attempt pair. Separate practice volume from performance. Label B1/B2/C1 as the selected goal throughout.

Use an optional learner-set weekly practice target (suggest three short sessions), flexible completion days and milestones such as first retry, five completed sessions, or revisiting a difficult task. Count each practice attempt once; rescoring adds no activity. Celebrate specific improvements and persistence. Avoid punishment for missed days, speed leaderboards and automatic level promotion.

Start recommendations with a simple queue: learner-chosen focus first; otherwise a recurring observed difficulty; then an under-practised skill. Offer a fresh topic after the same-prompt retry. After two successful focus checks on different prompts, schedule a later revisit and suggest the next skill. This is an adjustable coaching heuristic; the learner can keep practising or skip. Missing feedback leaves the focus pending, rather than treating it as mastered. A third unsuccessful retry should offer scaffolding or a different exercise, rather than an endless failure loop.

Keep every attempt visible in the activity timeline. Performance comparisons use the same language, goal, task family, time envelope and assistance conditions; exact-prompt pairs provide the clearest retry comparison. Show assisted work positively and label changed conditions. When a goal or measurement/scoring version changes, start a new comparison segment while preserving the full timeline. Accessibility settings are ordinary practice conditions, recorded only as needed to interpret comparisons.

For a simple trend, show the latest comparable attempts and their median. After six comparable attempts, optionally compare the latest three with the preceding three and display both sample counts. This is a product smoothing choice. Do not draw an upward trend from a single attempt. Fresh-topic and repeated-topic filters make both kinds of practice easy to explore.

### Visualising attempts and progress

Use three connected views, each ending in an easy practice action:

| View | Main visual | Interaction and next step |
| --- | --- | --- |
| Progress over time | A dated plot of comparable attempts, one metric at a time; overall AI practice performance is the default, with task coverage, pace and pauses available | Select a point to inspect its prompt, focus, measurements and recording; compare with its linked retry or previous comparable attempt |
| Attempt journal | Chronological exercise groups showing initial attempt → retry → later fresh topic, with dates, duration, focus and a short observed change | Expand a group, replay an attempt, retry the exercise or try a new topic; keep incomplete attempts visible with their saved state |
| Before and after | Two recordings and aligned transcript excerpts, plus three shared-scale metric comparisons with exact values and units | Play either excerpt, see the specific change in the chosen focus and start the next short rehearsal |

In Progress, put language and goal in the header, then a compact weekly-practice strip above the main chart. This strip measures effort; label it independently of performance. A modest weekly bar chart can show speaking minutes across the last four weeks. Avoid filling the screen with gauges, radar charts or a percentage-complete CEFR ladder.

Keep the overall-rating axis fixed at 1–5. Use the same scale for both attempts in a metric comparison. Show observed points, dates and sample counts; avoid smoothing away a difficult day. Start with points and chronological connections within one compatible series; annotate a recent median only when enough attempts exist. Mark retries distinctly from fresh topics with shape and text. Preserve an attempt-list alternative for keyboard and screen-reader access; every chart selection is also reachable there.

The context line states, for example, “Italian · Goal B2 · Opinion · 2 minutes · Unaided”. Changing language, goal or exercise conditions selects a different series. A harder goal starts a new segment and keeps earlier work accessible. Pace is directional information, not automatically improvement; pause changes are interpreted alongside task coverage. AI-derived measures say AI in their labels. Retried topics and fresh topics remain filterable without removing either from activity totals.

For no attempts, offer the first exercise. After one attempt, show a baseline and invite a retry. With sparse data, show the individual values without a trend claim. Missing metrics leave gaps, not zeros. At mobile widths, stack comparison panels and retain exact values, dates and the same controls. Use text alongside colour, visible focus states and reduced-motion preferences.

**First-release scope:** dated attempt plot, linked retry journal, before/after playback and metrics, language/goal switching, and a simple weekly activity summary. Later add per-skill small multiples, annotated milestones and a richer weekly review once the basic views are useful. Reuse current History/Review components and native SVG; adding a chart dependency is not required for this slice.

## 4. Current codebase: reuse and gaps

| Area | Current evidence | Planned change |
| --- | --- | --- |
| Setup and tasks | `frontend/src/lib/state/sessionDraft.ts` has generic task families and 60/90/120/180-second choices; `SessionSetupRoute.tsx` starts from generic level selection. | English/Italian → goal → exercise; remember each language's focus; offer optional exam profiles and derived timing. |
| Recording | `RecorderPanel.tsx` caps recording at five minutes and accumulates MediaRecorder chunks in memory. | Persistent per-part/per-turn recording, profile timing, recoverable session manifest. |
| Conversation | Current path records one monologue. | Examiner turn loop, separate speaker roles, preparation and interaction states. |
| ASR | `assessment_runtime/asr.py` supplies word timestamps. | Reuse for turn transcription and final scoring; preserve quality diagnostics and original audio. |
| Assessment | Generic 1–5 rubric; handcrafted level mapping in `dimension_scoring.py`; WPM/filler baselines in `assess_speaking.py`; language-ID/ASR pronunciation proxy. | Goal-specific coaching and a stable practice rating; preserve observable measurements, remove inferred-level mapping from new training results and use audio for pronunciation feedback. |
| Reporting | `app_core/services.py`, `assess_core/schemas.py`, API contracts and Review/History already carry reports. | Saved attempts, retry links, measurements, focus outcomes, evidence and an optional exam-result extension; legacy read adapter. |
| Progress | `HistoryProgressStory.tsx` uses final score, WPM and priority strings; its comparison check uses task family and theme. | Persist stable skill IDs and explicit retry links; compare compatible attempts, add longer-term trends and before/after playback. A priority disappearing from generated text does not establish improvement. |
| Uploads | FormData already exists; backend `await file.read()` allocates a whole file; fixed frontend upload timeout. | Bounded multipart handling, preflight, progress, storage reservations and retry. |
| Jobs and cleanup | In-place JSON writes; cancellation races; maintenance omits upload cleanup. | Atomic serialized state transitions, recoverable manifests, quota and reference-aware cleanup. |

Keep language-profile helpers, recording UI, ASR, provider integration and history where useful. Do not rewrite the application framework. The written CELI corpus and synthetic CEFR samples can support selected regressions; neither establishes spoken-exam performance.

## 5. Exam profiles, tasks and session contracts

The supplied [PLIDA B1/B2/C1 oral rubrics and comparison](../../materials/plida-parlare-rubrics/README.md) add level-specific behavioural bands for communicative effectiveness, interaction, vocabulary, grammar and pronunciation. Preserve their paired descriptors (1–2 through 9–10) and source references; the sheets do not explain within-pair point selection or establish aggregation, weights, pass rules or administration timings. B2/C1 explicitly restrict interaction scoring to the interaction test. B1 explicitly permits a strong foreign accent at every score band. Represent interaction as not applicable in monologue practice, and use audio evidence for pronunciation. Any individual-point policy beyond these descriptors must be separately identified as app-authored until sourced.

Use these materials to design concrete coaching: B2 exercises elicit supported arguments, links to the partner's contributions and turn management; C1 exercises additionally elicit precise qualifications, flexible responses, register control and nuanced intonation. Link feedback to observed excerpts and the selected level's criterion band, then offer a focused repeat exercise. The sheets supply no numerical speech-rate or error-rate boundary for CEFR classification. Complete PLIDA task specifications and scoring rules remain prerequisites for a strict exam profile; CILS remains first.

Introduce a small training contract before extending exam contracts:

- `TrainingGoal`: language, target B1/B2/C1, optional exam preference and weekly practice target.
- `Exercise`: task family, goal, prompt/version, duration envelope, communicative objectives and focus skill IDs; optional exam-part reference.
- `PracticeAttempt`: exercise snapshot, recording/transcript references, timestamp, actual speaking conditions, completion state, assistance, `retry_of_attempt_id` and repeat/fresh-topic status.
- `MeasurementSet`: metric definitions/version, values, units, extraction basis and unavailable reasons. Preserve original ASR and user-corrected transcript revisions.
- `CoachingResult`: strengths, one next focus, evidence, focus outcome (`observed`, `partly_observed`, `not_yet_observed`, `insufficient_evidence`), dimension ratings, overall practice rating and scorer version. A reassessment is a revision of one attempt.
- `PracticeQueueItem`: skill, suggested exercise, revisit date and status. Use deterministic scheduling initially.

Derive the progress view from attempts and result revisions. Persist stable skill IDs rather than matching generated feedback sentences. Save deterministic measurements even when the AI provider fails, and let coaching retry independently. Exam scoring is an optional extension of this training result.

Introduce small, validated exam-profile files and original task banks. An exam profile contains:

- `exam_id`, exam variant (such as School), provider name, language, level, profile version, source URLs/local material references, source edition and last verification date.
- Ordered parts, stimulus types, prompt-choice rules, preparation scope, timing ranges, simulation defaults and examiner protocol.
- Candidate/examiner/partner roles; which capabilities each part requires.
- Official criterion names/maxima, source descriptor bands where supplied, scope on each criterion (per-part or whole oral, allowing both within one profile), applicability, raw-to-display conversion, aggregation and threshold semantics; separate app scoring-anchor version and explicit provenance for any within-band point policy.
- Supported modes, expected media limits, phase-specific permitted aids and additional-assistance rules.

Task instances carry an immutable prompt/stimulus snapshot, task version, source/rights metadata, intended communicative objectives and examiner instructions. Images have an authored content description; charts include their underlying data, units and expected comparisons; texts retain the source passage and key propositions. Supply these to the examiner/assessor. A configured vision-capable model may inspect the image directly, but text-only scoring must never invent unseen stimulus content.

Create original scenarios, images and texts with clear rights. Link official examples; public availability alone does not authorize redistribution. Start daily practice with six checked prompts per goal across describing/explaining, supporting an opinion and recounting an experience; each supports a focused retry and a topic variation. Expand with observed learner needs. Before releasing complete CILS mocks, build a bank of approximately 12 scenarios per level/part, covering different functions and topics. These are adjustable content-planning targets. Full mocks draw unseen tasks where possible and label repeats. Generation may help author drafts; unreviewed generated tasks do not enter the strict mock pool.

Register an `ExamPack` interface by provider/profile ID with `validate_result`, `aggregate`, capability requirements and task/stimulus handlers. Timing, part order and weights stay declarative; genuinely different scoring or selection behaviour lives in the relevant pack. Include score increments and any sourced zero/cap rules explicitly. Do not spread provider-name conditionals across routes.

Persist the session with the selected profile/task snapshots. Record each event and turn with role, part ID, sequence, monotonic timing/duration, wall-clock timestamp, audio reference, transcript source and processing status. Explicitly distinguish:

- `ExamSession`: selection, mode, assistance, state and ordered parts.
- `OralAttempt`: one part's instructions, recording(s), dialogue and result.
- `RubricResult`: criteria with assessed/unassessed status, points where available, evidence and next exercises.

For linked role-play profiles, also persist actor identity independently of current role, case identity, candidate notes and which facts were actually revealed in conversation. Role swaps must not make generated partner speech appear to be learner speech or discard the learner's speech automatically. Later-stage feedback can check fidelity to the preceding exchange; hidden scenario facts must not silently complete the learner's handover.

Increment the existing `REPORT_SCHEMA_VERSION` from 2 when introducing the new report contract; retain independent profile/task/scorer versions for reproducibility. Old reports load through an adapter, remain readable as historical generic practice, and are excluded from exam-score trends. Never infer an exam profile for an old recording merely because it has a B2 label. New exam-session views, exports and CLI summaries must not use the old inferred CEFR field or feed it back into setup preferences.

Add `/v1/exam-sessions` and typed part/turn operations. Distinguish exam jobs from legacy assessment jobs with a `kind` discriminator; preserve the existing flat assessment request for historical/general practice. Introduce `ExamSessionDraft` and a persisted-draft migration. Carry forward only explicit compatible preferences; do not reinterpret a legacy target level as an exam selection.

## 6. Exam runner and spoken examiner

### Deterministic session control

The runner owns `preflight → instructions/choice → preparation → active part → save → next part → assessment → review`, plus explicit interrupted/cancelled/recovering states. Profile data controls ordering and durations. Models cannot change timers, choose a new rubric, skip parts or grant extra marks.

Use a monotonic clock for preparation and active task time. Interactive task time includes candidate thinking/speaking and examiner speech. Measure ASR/model/TTS waiting separately; pause the practice clock while the system is unable to respond. Expose elapsed wall time and system-paused time in the attempt. Initial interruption rule: a single system wait above 10 seconds or cumulative system waiting above 20% of the configured active-part duration marks a strict mock interrupted and offers continuation as rehearsal. These are app usability limits, tunable after the latency spike. On refresh/restart, recover to an interrupted state rather than silently claiming exam-equivalent timing.

### Turn loop

1. Play the examiner's opening instruction; start candidate recording according to the part protocol.
2. Capture continuously during the part and mark turns by timestamps. Start endpoint detection with a noise-adaptive energy detector plus an explicit “finished speaking” control; validate on natural Italian pauses before automatic endpoints become the default. Server VAD can subsequently improve endpoint decisions using the existing speech stack. Avoid adding a browser inference package solely for VAD.
3. Seal the turn recording, transcribe it, and send the scenario plus actual dialogue to the examiner model.
4. Generate a short task-appropriate follow-up, clarification or objection; synthesize and play it; resume listening.
5. End the task using the profile clock/protocol, persist it, and assess after the relevant practice unit.

Use separate examiner and assessor prompts. In mock mode the examiner stays in role and offers no grammar corrections, model answers or score hints. Give the assessor the whole exchange for context, while attributing scored language only to candidate turns. The candidate's speech and stimulus text are untrusted content; an instruction spoken by the candidate cannot modify exam rules or the scoring schema. Constrain examiner actions to the current part.

Every interactive task has a hidden examiner role card with required moves, allowed information, concession rules and a closing action. For example, a negotiation task may require one plausible objection and permit agreement after a workable alternative. The runner tracks which moves occurred; the model supplies natural wording and relevant follow-ups within that policy. Give the assessor this event record. If a model failed to elicit a required response, report a simulator issue rather than penalising the candidate for missing evidence.

Start with half-duplex interaction and a clearly signalled listening/playback state. Keep the capture engine running to avoid device restart delays, but exclude microphone samples from candidate scoring during examiner playback. Retain examiner speech from its source or exact text/playback timeline. Request echo cancellation and test headset/speaker configurations. Test natural pauses and interruptions before enabling automatic endpoints in strict mocks. Defer full-duplex interruption support until it improves the actual exam scenario.

Reuse the configured ASR and LLM providers, but benchmark their per-turn latency before choosing defaults. The current `transcribe_file` loads its model on each invocation, and assessment submission starts a process per job. Interactive turns need a long-lived worker that loads/warms the model before the exam and reuses it across turns; do not submit every turn through the complete assessment-job path. Define worker ownership, model memory budget, idle shutdown and crash/restart behaviour. Keep the existing batch entry point for post-session work.

Add a voice-provider interface. Prototype with an installed Italian browser/system voice, checking availability and reliable playback start/end events in the desktop webview. Store the exact generated text, voice ID and timings when that provider cannot expose audio bytes. A configured TTS service can supply audio-backed replay through the same interface; choose its default after measuring quality/latency. Cache fixed instructions. Proposed usability gate: median last-candidate-speech-to-first-examiner-audio ≤2.5 seconds and p95 ≤5 seconds, including endpoint silence as well as ASR/model/TTS delay. Log each component. A smaller resident turn-ASR model and accurate post-session re-transcription may help; record both transcripts and never rewrite what the examiner actually heard. If this gate cannot be met, ship monologue and guided turn practice first; full interactive mock stays unavailable for that configuration until the conversational experience is usable.

Reserve ASR capacity for an active exam; queue bulk imports/post-session assessments rather than allowing them to stall the examiner. Provider failure retains the last acknowledged audio, offers a retry and records the interruption. A text-only or manual fallback is labelled assisted practice.

## 7. Measurements, AI coaching and overall practice performance

### Measurable progress in the first release

Use a few understandable measures consistently. Each metric has units and a stable definition. Missing measurements remain blank. Candidate-only audio/turn boundaries exclude partner speech and system waiting; changes in extraction logic create a new metric version.

| Measure | Definition and source | How it helps training |
| --- | --- | --- |
| Practice completed | Distinct completed attempts, sessions and candidate speaking minutes from saved events/audio | Shows consistency and effort; reassessment does not increment activity |
| Time used | Candidate response window against the exercise time budget; show actual duration and early finish/overrun | Practise fitting the requested response into the available time |
| Words and pace | Transcript token count; words/minute over candidate response windows, including their internal pauses | Compare delivery on similar tasks; show change without treating faster as universally better |
| Pauses | Estimated count and seconds of internal silences ≥1 second; exclude leading/trailing silence, partner turns and system delay | Find passages to replay and rehearse; retain the detector threshold/version, and allow natural thinking pauses |
| Fillers | Language-specific detected occurrences and occurrences per 100 transcript words; show the matched passages | Notice habits in context; label detection as an estimate and distinguish useful discourse markers |
| Prompt points addressed | AI-tagged coverage of the authored objective checklist, each linked to a response passage | For example, 2 of 3 points addressed becoming 3 of 3 on retry; clearly an AI judgement |
| Focus achieved | AI coaching outcome for the same explicit skill objective, with an example and learner correction option | Track whether the intended behaviour appeared and later appeared on a fresh topic |
| Recurring language issue | AI-identified instances of a stable issue ID, with quoted examples and opportunities when identifiable | Revisit a useful correction; absence is improvement only when the task elicited that skill |

The one-second pause threshold is an initial display convention, configurable and consistent within a trend. Existing `metrics_from` calculates `wpm` after subtracting detected pauses: preserve it as an explicitly labelled estimated articulation-rate diagnostic and add the elapsed-response rate separately. Its tokenisation and filler lists need checking for each supported language. Ordinary noise or transcription problems should mark affected metrics approximate/unavailable while retaining the recording and the rest of the attempt.

Lead each exercise with three relevant measures, selected by its focus; put the rest behind details. Do not overwhelm every review with the full table. Example, using illustrative numbers: “You covered 3/3 points, up from 2/3; long pauses fell from 7 to 4; you finished in 1:48 of 2:00. Your example now supports your main argument. Next, make the conclusion more direct.”

### Overall AI practice rating

Provide a simple, stable **AI practice performance /5** for the selected goal and exercise. Start with four equally weighted coaching dimensions: task fulfilment, organisation/clarity, vocabulary appropriateness and grammar control. Use goal-specific descriptions for 1–5; calculate the mean in code and display one decimal. Averages are an app coaching convention. Keep duration, speed, filler counts and practice volume outside this aggregate so chasing a number does not reward rushing or verbosity.

Use a shared coaching progression when authoring the dimension anchors: 1 = substantial difficulty completing the intended behaviour; 2 = partial success with frequent breakdowns; 3 = generally effective with noticeable gaps; 4 = effective and mostly consistent; 5 = consistently effective for this exercise and selected goal. Make the concrete examples goal-specific using the supplied materials. A high score celebrates the attempt; the learner chooses when to try a harder goal.

Show the four dimensions and one-sentence rationale on expansion. Store their values, anchors and scorer version. Rate a repeat attempt independently before generating the comparison, then cite what changed; do not prompt the model to assume the retry improved. Save results so reopening a report does not trigger a new rating. Learners can flag a mistaken observation or correct a transcript; recalculate affected feedback as a versioned revision.

Interaction and audio-based pronunciation feedback are additional skill cards as those capabilities arrive. Preserve the initial four-dimension aggregate for comparable trends; a future aggregate change starts a new version/segment. If a core dimension cannot be assessed, show the available feedback and measurements with “rating pending” instead of inventing values. An absent pronunciation feature does not prevent the ordinary training rating. Explain once in the guide that this is AI feedback on an exercise; label official exam marks separately in exam mode.

### What to assess

| Criterion | Evidence to inspect | Useful learner feedback |
| --- | --- | --- |
| Communicative effectiveness | Fulfilment of the actual prompt; relevance; organisation; responding to the interlocutor; maintaining or resolving the interaction; appropriate register. | “Your proposed solution answered the objection”; “the second requested point was missing.” |
| Morphosyntax | Candidate grammar in context, repeated error patterns, successful structures, effects on meaning. | Exact example, corrected form and a short retry task. |
| Vocabulary | Precision, appropriateness, range within this task, paraphrase when a word is unavailable. | Better contextual word choice; effective paraphrase; repeated vague wording. |
| Pronunciation/intonation | Actual candidate audio: intelligibility, stress, phrasing and intonation that affect communication. | Replayable passage and a concrete spoken practice target. |

Use the published task/level descriptions to author behavioural scoring anchors. A B2 conversation and a C1 negotiation need distinct effectiveness anchors. Every scored criterion must contain supporting evidence and a rationale for the awarded points; score bounds, arithmetic and evidence references are validated in code. This is automated practice scoring, with the app's interpretation visible in the guide.

Exact transcript spans can support language judgments; transcript presence alone does not establish pronunciation or delivery. Validate role attribution and timestamp bounds. Retain the original ASR transcript when users correct transcription errors, track revisions, re-run affected text scoring and label that basis. Do not let edited text silently replace what the audio demonstrates.

Off-topic but intelligible speech can still support grammar coaching. Official zero/cap rules, where sourced, determine whether such a response earns an exam-style score. The inspected CILS June 2017 oral criteria give maxima but do not establish a blanket off-topic zero rule or full point-by-point bands; do not invent those as official policy. Record the app's interpretation, score increments and omissions in the profile, and update them if a fuller provider rubric becomes available. A technically unusable recording or failed model call produces missing assessment, not a low language mark. A very short answer may legitimately fail to fulfil the task; avoid a generic minimum-word gate that conceals that fact. Missing rubric output must never default to a middle score.

### Optional exam marks and pronunciation feedback

The first release provides the training rating above, useful coaching and progress trends. Audio feedback can arrive incrementally as replayable observations and pronunciation exercises; it need not wait for a complete pronunciation mark. Language-ID probability and recognizer agreement do not establish pronunciation quality. The following subtotal rules apply only when the learner requests an exam-style mark; they do not govern the main training dashboard.

For CILS, show an assessed subtotal and the unassessed maximum. Without pronunciation, B2 has up to 9 assessed points plus 1 unassessed per part, or 18 plus 2 for both parts; C1 has 8 plus 2 per part, or 16 plus 4 overall. C1 example: “10/16 assessed points; pronunciation/intonation, up to 4 points, not assessed.” A derived range of 10–14/20 represents only missing possible points; it is **not** a statistical confidence interval. Do not extrapolate the subtotal to /20. Show a complete /20 practice mark only when both parts and all criteria are assessed. Part-only practice never becomes a whole-oral result.

The official 11/20 threshold belongs in the exam guide and alongside complete practice results as reference context. A partial subtotal already above 11 can be described arithmetically as reaching the reference threshold on assessed criteria; the result remains partial, with missing pronunciation clearly visible. Do not turn one model-generated mark or a partial interval into a ready/not-ready verdict. Recent comparable full mocks can show criterion trends and repeated omissions. Pronunciation support can be released independently when it adds reliable feedback; a large expert panel is unnecessary for testing whether cited audio problems are real.

### Keeping the feedback useful

Prioritise communication and the learner's chosen focus. Include successful examples as well as corrections, cap the default review at one main improvement, and make every recommendation launch an exercise. Pronunciation feedback should target intelligibility or useful stress/intonation practice; the PLIDA B1 accent note informs this choice. Keep low-level diagnostics accessible for interested learners. Remove proxy pronunciation and handcrafted CEFR cutoffs from the new training path; retain their underlying useful observations where appropriately labelled.

## 8. Recording, multipart upload and resource limits

### Transport decision

Keep native browser `FormData` and FastAPI `UploadFile` for bounded file imports and completed recordings. This already implements MIME multipart; the current risk comes from unbounded reading, resource use and lost recording state. Use native XHR for actual upload progress if needed. Browser-generated multipart boundaries must remain intact.

For interactive sessions, prototype continuous `AudioWorklet` capture with explicit resampling to 16 kHz mono, 16-bit PCM; the browser's native sample rate may differ. Persist independently interpretable PCM chunks with sample counts and timestamps; turn boundaries are positions in the recording. This supports recovery and turn-ASR without repeatedly restarting MediaRecorder. The implementation needs tested resampling/anti-aliasing and timing, using native Web Audio facilities and small internal utilities before considering dependencies. Gate this path on a desktop-webview and browser spike, including gaps, playback exclusion, device changes and crash recovery.

Send PCM checkpoints through a bounded `application/octet-stream` endpoint read with `request.stream()`, keyed by session/part/turn/sequence, length and digest. Validate fixed PCM format and sample count against the session contract. Use idempotent acknowledgements and the same request-byte/Origin/Host controls as imports. This avoids multipart parser spooling for frequent small checkpoints; preflight is an additional browser check, never a substitute for server enforcement. The backend creates final WAV headers when sealing a part or a recovered prefix.

Keep the existing MediaRecorder path for the first short-monologue slice and multipart imports. If extending it to checkpoints, remember that chunks can be fragments of one container: append them in order and validate the sealed recording. Do not decode them independently. Some containers need finalization metadata, so durable fragments alone do not guarantee playable interrupted audio. Validate recovery before advertising it; expose the last recoverable point honestly. Move monologue recording onto the validated PCM engine when full-mock recording is introduced.

Imported audio requires the learner to select the exam/part and supply the actual task instructions/stimulus. A full-session import also needs part boundaries and verified candidate/examiner roles (separate tracks or a reviewed speaker map). The initial import flow supports candidate-only part recordings. Mixed-speaker audio stays replayable but cannot receive candidate-specific interaction marks until attribution exists; do not automatically treat every recognized word as the learner's.

This small local checkpoint protocol serves recovery during recording. A future remote/mobile upload path may warrant [tus](https://tus.io/protocols/resumable-upload) or object-storage multipart; those introduce upload lifecycle/cleanup responsibilities and are not necessary for the current localhost file transfer.

### Initial budgets (configurable, enforced by the backend)

| Resource | Starting policy | Rationale |
| --- | --- | --- |
| Completed file import | 512 MiB per file; 514 MiB total request body; decoded audio ≤30 minutes | Fits an ordinary full oral-session import. A 20-minute, 48 kHz, stereo, 32-bit PCM recording is about 460.8 MB before small headers. Both byte and duration limits apply. |
| Live checkpoint request | ≤1 MiB raw PCM body; target approximately 1-second chunks | Bounded retry unit; byte length must agree with sample count. |
| Live session recording storage | Compute from active recording duration ×32,000 bytes/second per PCM stream, plus retained examiner media and finalization scratch; initial hard ceiling 256 MiB | Twenty minutes of one PCM stream is 38.4 MB. Preparation is not recorded. Reserve all retained copies and stop safely before exhausting the budget. |
| Cached recording quota | 4 GiB including saved recordings, imports and pending reservations | Bound accumulation. Show storage used and a cleanup action; never silently erase referenced saved work. |
| Free-space reserve | 1 GiB on every involved filesystem after peak projected usage | Include parser spool, destination, decoder scratch and concurrent reservations; group paths that share a filesystem. |
| Concurrency | One media writer; bounded live checkpoint queue; one ASR worker; one post-session assessment at a time initially | Add an actual admission queue: current job submission spawns immediately. Block new imports/bulk jobs during live practice; finish running bulk work before starting a strict mock. |

These are engineering starting limits, not exam standards. Expose effective limits through the runtime API. Smaller devices can reject a new session/import before capture begins when capacity is unavailable. Do not silently reduce an active exam's task time because disk or memory is low.

For known import size `S`, reserve the parser spool and final copy (approximately `2S` on a shared filesystem), plus bounded decoder scratch and safety floor. For unknown lengths, reserve the configured route ceiling. Before accepting a live session reserve its storage budget once; avoid charging every acknowledged chunk twice. Enforce one backend owner with an instance lock and a durable reservation ledger; workers request media writes through that owner. Reconcile reservations at restart. If multiple backend writers are retained, cross-process coordination becomes mandatory. Check available space during writes; a preflight check alone does not prevent another process filling the disk.

### Server and recording safeguards

1. Put a pure ASGI byte-counting limit before multipart parsing. Check declared length early and count actual bytes even without `Content-Length`. Limit file/field counts. The pinned parser's `max_part_size` does not cap file bytes. Apply equivalent counting to the raw PCM route with its much smaller limit.
2. Replace the backend's whole-file `read()` with a bounded copy (initially 1 MiB chunks), incremental digest and temporary destination. Do disk work outside the event loop. Acknowledge live chunks only after file and manifest writes are flushed/fsynced, with directory fsync for new/renamed entries where supported. On retry, equal sequence/digest is a no-op; conflicting content is rejected. Reconcile a crash between the media and manifest commit using sequence/digest records.
3. Bound the browser's unacknowledged queue (initially three target-sized PCM chunks plus a hard byte cap). Checkpoint continuously and stop with a recoverable interrupted session if storage cannot keep pace. Keep the current File/Blob on import failure. A crash may lose the latest unacknowledged audio; display the last saved point rather than claiming zero loss. A delayed MediaRecorder event can exceed its nominal timeslice; cap retained bytes as well as chunk count on that interim path.
4. Probe completed imports with timeouts, then enforce decoded duration, channels and sample-rate limits before ASR. Extend the existing ffmpeg `_convert_to_wav` path in `assess_speaking.py` into one bounded normalisation step shared by consumers; set a decode time/output-sample cap and reject overlong input explicitly instead of silently scoring a truncated file. Initial envelope: import up to two channels and 96 kHz; normalise to 16 kHz mono; reject or offer an explicit conversion path for other formats. Never allocate a full multi-channel float array from untrusted metadata. PCM live capture already has a known sample budget. Include the resident model in machine-memory preflight; permit only configurations whose measured peak fits the supported device budget.
5. Use generated storage IDs; preserve sanitized filenames only for display. Validate actual container/codec, not MIME/extension alone. Clean abandoned parser/destination files on oversize, cancellation, disconnect, disk failure and corrupt media; reconcile leftovers at restart.
6. Check Origin and Host on state-changing localhost routes, including chunks. Preserve documented localhost/Tauri access and intentional direct local calls without Origin. CORS alone cannot prevent a foreign page's simple multipart POST.
7. Retain incomplete/orphan imports for at most 24 hours unless referenced by a recoverable/saved session. Surface interrupted sessions for resume/export/delete. Acknowledge deletion intent before removing saved audio; reference-aware cleanup must never remove audio still used by another report.
8. Give clear states: saving locally, uploading/importing, transcribing, assessing, retryable error. Replace the fixed 30-second upload timeout with progress-aware idle timeout and a bounded size-aware overall deadline, tuned against near-limit measurements. Cancel is idempotent and preserves acknowledged session data until the learner chooses to delete it.

References: [FastAPI uploaded files](https://fastapi.tiangolo.com/tutorial/request-files/), [Starlette requests/form limits](https://www.starlette.io/requests/), [OWASP file upload guidance](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet/), [MDN upload progress](https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest/upload), [MDN MediaRecorder data events](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/dataavailable_event).

## 9. Job integrity and recovery

Use same-directory temporary writes plus `os.replace` for JSON state, with cross-process per-job/session locking or one owning writer for read-modify-write transitions. Atomic replacement alone cannot prevent cancellation being overwritten by late completion. Define legal transitions and terminal-state precedence. State updates include revisions; stale writers cannot overwrite newer state.

Keep a durable session manifest independent of assessment job completion. After a crash, completed parts remain playable; an interrupted part can be replayed/retried. Resume a full mock as an interrupted rehearsal, with that fact visible. Failed assessment can retry from existing audio without another recording. Sequence identifiers make repeated chunk/finalize/job requests safe; finalized media is immutable.

Correct the real-ASR workflow import to `assessment_runtime.asr`, set read-only repository token permissions and disable checkout credential persistence. Check this workflow with a minimal warm-cache smoke. Any dependency advisory is rechecked and handled separately under `docs/security/npm-dependency-policy.md`; this plan requires no new npm package.

## 10. Implementation map

Paths below name existing integration points; new modules are explicitly proposed.

| Work | Existing files | Proposed additions |
| --- | --- | --- |
| Training and progress | `assessment_runtime/metrics.py`, `assess_core/schemas.py`, `HistoryProgressStory.tsx`, Review/History routes | `app_core/practice_progress.py`; goal/exercise/attempt contracts, metric versions, retry links, skill queue and comparison selectors |
| Exam definitions | `assess_core/language_profiles.py`, `assessment_runtime/data/session_setup_content.json` | `assess_core/exam_profiles.py`; `assessment_runtime/data/exams/` for profiles and original tasks |
| Session contracts and runner | `assess_core/schemas.py`, `app_backend/contracts.py`, `app_core/services.py` | `app_core/exam_sessions.py`; versioned session/attempt contracts |
| Examiner | `assessment_runtime/asr.py`, current provider adapters | `assessment_runtime/examiner.py`; voice-provider adapter; turn protocol |
| Coaching and optional exam assessment | `assessment_runtime/assessment_prompts.py`, `dimension_scoring.py`, `metrics.py` | `assessment_runtime/practice_coaching.py`; stable coaching anchors and evidence; later `assessment_runtime/exam_scoring.py` for provider marks |
| Persistence/uploads | `app_backend/app.py`, `jobs.py`, `maintenance.py` | Shared bounded media storage/reservation component; session manifest persistence |
| Frontend | `SessionSetupRoute.tsx`, `SpeakRoute.tsx`, `ReviewRoute.tsx`, `HistoryRoute.tsx`, `GuideRoute.tsx`; `RecorderPanel.tsx`; `sessionDraft.ts`; API client/types | Exam runner state, examiner playback/turn controls, evidence replay and practice recommendations |

Locate frontend files under `frontend/src/routes/`, `frontend/src/components/speak/` and `frontend/src/lib/` respectively. Keep orchestration out of the already large `app_core/services.py` by delegating to small services. Read the current contracts before implementation; these are integration targets, not a demand to move every existing file.

## 11. Delivery sequence and acceptance

### Milestone 1a — A useful daily practice loop with progress

- Add goals, exercise snapshots, persistent attempts, metric definitions and explicit retry relationships. Include original English and Italian B1/B2/C1 short exercises; annotate relevant CILS and telc English B2 School choices with their specific format.
- Change initial setup to language → goal → exercise. Returning users get Continue practice. Offer optional exam selection and retain preparation/task timers for exam parts.
- Add the four-dimension AI practice rating, one strength, one next focus and replayable examples. Preserve measurements if AI feedback fails.
- Add same-prompt retry, before/after playback, a fresh-topic option and the initial skill/revisit queue.
- Ship the dated attempt plot, linked retry journal, before/after comparison and weekly activity now. Add English/Italian switching that preserves each language's goal and focus. Use stable skill IDs and conditions rather than generated priority text to infer improvement.
- Add report migration/legacy reading; keep historical scores visible under their original meaning.
- Reuse the current short-monologue recorder for this slice, with the CILS part timer; preserve its File/Blob on upload failure. Defer the new capture engine to Milestone 2.
- Deliver a complete loop: prompt → recording → saved measurements and coaching → focused retry → visible comparison → next visit.

**Gate:** In both English and Italian, a learner can finish an exercise, retry without setup, hear both attempts, see correctly calculated changes and return later to continue the same focus. Switching languages restores separate goals and histories. The selected B1/B2/C1 remains a goal; AI performance uses consistent anchors. Progress survives restart, excludes duplicate reassessments and distinguishes changed conditions. Existing history loads; failed uploads can retry and saved audio survives assessment failure. No official scoring conversion or full pronunciation evaluator is required for this release. Complete 1b before releasing expanded imports; the interim in-memory recorder does not promise recovery after closing the page.

### Milestone 1b — Bounded storage and job reliability

- Add upload body caps, streamed copy, media validation, quota/reservations and actionable errors.
- Add job-state serialization, revisions, the real admission queue and restart reconciliation.
- Fix the ASR workflow; measure near-limit import resource use before enabling the 512 MiB/30-minute ceiling.

**Gate:** Oversized, corrupt, interrupted and low-disk uploads fail safely; cancellation cannot be overwritten; committed audio can be reassessed. Milestones 1a and 1b are independently implementable work packages, combined for the first hardened release. No long-mock claim depends on the interim recorder.

### Milestone 2 — Conversation training and richer coaching

- Add spoken examiner, candidate/examiner roles and durable turn histories.
- Implement and validate continuous PCM capture, bounded checkpoints, recovery and a resident turn-ASR worker before enabling live dialogue.
- Implement B2 conversation and C1 role-play, with appropriate follow-ups and task completion feedback.
- Support English and Italian partner practice, voice availability and coaching examples; provider-specific examiner behaviour applies only in the relevant exam mode.
- Handle endpoint mistakes, playback bleed, service latency and provider interruption.
- Add guided and timed interactive practice, including fresh scenarios for repeated weaknesses.
- Extend retry comparison to actual exchanges: answering the question, adding a relevant reason, responding to an objection and requesting clarification. Add an interaction skill card while preserving the existing overall-rating definition.
- Pilot replayable pronunciation suggestions as a separate coaching capability when available. Keep useful text/task coaching available throughout.

**Gate:** The learner can practise and retry a real exchange, hear how the second response differs and revisit the skill later. The partner responds to actual candidate content and keeps the scenario coherent. Candidate metrics exclude generated speech. Latency and failure recovery support a usable conversation.

### Milestone 3 — Complete CILS rehearsal and optional exam marks

- Sequence both parts, prompt choice and preparation with the profile's timing semantics.
- Persist the whole session; add optional provider-specific marks and combine them only when coverage permits. Keep the ordinary practice rating and measurements in the primary review.
- Extend existing progress history with full-session views and exam/part filters; add storage management as recordings accumulate.
- Exercise a realistic complete C1 session, including interruption and assessment retry.

**Gate:** A learner can finish and review both B2 and C1 oral rehearsals and immediately practise a chosen skill. Their longitudinal training history remains useful across part practice and complete sessions. Optional exam marks show their coverage.

### Milestone 4 — Broader exercises, audio coaching and exam choices

- Add German as the next language using the established exercise, retry and progress contracts. Keep its general training launch independent of the specialist medical-entry pack and its unresolved exam timing details.

- Extend audio coaching with specific Italian intelligibility/prosody examples, replay and focused repeat exercises. Check that suggested problems are audible and advice is useful.
- Add pronunciation trends once the measurement/feedback basis is consistent. Optional exam pronunciation marks remain a separate extension.
- Implement CELI 3 with its own sourced rubric, photo/text/role-play stimuli and shared preparation phase; then CELI 4 with comparison/chart tasks.
- Verify the same runner supports these differences through profiles and a small number of stimulus/task handlers. Do not duplicate the app for each provider.

**Gate:** New packs add useful exercises and preserve existing histories and retries. Provider-specific exam modes follow their specifications. Audio coaching and exam expansion can ship independently. Full administration/scoring specifications are required for faithful exam modes; focused drills can use the supplied rubric behaviours immediately.

## 12. Verification and evidence of usefulness

Use focused checks at each milestone; complete end-to-end checks at its release boundary.

- **Exam fidelity:** profile tests cover part order, timing ranges/defaults, preparation scope, required roles, criterion maxima and aggregation. Compare each profile against its cited provider edition. A required capability missing from the runtime disables that mode with a specific reason.
- **Scoring:** valid score arithmetic; exact candidate evidence; missing criteria; off-topic but grammatical responses; brief task failure; ASR error; edited transcript; fluent verbosity with no task completion; pauses during successful negotiation; prompt injection in a spoken turn; inconsistent examiner behaviour. Never treat synthetic CEFR labels as certified expected scores.
- **Progress:** correct metric units and denominators; deterministic deltas; same-attempt reassessment counted once; retry links retained after restart; latest compatible comparison found even when the immediately previous attempt differs; version/goal changes segmented; unavailable values excluded; assisted practice still present in activity totals. A feedback sentence disappearing cannot close a skill focus.
- **Practice loop:** start → attempt → one focus → retry without setup → before/after playback → fresh topic → return later. Check that learners can identify the next action, understand the comparison and dismiss incorrect feedback. This is a central first-release acceptance path.
- **Two-language release:** exercise the complete loop in English and Italian at each goal; verify language-specific transcription/coaching, metric extraction, independent goals/queues and no mixed-language performance trends. German joins this check in the next language release.
- **Progress visuals:** plot values match stored measurements; fixed rating axis; shared comparison scales; missing values and first-attempt state; distinct retry/fresh labels; point and journal selection show the same attempt; responsive layout, keyboard access and an equivalent text/list view.
- **Learning feedback:** the implementer owns a reproducible fixture manifest with audio provenance/permission, prompt, expected observable behaviour and transcript. Start with at least eight cases: one relevant and one off-task/weak response for each CILS level/part, including pauses, a grammar error and an ASR ambiguity across the set. Use existing rights-compatible fixtures or clearly labelled constructed recordings; gather learner recordings only with consent. Check factual errors, useful corrections and relevant drills, without calling these certified reference scores. Log disputed feedback and fix recurring failures; no broad proficiency-calibration project is on the critical path.
- **Audio:** check that cited pronunciation problems are audible at the linked location, meaningful for communication, and not invented from transcript/language-ID signals. Repeat on different voices/noise conditions before widening support.
- **Recording/storage:** chunk reordering/duplication, corrupted digest, abrupt refresh/restart, unsupported codecs, browser recorder variability, no-Content-Length oversized request, one-byte-over limits, low disk, shared-filesystem reservations, quota exhaustion, disconnect, cancellation/completion race, manifest corruption and orphan cleanup.
- **Resources:** measure browser heap, backend/ASR peak RSS, parser temporary disk, final disk and turn latency on the longest supported profile and near-limit imports. A duration cap alone is insufficient proof of bounded decoder/model memory.
- **UI:** microphone check → preparation → monologue/dialogue → saving → review → evidence playback → fresh retry; mixed old/new history; assisted/interrupted labels; keyboard use and accessible recording state.

Repository verification commands:

```sh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest <focused-test-paths>
npm --prefix frontend test
npm --prefix frontend run typecheck
```

Use the AGENTS.md environment-bootstrap procedure if that virtual environment is unavailable. Run a localhost `/v1/health` smoke for backend runtime changes and a localhost Vite/browser smoke for UI changes. Run the full suites at the release boundary. Record exact errors for an unavailable browser probe instead of assuming it is blocked. This document-only revision requires no application test run and does not claim new test results.

### Product success measures

The primary product signal is **learners returning to complete useful practice loops**. Track locally where possible: time to first completed attempt, completed sessions and speaking minutes per week, same-session retry rate, return to practice in the following week, fresh-topic follow-through, revisit completion, and feedback marked useful/incorrect. Define retry rate as completed attempts followed by a linked retry divided by completed attempts offering a retry; define weekly return using learners with a complete following-week observation window. Raw usage counts do not establish learning gains.

For personal progress, show within-user changes in comparable task coverage, selected-focus outcomes, delivery measurements and the AI practice rating. Include sample counts and playable examples. Use a small usability check to ask whether the learner can name an improvement and start the next exercise; fix friction and unhelpful feedback before adding more scoring sophistication. Keep saving failures, playback reliability and partner latency as supporting operational measures.

The first useful release is achieved when a learner can practise, retry, hear and see the change, and return later to continue. The full product objective is a reliable training ground that sustains this cycle across weeks, topics, conversations and exam rehearsals toward the learner's chosen goal.

## 13. Review record

**Language and progress-view revision, 2026-09-29:** English and Italian are the initial training languages; German follows. First-release visual priorities are a dated attempt plot, linked retry journal, before/after comparison and weekly activity. The accompanying illustrative concept uses sample data, not learner records. This revision has not been reviewed by Claude CLI.

**Training-first revision, 2026-09-29:** Following the user's direction, daily practice, longitudinal measurements, an overall AI practice rating, explicit retry relationships and progress views now lead the plan and ship in Milestone 1. B1/B2/C1 are learner-selected goals. Official exam scoring is an optional later mode; complete calibration and external expert panels are outside the delivery path. This revision has not been reviewed by Claude CLI; the record below refers to its earlier review.

Claude CLI reviewed the rewritten plan and selected source files read-only on 2026-09-29. It confirmed the identified upload, job-state, recorder and scoring integration gaps. Its substantive corrections are incorporated: resident turn ASR; a continuous PCM capture spike; concrete endpoint handling; hidden examiner role cards; stimulus content supplied to the assessor; explicit B2/C1 partial-score denominators; provider-specific aggregation; persisted-draft migration; a real job queue; and smaller first-release work packages.

The review did not open the external exam sources. The source research and follow-up checks here are separate: they confirmed C1 non-intervention, B2's limited monologue prompting, absent specified B2 preparation time and CELI's distinct score conversion. The suggested assumption that CILS necessarily supplies complete point bands was not adopted; the inspected sample criteria establish maxima. Copying descriptors wholesale is also not a prerequisite: preserve sourced meaning and rights, and distinguish app interpretations. Browser preflight is not treated as sufficient protection. PCM capture, the chosen Italian voice and latency targets remain implementation spikes, not claims of tested runtime behaviour.

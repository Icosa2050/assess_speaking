# Spoken Corpus Categorization Plan

Last updated: 2026-03-19
Status: Detailed proposal

## Goal

Use real Italian L2 spoken corpora to improve spoken-input categorization without destabilizing the current monologue-centric assessment flow.

This plan stays intentionally conservative:
1. get value from transcript-rich data first
2. keep raw corpus facts separate from guessed repo labels
3. avoid adding new public taxonomy until the corpus proves it is needed
4. keep phase 1 thin enough to answer one product question before building a larger subsystem

## Phase-1 Product Question

Phase 1 exists to answer one explicit question:

Can LIPS monologue sections be reliably extracted and mapped into the current `task_family` vocabulary well enough to create an analysis-grade Italian spoken categorization dataset, without changing runtime taxonomy, scorer behavior, or benchmark behavior?

If phase 1 cannot answer that question clearly, the work should stop at analysis artifacts and not spread further into runtime or benchmark code.

## Current Repo Constraints

The repo currently behaves as if spoken tasks are mostly monologue-like:
1. current `task_family` options are `travel_narrative`, `personal_experience`, `opinion_monologue`, `picture_description`, and `free_monologue`
2. the checked-in Italian benchmark and seeds are synthetic and centered on `opinion_monologue`
3. the coaching taxonomy already has controlled grammar, coherence, and lexical categories
4. the repo already has reusable patterns for typed manifests, small CLI scripts, and generated artifacts under `tmp/`

Implication:
1. phase 1 should improve categorization quality without changing the public task taxonomy
2. dialogue data should be preserved as raw evidence, not forced into monologue families
3. outputs should remain offline analysis artifacts until the corpus proves it is useful

## Corpus Findings

### Corpus LIPS

Local inspection of `/tmp/Corpus LIPS/Corpus LIPS` found:
1. 1420 transcript files
2. ISO-8859-1 encoding
3. CEFR coverage across A1, A2, B1, B2, C1, C2
4. section markers such as `SE1` and `SE2`
5. mode markers such as `M` and `D`
6. explicit examiner and candidate turns using `E:` and `C:`
7. learner-style disfluencies, repairs, and spoken-like fragments

Important nuance:
1. some files include both dialogue and monologue sections
2. this makes LIPS useful for section-level extraction, not just file-level labeling
3. raw `M` labels should be treated as a strong hint, not as the only monologue check

### DILS

Local inspection of `/tmp/2_794_ITL2_GNE` found:
1. 6 WAV files only
2. Gent subset only
3. B1, B2, and C1 coverage only
4. long recordings, about 650 to 737 seconds each
5. no transcripts in the local download

Implication:
1. DILS is too small and too transcript-poor for phase-1 categorization work
2. DILS is better treated as later evaluation material for dialogue or collaborative picture-task work
3. DILS should not shape the phase-1 schema beyond preserving space for future raw metadata

## PAL Consensus Summary

I discussed the direction with:
1. `anthropic/claude-sonnet-4.5`
2. `google/gemini-3.1-pro-preview`
3. `mistralai/devstral-2512` in a skeptical pass

Consensus:
1. prioritize LIPS before DILS
2. keep phase 1 transcript-first and monologue-only
3. preserve raw interaction metadata separately from mapped repo labels
4. require explicit parse and label-quality gates before using the corpus downstream
5. make the plan more actionable by defining the schema, cleaning rules, mapping heuristics, review protocol, and stop conditions up front

Most useful caution:
1. do not force dialogue sections into monologue families
2. do not build a large permanent side system before proving that phase 1 answers the product question
3. define the thin initial slice clearly so implementation remains reversible

Repo-shape note from consensus:
1. phase-1 artifacts can stay JSONL because they are research outputs, not runtime manifests
2. the implementation should still feel like the rest of the repo: small Python module, thin CLI scripts, generated artifacts under `tmp/`, small gold fixtures in tests only

## Final Decision

1. prioritize LIPS first
2. start with monologue sections only
3. do not add a new dialogue-specific `task_family` in phase 1
4. store raw interaction facts separately from mapped task family
5. defer DILS to a later feasibility or evaluation phase
6. keep phase-1 outputs as analysis artifacts only until the go or no-go gates pass

This gives the repo useful real spoken data now while keeping later dialogue work possible and minimizing the risk of destabilizing existing scoring and benchmark flows.

## Phase 1 Scope: LIPS Monologue Extraction

### Objective

Build a normalized section-level LIPS artifact set for categorization experiments using only sections that are monologue-like after both raw-mode filtering and turn-structure sanity checks.

### Thin-Slice Constraint

Phase 1 should stay thin:
1. one core parser module
2. one build script
3. one validation script
4. small focused tests
5. generated artifacts only under `tmp/lips_manifest/`

Do not add runtime wiring, product UI changes, scorer changes, benchmark changes, or broad corpus framework abstractions in phase 1.

### Deliverables

1. a parser that:
   - reads ISO-8859-1 safely
   - splits files on `SE` section markers
   - extracts CEFR level, section id, raw mode, topic, and turn structure
   - computes section-level audit fields
2. a normalized JSONL artifact set with:
   - included monologue-like sections
   - excluded sections with reasons
3. a task-family mapper that maps only when confidence is acceptable
4. a QC script and review-sample generator
5. a validation report that makes the go or no-go decision explicit

## Phase 1 Artifact Model

### Included Record Shape

Each normalized included LIPS section should contain at least:

```json
{
  "manifest_version": "lips_v1_monologue_only",
  "source_corpus": "lips",
  "source_file": "06091200006B2.txt",
  "section_id": "SE2",
  "raw_mode": "M",
  "turn_structure_flag": "monologue_like",
  "parse_status": "full",
  "exclusion_reason": null,
  "cefr_level": "B2",
  "prompt_topic": "descrizione foto signore con cane",
  "candidate_text_raw": "...",
  "candidate_text_clean": "...",
  "examiner_context": "...",
  "section_token_count": 154,
  "candidate_token_count": 143,
  "candidate_turn_count": 1,
  "examiner_turn_count": 1,
  "mapped_task_family": "picture_description",
  "mapping_source": "heuristic_v1",
  "mapping_confidence": "high",
  "needs_review": false
}
```

### Excluded Record Shape

Each excluded section should still be written with enough information to audit the drop:

```json
{
  "manifest_version": "lips_v1_monologue_only",
  "source_corpus": "lips",
  "source_file": "06091200006B2.txt",
  "section_id": "SE1",
  "raw_mode": "D",
  "turn_structure_flag": "dialogue_like",
  "parse_status": "full",
  "exclusion_reason": "raw_mode_dialogue",
  "cefr_level": "B2",
  "prompt_topic": "discussione guidata",
  "candidate_text_raw": "...",
  "candidate_text_clean": "...",
  "examiner_context": "...",
  "section_token_count": 98,
  "candidate_token_count": 71,
  "candidate_turn_count": 4,
  "examiner_turn_count": 3,
  "mapped_task_family": null,
  "mapping_source": null,
  "mapping_confidence": null,
  "needs_review": false
}
```

### Field Rules

1. keep both raw and cleaned candidate text
2. keep `raw_mode` even when the section is excluded
3. allow `mapped_task_family` to be `null`
4. never merge examiner text into candidate text
5. keep `turn_structure_flag` as derived evidence, not a public taxonomy
6. treat `mapping_confidence` as one of `high`, `medium`, `low`, or `null`

## Parse Status And Error Taxonomy

Use these `parse_status` values:
1. `full`
   - section markers, turns, CEFR, and candidate text all recovered as expected
2. `partial_metadata`
   - candidate text recovered, but one or more metadata fields are missing or inferred
3. `text_only`
   - some section text recovered, but the record is too incomplete for inclusion
4. `failed`
   - decoding or structural parsing failed badly enough that the section cannot be trusted

Track `exclusion_reason` using a controlled set:
1. `raw_mode_dialogue`
2. `turn_structure_not_monologue_like`
3. `insufficient_candidate_text`
4. `missing_candidate_text`
5. `decode_error`
6. `missing_section_marker`
7. `unparseable_turn_structure`
8. `metadata_incomplete`
9. `manual_holdout_review`

These values should be treated as reporting and audit tools, not as public user-facing labels.

## Cleaning Rules

`candidate_text_clean` should be intentionally minimal in phase 1.

Allowed cleaning:
1. remove section headers and speaker prefixes such as `SE1`, `E:`, and `C:`
2. concatenate candidate turns in order
3. normalize whitespace
4. normalize obvious punctuation spacing
5. remove structural transcription artifacts only when they are clearly markup and not learner language

Not allowed in phase 1:
1. correcting grammar
2. correcting spelling
3. removing disfluencies merely because they sound spoken
4. paraphrasing or rewriting learner language
5. using examiner text to repair or infer missing candidate meaning

Practical rule:
1. `candidate_text_raw` preserves corpus evidence
2. `candidate_text_clean` is only a lightly normalized version for categorization experiments

## Monologue Inclusion Rules

Include a section in the main phase-1 included artifact only if all of the following are true:
1. `raw_mode = "M"`
2. `parse_status` is `full` or `partial_metadata`
3. candidate text is present
4. `candidate_token_count >= 20`
5. turn structure is monologue-like

Turn-structure sanity check:
1. use `raw_mode` as the first filter
2. then verify the section does not show sustained examiner-candidate alternation
3. if the turn pattern still looks dialogic, exclude it with `exclusion_reason = "turn_structure_not_monologue_like"`

Outlier handling:
1. if `candidate_token_count > 500`, keep the section but set `needs_review = true`
2. if metadata is partially missing but the monologue text is usable, allow inclusion with `parse_status = "partial_metadata"` and `needs_review = true`

## Mapping Strategy

Map to existing task families only when the section clearly fits:
1. `travel_narrative`
2. `personal_experience`
3. `opinion_monologue`
4. `picture_description`
5. `free_monologue`

Do not invent a new mapped family in phase 1.

### Heuristic Starter Table

The first mapping pass should use an explicit table, not ad hoc guessing:

| Evidence | Candidate family | Starting confidence |
| --- | --- | --- |
| `foto`, `immagine`, `figura`, `descrivi` in prompt or topic | `picture_description` | `high` |
| `viaggio`, `vacanza`, `paese`, `turismo` | `travel_narrative` | `medium` |
| `esperienza`, `ricordo`, `famiglia`, `infanzia` | `personal_experience` | `medium` |
| `opinione`, `pensi`, `sei d'accordo`, `vantaggi`, `svantaggi` | `opinion_monologue` | `high` |
| open monologue prompt with no stronger family cues | `free_monologue` | `low` |

Heuristic rules:
1. `high` means the prompt/topic evidence is explicit and specific
2. `medium` means there is a strong cue but alternative families remain plausible
3. `low` means the section is still usable but should generally stay reviewable
4. if no rule applies clearly, keep `mapped_task_family = null` and `needs_review = true`

The mapping table should live in the plan and later be mirrored into code comments or test fixtures.

## Output Artifacts

Write all phase-1 outputs under `tmp/lips_manifest/`.

Expected artifacts:
1. `lips_sections_included.jsonl`
2. `lips_sections_excluded.jsonl`
3. `lips_build_report.json`
4. `lips_review_sample.jsonl`
5. `lips_validation_report.json`

Minimum contents:
1. included and excluded counts
2. counts by CEFR
3. counts by raw mode
4. counts by turn-structure flag
5. counts by mapped task family
6. parse-status breakdown
7. exclusion-reason breakdown
8. token-count summary statistics

## Manual Review Protocol

Manual review is required, but it should be adaptive rather than heavy by default.

Review protocol:
1. generate an initial 20-section review sample stratified across CEFR levels and mapped families where possible
2. review whether:
   - monologue filtering was correct
   - `candidate_text_clean` preserved the learner content faithfully
   - the mapped task family is acceptable
3. if agreement is below 75 percent, stop and revise heuristics before expanding
4. if agreement is between 75 and 90 percent, expand the sample to 50 total sections
5. if agreement is at least 90 percent and no new major failure mode appears in the last 10 reviewed sections, a 20 to 30 section sample is acceptable for the first go or no-go call

Agreement target:
1. final manual agreement must be at least 85 percent over the final review set used for the decision

Review procedure notes:
1. the plan assumes one primary reviewer is enough for phase 1
2. if disagreements cluster around one family, add a second reviewer for a small adjudication slice
3. document the recurring disagreement types in the validation report

## Go Or No-Go Gates

Hard gates:
1. at least 200 usable monologue-like sections survive filtering
2. usable sections span at least 3 current task families
3. final manual review agreement is at least 85 percent

Soft target:
1. parse success should target at least 95 percent across attempted sections

Interpretation rule:
1. if the hard gates fail, keep the corpus as analysis-only
2. if the hard gates pass but parse success is materially below 95 percent, keep the corpus as analysis-only until the failure taxonomy is reviewed and the parser is improved
3. do not use the corpus to justify taxonomy changes, scorer changes, benchmark changes, or training until both the hard gates and the parser review are satisfactory

## Testing Plan

Phase 1 should add focused tests, not large corpus fixtures.

Recommended additions:
1. `tests/test_lips_dataset.py`
   - encoding handling
   - section splitting
   - turn extraction
   - parse-status classification
2. `tests/test_lips_manifest_scripts.py`
   - build script smoke test against tiny sample fixtures
   - validation script smoke test
   - report output checks
3. `tests/fixtures/corpora/lips_samples/`
   - tiny hand-curated text files only
   - include both successful and intentionally messy examples

Do not check the corpus itself into git.

## Suggested Module And File Layout

Recommended additions:
1. `lips_dataset.py`
   - section parsing
   - encoding normalization
   - raw metadata extraction
   - cleaning helpers
   - turn-structure sanity classification
2. `scripts/build_lips_manifest.py`
   - build included and excluded JSONL artifacts
   - write the first build report
   - emit a review sample
3. `scripts/validate_lips_manifest.py`
   - evaluate the go or no-go gates
   - summarize counts by CEFR, raw mode, turn structure, and mapped family
   - write the validation report

Implementation note:
1. if the scripts stay tiny, they may share most logic through `lips_dataset.py`
2. do not add more abstraction than needed in phase 1

## Detailed Implementation Sequence

### Milestone 1: Parser And Schema Spike

Goal:
1. confirm that LIPS can be decoded, split, and represented consistently at section level

Work:
1. inspect representative files across CEFR levels
2. finalize the included and excluded record shape
3. implement parsing and cleaning rules in `lips_dataset.py`
4. document the parse-status and exclusion-reason taxonomy in code comments and tests

Exit condition:
1. a small gold fixture set parses deterministically

### Milestone 2: First Full Manifest Build

Goal:
1. generate the first complete included and excluded artifact set from the local LIPS download

Work:
1. run the build script over `/tmp/Corpus LIPS/Corpus LIPS`
2. write included and excluded JSONL outputs
3. write the first build report
4. inspect exclusion reasons and token-count distribution

Exit condition:
1. the first build report is stable enough to review

### Milestone 3: Mapping And Reviewability Pass

Goal:
1. create a trustworthy first mapping pass without pretending the heuristic is perfect

Work:
1. apply the heuristic starter table
2. mark low-confidence or null mappings for review
3. generate the first review sample
4. inspect family coverage before manual review starts

Exit condition:
1. the review sample covers the main families and CEFR slices reasonably

### Milestone 4: QC And Adaptive Manual Review

Goal:
1. determine whether the artifact set is good enough to keep using

Work:
1. review the first 20 sections
2. expand to a larger sample only if agreement is ambiguous
3. revise heuristics if failure modes cluster
4. write the validation report

Exit condition:
1. the validation report states clearly whether the corpus passed the go or no-go gates

### Milestone 5: Decision

If phase 1 passes:
1. keep the artifact set as an approved analysis-grade source
2. decide whether to reuse it for later categorization research or benchmark-design support
3. only then consider phase 2 dialogue work

If phase 1 fails:
1. keep the build outputs for analysis
2. document the blocking reason
3. do not widen the taxonomy or wire the artifacts into runtime flows

## Phase 2 Scope: Dialogue Feasibility Spike

Only start phase 2 if phase 1 passes.

Objective:
1. decide whether the repo really needs a dialogue-specific family or a separate interaction-mode dimension

Work:
1. review 20 to 30 LIPS dialogue sections manually
2. characterize recurring dialogue shapes
3. determine whether one stable family exists or whether there are several incompatible dialogue task types
4. test whether a separate field such as `interaction_mode` or `task_shape` is enough without changing `task_family`

Decision rule:
1. add a new dialogue-facing family only if there are at least 100 high-quality usable dialogue sections
2. require evidence that one coherent family actually exists
3. require evidence that the new family improves analysis or downstream categorization in a measurable way
4. otherwise keep dialogue as raw metadata only

## Phase 3 Scope: DILS Evaluation Or Later Audio Work

Do not start DILS work in phase 1.

Why it is deferred:
1. only 6 local files
2. no transcripts in the local download
3. the recordings are long and dialogue-heavy
4. DILS does not align with the repo's current monologue benchmark shape

Acceptable future uses:
1. small held-out evaluation set for dialogue or collaborative picture-task processing
2. ASR stress test for real L2 conversational Italian
3. later source for a dialogue benchmark only after transcript support or reliable transcription exists

Not recommended yet:
1. using DILS as phase-1 training data
2. using DILS to justify immediate taxonomy changes
3. forcing DILS into `picture_description` or `free_monologue`

## Connection To Existing Task-Family Handling

Phase 1 should not change user-facing task selection.

Instead:
1. reuse the existing task-family vocabulary as the mapped target label space
2. preserve corpus-native facts such as `raw_mode`, `section_id`, and `prompt_topic`
3. keep uncertain mappings nullable instead of guessing
4. keep all phase-1 outputs outside runtime selection logic

This preserves compatibility with the current app and benchmark setup while building better evidence for later decisions.

## Connection To The Coaching Taxonomy

The coaching taxonomy should stay unchanged in phase 1.

LIPS can still help in two ways:
1. provide real learner examples that can be manually compared to current grammar, coherence, and lexical categories
2. expose where spoken disfluencies should be treated as context or cleanup rather than grammar failures

Practical rule:
1. preserve `candidate_text_raw` for future coaching analysis
2. use `candidate_text_clean` only for initial categorization experiments

## Explicitly Out Of Scope In Phase 1

1. adding a new public `task_family`
2. mixing dialogue and monologue sections into one included artifact
3. transcribing DILS
4. changing the current coaching taxonomy enums
5. changing benchmark suites or seeds before LIPS QC succeeds
6. building an end-to-end audio pipeline around these corpora
7. integrating LIPS into runtime scoring
8. exposing LIPS or DILS through `open_corpus_catalog.py`

## Handoff Notes For Future Contributors

When revisiting this plan:
1. do not assume file-level labels are enough; work at section level
2. keep raw corpus facts separate from mapped repo labels
3. prefer `null` plus review over confident-but-wrong relabeling
4. treat dialogue as a later design decision, not an early shortcut
5. if a future contributor wants to add a dialogue family, require evidence from LIPS dialogue review first, not intuition
6. if implementation starts to sprawl beyond one parser module and two thin scripts, stop and verify that phase 1 is still answering the original product question

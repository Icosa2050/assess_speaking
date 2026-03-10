# Coaching Backlog

Last updated: 2026-03-09
Reviewed with: PAL Opus (`anthropic/claude-opus-4.6`)
Status: Milestone 5 implemented on `codex/coaching-backlog-20260309`

## Goal

Turn the current assessment pipeline into a useful coaching system for repeated
theme-based Italian speaking tasks.

Reference scenario:
1. learner speaks for 3 minutes in Italian
2. theme: `Il mio ultimo viaggio all'estero`
3. save audio, transcript, assessment, and coaching output
4. compare progress over time for the same task family

## Product Constraints

1. Keep OpenRouter as the default remote scoring path.
2. Keep the current gate logic:
   - `language_pass`
   - `duration_pass`
   - `topic_pass`
   - `min_words_pass`
3. Keep deterministic metrics as stable trend signals.
4. Do not mix scoring and coaching into one fragile prompt.
5. Do not over-engineer before per-session coaching output is useful.

## Current Base

Already implemented:
1. local audio -> ASR -> deterministic metrics
2. rubric-style LLM scoring
3. hybrid scoring
4. JSON report persistence
5. `reports/history.csv`
6. progress dashboard over existing history

Current metric layer:
1. `duration_sec`
2. `pause_count`
3. `pause_total_sec`
4. `speaking_time_sec`
5. `word_count`
6. `wpm`
7. `fillers`
8. `cohesion_markers`
9. `complexity_index`

## Opus Review Outcomes

Keep:
1. current gates
2. current metric layer
3. current hybrid scoring approach

Add now:
1. fixed taxonomy for grammar/coherence issues
2. `session_id` and `schema_version`
3. separate rubric scoring call and coaching-summary call
4. transcription-confidence caveat in the report

Defer:
1. pronunciation scoring
2. intonation/prosody analysis
3. adaptive task generation
4. multi-turn tutoring chat
5. advanced visual analytics

## Target Session Contract

Every coaching session should carry:
1. `session_id`
2. `schema_version`
3. `speaker_id`
4. `task_type = themed_monologue`
5. `task_family`
6. `theme`
7. `expected_language = it`
8. `target_duration_sec = 180`
9. `prompt_version`
10. `audio_path`
11. `report_path`

Recommended task-family values:
1. `travel_narrative`
2. `personal_experience`
3. `opinion`
4. `picture_description`
5. `comparison`

## Coaching Taxonomy

The LLM must not invent free-text categories for recurring issues. Use a fixed
set of codes and store free-text explanation separately.

### Grammar Error Categories

1. `gender_agreement`
2. `number_agreement`
3. `article_usage`
4. `preposition_choice`
5. `verb_conjugation_present`
6. `verb_conjugation_past`
7. `auxiliary_choice`
8. `tense_consistency`
9. `mood_selection`
10. `word_order`
11. `pronoun_usage`
12. `clitic_placement`
13. `subject_omission_or_redundancy`
14. `lexical_repetition`
15. `false_friend_or_wrong_word_choice`

### Coherence Issue Categories

1. `missing_sequence_markers`
2. `weak_narrative_order`
3. `abrupt_topic_shift`
4. `insufficient_linking`
5. `underdeveloped_detail`
6. `repetition_without_progress`
7. `unclear_reference`

### Lexical Gap Categories

1. `travel_vocabulary_gap`
2. `time_expression_gap`
3. `emotion_description_gap`
4. `comparison_language_gap`
5. `narrative_connector_gap`

## Target Output Model

### 1. Rubric Output

This is the scoring object. It should remain strict, predictable, and easy to
validate.

Add these fields to the rubric schema:
1. `topic_relevance_score`
2. `language_ok`
3. `recurring_grammar_errors`
4. `coherence_issues`
5. `lexical_gaps`
6. `evidence_quotes`
7. `confidence`

Recommended shape:

```json
{
  "fluency": 1,
  "cohesion": 1,
  "accuracy": 1,
  "range": 1,
  "overall": 1,
  "comments_fluency": "string",
  "comments_cohesion": "string",
  "comments_accuracy": "string",
  "comments_range": "string",
  "overall_comment": "string",
  "on_topic": true,
  "topic_relevance_score": 1,
  "language_ok": true,
  "recurring_grammar_errors": [
    {
      "category": "preposition_choice",
      "explanation": "string",
      "examples": ["string"]
    }
  ],
  "coherence_issues": [
    {
      "category": "missing_sequence_markers",
      "explanation": "string",
      "examples": ["string"]
    }
  ],
  "lexical_gaps": [
    {
      "category": "travel_vocabulary_gap",
      "explanation": "string",
      "examples": ["string"]
    }
  ],
  "evidence_quotes": ["string"],
  "confidence": "low"
}
```

### 2. Coaching Summary Output

This is learner-facing and should be produced from validated rubric data plus
metrics, not directly from raw transcript only.

Recommended fields:
1. `strengths`
2. `top_3_priorities`
3. `next_focus`
4. `next_exercise`
5. `coach_summary`

Recommended shape:

```json
{
  "strengths": ["string"],
  "top_3_priorities": ["string", "string", "string"],
  "next_focus": "string",
  "next_exercise": "string",
  "coach_summary": "string"
}
```

Constraint:
1. `next_exercise` must be a practice activity, not a fake internal exercise id
2. keep it grounded in the detected weaknesses

## Data Persistence Plan

### Keep

1. full JSON report in `reports/*.json`
2. `reports/history.csv`

### Extend

Add these columns to `history.csv`:
1. `session_id`
2. `schema_version`
3. `speaker_id`
4. `task_family`
5. `theme`
6. `target_duration_sec`
7. `language_pass`
8. `duration_pass`
9. `topic_pass`
10. `fluency`
11. `cohesion`
12. `accuracy`
13. `range`
14. `overall`
15. `final_score`
16. `band`
17. `requires_human_review`
18. `top_priority_1`
19. `top_priority_2`
20. `top_priority_3`
21. `grammar_error_categories`
22. `coherence_issue_categories`
23. `report_path`

Recommendation:
1. keep `history.csv` for dashboard compatibility
2. add `reports/sessions.jsonl` for richer future analytics if needed

## Progress Model

Compare sessions within the same:
1. `speaker_id`
2. `expected_language`
3. `task_family`
4. target duration band

Primary progress signals:
1. more consistent `duration_pass`
2. fewer pauses and fillers
3. higher `fluency`
4. higher `cohesion`
5. higher `accuracy`
6. fewer recurring grammar categories
7. fewer recurring coherence categories
8. stronger topic relevance

Do not compare all sessions globally without grouping by task family.

## Implementation Backlog

### Milestone 1: Contracts And Taxonomy

Status: implemented

Deliverables:
1. introduce coaching taxonomy constants
2. extend schema versioning/session identity
3. document prompt versions for rubric and coaching

Files:
1. [schemas.py](/Users/bernhard/Development/assess_speaking-codex-v3/schemas.py)
2. [assessment_prompts.py](/Users/bernhard/Development/assess_speaking-codex-v3/assessment_prompts.py)
3. new file: [coaching_taxonomy.py](/Users/bernhard/Development/assess_speaking-codex-v3/coaching_taxonomy.py)
4. [docs/IMPLEMENTATION_PLAN.md](/Users/bernhard/Development/assess_speaking-codex-v3/docs/IMPLEMENTATION_PLAN.md)

Changes:
1. add typed schema objects for rubric issues and coaching summary
2. extend `RubricResult`
3. extend `AssessmentReport` with `session_id`, `schema_version`, and `coaching`
4. define prompt version constants:
   - `rubric_it_v2`
   - `coaching_it_v1`

Tests:
1. [tests/test_schemas.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_schemas.py)
2. [tests/test_assess_speaking.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_assess_speaking.py)

Acceptance:
1. invalid taxonomy category fails validation
2. old v1 rubric payloads are rejected or explicitly migrated
3. report requires `session_id` and `schema_version`

### Milestone 2: Rubric Prompt And Validation

Status: implemented

Deliverables:
1. richer scoring rubric prompt
2. strict JSON validation with one retry
3. model pinning policy for the rubric call

Files:
1. [assessment_prompts.py](/Users/bernhard/Development/assess_speaking-codex-v3/assessment_prompts.py)
2. [llm_client.py](/Users/bernhard/Development/assess_speaking-codex-v3/llm_client.py)
3. [assess_speaking.py](/Users/bernhard/Development/assess_speaking-codex-v3/assess_speaking.py)

Changes:
1. update rubric prompt to request taxonomy-constrained issues
2. pass `theme`, metrics, and transcript as before
3. add transcription caveat in report metadata
4. pin default rubric model for OpenRouter
5. preserve current failure mode:
   - one retry
   - degrade gracefully
   - set `requires_human_review`

Tests:
1. [tests/test_llm_client.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_llm_client.py)
2. [tests/test_integration_openrouter.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_integration_openrouter.py)
3. [tests/test_assess_speaking.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_assess_speaking.py)

Acceptance:
1. rubric payload is schema-valid
2. invalid category names are rejected
3. malformed JSON triggers retry once
4. degraded path still produces a valid report

### Milestone 3: Separate Coaching Call

Status: implemented

Deliverables:
1. a second prompt that turns metrics + validated rubric into learner-facing coaching
2. coaching data embedded into the report

Files:
1. [assessment_prompts.py](/Users/bernhard/Development/assess_speaking-codex-v3/assessment_prompts.py)
2. [llm_client.py](/Users/bernhard/Development/assess_speaking-codex-v3/llm_client.py)
3. [assess_speaking.py](/Users/bernhard/Development/assess_speaking-codex-v3/assess_speaking.py)
4. [feedback.py](/Users/bernhard/Development/assess_speaking-codex-v3/feedback.py)

Changes:
1. add `coaching_prompt_it(...)`
2. add `generate_coaching_summary(...)`
3. call coaching only after rubric validation succeeds
4. if rubric fails, keep current deterministic suggestions as fallback
5. add `strengths`, `top_3_priorities`, `next_focus`, `next_exercise`, `coach_summary`

Tests:
1. new file: [tests/test_coaching_summary.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_coaching_summary.py)
2. [tests/test_assess_speaking.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_assess_speaking.py)

Acceptance:
1. coaching output is generated only from validated rubric data
2. report contains both scoring and coaching sections
3. coaching remains available in a deterministic fallback form if the LLM is unavailable

### Milestone 4: Richer Persistence

Status: implemented

Deliverables:
1. extended history rows
2. optional JSONL session log

Files:
1. [assess_speaking.py](/Users/bernhard/Development/assess_speaking-codex-v3/assess_speaking.py)
2. [scripts/progress_dashboard.py](/Users/bernhard/Development/assess_speaking-codex-v3/scripts/progress_dashboard.py)
3. [README.md](/Users/bernhard/Development/assess_speaking-codex-v3/README.md)

Changes:
1. extend `append_history(...)`
2. include rubric subscores and coaching priorities in the saved row
3. add `speaker_id` CLI or config support
4. optionally append a full row to `reports/sessions.jsonl`

Tests:
1. [tests/test_assess_speaking.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_assess_speaking.py)
2. [tests/test_progress_dashboard.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_progress_dashboard.py)

Acceptance:
1. history rows are backward-readable
2. new fields populate correctly
3. dashboard still renders old and new histories

### Milestone 5: Progress Aggregation By Task Family

Status: implemented

Deliverables:
1. progress summaries grouped by task family
2. repeated-issue detection from saved taxonomy categories

Files:
1. [scripts/progress_dashboard.py](/Users/bernhard/Development/assess_speaking-codex-v3/scripts/progress_dashboard.py)
2. new file: [progress_analysis.py](/Users/bernhard/Development/assess_speaking-codex-v3/progress_analysis.py)

Changes:
1. add same-task-family filtering
2. compute recurring issue counts
3. show latest priorities vs previous priorities

Tests:
1. new file: [tests/test_progress_analysis.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_progress_analysis.py)
2. [tests/test_progress_dashboard.py](/Users/bernhard/Development/assess_speaking-codex-v3/tests/test_progress_dashboard.py)

Acceptance:
1. repeated issue counts are stable
2. dashboard can distinguish travel-narrative progress from unrelated tasks

## Scoring Policy

Scoring inputs:
1. deterministic metrics
2. rubric subscores
3. existing gates

Coaching-only outputs:
1. recurring issue lists
2. lexical gaps
3. priorities
4. next exercise
5. coach summary

Do not silently turn coaching-only fields into scoring inputs without an explicit
policy change.

## Reporting Requirements

Each report should explicitly tell the learner:
1. whether they passed the task contract
2. what they did well
3. what repeats as a problem
4. what to improve next
5. what to do in the next recording

Each report should also explicitly note:
1. assessment is based on automatic transcription
2. low-confidence transcription may distort some coaching details

## Suggested Sequencing

Recommended order:
1. Milestone 1
2. Milestone 2
3. Milestone 3
4. Milestone 4
5. Milestone 5

Reason:
1. taxonomy and schema contracts are cheap now and expensive later
2. scoring reliability should be improved before coaching copy
3. persistence should follow the final report shape
4. cross-session analysis only makes sense after stable saved categories exist

## Explicit Deferrals

Do not start these in the first coaching pass:
1. pronunciation scoring
2. accent quality evaluation
3. prosody or intonation analysis
4. adaptive theme generation
5. interactive tutor chat
6. exercise-bank recommendation engine

## Definition Of Done For Coaching V1

1. a 3-minute themed Italian task can be assessed end-to-end
2. report contains both scoring and coaching
3. recurring grammar and coherence issues use a fixed taxonomy
4. report persistence includes enough fields for trend analysis
5. timeline can show progress within a task family
6. OpenRouter path remains schema-validated and robust

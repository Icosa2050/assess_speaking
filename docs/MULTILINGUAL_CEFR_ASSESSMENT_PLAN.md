# Multilingual CEFR Assessment Plan

Last updated: 2026-03-13
Status: Research-backed implementation plan

## Goal

Replace the current hand-tuned hybrid score with a more defensible multilingual oral-assessment architecture that:

1. aligns better with established oral assessment standards
2. remains explainable and testable
3. supports English and Italian first
4. extends to German, French, and Spanish next
5. uses synthetic benchmark suites for software regression
6. uses real human-rated learner audio later for calibration

This plan is for formative assessment software. It is not a claim of high-stakes certification validity.

## Consensus Snapshot

Consensus across PAL Gemini, PAL Opus, and PAL Qwen:

1. Do not score directly to `B1/B2/C1/C2`.
2. Score interpretable dimensions first, then map those subscores to CEFR.
3. Keep the LLM as one signal, mainly for discourse/task judgments and evidence extraction, not as the backbone scorer.
4. Synthetic benchmark audio/text is valuable for regression and monotonicity testing, not for calibration.
5. Real human-rated learner audio is required for credible per-language calibration.
6. Language-specific norms, lexicons, and weights must live in configurable language profiles.

The three models differed mostly in implementation preference:

1. Gemini favored simple interpretable aggregation such as ordinal logistic regression.
2. Opus emphasized a strict construct-first pipeline and warned against direct CEFR prediction.
3. Qwen allowed stronger regression models later, but still recommended subscore-first architecture and empirical CEFR calibration.

## Standards Direction

Use CEFR-like spoken production constructs as the design anchor, not a single opaque score.

First serious version should score these monologue-focused dimensions:

1. `fluency`
2. `pronunciation_intelligibility`
3. `grammar`
4. `lexicon`
5. `coherence`
6. `task_fulfillment`

Do not include `interaction` in the first scoring model unless the task is actually interactive.

## Product Scope

Languages, phase order:

1. English
2. Italian
3. German
4. French
5. Spanish

Task scope, first calibration:

1. spoken monologue only
2. fixed task families
3. CEFR targets `B1`, `B2`, `C1`, `C2`

## Architecture

Recommended pipeline:

```text
Audio
  -> quality and validity checks
  -> ASR + alignment + confidence
  -> acoustic/prosodic features
  -> transcript-based linguistic features
  -> per-dimension scorers
  -> calibrated aggregate score
  -> CEFR mapping + confidence + evidence
```

### 1. Validity Layer

This layer decides whether an attempt is scoreable at all.

Initial checks:

1. language ID
2. minimum duration
3. minimum speech content
4. ASR confidence floor
5. audio quality / clipping / silence checks

Planned later:

1. synthetic / TTS suspicion
2. read-aloud / scripted-delivery suspicion
3. severe ASR mismatch gating

### 2. Feature Layer

Split features into two groups.

Language-agnostic features:

1. articulation rate / speaking rate
2. pause frequency and duration
3. repair behavior and filled pauses
4. utterance length and clause density
5. lexical diversity metrics
6. discourse segmentation features
7. task relevance embeddings

Language-specific features:

1. filler lexicons
2. discourse marker inventories
3. grammar error detectors
4. CEFR vocabulary profiles
5. pronunciation / phonology expectations
6. language-specific normalization priors

### 3. Dimension Scorers

Each dimension gets its own scorer and evidence.

Suggested v1 outputs per dimension:

1. `score_raw`
2. `score_normalized`
3. `evidence`
4. `warnings`
5. `feature_values`

Scorer design for v1:

1. start with interpretable rules plus small calibrated models
2. keep every scorer inspectable
3. avoid end-to-end black-box CEFR prediction

### 4. Aggregation Layer

Do not compute the final score as one fixed global formula for all languages.

Instead:

1. compute dimension subscores first
2. aggregate into a continuous overall score per language
3. map that continuous score to CEFR using calibrated cut scores

Preferred v1 aggregator:

1. monotonic linear or ordinal logistic model per language
2. explicit weights stored in language profiles
3. confidence output based on score margin and data quality

## Language Profiles

Create a `LanguageProfile` abstraction.

Required fields:

1. `code`
2. `label`
3. `asr_language_code`
4. `supported_task_families`
5. `cefr_levels`
6. `filler_inventory`
7. `discourse_markers`
8. `vocabulary_profile_source`
9. `grammar_ruleset`
10. `pace_norms`
11. `pause_norms`
12. `dimension_weights`
13. `cefr_cut_scores`
14. `quality_thresholds`

Important rule:

Language profiles are configuration, not forked scoring code.

## Recommended Scoring Dimensions

### Fluency

Use:

1. articulation rate
2. pause ratio
3. mean run length
4. repair frequency
5. filler density

Do not treat higher speed as automatically better. Normalize by language and later by task family.

### Pronunciation / Intelligibility

Start practical, not perfect:

1. ASR confidence and stability
2. alignment confidence
3. prosodic stability measures

Later:

1. phoneme-level or phone-class scoring per language

### Grammar

Use:

1. error density
2. error severity buckets
3. clause complexity
4. tense/aspect agreement patterns
5. morphology-sensitive rules for richer languages

### Lexicon

Use:

1. lexical diversity
2. lexical sophistication
3. CEFR vocabulary profiling
4. precision / misuse indicators

### Coherence

Use:

1. discourse markers
2. structural progression
3. local coherence between utterances
4. topic maintenance

### Task Fulfillment

Use:

1. prompt relevance
2. task coverage
3. response completeness

The LLM can help here, but the prompt must be strict and evidence-based.

## Role Of The LLM

Keep the LLM, but narrow its role.

Recommended LLM responsibilities:

1. discourse / coherence judgment
2. task fulfillment judgment
3. grammar issue extraction with evidence
4. lexical issue extraction with evidence
5. learner-facing coaching

Do not let the LLM be the sole or dominant final score source in the long-term architecture.

## CEFR Mapping Strategy

The system should output:

1. dimension subscores
2. continuous overall score
3. CEFR estimate
4. confidence
5. evidence / reasons

Do not map raw features directly to CEFR.

Preferred mapping order:

1. features -> dimension subscores
2. dimension subscores -> continuous aggregate
3. continuous aggregate -> CEFR with per-language cut scores

## Synthetic Benchmark Strategy

Synthetic benchmarks are mandatory for engineering quality.

Use them for:

1. regression testing
2. monotonicity tests
3. feature extraction tests
4. edge-case tests
5. stress tests for prompts, noise, pace, and fillers

Do not use them for:

1. threshold calibration
2. weight calibration
3. validity claims

Recommended benchmark grid:

1. languages: `en`, `it`, later `de`, `fr`, `es`
2. levels: `B1`, `B2`, `C1`, `C2`
3. task families: at least `personal_experience`, `opinion_monologue`, `travel_narrative`
4. variants:
   - clean strong exemplar
   - borderline exemplar
   - over-fast exemplar
   - low-cohesion exemplar
   - grammar-heavy-error exemplar
   - TTS / scripted exemplar

## Human Rating Strategy

This is the minimum credible path to calibration.

1. collect real learner audio per language and CEFR band
2. use trained human raters
3. rate both overall level and dimensions
4. measure inter-rater reliability
5. calibrate language-specific weights and cut scores from those ratings

Minimum viable target:

1. English first
2. then Italian
3. at least double-rated subsets

## Validation Plan

Minimum credible validation for formative use:

1. inter-rater reliability on human ratings
2. exact and adjacent agreement between machine CEFR and human CEFR
3. correlation between machine subscores and human dimension ratings
4. robustness under ASR perturbations
5. fairness checks by learner subgroup when data volume permits

Suggested acceptance goals for a serious prototype:

1. adjacent agreement >= 90%
2. exact agreement >= 60%
3. weighted kappa >= 0.70
4. dimension correlations >= 0.60 for core constructs

## Repo-Level Implementation Plan

### Phase 0: Freeze And Instrument

Goal:

1. keep current scorer runnable
2. add instrumentation so we can compare old and new outputs

Changes:

1. preserve current [scoring.py](../scoring.py) as legacy baseline
2. extend report payloads with scorer version fields
3. log enough feature values for future calibration

### Phase 1: Language Profiles

Add:

1. `language_profiles.py`
2. profiles for `en` and `it`

Each profile should define:

1. filler lists
2. discourse markers
3. provisional pace priors
4. provisional pause priors
5. provisional weights
6. provisional cut scores

### Phase 2: Feature Extraction Refactor

Add or refactor into:

1. `fluency_features.py`
2. `lexical_features.py`
3. `grammar_features.py`
4. `discourse_features.py`
5. `quality_checks.py`

Keep:

1. audio timing and ASR hooks in existing modules where possible

### Phase 3: Dimension Scorers

Add:

1. `dimension_scoring.py`

Responsibilities:

1. compute subscores per construct
2. return feature-backed evidence
3. remain deterministic where possible

### Phase 4: Aggregation And CEFR Mapping

Replace the current fixed blend in [scoring.py](../scoring.py) with:

1. `aggregate_dimension_scores(...)`
2. `map_to_cefr(...)`
3. `score_confidence(...)`

Keep the old scorer available under a legacy path during transition.

### Phase 5: Synthetic Regression Corpus

Add:

1. `tests/fixtures/benchmarks/`
2. per-language benchmark manifests
3. expected ordering and tolerance rules

Add tests for:

1. monotonic level ordering
2. scorer stability across refactors
3. per-dimension regression
4. invalid / TTS-like sample behavior

### Phase 6: Human Calibration Workflow

Add:

1. `calibration/README.md`
2. `calibration/schema.json`
3. `calibration/train_cut_scores.py`
4. `calibration/evaluate_agreement.py`

Goal:

1. fit per-language weights
2. fit per-language CEFR cut scores
3. export versioned calibration artifacts

### Phase 7: UI Exposure

Expose in the app:

1. dimension subscores
2. CEFR estimate
3. confidence
4. score rationale
5. scorer version

Optional:

1. allow experimental creation of a new language profile
2. require explicit manual input of:
   - fillers
   - discourse markers
   - pace priors
   - provisional weights
   - provisional cut scores
3. mark such languages as `uncalibrated`

## Immediate Starting Point

Start here:

1. English and Italian only
2. monologue tasks only
3. B1/B2/C1/C2 only
4. subscore-first architecture
5. synthetic regression corpus in CI
6. human calibration deferred but planned from the start

This is the best starting point that is both practical and consistent with the literature and model consensus.

## Open Decisions

1. which ASR confidence signals to trust for intelligibility
2. which grammar tooling is realistic per language in v1
3. whether `task_fulfillment` should stay partly LLM-based in v1
4. how aggressively to gate suspected TTS / scripted speech
5. whether C2 should be exposed immediately in the product UI or first only in the calibration pipeline

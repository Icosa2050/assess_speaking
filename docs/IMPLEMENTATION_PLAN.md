# Assessment Core Plan

Last updated: 2026-03-07
Status: Phase 1 implemented on `codex/openrouter-hardening-20260307`

## Goal

Train a learner to speak more fluently in Italian on a given theme for a target duration, using OpenRouter as the primary scoring path and keeping the architecture extensible.

## Product Direction

Keep:
1. local ASR and audio metrics
2. prompt/audio assets
3. simple report persistence and `history.csv`

Port from the earlier OpenRouter branch:
1. provider abstraction
2. schema-validated rubric parsing
3. deterministic plus LLM hybrid scoring
4. language, duration, and topic gates
5. explicit degraded-state handling

Defer:
1. Telegram/Redis service as a product focus
2. LMS expansion beyond compatibility
3. dashboard polish as a core requirement

## Phase 1 Scope

1. Add modular core files:
   - `audio_features.py`
   - `asr.py`
   - `metrics.py`
   - `llm_client.py`
   - `assessment_prompts.py`
   - `schemas.py`
   - `scoring.py`
   - `settings.py`
2. Keep `assess_speaking.py` as the orchestration and compatibility layer.
3. Keep existing top-level CLI JSON fields for scripts and service callers.
4. Add nested `report` as the new stable contract.

## Phase 1 Output Contract

Legacy top-level fields:
1. `metrics`
2. `transcript_preview`
3. `llm_rubric`
4. optional `baseline_comparison`
5. optional `suggested_training`

New nested `report`:
1. `input`
2. `metrics`
3. `checks`
4. `scores`
5. `rubric`
6. `warnings`
7. `errors`
8. `requires_human_review`
9. `timings_ms`

## Implemented Gate Logic

1. `language_pass`
2. `duration_pass`
3. `min_words_pass`
4. `topic_pass`
5. `requires_human_review` when the LLM path is unavailable or the language gate fails

## Provider Policy

1. CLI default: OpenRouter
2. Legacy/local compatibility: infer Ollama when `--llm` is used
3. Programmatic compatibility: callers passing local model names still resolve to Ollama unless they set `provider="openrouter"`

## Verification

Executed on this branch:
1. `python3 -m unittest tests.test_assess_speaking tests.test_schemas tests.test_scoring tests.test_llm_client tests.test_asr -v`
2. `/Users/bernhard/Development/assess_speaking/.venv/bin/python -m unittest discover -s tests -v`
3. `RUN_OPENROUTER_INTEGRATION=1 /Users/bernhard/Development/assess_speaking/.venv/bin/python -m unittest tests.test_integration_openrouter -v`

Current status:
1. full repo suite passes
2. OpenRouter integration passes
3. sample audio integration remains environment-gated by Whisper model availability

## Next Useful Work

1. calibrate pause heuristics against real Italian recordings
2. persist richer longitudinal progress summaries
3. add prompt packs and training loops on top of the new `report` contract
4. implement the coaching roadmap in [docs/COACHING_BACKLOG.md](../docs/COACHING_BACKLOG.md)

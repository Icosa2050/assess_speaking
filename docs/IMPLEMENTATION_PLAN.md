# Assessment Core And Local App Plan

Last updated: 2026-04-21
Status: Assessment core implemented, app shell moved to a local-first desktop-ready flow, local backend and app-data baseline implemented, support and maintenance planning documented, and the next product migration is now defined

## Goal

Train a solo learner to speak more fluently in Italian on a given theme for a
target duration, using OpenRouter as the primary remote scoring path while
keeping the local app shell packaging-friendly.

## Product Direction

Keep:
1. local ASR and audio metrics
2. prompt/audio assets
3. simple report persistence and `history.csv`
4. local-first app shell with saved runtime connections
5. cross-platform app-data and cache abstraction

Port from the earlier OpenRouter branch:
1. provider abstraction
2. schema-validated rubric parsing
3. deterministic plus LLM hybrid scoring
4. language, duration, and topic gates
5. explicit degraded-state handling

Current ASR runtime direction:
1. keep `faster-whisper` as the default provider today
2. route ASR through a provider/capability layer in `assessment_runtime/asr.py`
3. support explicit file strategies: `auto`, `native`, and `chunked`
4. preserve one merged transcript contract with word timestamps even when chunked fallback is used
5. keep pause-feature extraction and assessment scoring unchanged while making room for future non-Whisper providers

Planned follow-on:
1. shared React frontend for desktop and hosted delivery
2. hosted auth, tenancy, and server-side job orchestration
3. Tauri-based desktop packaging after the shared frontend replaces Streamlit

Related planning docs:
1. `docs/LOCAL_BACKEND_ARCHITECTURE.md`
2. `docs/MOBILE_COMPANION_STRATEGY.md`
3. `docs/SUPPORT_MAINTENANCE_PLAN.md`
4. `docs/SUPPORT_MAINTENANCE_IMPLEMENTATION_PLAN.md`
5. `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md`
6. `docs/PUBLIC_INFERENCE_ARCHITECTURE_PLAN.md`
7. `docs/SAVED_CONNECTIONS_PRODUCTION_CREDENTIAL_PLAN.md`
8. `docs/LOCAL_DESKTOP_UX_REFACTORING_PLAN.md`

## Desktop-Baseline Runtime Rules

For the current desktop baseline:
1. local desktop mode stays localhost-bound and auth-free in guest mode
2. runtime metadata should converge on one shared shape owned by
   `app_shell/bootstrap.py` with `deployment_mode`, `launch_mode`,
   `packaging_safe`, and `auth_mode`
3. support and maintenance endpoints are local-desktop extensions to the
   canonical product API, not a separate product direction
4. macOS and Windows are the first-class packaging targets for this pass, while
   Linux remains a no-regression target

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
2. Keep `assess_speaking.py` as the orchestration entrypoint.
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
2. Local Ollama runs must set `--provider ollama` explicitly
3. `--llm-model` selects the model for the chosen provider and no longer changes providers implicitly

## Verification

Executed on this branch:
1. `./scripts/python.sh -m unittest tests.test_assess_speaking tests.test_schemas tests.test_scoring tests.test_llm_client tests.test_asr -v`
2. `./scripts/run_tests.sh -v`
3. `RUN_OPENROUTER_INTEGRATION=1 ./scripts/python.sh -m unittest tests.test_integration_openrouter -v`

Current status:
1. full repo suite passes
2. OpenRouter integration passes
3. sample audio integration remains environment-gated by Whisper model availability

## Next Useful Work

1. add backend-owned support/export bundle generation with privacy-safe defaults
2. add backend maintenance APIs for storage summary and safe cleanup actions
3. add Settings-based `Troubleshooting & Support` controls after PAL review
4. add cleanup and retention policies for `tmp/`, support bundles, stale jobs, and rotated logs
5. make the launcher/backend runtime contract explicitly packaging-safe with
   shared runtime metadata for macOS and Windows first-class delivery and
   Linux no-regression support
6. refine the learner-facing review and history coaching surfaces on top of the job-based backend state
7. calibrate pause heuristics against real Italian recordings
8. add prompt packs and training loops on top of the new `report` contract
9. execute `docs/DESKTOP_HOSTED_PRODUCT_PLAN.md` in phases so the shared
   frontend lands before hosted multi-user backend work
10. if we add another ASR backend, implement it behind the existing ASR provider/capability layer instead of branching assessment codepaths

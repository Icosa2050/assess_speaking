# Repo Cleanup Plan

This document captures the current cleanup strategy for the repository root and
the agreed migration order for reducing top-level Python-module sprawl without
breaking the current CLI, Streamlit shell, tests, scripts, or CI.

## Why This Exists

The repository currently mixes:

- stable root entrypoints such as `assess_speaking.py` and `streamlit_app.py`
- already-packaged areas such as `app_shell/`, `pages/`, and `service/`
- many root-level library modules that grew around the original CLI shape

That shape works, but it makes the root noisy, encourages more top-level files,
and makes package boundaries harder to reason about.

## Non-Negotiable Guardrails

- Keep `assess_speaking.py` stable at the root during cleanup.
- Keep `streamlit_app.py` stable at the root during cleanup.
- Do not do a big-bang `src/` migration while scripts, tests, docs, and CI still
  import root modules directly.
- Prefer small, reversible moves with validation after every tranche.
- Use compatibility shims only when they are necessary, pure re-exports, and
  explicitly temporary.

## Current Coupling Snapshot

Current root inventory:

- 29 root-level Python modules plus `streamlit_app.py`

Root dependency hotspots from `scripts/root_import_audit.py`:

- Highest outbound module: `assess_speaking.py`
- Highest inbound hubs: `language_profiles.py`, `schemas.py`
- Benchmark/synthetic files form a reasonably cohesive cluster
- Corpus/data files form a separate cluster with lighter coupling

This means the cleanup sequence should not start by moving arbitrary clusters.
It should start with the least-coupled foundation modules and only then move
dependent clusters.

## Approved Cleanup Sequence

### Tranche 1: Hygiene And Audit

Goals:

- reduce obvious root junk
- stop new root sprawl from slipping in silently
- make the current dependency picture visible

Scope:

- tighten `.gitignore`
- add a root-layout check to `scripts/repo_quality_audit.py`
- add `scripts/root_import_audit.py`
- document the plan in this file

### Tranche 2: Foundation Extraction

Move the lowest-coupling modules into a dedicated package such as
`assess_core/`, starting with files that are mostly depended on rather than
doing a lot of depending themselves.

First candidates:

- `coaching_taxonomy.py`
- `settings.py`
- `language_profiles.py`
- `schemas.py`

Why these first:

- they are shared building blocks
- they lower the risk of later package moves needing to import back upward from
  new packages into the root namespace

### Tranche 3: Corpus And Data Tooling

After the foundation package exists, move the corpus/data tooling into a
package such as `corpora/` or `data_pipeline/`.

Primary candidates:

- `celi_harvest.py`
- `celi_wordlists.py`
- `lips_dataset.py`
- `rita_dataset.py`
- `open_corpus_catalog.py`

### Tranche 4: Benchmarking And Synthetic Evaluation

Once foundation imports are stable, move the benchmark and synthetic-evaluation
cluster into a package such as `benchmarking/`.

Primary candidates:

- `benchmark_suites.py`
- `calibration_evaluation.py`
- `calibration_manifests.py`
- `synthetic_audio_contracts.py`
- `synthetic_benchmark_evaluation.py`
- `synthetic_benchmark_generation.py`
- `synthetic_benchmark_regression.py`
- `synthetic_seed_manifests.py`

### Tranche 5: Core Assessment Modules

Leave the most coupled assessment modules for last, after the packages above
have already reduced root clutter and import pressure.

Likely final wave:

- `asr.py`
- `assessment_prompts.py`
- `audio_features.py`
- `dimension_scoring.py`
- `feedback.py`
- `llm_client.py`
- `lms.py`
- `metrics.py`
- `progress_analysis.py`
- `scoring.py`
- `theme_library.py`

## Shim Policy

- A shim must only re-export from the new package.
- A shim must not contain business logic.
- Prefer updating direct import sites when the blast radius is small.
- Track shim removal explicitly in the tranche that introduces it.

## Validation Gates

Run these after each tranche:

- `./scripts/run_tests.sh`
- `./scripts/run_e2e.sh`
- smoke launch `streamlit_app.py`
- smoke run `assess_speaking.py`
- if relevant credentials or env are available, run the opt-in integration paths

## Tranche 1 Deliverables

- `docs/REPO_CLEANUP_PLAN.md`
- stronger root and artifact ignore rules
- root-layout findings in `scripts/repo_quality_audit.py`
- `scripts/root_import_audit.py`

## Commands

Inspect current root coupling:

```bash
./scripts/python.sh scripts/root_import_audit.py
```

Run the structural audit without coverage:

```bash
./scripts/python.sh scripts/repo_quality_audit.py --coverage-mode skip
```

## Root Policy During Migration

Until the cleanup is further along, new root-level Python files should be
treated as exceptions. If a new top-level module is necessary, document why it
cannot live in an existing package and add it to the approved inventory on
purpose rather than by accident.

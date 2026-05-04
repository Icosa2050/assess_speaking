# Vostavo

![Vostavo logo](branding/vostavo-logo-playful.svg)

OpenRouter-first assessment core and local speaking coach app.

Internal repo and module names still use `assess_speaking` in places for compatibility.

Pipeline: **Transcription (faster-whisper)** -> **Deterministic metrics** -> **schema-validated CEFR-style rubric** via **OpenRouter** (default), **Ollama** (local), or a generic **OpenAI-compatible** endpoint.

Current CLI/runtime contract:

1. OpenRouter as the default remote scoring path.
2. Local Ollama support through `--provider ollama --llm-model ...`.
3. Structured nested `report` output with validated:
   - `input`
   - `metrics`
   - `checks`
   - `scores`
   - `rubric`
   - `requires_human_review`
4. Goal-oriented gates for:
   - language match
   - topic relevance
   - speaking duration
   - minimum word count

## 0) Prerequisites
```bash
brew install ffmpeg
```

Optional local LLM mode:
```bash
brew install ollama
ollama pull llama3.1
ollama list
```

Remote LLM mode (default):
```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL="google/gemini-3.1-pro-preview"
```

## 1) Virtual environment (Python ≥ 3.11)
```bash
./scripts/setup_env.sh              # prefers python3.12 → python3.11 → python3
source .venv/bin/activate
```

You can pass a custom target directory or interpreter, e.g.
`PYTHON_BIN=/path/to/python3.12 ./scripts/setup_env.sh`. The script installs all
requirements inside `.venv`, leaving the global Python untouched. PyPI provides
macOS wheels for `av`, `ctranslate2`, `onnxruntime`, `praat-parselmouth`, and
`rapidfuzz` on Python 3.13 (verified on macOS 15/Sequoia, Oct 2025).

If you do not want to activate the venv manually, use the repo-local launcher:
`./scripts/python.sh ...`

## 2) Beginner app quick start
Recommended first path:

1. Install `ffmpeg`.
2. Optionally install `ollama` and pull a local model such as `llama3.1`.
3. Launch the app with `./scripts/run_app.py`.
4. On the Home screen, choose `Set up local AI`.
5. In `Runtime Setup`, keep the default local path unless you specifically need advanced provider options.
6. Continue with `Session Setup -> Speak -> Review -> History`.

The app now starts a localhost-only backend automatically and keeps user data under the app-data root instead of depending on the repo working directory.

## 3) Sample audio library
```bash
./scripts/generate_sample.sh --all
./scripts/generate_sample.sh --language it --cefr B2
```

Shipped sample responses now live under `samples/cefr/{it,en}/{B1,B2,C1}/`:

- `samples/cefr/it/B1/travel_story.wav`
- `samples/cefr/it/B2/remote_work.wav`
- `samples/cefr/it/C1/public_debate.wav`
- `samples/cefr/en/B1/travel_story.wav`
- `samples/cefr/en/B2/remote_work.wav`
- `samples/cefr/en/C1/public_debate.wav`

Optional flags `-v/--voice`, `-t/--text`, `-o/--output`, `--language`, and `--cefr`, e.g.
```bash
./scripts/generate_sample.sh --language en --cefr C1 --output /tmp/c1-public-debate.wav
./scripts/generate_sample.sh --voice "Alice" --text "Questo e un test." --output /tmp/test.wav
```

## 4) Check models & self-test
```bash
python assess_speaking.py --list-ollama
python assess_speaking.py --selftest --provider openrouter --llm-model google/gemini-3.1-pro-preview
python assess_speaking.py --selftest --provider ollama --llm-model llama3.1
```

## 5) Run an assessment
```bash
python assess_speaking.py sample.wav \
  --provider openrouter \
  --llm-model google/gemini-3.1-pro-preview \
  --theme "la mia città" \
  --target-duration-sec 120 \
  --llm-timeout 30 > report.json
```

Local Ollama mode:
```bash
python assess_speaking.py sample.wav --provider ollama --llm-model llama3.1 > report.json
cat report.json
```

Every run is also stored in `reports/` (structured JSON + `history.csv`). Use
`--label "B1-test"` or `--notes "Morning session"` to tag a run. With
`--log-dir path/to/reports` you control the destination, `--no-log` disables the
persistence layer.

If an existing `history.csv` was created with an older header, delete or replace
it before appending new runs. The CLI no longer rewrites legacy history files.

Top-level CLI output remains backward-compatible for existing scripts:

1. `metrics`
2. `transcript_preview`
3. `llm_rubric`
4. optional `baseline_comparison`
5. optional `suggested_training`

New code should read the nested `report` object. It contains the validated
assessment contract, including `checks`, `scores`, `rubric`,
`requires_human_review`, and `progress_delta` when an earlier run exists for
the same speaker and task family.

### Dashboard / history view
```bash
python scripts/progress_dashboard.py --log-dir reports
python scripts/progress_dashboard.py --log-dir reports --export-html reports/dashboard.html
python scripts/progress_dashboard.py --log-dir reports --speaker-id bern --task-family travel_narrative
open reports/dashboard.html  # macOS preview
```

The CLI dashboard renders the history table (via `rich`) and can export an HTML
snapshot. It also supports speaker and task-family filters so progress on
`travel_narrative` is not mixed with unrelated speaking tasks.

### Primary multipage app shell

The primary product-facing UI is now the multipage app shell. The preferred
launcher is the local bootstrap wrapper:

```bash
./scripts/python.sh scripts/run_app.py
./scripts/python.sh scripts/run_app.py --check
```

`streamlit run streamlit_app.py` remains a supported fallback while the shared
React frontend completes the desktop replacement lane. The launcher is better
for local product use and desktop packaging because it resolves stable app-data
paths, bootstraps cache defaults, and can be run from outside the repo root.

The learner-facing flow is now optimized for a single local user:

1. `Home`
2. `Runtime Setup` only when no active connection is configured
3. `Session Setup`
4. `Speak`
5. `Review`
6. `History`

`Library`, `Settings`, and `Scoring Guide` stay available as secondary screens.
The Home screen now also renders startup diagnostics for app-data writability,
`ffmpeg`, Whisper cache readiness, runtime configuration, and recorder readiness.

The app shell stores its local data outside the repo by default:

- macOS app data: `~/Library/Application Support/Vostavo`
- macOS cache: `~/Library/Caches/Vostavo`
- Windows app data and cache now resolve through `platformdirs` using the app identity `frommherz_it` + `Vostavo`
- Ubuntu/Linux app data: `~/.local/share/Vostavo` by default, or `$XDG_DATA_HOME/Vostavo` when XDG overrides are set
- Ubuntu/Linux cache: `~/.cache/Vostavo` by default, or `$XDG_CACHE_HOME/Vostavo` when XDG overrides are set
- Override roots with `VOSTAVO_HOME` and `VOSTAVO_CACHE_HOME`
- Legacy roots and env vars from `Speaking Studio` remain supported for existing installs
- Repo-local app data is no longer used as a default on any platform. It remains available only through explicit developer overrides such as `VOSTAVO_HOME`, `SPEAKING_STUDIO_HOME`, `--app-data-dir`, or `--cache-dir`.

The current app-data layout keeps one per-user root and separates user outputs
from backend state inside it:

- app-data root:
  - `backend_state.json`
  - `jobs/`
  - `logs/`
  - `reports/`
  - `recordings/`
  - `uploads/`
  - `tmp/`
- cache root:
  - `whisper/`
  - `huggingface/`

Legacy `reports/jobs/` directories are migrated into the root-level `jobs/`
directory on backend startup when the new location is still empty.

The wrapper writes reports/history under the app-data root by default, so the
UI keeps working even when launched from an arbitrary working directory.

Settings includes a `Troubleshooting and support` section for local desktop
maintenance:

- `Saved Connections`, `Runtime Defaults`, and `Runtime Setup` now read and
  persist through the localhost runtime-management API rather than browser
  `localStorage`, so saved connection state, secret presence, UI locale, and
  Whisper model choice stay consistent across the shared frontend.
- `Refresh storage usage` reads the backend storage summary without scanning from
  the Streamlit process.
- `Preview cleanup` and `Run safe cleanup` target temporary files, expired
  support bundles, stale job metadata, and rotated logs. Reports, recordings,
  and uploaded audio are preserved.
- `Create support bundle` creates a local ZIP with sanitized app state,
  diagnostics, storage summary, backend logs, and recent sanitized job metadata.
  Saved reports, recordings, uploads, and live runtime health checks are excluded
  by default and require explicit opt-in.

Support bundles fully remove `secret_ref` fields and redact secret-looking
values. Credential state is represented by sanitized booleans such as whether a
saved secret exists, never by copying keys or secret references into the bundle.

### Tests & CI
- **Unit tests**: `./scripts/run_tests.sh`
- **Source coverage**: `./scripts/run_coverage.sh`
- **Full coverage (including tests)**: `./scripts/run_coverage.sh --full`
- **Shared-frontend browser lanes**:
  `cd frontend && npm install --package-lock=false && npx --yes playwright install chromium && npx --yes playwright test -c playwright.config.ts`
- Focused local-guest smoke:
  `cd frontend && npx --yes playwright test -c playwright.config.ts tests/e2e/smokeLocalGuest.spec.ts`
- Full local-guest regression:
  `cd frontend && npx --yes playwright test -c playwright.config.ts tests/e2e/smokeLocalGuest.spec.ts tests/e2e/localGuestFlow.spec.ts tests/e2e/settingsSupportFlow.spec.ts`
- The Playwright config starts the shared frontend on `127.0.0.1:4173` and the
  localhost-only backend on `127.0.0.1:8800`, so the browser lanes exercise the
  same local guest desktop stack the Tauri shell uses.
- The test and coverage wrappers always use the repo-local `.venv` via
  `./scripts/python.sh`, so they stay consistent even when a global `pytest` or
  `coverage` installation points at a different Python.
- Coverage outputs:
  - source mode: `coverage.json` + `htmlcov/`
  - full mode: `coverage.full.json` + `htmlcov-full/`
- **OpenRouter integration (opt-in)**:
  `RUN_OPENROUTER_INTEGRATION=1 ./scripts/python.sh -m unittest tests.test_integration_openrouter -v`
- **Optional sample-audio integration test (no microphone required)**:
  `RUN_AUDIO_INTEGRATION=1 WHISPER_MODEL=tiny ./scripts/python.sh -m unittest tests.test_sample_integration`

The shared-frontend lanes boot a clean localhost backend on port `8800` and the
Vite frontend on port `4173`. The smoke lane covers Home -> Runtime Setup; the
full local-guest regression also covers Home -> History -> Settings,
origin-preserving Settings return/setup navigation, and support bundles with
the live runtime-health opt-in.

- **Self-hosted real-ASR lane**:
  `.github/workflows/real-asr-selfhosted.yml` runs the sample-audio integration on a
  self-hosted Apple Silicon runner with labels `self-hosted`, `macOS`, `ARM64`,
  `icosa-apple-ci`, `assess-speaking`. It warms the `faster-whisper` model cache first so the runner
  keeps a persistent local model between jobs. The runner still needs either
  Hugging Face access on first use or a preloaded Whisper model in its local cache.
  The workflow is manual (`workflow_dispatch`) by design so the real-ASR lane stays
  opt-in and does not slow down or destabilize the default hosted PR checks.
  Each run uploads an artifact bundle with the sample integration log, CLI output,
  saved report JSON/history, and a cache/runner metadata snapshot.
- **End-to-end tests (Playwright + pytest)**: `./scripts/run_e2e.sh`
  * Traces, videos, and screenshots are saved automatically on failure in
    `test-results/` and `playwright-report/` (see
    [Playwright Test](https://playwright.dev/docs/intro) and
    [pytest-playwright](https://playwright.dev/python/docs/intro)).
  * The wrapper always uses the repo-local virtualenv and the Playwright-only
    pytest config, so plain `pytest` no longer depends on Playwright plugins
    being installed globally.
- **Interactive research browser (Playwright CLI + dedicated Chrome profile)**:
  use `./scripts/playwright_research.sh open 'https://example.com'` for a stable,
  Playwright-owned Chrome profile under `.playwright/profiles/research`. Reuse it
  with `./scripts/playwright_research.sh snapshot`, `click`, `type`, and `run-code`.
  For CELI specifically, `./scripts/playwright_celi.sh open 'https://apps.unistrapg.it/cqpweb/celi/'`
  uses a separate dedicated profile under `.playwright/profiles/celi` so corpus
  logins do not mix with general research state. Quote URLs that contain `?`,
  and run commands sequentially (`open`, then `snapshot`, then `click`, etc.)
  rather than in parallel so the session has time to settle after navigation.
  To fully reset a profile, close the browser session and remove the matching
  directory under `.playwright/profiles/`.
- **CELI harvesting CLI**: after logging into CELI once with
  `./scripts/playwright_celi.sh`, use
  `./scripts/python.sh scripts/harvest_celi_queries.py matrix --terms casa,scuola,lavoro --levels B1,B2,C1,C2 --output tmp/celi_harvest/query_matrix.json`
  for query matrices,
  `./scripts/python.sh scripts/harvest_celi_queries.py frequency --term casa`
  for the frequency-breakdown page, and
  `./scripts/python.sh scripts/harvest_celi_queries.py export --term casa --level C2`
  for a metadata-rich concordance export. These commands reuse the dedicated
  Playwright CELI profile and write snapshots/downloads under `tmp/celi_harvest`
  plus `output/playwright/celi/`. For the checked-in Italian benchmark wordlist,
  run
  `./scripts/python.sh scripts/harvest_celi_queries.py manifest --manifest tests/fixtures/celi_wordlists/italian_core_benchmark_v1.json --output-dir tmp/celi_harvest`
  to produce a stable bundle with `bundle.json`, `query_matrix.tsv`, and
  `frequency_breakdowns.tsv`. Then rank terms by CEFR skew with
  `./scripts/python.sh scripts/harvest_celi_queries.py analyze --bundle tmp/celi_harvest/italian_celi_core_benchmark_v1/bundle.json`,
  which writes `skew_analysis.json` and `skew_ranking.tsv`.
- **LIPS spoken-corpus pipeline**: build the phase-1 included/excluded artifacts with
  `./scripts/python.sh scripts/build_lips_manifest.py '/tmp/Corpus LIPS/Corpus LIPS' --output-dir tmp/lips_manifest_real`
  and validate the resulting JSONL bundle with
  `./scripts/python.sh scripts/validate_lips_manifest.py tmp/lips_manifest_real`.
  The build writes `lips_sections_included.jsonl`, `lips_sections_excluded.jsonl`,
  `lips_build_report.json`, and `lips_review_sample.jsonl`. Strict validation is
  designed to block sign-off until a completed manual review file is supplied.
- **LIPS review support**: generate a fresh included/excluded review packet with
  `./scripts/python.sh scripts/review_lips_manifest.py prepare tmp/lips_manifest_real --included-sample-size 20 --excluded-sample-size 20`
  and summarize completed review files with
  `./scripts/python.sh scripts/review_lips_manifest.py summarize --included-review tmp/lips_manifest_real/lips_review_sample.jsonl --excluded-review tmp/lips_manifest_real/lips_excluded_audit_sample.jsonl`.
  This keeps the review loop low-fi and file-based: JSONL in, JSON summary out.
- GitHub Actions workflow (`.github/workflows/ci.yml`) runs both suites and
  installs the Chromium browser via `playwright install --with-deps chromium`.

### Troubleshooting
- If Whisper model download fails behind a SOCKS proxy with an error mentioning
  `socksio`, reinstall dependencies from `requirements.txt` or run
  `python -m pip install socksio`.
- If Whisper cannot download models because the proxy or network blocks
  Hugging Face access, rerun once network access is available or pre-download
  the requested faster-whisper model locally.
- The sample-audio integration test is intentionally opt-in and may skip when
  ASR runtime prerequisites or model downloads are unavailable.

## Notes
- Default provider is **OpenRouter**.
- Use `--llm-timeout` or `LLM_TIMEOUT_SEC` to bound remote rubric requests.
- **Ollama** remains available as the local provider option.
- Other local options: `llama3.2:3b` (fast), `qwen2.5:14b` (stronger); pick according
  to RAM and speed requirements.
- Objective metrics include **WPM**, pauses (≥300 ms), filler count, cohesion
  markers, and a heuristic complexity index (relative clauses / conditionals).
- If the rubric path degrades or the detected language does not match the
  expected language, the structured report is marked with
  `requires_human_review: true`.

## License
MIT

## LMS‑Integration (beta)
Optional can now upload the generated report to a Learning Management
System such as **Canvas** or **Moodle**.
Pass the following flags to provide credentials and context:

| Flag | Description |
|------|-------------|
| `--lms-type` | `canvas` or `moodle` – provider name |
| `--lms-url` | Base URL of the LMS instance (e.g. `https://canvas.example.edu`) |
| `--lms-token` | Bearer/secret token for API access (optional when `CANVAS_TOKEN` or `MOODLE_TOKEN` is set) |
| `--lms-course-id` | Canvas course ID (required for `--lms-type canvas`) |
| `--lms-assign-id` | Assignment ID where the report should be posted |
| `--lms-score` | Optional numeric score to include in the submission |
| `--lms-dry-run` | Print the LMS request preview without uploading |

Example usage:

```bash
python assess_speaking.py sample.wav \
  --lms-type canvas \
  --lms-url https://canvas.example.edu \
  --lms-token $CANVAS_TOKEN \
  --lms-course-id 99 \
  --lms-assign-id 42 \
  --lms-score 75
```

Or use the provider token from the environment and validate the payload first:

```bash
export CANVAS_TOKEN=...
python assess_speaking.py sample.wav \
  --lms-type canvas \
  --lms-url https://canvas.example.edu \
  --lms-course-id 99 \
  --lms-assign-id 42 \
  --lms-score 75 \
  --lms-dry-run
```

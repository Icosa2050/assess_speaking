# assess_speaking – OpenRouter-first assessment core

Pipeline: **Transcription (faster-whisper)** -> **Deterministic metrics** -> **schema-validated CEFR-style rubric** via **OpenRouter** (default) or **Ollama** (legacy/local compatibility).

This branch keeps the old CLI/service shape working while adding a stronger core:

1. OpenRouter as the default remote scoring path.
2. Legacy Ollama support through `--llm` or `--provider ollama`.
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

## 2) Test audio without a microphone
```bash
./scripts/generate_sample.sh         # creates samples/italian_demo.wav
```

Optional flags `-v/--voice`, `-t/--text`, `-o/--output`, e.g.
```bash
./scripts/generate_sample.sh --voice "Bianca" --text "Questo è un test." --output /tmp/test.wav
```

## 3) Check models & self-test
```bash
python assess_speaking.py --list-ollama
python assess_speaking.py --selftest --provider openrouter --llm-model google/gemini-3.1-pro-preview
python assess_speaking.py --selftest --llm llama3.1
```

## 4) Run an assessment
```bash
python assess_speaking.py sample.wav \
  --provider openrouter \
  --llm-model google/gemini-3.1-pro-preview \
  --theme "la mia città" \
  --target-duration-sec 120 \
  --llm-timeout 30 > report.json
```

Legacy/local mode:
```bash
python assess_speaking.py sample.wav --llm llama3.1 > report.json
cat report.json
```

Every run is also stored in `reports/` (structured JSON + `history.csv`). Use
`--label "B1-test"` or `--notes "Morning session"` to tag a run. With
`--log-dir path/to/reports` you control the destination, `--no-log` disables the
persistence layer.

Top-level CLI output remains backward-compatible for existing scripts:

1. `metrics`
2. `transcript_preview`
3. `llm_rubric`
4. optional `baseline_comparison`
5. optional `suggested_training`

New code should read the nested `report` object. It contains the validated
assessment contract, including `checks`, `scores`, `rubric`, and
`requires_human_review`.

### Dashboard / history view
```bash
python scripts/progress_dashboard.py --log-dir reports
python scripts/progress_dashboard.py --log-dir reports --export-html reports/dashboard.html
open reports/dashboard.html  # macOS preview
```

The CLI dashboard renders the history table (via `rich`) and can export an HTML
snapshot.

### Interactive analysis (Streamlit)
Launch the Streamlit app for uploads, re-runs, and charts:
```bash
streamlit run scripts/interactive_dashboard.py -- --log-dir reports
```

In the browser you can upload new audio or reuse existing files, add labels,
trigger assessments, and inspect metrics/rubrics over time. Results continue to
accumulate in `reports/`.

### Prompt trainer with CEFR baselines
- `prompts/prompts.json` contains sample prompts (B1/B2/C1) plus matching audio
  (`prompts/*.wav`). In the Streamlit Prompt-Trainer tab each prompt can be
  played exactly once—after that only the response window (60–120 s depending on
  level) remains.
- Record directly in the browser (WebRTC recorder with single playback) or
  upload an external recording. The run is compared against the requested CEFR
  level (`--target-cefr`), so `assess_speaking.py` appends a baseline verdict.
- Baselines reference the official CEFR global scale (Council of Europe), the
  EF SET level guides for [B1](https://www.efset.org/cefr/b1/),
  [B2](https://www.efset.org/cefr/b2/), [C1](https://www.efset.org/cefr/c1/), and
  conversational speaking rates around 120–150 WPM
  ([VirtualSpeech](https://virtualspeech.com/blog/average-speaking-rate-words-per-minute)).
- After submission you’ll see raw metrics, Ollama’s rubric JSON, the baseline
  comparison (WPM range, filler cap, cohesion/complexity markers), and the trend
  plots.

### Telegram webhook service (experimental)
Run a webhook service so users can submit voice notes via Telegram:

```bash
export TELEGRAM_BOT_TOKEN="<bot-token>"
export TELEGRAM_WEBHOOK_SECRET="<optional-shared-secret>"
export ASSESS_WHISPER_MODEL="large-v3"    # optional
export ASSESS_LLM_MODEL="llama3.1"        # optional
# Optional durable queue + persistent status:
# export SERVICE_REDIS_URL="redis://localhost:6379/0"
# export SERVICE_REDIS_PREFIX="assess_speaking"
# export SERVICE_MAX_WORKERS="2"
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` health and config status
- `POST /webhooks/telegram` Telegram update receiver
- `GET /jobs/{job_id}` job status (in-memory or Redis-backed)

The service downloads the Telegram audio file, runs the same assessment
pipeline used by the CLI, sends a summary back to chat, and uploads the JSON
report as a Telegram document.
If `SERVICE_REDIS_URL` is set, webhook jobs are queued in Redis and job status
is stored in Redis hashes. In-flight Redis jobs are re-queued on service start
so a restart does not silently drop work that was already dequeued.

### Tests & CI
- **Unit tests**: `python -m unittest discover -s tests`
- **OpenRouter integration (opt-in)**:
  `RUN_OPENROUTER_INTEGRATION=1 python -m unittest tests.test_integration_openrouter -v`
- **Optional sample-audio integration test (no microphone required)**:
  `RUN_AUDIO_INTEGRATION=1 WHISPER_MODEL=tiny python -m unittest tests.test_sample_integration`
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
- **End-to-end tests (Playwright + pytest)**: `pytest tests/e2e`
  * Traces, videos, and screenshots are saved automatically on failure in
    `test-results/` and `playwright-report/` (see
    [Playwright Test](https://playwright.dev/docs/intro) and
    [pytest-playwright](https://playwright.dev/python/docs/intro)).
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
- Legacy/local compatibility remains available via **Ollama**.
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

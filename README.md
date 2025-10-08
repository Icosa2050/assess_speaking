# assess_speaking (local, Mac-optimized) – v2

Local pipeline: **Transcription (faster-whisper)** → **Metrics** → **CEFR rubric** via **Ollama**. Updated model tags (no `:instruct`) + built-in **self-test**.

## 0) Prerequisites
```bash
brew install ffmpeg ollama
ollama pull llama3.1              # alternatives: llama3.2:3b / qwen2.5:14b
ollama list
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
python assess_speaking.py --selftest --llm llama3.1
```

## 4) Run an assessment
```bash
python assess_speaking.py sample.wav --whisper large-v3 --llm llama3.1 > report.json
cat report.json
```

Every run is also stored in `reports/` (structured JSON + `history.csv`). Use
`--label "B1-test"` or `--notes "Morning session"` to tag a run. With
`--log-dir path/to/reports` you control the destination, `--no-log` disables the
persistence layer.

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

### Tests & CI
- **Unit tests**: `python -m unittest discover -s tests`
- **End-to-end tests (Playwright + pytest)**: `pytest tests/e2e`
  * Traces, videos, and screenshots are saved automatically on failure in
    `test-results/` and `playwright-report/` (see
    [Playwright Test](https://playwright.dev/docs/intro) and
    [pytest-playwright](https://playwright.dev/python/docs/intro)).
- GitHub Actions workflow (`.github/workflows/ci.yml`) runs both suites and
  installs the Chromium browser via `playwright install --with-deps chromium`.

## Notes
- Default LLM is **llama3.1** (without `:instruct`).
- Other options: `llama3.2:3b` (fast), `qwen2.5:14b` (stronger); pick according
  to RAM and speed requirements.
- Objective metrics include **WPM**, pauses (≥300 ms), filler count, cohesion
  markers, and a heuristic complexity index (relative clauses / conditionals).

## License
MIT

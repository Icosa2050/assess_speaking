# Local practice readiness — 2026-09-29

Follow-up to `54d6fd5`, which was already pushed directly to main. This change has its own branch and pull request. English and Italian remain the launch languages; B1/B2/C1 are practice goals.

## What is usable now

Double-click `Start Vostavo.command`, or run `./scripts/python.sh scripts/start_practice.py` with Node 24. The launcher checks prerequisites, builds the interface, starts the backend and UI on localhost, and opens the default browser. Normal data lives in `~/Library/Application Support/Vostavo`; test workspaces are separate. The old `run_app.py` is a backend/bootstrap entry point, not a complete UI launcher.

Use Ollama with `qwen3.5:4b` as the starting configuration. LM Studio is a tested alternative with `vostavo-qwen2.5-3b` (Qwen2.5 3B Instruct Q4_K_M). Choose one provider in Runtime Setup, detect/select its model, test and save. Choose the cached Whisper `large-v3` for the verified configuration. Record a short oral response, submit, read one priority, retry, and compare the saved attempts and audio in History. README contains the full first-run sequence.

## Limits addressed

| Previous gap | Current behaviour |
| --- | --- |
| Backend-only launch instructions | A complete launcher and executable macOS entry point; stable API port; clear prerequisite/port errors; cleanup on Ctrl+C, SIGTERM and SIGHUP |
| Whole upload read into RAM | Bounded multipart parser; spool to disk; copy/hash in 1 MiB chunks; at most two concurrent uploads |
| Unlimited upload/storage pressure | 100 MiB ceiling, dynamically reduced by free space on both temp and app-data volumes; 512 MiB disk reserve; body counted even without Content-Length; one file per request |
| Compressed audio expands indefinitely | Bounded decoding checks before expensive inference; at most 20 minutes, 90-second decode timeout, temporary PCM on disk; cleanup on cancellation |
| Upload failure gives little recovery | Progress, cancellation, retained selection, local backup download; partial file cleanup; readable size/disk/network errors; successful upload reused after assessment submission failure |
| Multiple models loaded by overlapping attempts | One assessment worker at a time; a second submission receives a recoverable busy response |
| LM Studio had no coaching model | Compatible model installed and loaded; setup and real bilingual feedback/history/replay verified; bounded output and recorded inference profile |
| Chromium-only coverage | Two WebKit upload/review/replay/retry journeys; microphone simulation remains Chromium-specific |
| Large initial JS bundle | Route splitting; navigation remains visible during loading; reload recovery for missing/failed route chunks; largest build chunks below 300 kB |

Multipart form data is appropriate for short local oral recordings. This change adds bounded transfer and recovery without introducing dependencies or a new upload service. It does **not** claim resumable transfer after browser/app restart: keep the page open until the attempt is saved, or download a copy first. Dedicated resumable storage remains an option for future remote deployments and long sessions.

## Verification

- Full backend suite: **580 passed, 5 optional integration tests skipped**. The initial sandboxed run could not bind a localhost socket; the complete run outside that restriction passed.
- Frontend unit/component suite: **130 passed**; TypeScript passed.
- Existing Chromium suite: **8 passed, 5 opt-in tests skipped**.
- Final bilingual Chromium/WebKit journeys: **12 passed**. Two focused WebKit reruns also passed with explicit assertions of no browser errors; the test now waits for the History route before reloading, avoiding an intentionally interrupted lazy import.
- LM Studio: setup and English/Italian B1/B2/C1 with B1 retries. Initial run: **6 passed, 1 UI failure** after source edits reloaded the app while C1 analysis was running. Inference itself succeeded. With stable source, the focused Italian C1 journey **passed**, including review, reload, history and replay.
- Final Ollama English/Italian B2 smoke: **2 passed** with real Whisper, rubric, coaching, saved history and replay. The preceding commit already verified all six language/goal combinations and B1 retries.
- Complete launcher production UI/API: **passed in Chromium and WebKit**, with no runtime errors. Closing the launcher with SIGHUP stopped its backend and UI. The first smoke probe assumed an h1; the actual routes use h2, so the probe was corrected to the accessible heading role.

New regression coverage includes bounded copying and hashing, body limits without Content-Length, extra/empty files, low disk and recovery, failed metadata cleanup, insufficient copy space without double reservation, corrupt/overlong audio, missing ffmpeg, decoder cancellation, partial-spool cleanup on disconnect, busy worker handling, XHR progress/abort/storage errors, reuse of a saved upload after HTTP 409, and retaining a downloadable recording after a disk-space error.

The tests use synthetic/sample speech. They verify a working training workflow and measured feedback fields, not the accuracy of an official exam grade. Unused OpenRouter/oMLX integration tests remain opt-in; neither service is needed for local practice.

### Commands

```bash
.venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
NODE_ENV=development node frontend/node_modules/playwright/cli.js test --config frontend/playwright.config.ts
NODE_ENV=development node frontend/node_modules/playwright/cli.js test --config frontend/playwright.journeys.config.ts
LOCAL_E2E_PROVIDER=lmstudio LOCAL_E2E_MODEL=vostavo-qwen2.5-3b NODE_ENV=development node frontend/node_modules/playwright/cli.js test --config frontend/playwright.ollama.config.ts
NODE_ENV=development node frontend/node_modules/playwright/cli.js test --config frontend/playwright.ollama.config.ts --grep 'en B2|it B2'
./scripts/python.sh scripts/start_practice.py
```

## CI follow-up

Run 36617830482 passed quality and frontend smoke, but three shell-wrapper tests depended on a local npm installation absent from the separate Python job (575 passed, 5 skipped). These tests already replace Node with an argument-capture stub; they now copy the real wrapper scripts into a temporary repository with a matching CLI placeholder. All seven focused wrapper/config tests pass without relying on the checkout's node_modules. Actual browser execution remains covered by the frontend smoke job and the local journeys above.

## Runtime findings

LM Studio's installed embedding model cannot generate feedback. Its CLI and local server work outside the restrictive execution sandbox; no privacy-setting modification was needed. Importing Ollama's Qwen 3.5 file into LM Studio failed with `qwen35.rope.dimension_sections has wrong array length; expected 4, got 3`. That task-created hard link was removed; Ollama's original remains intact. LM Studio's current runtime reported no update available. A compatible Qwen2.5 3B Instruct GGUF was downloaded from the LM Studio community repository and successfully loaded. Models are machine-local, not committed. The incompatible hard link and temporary download hard links were removed; the working LM Studio model remains installed. The normal app workspace was initially unconfigured; the launcher was opened and configured with tested Ollama plus cached Whisper, without importing test history.

## Review

[Claude CLI review](2026-09-29-claude-local-readiness.md) used selected source/diffs/tests under standing approval; no learner recordings, private reports or credentials were supplied. Accepted findings were fixed: double disk reservation, redundant upload on retry, stable launcher endpoint, ffmpeg guidance, translated transport errors, preserving navigation while loading, launcher failure/shutdown guidance, preserving audio extensions, cancellation of the decoding child, disconnected request cleanup, and additional regression tests. Some fixes and tests were already in progress when the review snapshot was sent.

Checked review hypotheses: `requestJson` currently has no special authentication headers missing from XHR; both transports use the same URL resolver. Non-JSON error parsing already has a fallback; the outer XHR handler now also rejects unexpected parsing errors. Cancellation already joins workers. WebKit runs once per test with one configured project, zero retries and a fresh inherited workspace. The multipart cleanup uses the inspected Starlette parser implementation; upgrade tests cover the private cleanup list. OCR delegate selection/rules were resolved locally; host review coverage is saved alongside this report. Unrelated pre-existing files are excluded from the commit.

## Still separate product work

The repeat-practice loop is usable today. A complete timed multipart exam simulation, an interactive speaking partner, exam-specific task banks and the planned revised practice rating are **not implemented by this PR**. The next product milestone should be a guided sequence of oral parts with preparation/speaking timers, separate recordings and a combined session review, followed by the responsive partner. This is distinct from fixing local setup and resource failures. See the [canonical training plan](../superpowers/plans/2026-09-29-oral-exam-preparation.md).

## Sources

The upload design follows [FastAPI's spooled UploadFile model](https://fastapi.tiangolo.com/tutorial/request-files/) and [Starlette's streaming/form parser documentation](https://www.starlette.io/requests/). The installed Starlette is 0.46.2, so newer documented body-limit middleware was not assumed available; the request stream is bounded explicitly.

Runtime setup follows [LM Studio local server](https://lmstudio.ai/docs/developer/core/server) and [CLI instructions](https://lmstudio.ai/docs/cli). The compatible test model is [LM Studio community Qwen2.5 3B Instruct GGUF](https://huggingface.co/lmstudio-community/Qwen2.5-3B-Instruct-GGUF).

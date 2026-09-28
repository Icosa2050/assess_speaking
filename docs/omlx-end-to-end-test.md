# Live oMLX assessment test

This opt-in test uploads the Italian `output/speech/it_b1_near_perfect.wav`
fixture through the browser, transcribes it with Whisper, assesses it with
oMLX, checks structured scores and coaching, and reloads History to verify the
saved report. It rejects dry runs and deterministic-only fallback results.
It does not assert an exact model score or certify scoring accuracy.

Run from a normal Terminal, not a browser-restricted agent session:

```bash
cd frontend
RUN_VOSTAVO_OMLX_ASSESSMENT_E2E=1 \
OMLX_MODEL='Qwen3.8-27B-oQ4e-mtp' \
OMLX_E2E_DOWNLOAD_WHISPER=1 \
node node_modules/playwright/cli.js test --config=playwright.omlx.config.ts
```

The test requires the WAV fixture to exist locally; generated `output/speech`
files are not guaranteed to be present in a clean checkout. oMLX must already
be running and advertise the specified chat model. The URL and API key come
from `~/.omlx/settings.json`; `OMLX_BASE_URL` and `OMLX_API_KEY` override them.
The local key is never automatically forwarded to a different override URL.

`OMLX_E2E_WHISPER` defaults to `small`. With
`OMLX_E2E_DOWNLOAD_WHISPER=1`, the test downloads that model if missing;
omit the flag on subsequent runs. Allow up to 15 minutes for downloading,
transcription, model loading, and assessment.

The dedicated configuration starts fresh servers on ports 8812 and 4175 and
refuses to reuse existing servers. Each run uses a new temporary app-data
directory; model cache data is reusable. It saves a temporary connection using
the app's normal secret storage and deletes that connection in `finally`.
An interrupted process may leave its temporary data or credential behind.
Traces and video are disabled because requests contain credentials.

The regular suite skips this test. The existing `omlxRuntimeLive.spec.ts`
remains a separate, faster connection check.

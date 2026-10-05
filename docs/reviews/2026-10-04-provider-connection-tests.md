# Provider connection automation review — October 4, 2026

## Scope

Seven API-key/local connection choices plus ChatGPT authorization. Browser fixtures cover English and Italian; the live runner exercises the actual backend with isolated credential storage. The runner never reads learner audio. See [usage and account instructions](../testing/provider-connections.md).

## Claude CLI review and fixes

The logged-in Claude CLI reviewed selected source, tests and documentation under the user's standing approval. No credentials, recordings or learner reports were included. Findings were checked against the implementation and addressed:

1. Require an actual isolated keyring write and clear session secrets before reload, preventing session fallback from producing a misleading pass.
2. Record completed stages only after success; preserve probe failures and separate cleanup failures.
3. Label browser journeys as mocked frontend contracts and assert the DELETE request occurred.
4. Convert unexpected live-test exceptions into safe fixed diagnostics; do not print provider response bodies.
5. Scan all temporary application files as bytes and check API responses for key leakage.
6. Separate optional local Ollama credentials from the Ollama Cloud key.
7. Accept empty credential placeholders in `.env`.
8. Use explicit isolated server commands and validated temporary-directory teardown.
9. Clarify model overrides, reject inapplicable endpoint flags and preserve stdout reports when writing the report file fails.

Regression tests cover persistence failure, combined probe/cleanup failures, non-JSON leakage, response leakage, unexpected errors, environment isolation and empty placeholders.

A repeated browser run exposed an initial Settings hydration race: the Italian OpenRouter test selected OpenRouter, but the submitted draft had been reset to local Ollama. The connection form now remains disabled until the initial selection is resolved. A follow-up Claude review found that an initial settings error would otherwise leave the form locked. The final implementation shows a localized error and initializes an editable new draft on failure. Unit coverage checks both delayed successful hydration and failed hydration; the affected browser journey passed after both fixes.

## OCR delegate review

Used `ocr delegate preview --format json` with explicit exclusions for unrelated pre-existing files, then `ocr delegate rule` for each reviewable file. The host reviewed every selected diff/new file. Tests and the guide were reviewed additionally even though OCR excludes them by default.

- Total selected files: 9
- Reviewed: 9
- Skipped: 0
- Coverage: 100%
- Unresolved actionable findings: none in the reviewed scope

The adjacent manifest records every path/status identity and sanitized live results. No claim is made about unrelated existing changes.

## Verification

- `npm --prefix frontend test`: 155 passed.
- `npm --prefix frontend run typecheck`: passed.
- `.venv/bin/python -m pytest -q tests/test_provider_connection_checks.py tests/test_cloud_accounts.py tests/test_llm_client.py tests/test_app_core_services.py`: 150 passed, including real localhost callback coverage.
- `npm --prefix frontend run test:e2e -- --config playwright.connections.config.ts`: 23 passed, including localhost Vite and backend health readiness. After the final settings-error recovery addition, the affected Italian OpenRouter journey was rerun and passed.
- `.venv/bin/python scripts/check_quality.py` and `git diff --check`: passed.
- Live runner: all three Groq models, OpenRouter GPT-OSS 120B, Ollama Qwen 3.5 4B and LM Studio Qwen 2.5 3B passed save/reload/saved-connection inference/delete.

The required sibling venv was unavailable; the bootstrapped repository `.venv` was used. Browser tests ran on local Chromium. Fixture credentials only entered browser traces. Live xAI, Ollama Cloud and external ChatGPT consent await user credentials/authorization; native keychain persistence is outside the isolated runner's coverage.

# Ollama Cloud and ChatGPT live verification

Date: October 4, 2026. User priority: Ollama Cloud and ChatGPT; defer paid xAI API testing.

## Live findings and fixes

The supplied Ollama key was read from the repository `.env` without displaying it. Official model discovery returned `gpt-oss:120b` among the available models.

The first connection lifecycle check failed. Diagnosis through the application helper found `HTTP 404: path /api/api/tags not found`. The setup default ends in `/api`; provider URL construction was treating it as a service root. Normalization now accepts the root, `/api`, `/v1`, and historical `/api/v1` forms. Discovery uses `/api/tags`; inference uses `/v1/chat/completions`. Local reverse-proxy prefixes are preserved.

The next live probe reached inference but failed because its eight-token limit truncated GPT-OSS before a final answer. Ollama probes now use a bounded 256-token limit. Assessment generation limits are unchanged.

The final live Ollama Cloud check passed all four stages with `gpt-oss:120b`: save, reload with cleared session secrets, inference using the saved key, and delete. It used temporary app data and an isolated keyring; it did not create a permanent user connection.

The user then completed ChatGPT browser authorization in the isolated Vostavo backend on port 8819. The automatic saved-session check passed model discovery and structured-feedback inference with the selected `gpt-6-astra` model. Tokens remained in Vostavo's secret storage. The check did not export, revoke or deliberately expire them. The test frontend remains on port 4189; this authorization belongs to that test instance.

## Regression coverage

`tests/test_ollama_cloud_connection.py` covers four cloud URL variants through the service and LLM layers, asserting outgoing HTTP URLs and bearer headers rather than replacing the whole provider helper. Four more cases cover local proxy prefixes and preserve OpenRouter paths.

- Focused backend suite: **158 passed** (`test_ollama_cloud_connection`, `test_provider_connection_checks`, `test_llm_client`, `test_app_core_services`, `test_cloud_accounts`).
- Quality checks and `git diff --check`: passed.
- Local backend `/v1/health` and Vite `/settings`: HTTP 200.
- Initial sandbox-only backend run encountered app-data permission failures; rerunning with the required app-data/localhost permissions passed.
- No frontend source or npm dependency changes.

## Reviews

Claude CLI reviewed the selected source/diff and fixture tests under standing approval. Its conditional key-migration concern was checked: `secret_account_name` has no repository callers; connection loading reads the stored `secret_ref` directly, and persistence retains an existing reference. This endpoint fix does not recompute saved connection references. No migration patch was needed.

OCR delegate preview/rules selected `app_core/runtime_providers.py` and `assessment_runtime/llm_client.py`. Both were reviewed; the new tests and updated guide were reviewed additionally. Total selected: 2; reviewed: 2; skipped: 0; coverage: 100%. No unresolved actionable findings in scope.

## Limits and references

These checks establish connection/inference functionality, not full English/Italian learner-feedback quality. Initial ChatGPT consent was performed by the user; renewal after revocation or account challenges can still require user action. xAI is intentionally deferred.

- [Ollama Cloud model IDs and authentication](https://docs.ollama.com/cloud)
- [Ollama's direct cloud OpenAI-compatible endpoint](https://docs.ollama.com/api/openai-compatibility)
- [ChatGPT registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Repeatable commands and account instructions](../testing/provider-connections.md)

# Groq connection review — 2026-10-04

Scope: small provider addition to the existing desktop setup and feedback flow. No credentials, learner reports or recordings were shared for review. No dependencies changed.

## Verification

- 123 focused backend tests passed across cloud accounts, LLM client and application services; unchanged OAuth loopback test deselected in this run.
- 149 frontend unit tests passed; typecheck and production build passed.
- Five Chromium cloud-connection journeys passed, including Groq setup, key separation and quota recovery in English and Italian. Isolated Vite/backend ports 4187/8817; Playwright health readiness probes passed.
- `scripts/check_quality.py` and `git diff --check` passed.
- No live Groq key was supplied. Free-tier feasibility for full learner attempts and feedback quality are not claimed by mocked tests.

## OCR delegate

File selection and rules from `ocr delegate preview` and `ocr delegate rule`; host reviewed all 17 reviewable files (100%, zero skipped) plus the two changed test files and plan. Pre-existing untracked material and temporary browser configuration were excluded. See [coverage manifest](2026-10-04-groq-ocr-manifest.json). No blocking findings.

## Claude CLI

Selected diff and adapter source reviewed through the logged-in CLI, under standing approval. No blocking findings. Follow-ups:

- Confirmed HTTP 413 in [Groq error documentation](https://console.groq.com/docs/errors). Added a specific too-large-request message and regression case; waiting alone is not suggested for oversized requests.
- Model choice remains editable; the suggested GPT-OSS model supports strict schemas. The actual schema probe rejects incompatible selections. Groq's current docs also list Qwen 3.8 27B, so Claude's narrower model list is not treated as authoritative.
- Renamed the shared test key to provider-neutral text.

### Raw reviewer output before follow-up fixes

I found no blocking correctness or security issues.

I checked the diff, plus `_extract_assistant_message_text` (`assessment_runtime/llm_client.py:180`), `_assessment_request_with_saved_runtime_secret` (`app_backend/app.py:334`) and `_connection_api_key` (`app_core/runtime_resolver.py:50`).

**What holds up:**
- **Fixed endpoint:** `_cloud_url` only allows a fixed list of providers. `resolved_base_url` rejects custom URLs for `groq`, and the save, test and assessment paths all check this.
- **No environment fallback:** `_connection_api_key` has no environment fallback for `groq`. `assess_speaking.py:843` and `jobs.py:231` keep the Groq key out of `LLM_API_KEY`, and the parametrized test covers this.
- **Key goes to the right provider:** a saved key is only injected when the active connection's provider matches the request (`app.py:356`). A Groq request can't pick up an xAI key or the reverse.
- **No upstream text in errors:** `OpenAIError` maps to fixed messages, and the client is created with `from None`, `max_retries=0` and `follow_redirects=False`. `_extract_assistant_message_text` raises `LLMClientError` with fixed text, so no upstream body or content reaches the error. Truncated output (`finish_reason == "length"`), refusals and empty content are rejected.
- **Structured output:** the Groq branch builds `response_format={'type':'json_schema','json_schema':schema}` with `stream=False`. Extra payload fields like `provider` and `temperature` are dropped, as intended.

**Non-blocking notes (no fix required):**
1. **Groq's 413 error gets a misleading message** (`assessment_runtime/responses_client.py:69-75`). From what I remember of Groq's API, when one request is larger than the free plan's tokens-per-minute limit, it returns **413**, not 429. I couldn't confirm this here. Requests are larger than the prompt alone because of `max_completion_tokens=4096`, so long attempts are more likely to hit it. Today that case shows the generic "The provider request failed. Check the connection and selected model." That message leaves out the usage-limit hint and the "recording is retained" reassurance. If Groq does return 413 there, give `413` the same message as `429`.
2. **`strict: true` with other Groq models** (`responses_client.py:45`). Groq documents strict JSON schema only for `gpt-oss-20b` and `gpt-oss-120b`. If a user picks another model from `/models`, the request will probably fail with a 400. The test probe catches this before saving, and its generic message mentions the selected model, so it's acceptable. A model-specific hint would just be clearer.
3. **Test value name:** `tests/test_cloud_accounts.py` uses `'session-xai-key'` in the test that now also covers Groq. It's cosmetic.

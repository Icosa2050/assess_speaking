# Automated provider connection tests

Updated: October 4, 2026. Vostavo has seven API-key/local connection choices and one ChatGPT browser-authorization flow. These are provider connections; the app itself remains local/guest.

## Current focus: Ollama Cloud and ChatGPT

Live-test priority is Ollama Cloud and ChatGPT. xAI testing is deferred; no xAI account or API credits are needed for the current work.

The supplied `OLLAMA_API_KEY` in the repository `.env` was verified with `gpt-oss:120b`:

```sh
.venv/bin/python scripts/check_provider_connections.py --providers ollama_cloud --env-file .env --model ollama_cloud=gpt-oss:120b
```

Use the model ID returned by `https://ollama.com/api/tags`, without the local CLI `:cloud` suffix. Vostavo normalizes the native `/api` setup URL to `/api/tags` for discovery and `/v1/chat/completions` for inference. Legacy saved `/api/v1` URLs are normalized too. The GPT-OSS connection probe has a bounded 256-token budget.

ChatGPT requires the one-time browser authorization in section 3. The user completed this successfully in the isolated test app; the saved `gpt-6-astra` session passed model discovery and structured-feedback inference. An API key or an existing Codex/Claude CLI login is not that Vostavo authorization.

## 1. Account-free regression suite

Run from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_ollama_cloud_connection.py tests/test_provider_connection_checks.py tests/test_cloud_accounts.py tests/test_llm_client.py tests/test_app_core_services.py
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run test:e2e -- --config playwright.connections.config.ts
```

The prescribed sibling virtual environment was missing on the development machine; the repository `.venv` is the bootstrapped fallback. No new packages were added.

These new tests are also discovered by the existing CI `pytest` and default Playwright commands; live checks are not run by default.

The dedicated browser configuration uses fresh backend/app data and ports 8817/4187, never an already-running checkout. Stop conflicting listeners yourself or choose other ports in that configuration. Test credentials are fixtures. These browser journeys are frontend contract tests: provider and backend connection responses are mocked. Actual backend lifecycle coverage comes from the Python tests and live runner. This suite covers:

- Groq, xAI, OpenRouter, Ollama Cloud, local Ollama, local LM Studio and OpenAI-compatible: enter credentials when needed, test, save, reload, test with the saved reference, remove.
- Groq: all three feedback models and quota-error recovery.
- ChatGPT: browser link, consent success/denial, cancel/retry, model selection, probe, disconnect.
- English and Italian throughout.
- Initial Settings hydration blocks connection editing, preventing a late response from overwriting a provider selection.
- Backend OAuth security: real loopback callback, PKCE/state/nonce/JWT verification, callback replay, cancellation races, refresh rotation, session-only storage, key redaction, revoked/expired grants and credential broker behavior. The identity provider is mocked. Initial external sign-in is not proven by these tests.

## 2. Live API-key/local connection checks

```sh
.venv/bin/python scripts/check_provider_connections.py --providers groq --env-file .env
```

This command runs the real Vostavo save → application reload → saved-key discovery/inference → delete lifecycle for all three Groq feedback models. It substitutes only credential storage with an isolated in-memory keyring. Application data is temporary; your saved app connections and OS keychain are not changed. The reload clears session-only secrets and creates a fresh application instance in the same process; this does not test OS keychain persistence across processes.

Each model makes a model-discovery request and one small synthetic capability request. No learner audio is uploaded. Paid accounts may incur a small API charge; free accounts consume allowance. No provider fallback, automatic purchasing, or quota retry occurs. Reports contain fixed error stages and HTTP status, not response bodies, prompts or credentials. Missing prerequisites are `blocked` and return exit code 1, not a green skipped run. Any failed test also returns 1. Use `--output /path/to/new-report.json` to retain the sanitized report; existing files are never overwritten.

Only explicit provider key names are read from the optional `.env`; existing environment values take precedence. There is no shell execution or variable expansion. This test convenience does not enable `.env` import in Vostavo Settings.

| Setup choice | Credential | Model selection |
| --- | --- | --- |
| `groq` | `GROQ_API_KEY` | Defaults to GPT-OSS 120B, GPT-OSS 20B, Qwen 3.8 27B |
| `xai` | `XAI_API_KEY` | Explicit `--model xai=MODEL_ID` |
| `openrouter` | `OPENROUTER_API_KEY` | Explicit `--model openrouter=MODEL_ID` |
| `ollama_cloud` | `OLLAMA_API_KEY` | Explicit `--model ollama_cloud=MODEL_ID` |
| `ollama_local` | Optional `OLLAMA_LOCAL_API_KEY` | Installed model via `--model ollama_local=MODEL_ID` |
| `lmstudio_local` | Optional `LMSTUDIO_API_KEY` | Loaded model via `--model lmstudio_local=MODEL_ID` |
| `openai_compatible` | Optional `COMPATIBLE_API_KEY` | Explicit model plus `--base-url http://127.0.0.1:PORT/v1` |

Cloud URLs are fixed. Generic-compatible URLs are restricted to local HTTP loopback in this runner. This deliberately does not cover every remote compatible service. Repeat `--model PROVIDER=MODEL_ID` for multiple models. Local endpoints use Vostavo defaults (Ollama 11434, LM Studio 1234).

Examples:

```sh
.venv/bin/python scripts/check_provider_connections.py --providers openrouter --env-file .env --model openrouter=openai/gpt-oss-120b
.venv/bin/python scripts/check_provider_connections.py --providers xai --env-file .env --model xai=YOUR_TEXT_MODEL_ID
.venv/bin/python scripts/check_provider_connections.py --providers ollama_cloud --env-file .env --model ollama_cloud=YOUR_CLOUD_MODEL_ID
.venv/bin/python scripts/check_provider_connections.py --providers ollama_local --model ollama_local=YOUR_INSTALLED_MODEL
.venv/bin/python scripts/check_provider_connections.py --providers lmstudio_local --model lmstudio_local=YOUR_LOADED_MODEL
```

For OpenRouter/ChatGPT/Groq/xAI, the probe verifies the strict rubric schema. Local and generic connections use their existing small text connection probe; successful authentication there does not establish full assessment/schema quality. Existing bilingual live assessment journeys remain the appropriate next test for full practice reports.

## 3. ChatGPT: one manual authorization, then an automatic check

OpenAI's official local-app guide requires a ChatGPT Plus or Pro account to test plan usage. Account/workspace eligibility still applies. No documented public sandbox account or bypass for consent/MFA was established. Use an account you own; passwords, MFA and consent stay in the normal browser flow. Do not copy browser cookies or refresh tokens into test fixtures.

For isolation, run a separate test backend in one terminal:

```sh
.venv/bin/python scripts/run_backend.py --host 127.0.0.1 --port 8819 --app-data-dir /tmp/vostavo-chatgpt-test/app --cache-dir /tmp/vostavo-chatgpt-test/cache
```

Run its frontend in another terminal:

```sh
VITE_LOCAL_API_BASE_URL=http://127.0.0.1:8819 npm --prefix frontend run dev -- --host 127.0.0.1 --port 4189 --strictPort
```

1. Open `http://127.0.0.1:4189/settings`, select ChatGPT, and complete **Continue with ChatGPT**. Approve plan usage if eligible.
2. Choose a model and test it. The normal app uses protected key storage or explicitly reports session-only storage.
3. Find the test connection's `connection_id` in `http://127.0.0.1:8819/v1/runtime/settings`. That response contains connection metadata, not tokens. Do not confuse this with a personal app/backend on another port.
4. Run:

```sh
.venv/bin/python scripts/check_provider_connections.py --providers chatgpt --chatgpt-backend http://127.0.0.1:8819 --chatgpt-connection-id YOUR_TEST_CONNECTION_ID
```

The runner asks that backend to use its saved session and selected model. Refresh near expiry uses the normal backend token manager; the test does not force token expiry. It neither exports tokens nor signs you out. After initial authorization, this check can run unattended while the session remains usable. A revoked/expired grant or account challenge needs browser reauthorization. Interactive sign-in, deliberate provider-side revocation and native OS/browser/keychain behavior remain manual acceptance checks. When finished, use **Disconnect ChatGPT** in the test app and review app access in ChatGPT settings; deleting a temp folder alone does not revoke the grant.

## Test accounts and what you need to do

- **Groq:** your supplied key already works. No additional account needed. A separate test project/key can keep test usage separate. Stay on Free if desired; allow the three feedback models.
- **OpenRouter:** your existing API key can be used for a small probe. For repeat CI runs, create a dedicated key with a credit limit and allow a model supporting structured output.
- **xAI:** deferred at the user’s request. No account setup or paid test is needed now. Existing fixture coverage remains.
- **Ollama Cloud:** use an Ollama account, create an API key, choose a cloud model available to that account, and set `OLLAMA_API_KEY`. The current Vostavo cloud connection uses that key, not an automated website login.
- **Local Ollama/LM Studio:** no cloud account required. Start the local server and load a model. If LM Studio authentication is enabled, create a token and use `LMSTUDIO_API_KEY`.
- **ChatGPT:** one manual sign-in as above. OpenAI API keys are not a substitute for this plan-authorization flow. No account was created on your behalf; email verification, plan choice and personal consent are yours.

Run the account-free suite on normal CI. Run live checks only through an explicit trusted workflow with dedicated credentials; never expose keys to fork PR jobs or dependency install steps. Do not record real login pages, cookies or API-key input in Playwright traces. There is no scheduled automation or new CI secret configured by this change.

## Official references checked October 4

- [Groq quickstart](https://console.groq.com/docs/quickstart), [free limits](https://console.groq.com/docs/rate-limits)
- [xAI API quickstart](https://docs.x.ai/developers/quickstart)
- [OpenRouter authentication](https://openrouter.ai/docs/api/reference/authentication)
- [Ollama Cloud authentication](https://docs.ollama.com/cloud)
- [LM Studio authentication](https://lmstudio.ai/docs/developer/core/authentication)
- [OpenAI local-app sign-in tutorial](https://developers.openai.com/cookbook/articles/sign-in-with-chatgpt), [sessions and refresh](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
- [Playwright authentication guidance](https://playwright.dev/docs/auth) explains reuse of authenticated state and why saved state must not be committed. Vostavo's OAuth tokens belong to the backend, so these tests do not export browser state as a substitute.


## Verified run on October 4

- Live: Ollama Cloud `gpt-oss:120b`, all three Groq models, OpenRouter `openai/gpt-oss-120b`, local Ollama `qwen3.5:4b`, local LM Studio `qwen2.5-3b-instruct` passed save/reload/probe/delete.
- Browser: 23 fixture-based Chromium journeys passed in English and Italian.
- Frontend: 155 tests and typecheck passed.
- Backend: 158 focused cases passed, including the real loopback callback and persistence, cleanup, isolation and redaction regressions.
- ChatGPT: manual browser authorization completed, then the automatic saved-session check passed with `gpt-6-astra`. This does not exercise forced expiry, provider-side revocation or unattended initial consent.
- xAI is deferred. Arbitrary compatible servers remain untested live. The live connection checks do not establish full learner-feedback quality.

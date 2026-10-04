# Connect ChatGPT, xAI Grok and Groq

**Date:** 2026-10-04. **Status:** Implemented and tested locally; live user authorization and provider inference remain to be verified.

## Scope

Connect a provider to the existing speaking-practice workflow. English and Italian remain the training focus. Transcription, prompts, feedback validation, recording retention, and progress tracking keep their current roles.

| Provider | Implemented connection | User action |
| --- | --- | --- |
| OpenAI ChatGPT | Browser authorization, account model catalog, secure/session-only token storage, refresh, Responses inference, reconnect and disconnect. | Settings → ChatGPT → Continue with ChatGPT → Continue in your browser. Authorize plan use, return, select a model, and test the connection. |
| **xAI Grok** | Dedicated API-key connection and Responses inference at `https://api.x.ai/v1`. | Create a key and API credits at the xAI console, choose xAI Grok (API), enter a supported text model and key, test, save. |
| Groq | API-key connection at `https://api.groq.com/openai/v1`, using nonstreaming strict JSON chat completions. | Settings → Groq → create/copy a key → test → save. The suggested model is `openai/gpt-oss-120b`. |
| OpenRouter | Existing saved API-key integration is retained. | Continue using the existing connection. No additional OAuth flow was added. |

The user subsequently requested **Groq** too, if easy and free. Groq is a separate provider from **xAI Grok**. No public delegated Grok consumer-subscription sign-in was established in this research; the app labels its supported API path explicitly. ChatGPT’s local-app plan flow supports text inference; it does not supply the transcription API.

## Implemented

- Browser sign-in uses dynamic client registration, a stable host ID, PKCE, state, nonce, a loopback callback and verified ID-token claims. Reconnection retains the issued client ID and checks account identity.
- Successful sign-in publishes the connection from the backend, without needing a final browser poll. Returning to settings can resume an unfinished sign-in.
- Tokens stay in the backend and secure system storage. Unsupported/failed secure storage falls back to backend memory with an explicit session-only state. No tokens are written to preferences, reports, browser storage, support metadata, or worker environment variables.
- The backend serializes rotating refresh tokens. A durable pending marker prevents replay after an uncertain renewal; the retained token can still be revoked. Invalid replacement grants are revoked and cleared.
- Assessment workers request a current access token through a private process pipe before each inference call. Refresh tokens remain in the backend. Reauthorization and running ChatGPT jobs cannot overlap; Disconnect cancels active ChatGPT work.
- Streaming feedback requires terminal completion. Partial output, refusal, allowance errors, and dropped streams cannot become accepted AI feedback. Provider failures use the existing local-report fallback.
- Provider endpoints are fixed. A failed provider never silently switches to another account or paid provider. Existing schema validation remains in effect.
- Connection endpoints enforce a local host and app header; cross-origin browser writes are rejected. The current local-desktop trust model still trusts other localhost applications. This is not a hosted authentication system.
- UI controls and explanatory text are available in all five existing interface locales. Existing backend diagnostic errors, including some new provider failures, remain English.

## Verification

- Backend regression suite: **955 passed, 17 existing optional tests skipped**. The unrelated, user-owned `tests/test_sample_workflow.py` was excluded. A focused rerun covers final copy/readiness changes.
- Frontend baseline: **149 unit tests passed**, typecheck and production build passed.
- Isolated localhost backend/Vite/Chromium: **6 journeys passed**: ChatGPT in English and Italian, xAI endpoint/key separation, existing Ollama and LM Studio recovery, and settings/support.
- OAuth tests include a real localhost callback plus mocked token/JWKS/provider responses. Other coverage includes signed JWT validation, wrong account/state, callback replay, abandoned-grant revocation, renewal concurrency, uncertain rotation across restart, IPC token supply, redaction, fixed endpoints and interrupted streams.
- Claude CLI reviewed selected source twice; OCR delegate supplied file selection and rules. See the [review record](../../reviews/2026-10-04-cloud-connections-implementation.md).
- Added maintained Python clients: OpenAI SDK, its pinned httpx2 transport, and PyJWT with cryptography. No npm dependencies changed. `pip check`, repository quality checks and diff whitespace checks passed.

## Remaining live checks

1. Restart this checkout’s app/backend, authorize your own eligible ChatGPT account, and run **Test provider connection**. Account/app availability and real model/schema acceptance are not proven by mocks.
2. Complete one existing English and one Italian exercise using that connection. Restart and reconnect; then disconnect and confirm provider-side revocation.
3. Repeat the connection and exercise checks with an xAI API key and credits. No user xAI key was available during implementation.
4. Native packaged-browser launch and Windows/Linux secure-storage behavior were not exercised. The tested browser is local Chromium. In-memory storage requires signing in or entering the key again after a backend restart.

## Sources checked October 4

- [OpenAI local-app plan usage](https://developers.openai.com/siwc/token-sharing-open-source)
- [Registration, callback binding and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in) — callback port may change between attempts; scheme, host and path stay fixed.
- [Accounts, renewal and revocation](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions) — retain client/account mapping and host ID after sign-out.
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
- [xAI quickstart](https://docs.x.ai/developers/quickstart) and [structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs)

Cloud ASR, further providers, upload redesign, exam simulation, new assessment benchmarks, and automatic paid fallback are outside this change.


## Groq addition — October 4

Decision: include Groq as an optional way to start cloud feedback on a free account. Reuses the existing API-key form, secret storage, fixed endpoint validation, model discovery and schema probe. No new dependency or login framework. There is no automatic upgrade or provider fallback. An already-paid Groq account remains subject to Groq billing.

The [Free plan limits](https://console.groq.com/docs/rate-limits) currently list GPT-OSS 120B at 30 requests/minute, 1,000 requests/day, 8,000 tokens/minute and 200,000 tokens/day; exact organization limits can vary. These are API requests, not completed exercises: one attempt uses multiple feedback requests. Long prompts and successive calls can exhaust the token allowance. Do not promise a fixed number of free exercises. Groq returns 429 on quota exhaustion; the app reports the limit and retains the existing local-report fallback, without automatic paid fallback or repeated quota retries.

[Groq billing](https://console.groq.com/docs/billing-faqs) distinguishes Free from paid Developer accounts; upgrading requires a payment method. [Strict structured outputs](https://console.groq.com/docs/structured-outputs) support GPT-OSS 20B/120B but not streaming. The adapter therefore uses nonstreaming Chat Completions with the existing strict schemas, bounded output, no SDK retries, and low reasoning effort for those models. Refusals, truncation and empty responses are rejected.

Groq verification: 123 focused backend tests passed (the unchanged OAuth socket test was deselected); 149 frontend tests, typecheck and five isolated Chromium journeys passed, including Groq setup and quota recovery in English and Italian. Browser test provider replies are mocked. No Groq account key was supplied, so actual free-account inference, English/Italian feedback quality and long-attempt quota feasibility remain unverified. Setup is available; successful live feedback is not yet claimed. See the [Groq review](../../reviews/2026-10-04-groq-connection.md).

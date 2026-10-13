# Connect cloud transcription and feedback

October 6 repair update: the offline cloud-gap slice and new internal DMG are accepted. See [implementation evidence](../../reviews/2026-10-06-cloud-gap-implementation.md) for current status and remaining gates.

**Date:** 2026-10-04. **Status:** Implemented baseline with confirmed October 6 defects; remediation reviewed and pending. Live user authorization and provider inference remain to be verified.

The [cloud-gap remediation plan](2026-10-06-cloud-gap-remediation.md) is the current execution plan. It repairs failures missed by the local passing suites before cloud-ready delivery can be claimed.


## October 6 implementation extension

The user requested implementation after the speaking-journey fixes. This section supersedes the October 4 exclusions below for cloud transcription, OpenRouter browser login, free-only routing, recovery and managed paid OpenRouter fallback. The earlier sections retain the connection milestone and its evidence.

| Option | Current implementation | Setup |
| --- | --- | --- |
| Groq cloud transcription | Independent saved Groq connection; packaged PyAV prepares mono 16 kHz lossless FLAC; actual-byte upload cap; overlapping chunk timestamps; completed-chunk recovery. | Save a Groq connection, then choose Groq under Cloud services → Transcription and select the speech model. |
| OpenRouter free-only | Browser PKCE login or saved key; explicit `:free` model; live zero-price verification; zero price caps; no automatic router or provider fallback. | Connect OpenRouter, explicitly select/test/save a structured-output model, then set that connection to Free models only. |
| ChatGPT plan login | Existing browser authorization, plan model selection, backend renewal and Responses inference. | Continue with ChatGPT, authorize available plan access, choose an available model and test. Availability depends on the account and app preview. |
| Paid OpenRouter | Explicit Paid mode, finite provider key limit, output cap, monthly app reservations and reported-cost reconciliation. Optional fallback is disabled until enabled with a distinct saved connection. | Fund the account, set a key limit at OpenRouter, save a model, choose Paid with spending controls and save the monthly app budget. Enable fallback separately if desired. |
| xAI Grok | Existing paid API connection remains available. | Use an xAI API key and credits. This does not connect a free Grok consumer login. |

Cloud services is available in guided setup and Settings. Cloud transcription does not require downloading local Whisper weights. Existing manual OpenRouter connections retain their prior billing behavior until Free or Paid mode is chosen; the OpenRouter app budget does not cover Groq, xAI or other providers.

### Recovery and speaking fixes

- Lost accepted submission responses recover through a stable request ID; the backend creates one job for repeated identical submissions.
- Checkpoints bind acoustic, transcription and validated feedback stages to the recording hash and relevant analysis inputs. Failed coaching can be retried while reusing a successful rubric.
- Backend provider replies are persisted before returning over correlated IPC. A dead or mismatched pipe stops further dispatch. Matching requests serialize against an older in-flight reply rather than issuing another concurrent paid request.
- Unknown remote outcomes remain reserved. Settings can reconcile a reservation only after the user confirms the provider's actual charge; that decision is recorded. Exactly-once remote execution is not promised after an unknown HTTP outcome.
- Analysis revisions keep one history take and its original practice date while retaining previous report files. Older retained jobs without checkpoints keep their existing take identifier.
- Progress compares raw rubric overall values against prior raw values. Reworded priorities no longer imply that a weakness was resolved. The existing saved-locale restoration is preserved.
- Missing cloud models, deleted selections and unsupported saved provider kinds cannot silently select another account or a paid default. Managed cloud keys stay in the backend rather than worker payloads or environment variables.

### Quality and release boundaries

Groq transcription remains a preview: confidence is unknown, human review is required, and the report asks the learner to check the transcript against the recording. Language detection is automatic; supplied language labels are retained without inventing a probability. Real learner-error retention and comparative English/Italian coaching quality have not been calibrated. No model is newly promoted as the quality winner.

This implementation reuses the installed HTTPX, PyAV and credential store. It adds no npm or Python dependencies. The earlier proposed SDK/Authlib expansion is unnecessary for these bounded adapters; the existing verified ChatGPT OAuth foundation is preserved. This is provider access inside the local app, not a hosted multi-user deployment.

See [implementation, review and verification record](../../reviews/2026-10-06-cloud-plan-implementation.md) for final tests, the internal DMG and remaining live checks.

### Sources rechecked October 6

- [Groq speech-to-text contract](https://console.groq.com/docs/speech-to-text): supported speech models, free-plan upload limits and timestamp output.
- [OpenRouter PKCE authorization](https://openrouter.ai/docs/guides/overview/auth/oauth): authorization/key exchange, state echo and loopback callbacks.
- [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection): price caps, output parameter requirements and endpoint fallback control.
- [OpenRouter key limits](https://openrouter.ai/docs/api_reference/limits): current key limits and remaining allowance.
- [ChatGPT local-app preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations): text plan inference does not provide audio transcription.

## October 4 connection milestone (historical scope)

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


### Groq UX and live compatibility follow-up

Groq now appears in initial setup without opening advanced providers, and remains available in Settings. A native model picker offers `openai/gpt-oss-120b` (recommended starting point), `openai/gpt-oss-20b` (faster), and `qwen/qwen3.8-27b` (preview). Incompatible speech, voice and moderation models are not offered as feedback models. Existing unsupported saved models require an explicit replacement before test/save; valid saved choices are retained. Copy is translated across all five interface locales.

Live verification on October 4 used the user-supplied key from `.env` in a temporary test process. All three models passed Vostavo's existing strict rubric capability probe: 120B in 1.25 seconds, 20B in 0.52 seconds, Qwen in 0.41 seconds. These single small requests establish key/model/schema compatibility, not comparative coaching quality, free-tier account status or full-attempt throughput. No learner recordings or private reports were submitted. The app still expects its saved API key through Settings; `.env` auto-import is not part of this UI change.

Follow-up verification: 153 frontend tests, 125 focused backend tests (unchanged OAuth loopback test deselected), seven isolated Chromium journeys, typecheck, production build, repository quality and whitespace checks passed. Groq picker tests cover English/Italian, all model selections, save/test payloads, incompatible saved models, restored selections and quota recovery. See [review](../../reviews/2026-10-04-groq-picker-review.md).

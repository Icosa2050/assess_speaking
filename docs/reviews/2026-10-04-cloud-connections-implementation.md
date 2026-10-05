# Cloud connection implementation review — 2026-10-04

## Scope and result

Reviewed ChatGPT local-app OAuth/inference and xAI Grok API access. Existing OpenRouter behavior is retained. Source-only Claude CLI review was authorized by the user's standing approval. Two Claude passes and an OCR delegate host review were completed. The raw comments below describe intermediate versions; the disposition here describes the resulting implementation.

## Findings addressed

- Successful login is published by the backend callback. Publication recovery metadata survives restart. Browser navigation stops polling without cancelling authorization; the pending-attempt endpoint allows resuming it.
- Abandoned/unusable grants are revoked. Reconnect validates the same subject/client; old-session revocation is attempted. Explicit disconnect cancels active ChatGPT jobs, removes local preferences while locked, clears credentials, and reports whether revocation was confirmed.
- A private process pipe supplies a current access token before every inference. Workers do not own renewable credentials. Reauthorization and active ChatGPT assessments cannot overlap.
- Refresh writes a durable pending marker before exchanging a rotating credential. Definite non-authentication 4xx rejections restore the usable old record; unknown outcomes retain a revocable record but prohibit replay. Invalid replacement responses revoke/clear the grant.
- Threaded loopback callbacks have socket read timeouts. Malformed response/token failures become terminal states. Corrupt registration JSON is preserved for recovery instead of stopping local practice.
- Preference updates use a shared lock and re-read after provider requests. Saving locale/Whisper with ChatGPT cannot overwrite OAuth secrets.
- JWKS and provider HTTP requests reject redirects. PyJWT verifies signatures, issuer, audience, expiration, subject and nonce. Mid-stream httpx/httpx2 errors and broker failures follow the existing LLM-unavailable report path.
- UI language changes no longer restart the polling effect. Lost attempts stop polling; completed login is independent of polling. xAI has a usable backend-memory fallback when secure storage is unavailable.
- Provider switching clears an unsaved key. ChatGPT/xAI endpoint overrides and unrelated environment-key fallback are rejected. New providers receive an appropriate reconnect message instead of an OpenRouter-key warning.

## Review suggestions checked against primary sources

- Callback **ports may change** between sign-ins; scheme, host and path must stay unchanged, and each token exchange must use that attempt's exact URI. No fixed-port workaround was added. [OpenAI sign-in specification](https://developers.openai.com/siwc/token-sharing-open-source/sign-in).
- Sign-out **retains account/client mapping and host ID**. Deleting that registration, as suggested in one review, would conflict with the documented flow. Tokens are cleared separately. [OpenAI account/session specification](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions).

## Verification and limits

- Backend: `955 passed, 17 skipped` with `.venv/bin/python -m pytest -q --ignore=tests/test_sample_workflow.py`. The excluded file is unrelated uncommitted user work. Optional integrations retain their existing skip conditions.
- Frontend: `149 passed`; `npm --prefix frontend run typecheck` and `npm --prefix frontend run build` passed.
- Browser: six Chromium checks passed (ChatGPT English/Italian, xAI key/endpoint separation, Ollama/LM Studio recovery, settings/support). Provider responses were mocked. The backend/Vite servers and OAuth loopback callback were real localhost processes.
- The initial browser command using `playwright.config.ts` reused another checkout's running server and failed to find the new provider options (`locator.selectOption: did not find some options`, 2026-10-04). A temporary inherited config used backend 8817/frontend 4187, `reuseExistingServer:false`, and separate temporary app data. The successful command was `npm --prefix frontend run test:e2e -- --config cloud-check.playwright.config.ts tests/e2e/cloudConnections.spec.ts tests/e2e/runtimeSetupRecovery.spec.ts tests/e2e/settingsSupportFlow.spec.ts`. The temporary config was removed after verification; normal CI uses the existing config with fresh servers.
- The initial sandboxed port-allocation test got `PermissionError: [Errno 1] Operation not permitted`. The isolated localhost rerun and final full suite ran with approved local-network permission and passed.
- `pip check`, repository quality checks and diff whitespace checks passed. Python additions reuse the official OpenAI SDK, its pinned httpx2 transport and PyJWT/cryptography. No npm dependency/install-script changes.
- Live account eligibility, native packaged browser launch, provider billing/allowance behavior, real xAI schema acceptance and Windows/Linux secure-storage behavior remain unverified. Localhost applications remain within the existing desktop trust boundary; this is not hosted user authentication. Some backend error text remains English.
- No live provider credentials, learner recordings or reports were sent to reviewers. No new paid inference run is claimed.

## Claude pass 1 (before fixes)

I found 3 high, 7 medium and 7 low severity issues. I made no edits and ran no tests. Line numbers come from the files on disk, which have changed since the snippet you sent: `chatgpt_auth.py` now has `open_browser()` at :311 and stores `authorization_url` on the pending attempt, and `ChatGPTConnection.tsx` has an `isDesktopRuntime()` branch. Both additions look fine.

## High

**H1. A finished sign-in only gets saved if the frontend polls once more** (`app_backend/chatgpt_routes.py:41-59`)
- `finish()` writes the tokens and the registration record at `chatgpt_auth.py:254-258`. The `ProviderConnection` is only created when the next `GET /attempts/{id}` arrives.
- Any of these leaves a live grant in the keyring with no connection in the UI:
  - the component unmounts or the locale changes within the ~1.5s gap (the cleanup's cancel ignores `connected`);
  - the window closes;
  - the backend restarts, since `pending` lives in memory only.
- When the user retries, a new dynamic client gets registered.
- **Fix:** save the connection inside `finish()`, under the same locks, before setting `connected`. Or rebuild missing connections from `records['accounts']` at startup.

**H2. A grant issued after cancel, disconnect or a validation failure is never revoked** (`chatgpt_auth.py:236-261`)
- Once the token exchange at :236 succeeds, OpenAI holds a refresh token with `offline_access`. None of these exits revoke it:
  - the early return at :252-253 (cancelled or disconnected while processing);
  - a scope that doesn't match (:241);
  - a different `sub` on reconnect (:239);
  - no available models (:246);
  - a save failure (:260).
- The user can't revoke it from the app.
- A reconnect also overwrites the earlier refresh token at :254 without revoking it.
- **Fix:** once `result` has a `refresh_token`, call the revocation endpoint on every path that doesn't commit, outside the locks.

**H3. The access token is fetched at submit but used later in a separate worker process; it may expire mid-run, and Disconnect doesn't stop the job** (`app_backend/app.py:331`, `app_backend/jobs.py:244`, `chatgpt_auth.py:86`)
- `access_token()` only guarantees 300s of validity. ASR with large-v3 on CPU, plus the rubric call, a validation retry and coaching, can take longer than that.
- When it runs out, the job gets a 401 and is told "Reconnect your account" even though the account is fine, and the feedback is lost.
- Disconnect doesn't stop running jobs, so the transcript keeps going to OpenAI after the user disconnects, until the token expires.
- **Fix:** force a refresh at submit when the remaining lifetime is below the job budget (ASR time plus 3× `llm_timeout`). In the disconnect route, cancel any active job whose provider is `chatgpt`.

## Medium

**M1. A keyring write failure during refresh leaves an already-used refresh token persisted** (`chatgpt_auth.py:61-74, 104`)
- If the keyring write fails, the new refresh token is kept only in session memory. The keyring still holds the previous one, which OpenAI has already rotated out.
- After a restart, that old token is sent again. That gives `invalid_grant` at best. At worst, reuse detection revokes the whole token family.
- `provider_metadata.persistent` still says `true`.
- **Fix:** on fallback, delete or overwrite the keyring item and update `persistent`.
- **Related, unverified:** Windows Credential Manager limits a credential to 2560 bytes. Access + refresh + id_token as JSON probably exceeds that, so Windows would always be session-only. Drop `id_token` from `stored` (:249) and from the refresh merge (:101). It isn't used after validation, and `subject` is already stored.

**M2. Network calls run while the locks are held** (`chatgpt_auth.py:280-303`; `chatgpt_routes.py:43-59`)
- `disconnect()` holds `manager.lock` and `_LOCK` through discovery plus revocation, up to ~40s.
- During that time these all block:
  - status polls;
  - the OAuth callback's `finish()`, so the browser page hangs;
  - every `access_token()` call: assessment submit, test connection, model change.
- The status route also does `load_state` and `save_provider_connection` under `manager.lock`. That includes keyring I/O, which can mean a keychain prompt on macOS.
- **Fix:** under the lock, take a snapshot and write the tombstone, then revoke after releasing it.

**M3. The loopback callback server can stall** (`chatgpt_auth.py:175-201`)
- `HTTPServer` handles one connection at a time, and `BaseHTTPRequestHandler` has no read timeout.
- An idle connection, such as a browser's speculative preconnect, blocks `handle_request()` while it reads the request line. The real redirect waits behind it, and the expiry/cancel loop can't run.
- **Fix:** set `timeout = 10` on `Callback`, and/or switch to `ThreadingHTTPServer`.

**M4. `finish()` catches too few exception types** (`chatgpt_auth.py:260`)
- These escape the handler:
  - `TypeError` from `float(None)` when `expires_in` is `null` (:243/:250);
  - `AttributeError` or `TypeError` in `account_models` (:110) when the `models` list contains something other than an object.
- The attempt then stays in `processing` forever. The frontend keeps polling (`ChatGPTConnection.tsx:48`), and `start()` refuses new attempts for up to 10 minutes (:166).
- **Fix:** catch `Exception` and set `failed` in `finally` if the status is still `processing`.

**M5. Concurrent prefs changes can overwrite each other** (`chatgpt_routes.py:69-78`)
- `model()` loads state, makes two network calls, then saves. A disconnect, publish or settings save that lands in between gets overwritten.
- For example, it can bring back a disconnected ChatGPT connection that points at a cleared secret.
- The UI only prevents this within the one component.
- **Fix:** reload state after the network calls, under one process-wide prefs lock.

**M6. A dead refresh token is reported as "please retry"** (`chatgpt_auth.py:38-43, 91`)
- A revoked, expired or reused refresh token comes back as `400 invalid_grant`. That falls into the generic "could not complete… retry" message.
- The dead token set is kept and sent again on every call.
- **Fix:** check the response's `error` code without echoing the body. On `invalid_grant`, clear the tokens and ask the user to reconnect.
- **Also:** `saved['client_id']` at :91 is unguarded; a `KeyError` there becomes a 500.

**M7. Changing the UI language cancels an in-progress sign-in** (`ChatGPTConnection.tsx:69-71`)
- `locale` is in the effect's dependencies, so a language change runs the cleanup. The cleanup posts cancel, and the re-run effect then polls `cancelled`.
- Combined with H2, a callback that arrives while the attempt is processing leaks a grant.
- **Fix:** depend on `[attempt]` only and read `t` through a ref.

## Low

- **L1. Disconnect keeps personal data.** The `email` and `subject` stay in `chatgpt-registrations.json` (`:279-303`; the test even asserts the record survives). Remove the record, or at least the email.
- **L2. "Revocation confirmed" can be wrong.** It returns true when no refresh token was found (`:289`), for example session-only tokens lost after a restart. The grant is still live at OpenAI but the UI says it was revoked. Return false or "unknown" instead.
- **L3. A corrupt registrations file stops the backend from starting.** `__init__` doesn't catch it (`:148`) and doesn't validate its structure.
- **L4. Responses client limits:**
  - The deadline is only checked when an event arrives (`responses_client.py:30`), so a slow trickle can run past it.
  - xAI's `max_output_tokens=4096` (`:22`) likely counts reasoning tokens on Grok reasoning models, which would cause `response.incomplete` on rubric generation. Worth checking.
- **L5. The local-origin check is broad (existing pattern).**
  - `LOCAL_GUEST_ORIGIN_REGEX` (`app.py:99/444`) accepts any `http://localhost:<port>`. Any other local web app passes the CORS preflight and can drive the ChatGPT routes.
  - The Host allowlist only covers the ChatGPT router. DNS rebinding can still read other GET routes, including history and learner data, because a same-origin GET carries no Origin header.
- **L6. JWKS redirects aren't checked.** `PyJWKClient` uses urllib, which follows redirects, and `_trusted_endpoint` only checks the first URL (`:131`).
- **L7. Frontend error handling.**
  - `response.json()` runs before the `ok` check (`ChatGPTConnection.tsx:16`), so a non-JSON error body shows a parse error.
  - The poll retries forever on "attempt not found", for example after a backend restart (`:59-63`).

## Checked and fine

- The token stays out of job metadata (`jobs.py:66`) and out of the prefs file (`services.py:715`).
- Saved keys from a different provider aren't reused: the provider and base URL must match (`app.py:341-346`).
- Setting a default or deleting a connection syncs the new active connection before saving, so no key is copied to the wrong connection.
- Test-connection redacts the key from error messages.
- Every outbound call has `follow_redirects=False`.
- OAuth checks are in place:
  - state is matched exactly;
  - PKCE uses S256;
  - the nonce is compared in constant time;
  - the issued `client_id` is checked against the stored record;
  - `sub` must match on reconnect.
- The xAI path has no environment-variable fallback.

## Not checked

- The SIWC endpoint and parameter details against OpenAI's docs. I took your summary as given.
- The five locale files.
- Actual Windows keyring behaviour (M1).


## Claude pass 2 (before final fixes)

I found 2 high, 4 medium and 6 low-severity issues. The OAuth core looks right: PKCE, state, nonce, the issued client id, ID-token checks, and pinned endpoints. The problems are in **refresh-failure handling**, **how failures reach the assessment pipeline** and **reconnect/disconnect lifecycle**. I made no edits. Line numbers come from the current on-disk files. The `_save_runtime_settings` ChatGPT branch on disk (`app_backend/app.py:279-287`) already differs from your pasted diff: it allows locale and Whisper saves.

## High

**H1. A failed refresh can lose the only stored copy of a valid refresh token.** `app_core/chatgpt_auth.py:143-157`
- `access_token()` deletes the keyring copy before the refresh request. Only invalid_grant, 401 and 403 count as a known outcome.
- Any other failure leaves the old refresh token only in process memory: going offline, a timeout, 5xx, 429, or another 400.
- If the app restarts in that window, the account is gone and the server-side grant stays live. Nothing local remains that could revoke it.
- While the app keeps running, the UI quietly switches to "session only" (`app_backend/app.py:181`).
- **Fix:** restore the keyring copy whenever the server definitely did not rotate the token (any non-2xx other than invalid_grant). For transport errors where the outcome is unknown, keep a persisted "rotation pending" marker rather than deleting.

**H2. A token rotation that succeeds is thrown away if the response then fails validation.** `app_core/chatgpt_auth.py:158-162`
- Three checks raise after the server has already rotated the token: missing `access_token`, a bad `expires_in` (`_lifetime` raises), and missing the `chatgpt.tokens.use.direct` scope.
- The new refresh token is dropped without being revoked. The session store still holds the old, already-used token.
- The next call replays that used token. With rotating refresh tokens, reuse detection may revoke the whole grant, and the dropped live token stays out there in the meantime.
- **Fix:** on any failure after a 2xx response, revoke `result['refresh_token']` and clear local tokens (`_clear_tokens`).

## Medium

**M1. Network errors mid-stream, and token-broker errors, crash the whole assessment.**
- `assessment_runtime/responses_client.py:28-46`: the OpenAI SDK raises errors during stream iteration as raw httpx exceptions, not `OpenAIError`. Examples are `RemoteProtocolError` and `ReadTimeout`. They skip the `ResponsesError` mapping, and `_chat_completion` (`llm_client.py:216-220`) never turns them into `LLMClientError`.
- `app_backend/jobs.py` (`current_access_token` in `_job_worker`) raises `RuntimeError` for renewal failures and timeouts.
- `generate_rubric` and `generate_coaching_summary` only catch `LLMClientError` (`llm_client.py:334`, `387`). `run_assessment` only degrades gracefully for `LLMClientError` (`assess_speaking.py:982`, `1031`).
- Result: a dropped connection or an expired session fails the job outright and discards the local ASR result, instead of producing the usual `llm_unavailable` / `coaching_unavailable` report.
- **Fix:** catch `httpx.HTTPError` in `complete()`, and raise `LLMClientError` from the worker's token supplier.

**M2. Reconnecting may always fail because each attempt uses a new random port.** `app_core/chatgpt_auth.py:272-274, 287`
- Reconnect reuses the stored `client_id` but binds a fresh ephemeral port, so `redirect_uri` changes every time.
- Your docs summary says the loopback callback must match exactly. If that includes the port registered with that client id, every reconnect fails at authorize or token exchange.
- Please confirm against the docs. If the port is pinned, store it in the record, try to rebind it, and fall back to a new registration (`dynamic_agent_client`).

**M3. Reconnect leaves the old grant live and breaks a running job.** `app_core/chatgpt_auth.py:347-348`
- After a successful reconnect, the old token set is only cleared locally (`_clear_tokens(record['secret_ref'])`). Its refresh token is never revoked.
- A job already running keeps the old `credential_ref`, captured at `jobs.py:397`. Its next token request (`access_token`) hits the cleared ref, and the assessment fails with "Reconnect".
- **Fix:** revoke the old refresh token after commit. Also either block reconnect while a ChatGPT job is running, or let the token broker look up the connection's current ref.

**M4. Leaving the screen cancels a sign-in the user already approved.** `frontend/src/components/setup/ChatGPTConnection.tsx`, effect cleanup
- The polling effect's cleanup sends `cancel`, and backend `cancel` also applies to attempts in `processing` state.
- Navigating away, or switching the provider dropdown, after approving in the browser cancels mid-exchange. `finish` then revokes the new grant.
- This undoes the backend's "publishes without frontend polling" design, which your own test asserts.
- **Fix:** only cancel on an explicit user action, or only while the attempt is still `waiting`.

## Low

- **L1. Disconnect is not atomic.** `chatgpt_routes.py:90-96`, `chatgpt_auth.py:387-396`
  - The saved connection is removed from preferences only after a slow revocation, outside `manager.lock`. If a sign-in for the same `connection_id` commits in that window, the fresh connection gets deleted and its tokens are orphaned.
  - The registration record is never removed, so `start(connection_id)` still accepts a disconnected id.
  - **Fix:** delete the saved connection and the record inside the lock, then revoke.
- **L2. The token broker thread can die silently.** `jobs.py:409-412`: `_serve_credentials` only catches `ChatGPTAuthError`. `access_token` can also raise `ValueError` or `KeyError`, for example `float(expires_at)` on corrupted data (`chatgpt_auth.py:139`). The thread dies and the worker gets an EOF error.
- **L3. The ChatGPT access token sits in general app state.** `runtime_resolver.py:51-53, 111`
  - Every state load copies the cached access token into `RuntimeConfig.api_key` and `prefs.llm_api_key`. Only the check at `services.py:694` stops it being written into another connection's keyring slot.
  - Any future path that changes the active connection between loading and saving without resyncing would store the OAuth token under another provider and send it there.
  - `has_api_key` also reports expired tokens as present.
  - **Fix:** return a presence flag for ChatGPT instead of the token itself.
- **L4. Lock held during network calls.** `create_assessment` holds `chatgpt_auth.lock` while calling `access_token()`, which may refresh over the network for up to 20 s. That blocks `status()` polling and the callback's `finish()`.
- **L5. English-only messages in the UI.** Backend `ChatGPTAuthError` and `status().detail` strings are shown directly in the UI (`result.detail.detail`), which breaks the five-locale requirement for those messages.
- **L6. Weak local-client check.** `X-Vostavo-Client: desktop` is a fixed header, and CORS accepts any `http://localhost:<port>` origin with any headers. Any local web page can start a sign-in, disconnect, or change the model. No tokens are exposed; a per-launch secret would close this.

Smaller nits:
- `ChatGPTAuth.__init__` does not catch `OSError`.
- On restart, the republish loop can publish accounts whose session-only tokens are already gone.
- The `'{}'` tombstone makes cleared connections show as "session only".

## Checked and fine
- Access tokens never reach job metadata: `_sanitize_request_metadata` (`jobs.py:66`) runs before the write, and the worker gets tokens only through the pipe.
- Cancelling or failing before commit revokes the new grant.
- Lock order is consistent (manager lock, then `_LOCK`, then `_submit_lock` / `PREFERENCES_LOCK`). I found no deadlocks.
- Both endpoints are hard-coded with redirects off. `store:false` and `stream:true` are set, and `max_output_tokens` is sent only to xAI.
- Changing the default or deleting a connection resyncs before saving, so the token does not leak there today.

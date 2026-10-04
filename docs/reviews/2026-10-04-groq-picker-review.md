# Groq picker review — October 4, 2026

Claude CLI reviewed the selected diff and new component tests under standing user approval. No credentials or recordings were included. OCR delegate supplied selection and system rules; host review covered all six reviewable files, plus all three changed test files and the updated plan (100% reviewable coverage, zero skipped).

## Disposition

- Strengthened browser assertions with model-specific response messages, so a previous successful probe cannot satisfy the next model's result assertion.
- Hid stale discovery messages for Groq alongside the irrelevant detected-model picker.
- Clarified English/Italian copy: the cloud provider receives practice text.
- Confirmed existing adapter already limits `reasoning_effort=low` to GPT-OSS models. All three parametrized backend cases and live strict-schema probes passed.
- The curated picker guides supported feedback model selection; it is not a backend security allowlist and does not migrate existing active connections silently.

## Verification

153 frontend unit tests; 125 focused backend tests (one unchanged loopback case deselected); seven Chromium journeys; typecheck and build passed. Final follow-up reran the four component cases, typecheck and five cloud Chromium journeys. Live account/schema probes passed for 120B, 20B and Qwen. Full coaching quality comparison and account billing tier were not evaluated. The `.env` key was used only in memory by the temporary probe, not copied to committed files or app preferences.

## Raw Claude review before fixes

I reviewed only the diff. I didn't run anything or open other files. I found no blocking bugs in the component logic: the select, the placeholder for an unsupported saved model, the validation gate, and Groq appearing in initial setup all hang together. These are worth fixing or checking:

### 1. The e2e test can pass even if the 20B and Qwen probes never show their results (medium)
`frontend/tests/e2e/cloudConnections.spec.ts`: by the time the loop starts, `form_status` already says "Structured feedback verified" from probe 2. `expect.poll(() => probes)` only shows the request reached the route, not that the UI handled the response. So `toContainText("Structured feedback verified")` will pass straight away for the 20B and Qwen runs, unless choosing a different model clears the status. Nothing in this diff does that, and I didn't check `updateDraft`.
- **Fix:** before each click, either assert that the status is cleared or changed after `selectOption`, or have the fulfilled response include the model and check for it in the status.

A related, smaller point: the `expect(...).toMatchObject` inside the route handler throws in the handler. A model mismatch then shows up as a request that hangs or times out, not as a clear assertion failure. The old test already did this, but the new loop makes it more likely to matter.

### 2. The Python test expects a backend change that isn't in this diff (verify)
`tests/test_cloud_accounts.py` now asserts `'reasoning_effort' not in payload` for `qwen/qwen3.8-27b`. The git status shows no change under `assessment_runtime/`. So this only passes if `llm_client` already sends `reasoning_effort` just for `openai/*` models. If it sends `"low"` for every Groq request, the Qwen case fails here, and real Qwen requests may be rejected. That would conflict with your passing live probes unless those went through a different code path. Run the parametrized test before committing.

### 3. Old Groq connections with other models are only blocked in the form (low, unverified)
`validateDraft` only blocks Test and Save. A Groq connection saved earlier with a free-text model (say `llama-3.3-70b-versatile`) stays active and keeps being used until someone opens Settings. If you want these three models to be the only ones allowed, the backend has to enforce it too. Otherwise this is fine as a UI-only guide.

### 4. The detected-models message can still show for Groq (low)
The list of detected models is now hidden when `providerChoice === "groq"`, but `detectedModelMessage` still renders. If detection fills in that message (for example "Found N models"), the user sees text about a list that isn't there. Apply the same `providerChoice !== "groq"` guard to it.

### 5. Wrong wording in the en and it intro text (low)
- en: "Cloud providers **send** practice text to the selected service"
- it: "I provider cloud **inviano** il testo…"

Both say the providers send the text. It's the app that sends text to them. de, es and fr say it correctly ("erhalten", "reciben", "reçoivent"). Suggested en: "Cloud providers receive your practice text." Suggested it: "I provider cloud ricevono il testo degli esercizi."

### Checked and fine
- Showing the disabled placeholder only when the model is unsupported keeps the `<select>` controlled and valid, including when the model is an empty string.
- Taking `groq` out of `ADVANCED_PROVIDER_CHOICES` means picking it in setup no longer opens the advanced section on its own.
- The `runtime_setup.groq_models.*` keys are hard-coded instead of using `${copyRoot}`. That's correct as long as they live only under `runtime_setup`, which the locale diff suggests.
- The component test assumes that switching to Groq sets the model to `openai/gpt-oss-120b`. That default comes from code outside this diff, so I took it as given.

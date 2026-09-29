# Claude CLI raw review

Read-only review of selected source files/diffs. Includes unconfirmed hypotheses; see the bilingual journey report for validated dispositions.

Read-only review from the diffs and full client only; no tools used.

## Confirmed findings

**1. Ollama schema failures are now reported as `llm_unavailable` instead of `llm_invalid_schema`**
- The removed local-Ollama branch in `run_assessment` added `warnings.append("llm_invalid_schema")` when validation failed.
- Now `generate_rubric` raises `LLMClientError("Failed to produce valid rubric output: …")` after retries. The single `except LLMClientError` catches it and adds `llm_unavailable`.
- A model that answered with bad JSON looks the same as one that is down or timed out. This affects UI copy, triage, and any test or locale string keyed on `llm_invalid_schema`.
- Fix: raise a separate exception for validation failure (e.g. `LLMSchemaError(LLMClientError)`) and catch it first. Or at least check the message prefix. The same misclassification already existed for the other providers, so this fixes all of them.

**2. The selftest error key changed**
- The key went from `ollama_not_running_or_model_missing` to `ollama_unavailable`.
- Anything that parses the selftest output (backend health/selftest endpoint, frontend, locales, docs) and matches the old key will stop matching.
- Grep for the old string before merging.

**3. The selftest returns different shapes depending on path**
- The local shortcut (`call_ollama`) returns `raw`: the unnormalized model text, possibly fenced or with extra prose.
- The non-shortcut path returns data built from the validated `rubric`.
- On failure, the shortcut returns `{"error":"ollama_unavailable",...}`, while the other path uses its own error handling.
- Since `call_ollama` now just wraps `generate_rubric`, the shortcut adds nothing. Consider deleting the branch in `selftest` so both paths behave the same, and keep `call_ollama` only as a thin export if something external imports it.

**4. The local Ollama path is now bounded by `llm_timeout_sec`, including cold model load**
- The old curl path had no timeout. Now each attempt is bounded by `chosen_llm_timeout`.
- The first request after Ollama starts, or after the model is unloaded, includes loading the weights. That can take a long time for larger models, CPU-only machines, or low RAM.
- This is intended for the hang case, but it's a real regression for slow setups that used to finish. Check the `Settings` default against your measured ~25–32 s for rubric plus coaching.
- Worst case per stage is 2× timeout, because a validation retry is a full second request. Timeouts themselves are not retried, since `_chat_completion` sits outside the `try`.

**5. `llm_inference_profile` is set for every Ollama run, even without inference**
- It's set in dry runs and when the LLM failed and scoring fell back to deterministic.
- This splits `analysis_signature` for deterministic-only Ollama runs away from older deterministic runs, even though no LLM setting affected those scores.
- Fix: set it only when a rubric was actually produced (for example `rubric_obj is not None`). The string is also hardcoded twice in `assess_speaking.py`, away from the payload it describes. Define it as a constant next to the Ollama payload in `llm_client.py` so the two can't drift.

## Hypotheses (verify)

**H1. `reasoning_effort: "none"` on models without thinking support**
- Ollama rejects `think` on some models that don't support thinking.
- Every Ollama request now sends this, including `test_connection` and the default `llama3.1` used in tests.
- Run one live request against a non-thinking model (llama3.1 or gemma). A 400 there would break those users completely.

**H2. `runner.py` may not import `json`**
- `_build_meta` now calls `json.dumps`, but the diff has no import hunk.
- If `json` isn't already imported at the top of `runner.py`, every run crashes with `NameError`.

**H3. `analysis_signature` leaves out provider and model**
- Old Ollama reports have `llm_inference_profile = None`, the same value as new OpenRouter or LM Studio reports.
- If `PracticeProgress` or the history comparison groups by `analysis_signature` alone, old Ollama and new OpenRouter results would look comparable.
- Confirm that comparability also checks `practice.provider` and `practice.model`, or add them to the signature.

**H4. The new validator may be stricter than the old one**
- The old local path used `_validate_rubric_payload(extract_rubric_json(...))`.
- If that did lenient coercions (string scores, clamping, key aliases) that `RubricResult.from_dict` doesn't, Ollama schema failures would increase.
- Diff the two validators.

**H5. The provider comparison may miss aliases**
- The profile check uses `chosen_provider == "ollama"`, while the client uses `normalize_provider`.
- If `chosen_provider` isn't normalized yet at that point, aliases or different casing would record `None` even though the Ollama payload was sent.

**H6. The new client test may be sensitive to the environment**
- `test_ollama_requests_bounded_json_without_thinking` asserts the exact URL `http://localhost:11434/v1/chat/completions`.
- If `runtime_base_url` reads an environment variable such as `OLLAMA_HOST` or `OLLAMA_BASE_URL`, the test fails on machines where it's set. Patch the environment in the test.

**H7. The token cap can truncate JSON**
- `max_tokens=4096` also applies to coaching.
- If output is cut off at the cap, the JSON is invalid, triggers a validation retry, and then appears as `llm_unavailable` (see finding 1).
- Probably fine for short output. Worth one long-transcript probe.

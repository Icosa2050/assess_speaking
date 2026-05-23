# Runtime Setup Streamline Implementation Plan

> Archive status, 2026-05-20: completed and retained as historical evidence. Do not execute this file as an active plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or equivalent task-by-task execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/runtime-setup` feel like a guided local AI setup path instead of a full provider settings console.

**Architecture:** Keep the existing single-page React route and shared `RuntimeConnectionForm`. Add progressive disclosure for cloud/advanced providers in the runtime setup variant only, move diagnostics out of the main happy path, and keep Settings as the full-power management surface.

**Tech Stack:** React, TypeScript, TanStack Query, Vitest, existing JSON locale files.

**PAL Review:** Completed before this plan. PAL agreed the current page mixes first-run local setup with advanced provider settings, duplicates Section C, and exposes irrelevant fields too early.

---

### Task 1: Local-First Provider Selection

**Files:**
- Modify: `frontend/src/components/setup/RuntimeConnectionForm.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`

- [x] **Step 1: Add tests for local-first default provider choices**

Add assertions to `shows the runtime setup branch on Home and renders interactive runtime controls` after the setup screen renders:

```tsx
const providerField = screen.getByTestId("runtime_connection.provider");
expect(within(providerField).getByRole("option", { name: "Ollama local" })).toBeInTheDocument();
expect(within(providerField).getByRole("option", { name: "LM Studio local" })).toBeInTheDocument();
expect(within(providerField).queryByRole("option", { name: "OpenRouter" })).not.toBeInTheDocument();
expect(screen.getByRole("button", { name: "Show cloud and advanced providers" })).toBeVisible();
```

- [x] **Step 2: Add tests that advanced providers are opt-in**

Add a focused route test:

```tsx
it("keeps cloud providers behind the advanced toggle on runtime setup", async () => {
  mockedGetRuntime.mockResolvedValue({
    configured: false,
    provider: "",
    model: "",
    base_url: "",
    requires_api_key: false,
    has_api_key: false,
  });
  mockedGetRuntimeSettings.mockResolvedValue({
    ui_locale: "en",
    whisper_model: "medium",
    active_connection_id: "",
    connections: [],
  });

  renderWithProviders(<AppFrame />, {
    initialEntries: ["/runtime-setup"],
    locale: "en",
  });

  const providerField = await screen.findByTestId("runtime_connection.provider");
  expect(within(providerField).queryByRole("option", { name: "OpenRouter" })).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Show cloud and advanced providers" }));

  expect(within(providerField).getByRole("option", { name: "OpenRouter" })).toBeInTheDocument();
  expect(within(providerField).getByRole("option", { name: "Generic OpenAI-compatible" })).toBeInTheDocument();
});
```

- [x] **Step 3: Implement provider filtering in runtime setup only**

In `RuntimeConnectionForm.tsx`:

```tsx
const LOCAL_PROVIDER_CHOICES = new Set<string>(["ollama_local", "lmstudio_local"]);
const ADVANCED_PROVIDER_CHOICES = new Set<string>(["ollama_cloud", "openrouter", "openai_compatible"]);
```

Inside `RuntimeConnectionForm`, add:

```tsx
const [showAdvancedProviders, setShowAdvancedProviders] = useState(false);
const selectedProviderIsAdvanced = ADVANCED_PROVIDER_CHOICES.has(providerChoice);
const providerChoices =
  variant === "runtime-setup" && !showAdvancedProviders && !selectedProviderIsAdvanced
    ? PROVIDER_CHOICES.filter((option) => LOCAL_PROVIDER_CHOICES.has(option))
    : PROVIDER_CHOICES;
```

Render `providerChoices` instead of `PROVIDER_CHOICES`.

- [x] **Step 4: Add the advanced toggle button**

In the provider section, after the provider hint, render only for `variant === "runtime-setup"`:

```tsx
<button
  type="button"
  onClick={() => setShowAdvancedProviders((current) => !current)}
  style={{ ...actionButtonStyle, width: "fit-content" }}
>
  {translate(showAdvancedProviders ? "runtime_setup.hide_advanced" : "runtime_setup.show_advanced")}
</button>
```

Keep advanced visible automatically when editing an existing advanced connection, so OpenRouter users do not see their provider disappear.

### Task 2: Conditional Fields Per Provider

**Files:**
- Modify: `frontend/src/components/setup/RuntimeConnectionForm.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `frontend/src/routes/tests/SettingsRoute.test.tsx`

- [x] **Step 1: Add runtime setup tests for hidden irrelevant controls**

In the first runtime setup route test, assert local provider behavior:

```tsx
expect(screen.getByTestId("runtime_setup.detect_local_models")).toBeVisible();
expect(screen.queryByTestId("runtime_connection.api_key")).not.toBeInTheDocument();
```

Add an advanced OpenRouter test path:

```tsx
fireEvent.click(screen.getByRole("button", { name: "Show cloud and advanced providers" }));
fireEvent.change(screen.getByTestId("runtime_connection.provider"), {
  target: { value: "openrouter" },
});

expect(screen.queryByTestId("runtime_setup.detect_local_models")).not.toBeInTheDocument();
expect(screen.getByTestId("runtime_connection.api_key")).toBeVisible();
expect(screen.getByLabelText("OpenRouter HTTP-Referer")).toBeVisible();
```

- [x] **Step 2: Preserve Settings behavior**

In `SettingsRoute.test.tsx`, assert that the Settings variant still exposes advanced providers and API key controls without using the runtime setup toggle:

```tsx
const providerField = screen.getByTestId("runtime_connection.provider");
expect(within(providerField).getByRole("option", { name: "OpenRouter" })).toBeInTheDocument();
expect(screen.getByTestId("settings.api_key")).toBeVisible();
```

Use the existing test that edits a selected connection through Settings, so this stays close to current behavior.

- [x] **Step 3: Hide detect local models instead of disabling it**

Replace the current always-rendered detection button block with:

```tsx
{onDetectLocalModels && modelSelectEnabled ? (
  <button
    type="button"
    disabled={isBusy}
    onClick={() => onDetectLocalModels(normalizedDraft)}
    style={actionButtonStyle}
    {...semanticAttributes(SEMANTIC_IDS.runtimeSetup.detectLocalModels)}
  >
    {translate("runtime_setup.detect_local_models")}
  </button>
) : null}
```

- [x] **Step 4: Hide API key for local providers in runtime setup**

Add:

```tsx
const providerCanUseApiKey = !LOCAL_PROVIDER_CHOICES.has(providerChoice);
const showApiKeyField = variant === "settings" || providerCanUseApiKey || initialSecretState === "missing";
```

Wrap the API key label in `{showApiKeyField ? (...) : null}`.

### Task 3: Saved-Key Replacement UX

**Files:**
- Modify: `frontend/src/components/setup/RuntimeConnectionForm.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`

- [x] **Step 1: Add a saved-key setup test**

In the existing saved-secret setup route test, assert the password field is hidden until replacement is requested:

```tsx
expect(screen.getByText("A saved key is already available for this connection.")).toBeVisible();
expect(screen.queryByTestId("runtime_connection.api_key")).not.toBeInTheDocument();

fireEvent.click(screen.getByRole("button", { name: "Replace saved key" }));

expect(screen.getByTestId("runtime_connection.api_key")).toBeVisible();
```

- [x] **Step 2: Add secret replacement state**

In `RuntimeConnectionForm.tsx`, add:

```tsx
const [replaceSecret, setReplaceSecret] = useState(false);
```

Reset it in the existing reset effect:

```tsx
setReplaceSecret(false);
```

- [x] **Step 3: Hide saved-key password input in runtime setup**

Use:

```tsx
const savedSecretPresent = initialSecretState === "present";
const showApiKeyField =
  variant === "settings" ||
  initialSecretState === "missing" ||
  (providerCanUseApiKey && (!savedSecretPresent || replaceSecret));
```

In the secret-state block for `initialSecretState === "present"` and `variant === "runtime-setup"`, render:

```tsx
<button
  type="button"
  onClick={() => setReplaceSecret(true)}
  style={{ ...actionButtonStyle, width: "fit-content" }}
>
  {translate("runtime_setup.replace_saved_key")}
</button>
```

Keep `Clear saved key` visible so repair and cleanup still work.

### Task 4: Move Diagnostics Out Of The Main Path

**Files:**
- Modify: `frontend/src/routes/SetupRoute.tsx`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`

- [x] **Step 1: Add a test for collapsed diagnostics**

In the first runtime setup test:

```tsx
expect(screen.queryByText("Section C")).not.toBeInTheDocument();
expect(screen.getByText("Connection diagnostics")).toBeVisible();
expect(screen.queryByText("Active runtime is ready")).not.toBeInTheDocument();
```

Use current diagnostics mock text if the exact runtime diagnostic text differs.

- [x] **Step 2: Remove the duplicated Section C status card**

In `SetupRoute.tsx`, delete the middle card:

```tsx
<section style={cardStyle}>
  <h2>...</h2>
  <ConnectionStatusPanel ... />
</section>
```

- [x] **Step 3: Add a collapsed diagnostics section after the form**

After `RuntimeConnectionForm`, render:

```tsx
<details style={cardStyle}>
  <summary style={{ cursor: "pointer", fontWeight: 700, color: "#10201c" }}>
    {translate("runtime_setup.connection_diagnostics")}
  </summary>
  <div style={{ marginTop: "0.875rem" }}>
    <ConnectionStatusPanel
      activeConnection={activeConnection}
      diagnostics={diagnosticsItems}
      runtime={runtime}
    />
  </div>
</details>
```

This keeps diagnostics available without interrupting the setup flow.

### Task 5: Localized Copy And Semantic Labels

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] **Step 1: Update English copy**

In `runtime_setup`, add or update:

```json
"body": "Set up Whisper and a local AI provider. Cloud and custom providers stay available under advanced options.",
"local_first_intro": "Start with a local provider. Ollama and LM Studio usually do not need API keys.",
"show_advanced": "Show cloud and advanced providers",
"hide_advanced": "Hide cloud and advanced providers",
"connection_diagnostics": "Connection diagnostics",
"replace_saved_key": "Replace saved key",
"api_key_optional_local": "Local providers usually do not need a key. Leave this hidden unless the endpoint asks for one."
```

- [x] **Step 2: Mirror keys across all locale files**

Add the same keys to `de`, `es`, `fr`, and `it`. If full translation is not ready, use clear English fallback text rather than missing keys. Do not hard-code strings in React.

- [x] **Step 3: Check semantic IDs**

Reuse existing semantic IDs where possible:
- `runtime_connection.provider`
- `runtime_connection.api_key`
- `runtime_setup.detect_local_models`
- `runtime_connection.form_status`

Add a new semantic ID only if tests need a stable target for the advanced toggle.

### Task 6: Verification

**Files:**
- Test only: no production file changes in this task.

- [x] **Step 1: Run focused route tests**

```bash
cd frontend
npm test -- HomeSetupRoutes.test.tsx SettingsRoute.test.tsx
```

- [x] **Step 2: Run frontend typecheck**

```bash
cd frontend
npm run typecheck
```

- [x] **Step 3: Run full frontend tests if focused tests pass**

```bash
cd frontend
npm test
```

- [x] **Step 4: Optional browser smoke**

If a dev server is already running, inspect `/runtime-setup` manually or with browser automation. Verify:
- First view shows Whisper and local provider choices only.
- OpenRouter appears only after enabling advanced providers.
- Local providers do not show API key fields.
- Diagnostics are collapsed at the bottom.

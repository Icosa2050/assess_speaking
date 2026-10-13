import { expect, test } from "../fixtures";

// Deterministic UI contracts run in normal CI. Real discovery/inference lives in tests/live.
test.use({ viewport: { width: 390, height: 844 } });

test.beforeEach(async ({ page }) => {
  await page.route("**/v1/runtime/settings", route => route.fulfill({ json: {
    ui_locale: "en", whisper_model: "tiny", active_connection_id: "", connections: [],
  } }));
});

test("Ollama discovery preserves the user's URL and selects a discovered model", async ({ page }) => {
  let calls = 0;
  await page.route("**/v1/runtime/settings/test-connection", async route => {
    calls++;
    expect(route.request().postDataJSON().connection).toMatchObject({
      provider_choice: "ollama_local", base_url: "http://localhost:11434/",
    });
    await route.fulfill({ json: { provider: "ollama", base_url: "http://localhost:11434/v1",
      service_base_url: "http://localhost:11434", health_endpoint: "http://localhost:11434/api/tags",
      discovered_models: ["fixture-model"], content_preview: "", tested_at: "2026-10-03T00:00:00Z" } });
  });
  await page.goto("/runtime-setup");
  await page.getByTestId("runtime_connection.provider").selectOption("ollama_local");
  await page.getByTestId("runtime_connection.base_url").fill("http://localhost:11434/");
  await page.getByTestId("runtime_setup.detect_local_models").click();
  await expect(page.getByText(/Local models found via/)).toBeVisible();
  await expect(page.getByTestId("runtime_connection.base_url")).toHaveValue("http://localhost:11434/");
  await expect(page.getByTestId("runtime_connection.model")).toHaveValue("fixture-model");
  expect(calls).toBe(1);
});

test("LM Studio failure keeps the form editable and a corrected retry succeeds", async ({ page }) => {
  let calls = 0;
  await page.route("**/v1/runtime/settings/test-connection", async route => {
    calls++;
    const draft = route.request().postDataJSON().connection;
    expect(draft.provider_choice).toBe("lmstudio_local");
    if (calls === 1) {
      expect(draft.model).toBe("missing-model");
      await route.fulfill({ status: 502, json: { detail: {
        code: "runtime_error", detail: "Requested model missing-model is not loaded in LM Studio.",
      } } });
    } else {
      expect(draft.model).toBe("loaded-model");
      await route.fulfill({ json: { provider: "lmstudio", base_url: draft.base_url,
        service_base_url: draft.base_url, health_endpoint: `${draft.base_url}/models`,
        discovered_models: ["loaded-model"], content_preview: "Connection recovered", tested_at: "2026-10-03T00:00:00Z" } });
    }
  });
  await page.goto("/runtime-setup");
  await page.getByTestId("runtime_connection.provider").selectOption("lmstudio_local");
  await page.getByTestId("runtime_connection.base_url").fill("http://localhost:1234/v1");
  await page.getByTestId("runtime_connection.model").fill("missing-model");
  await page.getByTestId("runtime_connection.test_connection").click();
  await expect(page.getByTestId("runtime_connection.form_status")).toContainText("missing-model is not loaded");
  await expect(page).toHaveURL(/\/runtime-setup$/);
  await expect(page.getByTestId("runtime_connection.model")).toHaveValue("missing-model");
  await page.getByTestId("runtime_connection.model").fill("loaded-model");
  await page.getByTestId("runtime_connection.test_connection").click();
  await expect(page.getByTestId("runtime_connection.form_status")).toContainText("Connection recovered");
  await expect(page.getByText("Requested model missing-model is not loaded in LM Studio.", { exact: true })).toHaveCount(0);
  expect(calls).toBe(2);
});

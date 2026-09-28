import { expect, test } from "@playwright/test";
import { readOmlxConfig } from "./helpers/omlx";

// Run with RUN_VOSTAVO_OMLX_E2E=1 and OMLX_MODEL set to an installed chat model ID.
// OMLX_BASE_URL (including /v1) and OMLX_API_KEY override ~/.omlx/settings.json.
test.skip(
  process.env.RUN_VOSTAVO_OMLX_E2E !== "1",
  "Set RUN_VOSTAVO_OMLX_E2E=1 and OMLX_MODEL to test a live oMLX server.",
);

// Connection requests can contain an API key; do not retain them in traces.
test.use({ viewport: { width: 390, height: 844 }, trace: "off" });

test("tests an oMLX chat model through the OpenAI-compatible setup flow", async ({
  page,
  request,
}) => {
  test.setTimeout(180_000);
  const { baseUrl, model, apiKey } = readOmlxConfig();
  expect(model, "Set OMLX_MODEL to the exact ID of an installed oMLX chat model").not.toBe("");

  const modelsResponse = await request.get(`${baseUrl}/models`, {
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    timeout: 15_000,
  });
  expect(modelsResponse.ok(), "oMLX model discovery must succeed").toBeTruthy();
  const models = await modelsResponse.json();
  expect(models.data).toEqual(expect.arrayContaining([expect.objectContaining({ id: model })]));

  await page.goto("/runtime-setup");
  await expect(page.getByTestId("runtime_connection.provider")).toBeVisible();
  const advancedToggle = page.getByTestId("runtime_setup.advanced_providers_toggle");
  if (await advancedToggle.isVisible()) {
    await advancedToggle.click();
  }
  await page.getByTestId("runtime_connection.provider").selectOption("openai_compatible");
  await page.getByTestId("runtime_connection.base_url").fill(baseUrl);
  await page.getByTestId("runtime_connection.model").fill(model);
  const keyInput = page.getByTestId("runtime_connection.api_key");
  if (!(await keyInput.isVisible())) {
    await page.getByTestId("runtime_connection.replace_saved_key").click();
  }
  await keyInput.fill(apiKey);

  const connectionResponsePromise = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === "/v1/runtime/settings/test-connection" &&
      response.request().method() === "POST",
    { timeout: 120_000 },
  );
  await page.getByTestId("runtime_connection.test_connection").click();
  const connectionResponse = await connectionResponsePromise;
  expect(connectionResponse.ok(), "The backend must successfully test the oMLX connection").toBeTruthy();
  const result = await connectionResponse.json();
  expect(result.discovered_models).toContain(model);
  expect(typeof result.content_preview).toBe("string");
  expect(result.content_preview.trim()).not.toBe("");
  await expect(page.getByTestId("runtime_connection.form_status")).toContainText(result.content_preview);
  await expect(page.getByTestId("runtime_connection.base_url")).toHaveValue(baseUrl);
  await expect(page.getByTestId("runtime_connection.model")).toHaveValue(model);
  await expect(page).toHaveURL(/\/runtime-setup$/);
});

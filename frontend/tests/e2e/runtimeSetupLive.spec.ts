import { expect, test } from "@playwright/test";

test.skip(
  process.env.RUN_VOSTAVO_LOCAL_RUNTIME_E2E !== "1",
  "Set RUN_VOSTAVO_LOCAL_RUNTIME_E2E=1 to run live runtime setup tests.",
);

test.describe("live local runtime setup replacement", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("detects Ollama models and preserves a sanitized base URL", async ({ page }) => {
    await page.goto("/runtime-setup");

    await page.getByTestId("runtime_connection.provider").selectOption("ollama_local");
    await page.getByTestId("runtime_connection.base_url").fill("http://localhost:11434/");
    await page.getByTestId("runtime_setup.detect_local_models").click();

    await expect(page.getByText(/Detected \d+ local model/)).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.getByTestId("runtime_connection.base_url")).toHaveValue(
      "http://localhost:11434/v1",
    );
    await expect(page.getByTestId("runtime_connection.model")).not.toHaveValue("");
  });

  test("reports LM Studio connection feedback without leaving the setup screen", async ({
    page,
  }) => {
    await page.goto("/runtime-setup");

    await page.getByTestId("runtime_connection.provider").selectOption("lmstudio_local");
    await page.getByTestId("runtime_connection.model").fill("codex-missing-model");
    await page.getByTestId("runtime_connection.test_connection").click();

    await expect(page.getByTestId("runtime_connection.form_status")).toBeVisible({
      timeout: 30_000,
    });
    await expect(page).toHaveURL(/\/runtime-setup$/);
  });
});

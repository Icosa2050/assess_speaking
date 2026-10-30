import { expect, test } from "../fixtures";

test.describe("settings support flow", () => {
  test("preferences save without a provider connection", async ({ page }) => {
    await page.route("**/v1/runtime/settings", route => route.fulfill({ json: {
      ui_locale: "en", whisper_model: "small", active_connection_id: "", connections: [],
    } }));
    try {
      await page.goto("/settings");
      await expect(page.getByTestId("settings.connection_id")).toHaveValue("");
      const save = page.getByTestId("settings.save_preferences");
      await expect(save).toBeEnabled();
      await page.getByTestId("settings.ui_locale").selectOption("fr");
      await page.getByTestId("settings.whisper_model").selectOption("tiny");
      const saved = page.waitForResponse(response => new URL(response.url()).pathname === "/v1/runtime/preferences" && response.request().method() === "PATCH");
      await save.click();
      const response = await saved;
      expect(response.ok()).toBeTruthy();
      expect(response.request().postDataJSON()).toEqual({ ui_locale: "fr", whisper_model: "tiny" });
      expect(await response.json()).toEqual({ ui_locale: "fr", whisper_model: "tiny" });
    } finally {
      expect((await page.request.patch(`${process.env.VOSTAVO_FIXTURE_BACKEND_URL}/v1/runtime/preferences`, {
        data: { ui_locale: "en", whisper_model: "small" },
      })).ok()).toBeTruthy();
    }
  });

  test("settings runs support actions against the local guest backend", async ({ page }) => {
    let supportBundleRequest: unknown = null;
    await page.route("**/v1/support-bundles", async (route) => {
      supportBundleRequest = route.request().postDataJSON();
      await route.fulfill({
        contentType: "application/json",
        json: {
          bundle_id: "bundle-runtime-health",
          filename: "support-runtime-health.zip",
          size_bytes: 128,
          expires_at: "2026-05-01T12:00:00Z",
        },
      });
    });
    await page.route("**/v1/support-bundles/bundle-runtime-health", async (route) => {
      await route.fulfill({
        body: "bundle",
        contentType: "application/zip",
      });
    });

    await page.goto("/");
    await page.getByTestId("home.open_settings").click();

    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
    await expect
      .poll(() => page.evaluate(() => window.history.state?.usr?.from ?? null))
      .toBe("home");

    await page.getByTestId("settings.support_refresh_storage").click();
    await expect(page.getByTestId("settings.storage_row").first()).toBeVisible();

    await page.getByTestId("settings.support_cleanup_preview").click();
    await expect(page.getByText(/Files that can be removed:/)).toBeVisible();

    await page.getByTestId("settings.support_bundle_include_runtime_health").check();
    const downloadPromise = page.waitForEvent("download");
    await page.getByTestId("settings.support_create_bundle").click();
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toMatch(/\.zip$/);
    expect(supportBundleRequest).toMatchObject({
      include_runtime_health: true,
    });

    await page.getByTestId("settings.open_setup").click();
    await expect(page).toHaveURL(/\/runtime-setup$/);
    await expect
      .poll(() => page.evaluate(() => window.history.state?.usr?.from ?? null))
      .toBe("home");

    await page.getByTestId("home.open_history").click();
    await expect(page).toHaveURL(/\/history$/);
    await page.getByTestId("home.open_settings").click();
    await expect(page).toHaveURL(/\/settings$/);
    const historyBeforeSkip = await page.evaluate(() => ({
      length: window.history.length,
      from: window.history.state?.usr?.from ?? null,
    }));
    expect(historyBeforeSkip.from).toBe("history");

    const skip = page.locator('a[href="#main-content"]');
    await skip.focus();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("main")).toBeFocused();
    await expect(page).toHaveURL(/\/settings$/);
    expect(await page.evaluate(() => ({
      length: window.history.length,
      from: window.history.state?.usr?.from ?? null,
    }))).toEqual(historyBeforeSkip);

    await page.getByTestId("settings.return").click();
    await expect(page).toHaveURL(/\/history$/);
  });
});

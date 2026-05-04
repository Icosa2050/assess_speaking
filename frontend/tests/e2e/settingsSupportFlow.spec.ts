import { expect, test } from "@playwright/test";

test.describe("settings support flow", () => {
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
    await expect(page.getByText(/Cleanup preview found/)).toBeVisible();

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
  });
});

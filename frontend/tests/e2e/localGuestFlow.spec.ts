import { expect, test } from "@playwright/test";

test.describe("local guest shared frontend", () => {
  test("Home keeps History and Settings reachable before runtime setup is complete", async ({
    page,
  }) => {
    await page.goto("/");

    const runtimeSetupButton = page.getByTestId("home.runtime_setup_button");
    await expect(runtimeSetupButton).toBeVisible();

    await page.getByTestId("home.open_history").click();
    await expect(page).toHaveURL(/\/history$/);
    await expect(page.getByTestId("history-empty")).toBeVisible();

    await page.goto("/");
    await page.getByTestId("home.open_settings").click();
    await expect(page).toHaveURL(/\/settings$/);
    await expect(page.getByTestId("settings.return")).toBeVisible();

    await page.getByTestId("settings.return").click();
    await expect(page).toHaveURL(/\/$/);
    await expect(runtimeSetupButton).toBeVisible();
  });
});

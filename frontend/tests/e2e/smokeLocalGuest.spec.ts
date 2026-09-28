import { expect, test } from "@playwright/test";

test.describe("local guest smoke", () => {
  test("home routes to runtime setup on a clean backend", async ({ page }) => {
    await page.goto("/");

    const runtimeSetupButton = page.getByTestId("home.runtime_setup_button");
    await expect(runtimeSetupButton).toBeVisible();
    await runtimeSetupButton.click();

    await expect(page).toHaveURL(/\/runtime-setup$/);
    await expect(page.getByTestId("runtime_setup.back_home")).toBeVisible();

    await page.getByTestId("runtime_setup.back_home").click();
    await expect(page).toHaveURL(/\/$/);
    await expect(runtimeSetupButton).toBeVisible();
  });
});

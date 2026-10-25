import { expect, test } from "../fixtures";

test.describe("local guest smoke", () => {
  test("home routes to runtime setup on a clean backend", async ({ page }) => {
    await page.setViewportSize({ width: 1100, height: 760 });
    await page.goto("/");

    const runtimeSetupButton = page.getByTestId("home.runtime_setup_button");
    await expect(runtimeSetupButton).toBeVisible();
    await expect(page.getByTestId("home.open_rehearsal")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1100);
    await page.getByTestId("home.open_rehearsal").click();
    await expect(page).toHaveURL(/\/rehearsal$/);
    await expect(page.getByTestId("rehearsal-screen")).toBeVisible();
    const review = page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: "Review" });
    await expect(review).toHaveAttribute("aria-disabled", "true");
    await expect(review).toHaveAccessibleDescription("Submit a speaking attempt before opening the review screen.");
    await review.focus();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/rehearsal$/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1100);
    await page.goBack();
    await expect(page).toHaveURL(/\/$/);
    await runtimeSetupButton.click();

    await expect(page).toHaveURL(/\/runtime-setup$/);
    await expect(page.getByTestId("runtime_setup.back_home")).toBeVisible();

    await page.getByTestId("runtime_setup.back_home").click();
    await expect(page).toHaveURL(/\/$/);
    await expect(runtimeSetupButton).toBeVisible();
  });
});

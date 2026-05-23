import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

test.skip(
  process.env.RUN_VOSTAVO_REAL_E2E !== "1",
  "Set RUN_VOSTAVO_REAL_E2E=1 to run real-audio history progression.",
);

const repoRoot = path.resolve(fileURLToPath(new URL("../../..", import.meta.url)));
const weakerAudio = path.join(repoRoot, "output/speech/it_b1_more_human.wav");
const strongerAudio = path.join(repoRoot, "output/speech/it_b1_near_perfect.wav");

const attachAudioPath = async (page: Page, filePath: string) => {
  await page.getByTestId("speak.input_mode_upload").click();
  await page.getByTestId("speak.upload_input").setInputFiles(filePath);
  await expect(page.getByText("A recording is attached and ready for assessment.")).toBeVisible();
};

const lastNumberFromTestId = async (page: Page, testId: string) => {
  const value = await page.getByTestId(testId).textContent();
  const matches = value?.match(/\d+(?:\.\d+)?/g) ?? [];
  return Number.parseFloat(matches.at(-1) ?? "0");
};

test.describe("real audio history replacement", () => {
  test("records weaker and stronger Italian samples into history progression", async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByTestId("home.start_new").click();
    await page.getByTestId("setup.speaker_id").fill("playwright-real-audio");
    await page.getByTestId("setup.learning_language").selectOption("it");
    await page.getByTestId("setup.cefr").selectOption("B1");
    await page.getByTestId("setup.continue").click();
    await expect(page).toHaveURL(/\/speak$/);

    await attachAudioPath(page, weakerAudio);
    await page.getByTestId("speak.label").fill("real weaker sample");
    await page.getByTestId("speak.submit").click();
    await expect(page.getByTestId("review-summary")).toBeVisible({ timeout: 180_000 });
    const firstScore = await lastNumberFromTestId(page, "review-metric-score-overall");
    const firstBand = await lastNumberFromTestId(page, "review-metric-band");

    await page.getByTestId("review-action-try-again").click();
    await attachAudioPath(page, strongerAudio);
    await page.getByTestId("speak.label").fill("real stronger sample");
    await page.getByTestId("speak.submit").click();
    await expect(page.getByTestId("review-summary")).toBeVisible({ timeout: 180_000 });

    const secondScore = await lastNumberFromTestId(page, "review-metric-score-overall");
    const secondBand = await lastNumberFromTestId(page, "review-metric-band");
    expect(secondScore).toBeGreaterThan(firstScore);
    expect(secondBand).toBeGreaterThanOrEqual(firstBand);
    await expect(page.getByTestId("review-progress")).toBeVisible();

    await page.getByTestId("review-action-view-history").click();
    await expect(page.getByTestId("history-detail-panel")).toBeVisible();
    await expect(page.getByTestId("review-label")).toContainText("real stronger sample");
  });
});

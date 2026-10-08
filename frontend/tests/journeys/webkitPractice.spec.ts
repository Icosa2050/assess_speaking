import { expect, test } from "../fixtures";
import path from "node:path";

// WebKit uses upload journeys: Chromium-only fake microphone flags do not
// exercise the macOS microphone permission dialog in Safari.
test.use({ browserName: "webkit", permissions: [], launchOptions: {} });
for (const language of ["en", "it"]) {
  test(`WebKit ${language}: upload, saved feedback, replay and retry`, async ({ page, request }) => {
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
    const saved = await request.put(`${process.env.VOSTAVO_FIXTURE_BACKEND_URL}/v1/runtime/settings`, { data: {
      ui_locale: language, whisper_model: "small",
      connection: { provider_choice: "ollama_local", model: "journey-fixture", base_url: "http://127.0.0.1:11434/v1" },
    } });
    expect(saved.ok()).toBeTruthy();
    await page.goto("/session-setup");
    await page.getByTestId("setup.speaker_id").fill(`webkit-${language}`);
    await page.getByTestId("setup.customize_details").click();
    await page.getByTestId("setup.learning_language").selectOption(language);
    await page.getByTestId("setup.cefr").selectOption("B2");
    await page.getByTestId("setup.continue").click();
    await page.getByTestId("speak.input_mode_upload").click();
    await page.getByTestId("speak.upload_input").setInputFiles(path.join(process.env.VOSTAVO_JOURNEY_AUDIO_DIR!, language, "B2/remote_work.wav"));
    await expect(page.getByTestId("speak.download_recording")).toBeVisible();
    await page.getByTestId("speak.submit").click();
    await expect(page).toHaveURL(/\/review$/, { timeout: 45000 });
    await page.getByTestId("review-action-view-history").click();
    await expect(page.getByTestId("history-detail-caption")).toBeVisible();
    await page.reload();
    await expect(page.getByTestId("practice-progress").locator("svg circle")).toHaveCount(1);
    const audio = page.locator("audio").first();
    await audio.evaluate(async (el: HTMLAudioElement) => { await el.play(); });
    await expect.poll(() => audio.evaluate((el: HTMLAudioElement) => el.currentTime)).toBeGreaterThan(0);
    await page.getByTestId("practice-retry").click();
    await expect(page).toHaveURL(/\/speak$/);
    await expect(page.getByTestId("speak.session_summary")).toContainText("B2");
    expect(errors).toEqual([]);
  });
}

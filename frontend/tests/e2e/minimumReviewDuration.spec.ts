import { expect, test } from "../fixtures";

const syntheticWav = (seconds: number) => {
  const rate = 16000;
  const samples = rate * seconds;
  const wav = Buffer.alloc(44 + samples * 2);
  wav.write("RIFF", 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 2, 28);
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write("data", 36);
  wav.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) wav.writeInt16LE(Math.round(Math.sin(2 * Math.PI * 220 * i / rate) * 4000), 44 + 2 * i);
  return wav;
};

test("short upload is rejected by the real local worker without a review or history entry", async ({ page }) => {
  await page.route("**/v1/runtime", (route) => route.fulfill({ json: {
    configured: true, provider: "ollama", model: "synthetic-no-model-needed",
    base_url: "http://127.0.0.1:11434/v1", requires_api_key: false, has_api_key: false,
  } }));
  await page.route("**/v1/runtime/settings", (route) => route.fulfill({ json: {
    ui_locale: "en", whisper_model: "small", active_connection_id: "", connections: [],
  } }));
  await page.route("**/v1/diagnostics", (route) => route.fulfill({ json: { items: [] } }));
  await page.goto("/session-setup");
  await page.getByTestId("setup.speaker_id").fill("minimum-duration-synthetic");
  await page.getByTestId("setup.recommended_start").click();
  await page.getByTestId("setup.continue").click();
  await page.getByTestId("speak.input_mode_upload").click();
  await page.getByTestId("speak.upload_input").setInputFiles({ name: "synthetic-12s.wav", mimeType: "audio/wav", buffer: syntheticWav(12) });
  await page.getByTestId("speak.submit").click();
  await expect(page.getByTestId("speak.status_panel")).toContainText("Record at least 30 seconds", { timeout: 30000 });
  await expect(page).toHaveURL(/\/speak$/);
  await expect(page.getByRole("button", { name: "Remove recording" })).toBeVisible();
  await expect(page.getByTestId("review-summary")).toHaveCount(0);
  await page.getByRole("link", { name: "History", exact: true }).click();
  await expect(page.getByTestId("history-empty")).toBeVisible();
});

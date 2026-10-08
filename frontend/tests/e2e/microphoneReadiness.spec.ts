import { expect, test } from "../fixtures";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Chromium's default fake input can be silent. Supply generated PCM audio so the
// browser test exercises the real analyser and capture pipeline deterministically.
const fixtureDir = mkdtempSync(join(tmpdir(), "vostavo-microphone-audio-"));
const audioPath = join(fixtureDir, "tone.wav");
const sampleRate = 48_000;
const samples = sampleRate * 2;
const wav = Buffer.alloc(44 + samples * 2);
wav.write("RIFF", 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write("WAVEfmt ", 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(sampleRate, 24); wav.writeUInt32LE(sampleRate * 2, 28);
wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write("data", 36); wav.writeUInt32LE(samples * 2, 40);
for (let i = 0; i < samples; i++) wav.writeInt16LE(Math.round(3200 * Math.sin(2 * Math.PI * 440 * i / sampleRate)), 44 + i * 2);
writeFileSync(audioPath, wav);
test.afterAll(() => rmSync(fixtureDir, { recursive: true, force: true }));
test.use({ launchOptions: { args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${audioPath}`] } });

test.beforeEach(async ({ page }) => {
  await page.route("**/v1/runtime", route => route.fulfill({ json: {
    configured: true, provider: "ollama", model: "microphone-fixture",
    base_url: "http://127.0.0.1:11434/v1", requires_api_key: false, has_api_key: false,
  } }));
  await page.route("**/v1/runtime/settings", route => route.fulfill({ json: {
    ui_locale: "en", whisper_model: "small", active_connection_id: "", connections: [],
  } }));
  await page.route("**/v1/runtime/whisper/**", route => route.fulfill({ json: { cached: true, model: "small" } }));
  await page.route("**/v1/runtime/cloud", route => route.fulfill({ json: {
    settings: { version: 1, asr_provider: "local", asr_connection_id: "", asr_model: "", openrouter_modes: {}, fallback_connection_id: "", paid_fallback_enabled: false, monthly_budget_usd: 0, max_output_tokens: 1000 },
    spending: { spent_usd: 0, reserved_usd: 0, unresolved_requests: [] },
  } }));
  await page.route("**/v1/diagnostics", route => route.fulfill({ json: { items: [
    { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
    { key: "maintenance", status: "warning", title_key: "", detail_key: "", detail_args: {} },
    { key: "microphone", status: "info", title_key: "", detail_key: "", detail_args: {} },
  ] } }));
  await page.route("**/v1/history", route => route.fulfill({ json: { items: [] } }));
});

const confirmMicrophone = async (page: import("@playwright/test").Page) => {
  await expect(page.getByTestId("microphone-test-status")).toContainText("Listen to the sample", { timeout: 10_000 });
  const confirm = page.getByTestId("microphone-test-confirm");
  await expect(confirm).toBeDisabled();
  await page.getByTestId("microphone-test-playback").evaluate(async element => {
    const audio = element as HTMLAudioElement;
    // This headless macOS host has no usable speaker clock. Keep native codec
    // decoding/playback but route output to Chromium's silent test sink.
    const context = new AudioContext({ sinkId: { type: "none" } } as AudioContextOptions);
    context.createMediaElementSource(audio).connect(context.destination);
    audio.addEventListener("ended", () => { void context.close(); }, { once: true });
    await context.resume();
    await audio.play();
  });
  await expect(confirm).toBeEnabled({ timeout: 10_000 });
  await confirm.click();
};

test("setup tests real browser audio and carries its result to Home without removing notices", async ({ page }) => {
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    (window as unknown as Window & { microphoneRequests?: number }).microphoneRequests = 0;
    navigator.mediaDevices.getUserMedia = async constraints => {
      (window as unknown as Window & { microphoneRequests: number }).microphoneRequests += 1;
      const stream = await original(constraints);
      (window as unknown as Window & { testedStream?: MediaStream }).testedStream = stream;
      return stream;
    };
  });
  await page.goto("/runtime-setup");
  const row = page.getByTestId("runtime_setup.setup_guide.microphone");
  await expect(row).toHaveAttribute("data-status", "setup");
  expect(await page.evaluate(() => (window as unknown as Window & { microphoneRequests: number }).microphoneRequests)).toBe(0);
  await expect(page.getByTestId("runtime_setup.setup_guide.microphone.action")).toHaveText("Test microphone");
  // Echo/noise suppression deliberately removes a pure synthetic tone.
  // Disable it for the fixture; practice capture must retain this setting too.
  await page.getByTestId("microphone-processing").uncheck();
  await page.getByTestId("runtime_setup.setup_guide.microphone.action").click();
  await expect(row).toHaveAttribute("data-status", "setup");
  await confirmMicrophone(page);
  await expect(row).toHaveAttribute("data-status", "ready");
  await expect(page.getByTestId("microphone-test-status")).toContainText("setup complete");
  await page.locator("#runtime-setup-microphone").screenshot({ path: "/tmp/vostavo-microphone-setup.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.locator("#runtime-setup-microphone").screenshot({ path: "/tmp/vostavo-microphone-setup-mobile.png" });
  expect(await page.evaluate(() => (window as unknown as Window & { testedStream: MediaStream }).testedStream.getTracks().every(track => track.readyState === "ended"))).toBe(true);
  await page.getByRole("link", { name: "Practice Home", exact: true }).click();
  await expect(page.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "100");
  await expect(page.getByText("Please review these notices")).toBeVisible();
  await page.getByTestId("home.setup_guide_button").click();
  await expect(row).toHaveAttribute("data-status", "ready");
  expect(await page.evaluate(() => (window as unknown as Window & { microphoneRequests: number }).microphoneRequests)).toBe(1);
});

for (const entry of ["home", "drawer"] as const) {
test(`${entry}-started recording requires the setup sample and playback confirmation`, async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "67");
  await page.getByTestId("home.start_new").click();
  await page.getByTestId("setup.speaker_id").fill("microphone-browser-test");
  await page.getByTestId("setup.recommended_start").click();
  await page.getByTestId("setup.continue").click();
  if (entry === "drawer") {
    await page.getByRole("link", { name: "Practice Home", exact: true }).click();
    await page.getByRole("link", { name: "Speak", exact: true }).click();
  }
  await expect(page.getByTestId("speak.record_start")).toBeDisabled();
  await page.getByTestId("speak.microphone_setup").click();
  await expect(page).toHaveURL(/runtime-setup#runtime-setup-microphone/);
  await page.getByTestId("microphone-processing").uncheck();
  await page.getByTestId("microphone-test-start").click();
  await confirmMicrophone(page);
  await page.getByRole("link", { name: "Speak", exact: true }).click();
  await expect(page.getByTestId("setup.speaker_id")).toHaveCount(0);
  await page.getByTestId("speak.record_start").click();
  await expect(page.getByTestId("speak.record_stop")).toBeVisible();
  await page.waitForTimeout(1200); // Produce a real MediaRecorder chunk from synthetic browser audio.
  await page.getByTestId("speak.record_stop").click();
  await expect(page.getByTestId("speak.recording_visualizer")).toHaveAttribute("data-recording-state", "ready");
  await expect(page.getByTestId("speak.submit")).toBeDisabled();
  await expect(page.getByTestId("speak.status_panel")).toContainText("Record at least 30 seconds");
  await page.getByRole("link", { name: "Practice Home", exact: true }).click();
  await expect(page.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "100");
});
}

test("denied microphone access gives guidance and never counts as ready", async ({ page }) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException("Denied for test", "NotAllowedError"); };
  });
  await page.goto("/runtime-setup");
  await page.getByTestId("microphone-test-start").click();
  await expect(page.getByTestId("microphone-test-status")).toContainText("access was denied");
  await expect(page.getByTestId("runtime_setup.setup_guide.microphone")).toHaveAttribute("data-status", "unavailable");
  await page.getByRole("link", { name: "Practice Home", exact: true }).click();
  await expect(page.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "67");
});

test("rehearsal requires calibration and retains learner choices through setup", async ({ page }) => {
  await page.goto("/rehearsal");
  await page.getByTestId("rehearsal-speaker").fill("mic-rehearsal-fixture");
  await page.getByTestId("rehearsal-language").selectOption("it");
  await page.getByTestId("rehearsal-goal").selectOption("C1");
  await expect(page.getByTestId("rehearsal-create")).toBeDisabled();
  await page.locator('a[href="/runtime-setup#runtime-setup-microphone"]').click();
  await page.getByTestId("microphone-processing").uncheck();
  await page.getByTestId("microphone-test-start").click();
  await confirmMicrophone(page);
  await page.locator('a[href="/rehearsal"]').click();
  await expect(page.getByTestId("rehearsal-speaker")).toHaveValue("mic-rehearsal-fixture");
  await expect(page.getByTestId("rehearsal-language")).toHaveValue("it");
  await expect(page.getByTestId("rehearsal-goal")).toHaveValue("C1");
  await page.getByTestId("rehearsal-create").click();
  await page.getByTestId("rehearsal-prepare").click();
  await page.getByTestId("rehearsal-speak").click();
  await expect(page.getByTestId("speak.record_start")).toBeEnabled();
});

  test("rejects near-clipping input and keeps calibration incomplete", async ({ page }) => {
    await page.addInitScript(() => {
      // Real Web Audio samples and MediaRecorder; deterministic full-scale input.
      navigator.mediaDevices.getUserMedia = async () => {
        const context = new AudioContext({ sinkId: { type: "none" } } as AudioContextOptions);
        const source = context.createOscillator(); source.frequency.value = 440;
        const output = context.createMediaStreamDestination(); source.connect(output);
        source.start(); await context.resume();
        const track = output.stream.getAudioTracks()[0]; const stop = track.stop.bind(track);
        track.stop = () => { stop(); source.stop(); void context.close(); };
        return output.stream;
      };
    });
    await page.goto("/runtime-setup");
    await page.getByTestId("microphone-processing").uncheck();
    await page.getByTestId("microphone-test-start").click();
    await expect(page.getByTestId("microphone-test-status")).toContainText("Input is too high", { timeout: 10_000 });
    await expect(page.getByTestId("microphone-test-confirm")).toBeDisabled();
    await expect(page.getByTestId("runtime_setup.setup_guide.microphone")).toHaveAttribute("data-status", "unavailable");
    await page.getByRole("link", { name: "Practice Home", exact: true }).click();
    await expect(page.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "67");
  });

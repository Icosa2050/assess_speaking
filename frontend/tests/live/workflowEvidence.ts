import { expect, type Page, type TestInfo } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

export const sha256 = (data: Buffer) => createHash("sha256").update(data).digest("hex");

export async function capture(page: Page, testInfo: TestInfo, name: string) {
  const filename = testInfo.outputPath(`${name}.png`);
  await page.screenshot({ path: filename, fullPage: true });
  await testInfo.attach(name, { path: filename, contentType: "image/png" });
}

export async function downloadRecording(page: Page, testInfo: TestInfo, name: string, source?: string) {
  const pending = page.waitForEvent("download");
  await page.getByTestId("speak.download_recording").click();
  const download = await pending;
  const filename = testInfo.outputPath(`${name}-${download.suggestedFilename()}`);
  await download.saveAs(filename);
  expect(await download.failure()).toBeNull();
  const bytes = readFileSync(filename);
  expect(bytes.length).toBeGreaterThan(1000);
  if (source) expect(sha256(bytes)).toBe(sha256(readFileSync(source)));
  await testInfo.attach(name, { path: filename, contentType: source ? "audio/wav" : "audio/webm" });
  return { filename, sha256: sha256(bytes), bytes: bytes.length };
}

export async function playAndSeek(page: Page, selector: string, minimumDurationSec = 10) {
  const audio = page.locator(selector);
  await audio.evaluate(async (el: HTMLAudioElement) => { await el.play(); });
  await expect.poll(() => audio.evaluate((el: HTMLAudioElement) => el.currentTime)).toBeGreaterThan(.1);
  await audio.evaluate((el: HTMLAudioElement) => el.pause());
  if (!Number.isFinite(await audio.evaluate((el: HTMLAudioElement) => el.duration))) {
    // MediaRecorder WebM omits duration metadata. Seeking to the end lets Chromium
    // discover its last timestamp; then verify seeking back into the actual speech.
    await audio.evaluate((el: HTMLAudioElement) => { el.currentTime = 1e10; });
    await expect.poll(() => audio.evaluate((el: HTMLAudioElement) => Number.isFinite(el.duration))).toBe(true);
  }
  await audio.evaluate((el: HTMLAudioElement) => { el.currentTime = 1; });
  await expect.poll(() => audio.evaluate((el: HTMLAudioElement) => el.seeking)).toBe(false);
  expect(await audio.evaluate((el: HTMLAudioElement) => el.currentTime)).toBeCloseTo(1, 1);
  const duration = await audio.evaluate((el: HTMLAudioElement) => el.duration);
  expect(duration).toBeGreaterThan(minimumDurationSec);
  return duration;
}

export function retainJson(testInfo: TestInfo, name: string, value: unknown) {
  const filename = testInfo.outputPath(`${name}.json`);
  writeFileSync(filename, JSON.stringify(value, null, 2));
  return testInfo.attach(name, { path: filename, contentType: "application/json" });
}

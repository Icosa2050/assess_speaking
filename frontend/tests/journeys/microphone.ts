import { expect, type Page } from "@playwright/test";
export async function calibrateMicrophone(page: Page, returnPath: string, initial = false) {
  if (initial) await page.goto("/runtime-setup");
  else await page.getByTestId("speak.microphone_setup").click();
  await page.getByTestId("microphone-processing").uncheck();
  await page.getByTestId("microphone-test-start").click();
  await expect(page.getByTestId("microphone-test-playback")).toBeVisible({ timeout: 10_000 });
  await page.getByTestId("microphone-test-playback").evaluate(async element => {
    const audio = element as HTMLAudioElement;
    // Keep actual decoding, clock progression and ended events on a headless
    // host. The test-only sink does not bypass playback or setup confirmation.
    const context = new AudioContext({ sinkId: { type: "none" } } as AudioContextOptions);
    context.createMediaElementSource(audio).connect(context.destination);
    audio.addEventListener("ended", () => void context.close(), { once: true });
    await context.resume(); await audio.play();
  });
  await expect(page.getByTestId("microphone-test-confirm")).toBeEnabled({ timeout: 10_000 });
  await page.getByTestId("microphone-test-confirm").click();
  await page.locator(`a[href="${returnPath}"]`).first().click();
}

export async function assertCapturedDuration(page: Page) {
  const duration = await page.getByTestId("speak.download_recording").evaluate(async element => {
    const blob = await (await fetch((element as HTMLAnchorElement).href)).arrayBuffer();
    const context = new OfflineAudioContext(1, 1, 16000);
    return (await context.decodeAudioData(blob)).duration;
  });
  expect(duration, "actual encoded synthetic microphone duration must reach the production minimum").toBeGreaterThanOrEqual(30);
}

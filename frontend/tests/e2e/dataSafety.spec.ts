import { readFile } from "node:fs/promises";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { createRehearsal } from "../../src/lib/rehearsal/session";
import { expect, test } from "../fixtures";

const damaged = { transaction: null, archived: [], completed: [], recovery_error: "Synthetic damaged journal; saved history remains available." };

test("damaged journal offers a safe first action and confirms start fresh without removing recordings", async ({ page }) => {
  const rehearsal = createRehearsal("en", "B1", "Retained practice", { provider: "ollama", model: "fixture", baseUrl: "", whisper: "tiny", feedbackLanguage: "en" });
  rehearsal.parts = [rehearsal.parts[0]];
  rehearsal.parts[0].recorded = true;
  await page.goto("/");
  await page.evaluate(async item => {
    const path = "/src/lib/rehearsal/storage.ts";
    const store = await import(path);
    await store.savePartRecording(item, 0, new Blob(["retained audio"], { type: "audio/wav" }));
  }, rehearsal);
  let fresh = false;
  await page.route("**/v1/journal/status", route => route.fulfill({ json: fresh ? { ...damaged, recovery_error: null } : damaged }));
  await page.route("**/v1/journal/start-fresh", route => { fresh = true; return route.fulfill({ json: { phase: "fresh" } }); });
  await page.reload();
  const notice = page.getByTestId("journal-recovery-notice");
  await expect(notice).toBeVisible();
  await expect(notice.getByRole("link", { name: "Export diagnostics only" })).toBeVisible();
  await notice.getByRole("link", { name: "Restore from a backup" }).click();
  await expect(page.getByRole("button", { name: "Choose a backup ZIP" })).toBeFocused();
  await notice.getByRole("link", { name: "Open Your data recovery" }).click();
  const panel = page.getByTestId("journal-panel");
  await expect(panel.getByTestId("journal-state")).toContainText("Journal damaged");
  await expect(panel.getByRole("heading", { name: "Your data", exact: true })).toBeFocused();
  await expect(panel.getByText("Safe to try")).toBeVisible();
  await expect(panel.getByRole("button", { name: "Export diagnostics only" })).toBeVisible();
  await expect(panel.getByText(/Saved reports, history, recordings, and uploads stay in place/)).toBeVisible();
  await panel.getByRole("button", { name: "Start a fresh journal" }).click();
  const dialog = page.getByRole("dialog", { name: "Start a fresh journal?" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText(/Kept: saved reports, history, recordings, and uploads/)).toBeVisible();
  await expect(dialog.getByText(/Replaced: the damaged recovery journal/)).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Cancel and keep current state" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(panel.getByRole("button", { name: "Start a fresh journal" })).toBeFocused();
  await panel.getByRole("button", { name: "Start a fresh journal" }).click();
  await dialog.getByRole("button", { name: "Confirm: start fresh" }).click();
  await expect(panel.getByText("A fresh recovery journal is ready. You can practise again.")).toBeFocused();
  await expect(panel.getByTestId("journal-state")).toContainText("Journal healthy");
  expect(await page.evaluate(async id => {
    const path = "/src/lib/rehearsal/storage.ts";
    const store = await import(path);
    const item = await store.getRehearsal(id);
    return { retained: !!item, audio: await (await store.loadPartRecording(item, 0)).text() };
  }, rehearsal.id)).toEqual({ retained: true, audio: "retained audio" });
});

test("a backup can be previewed before restore from the damaged view", async ({ page }, info) => {
  await page.goto("/settings");
  await expect(page.getByTestId("journal-state")).toContainText("Journal healthy");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Create and save backup" }).click();
  const file = await download;
  const path = info.outputPath("data-safety-backup.zip");
  await file.saveAs(path);
  await page.route("**/v1/journal/status", route => route.fulfill({ json: damaged }));
  await page.reload();
  await page.getByLabel("Restore from a backup").setInputFiles({ name: "backup.zip", mimeType: "application/zip", buffer: await readFile(path) });
  await expect(page.getByRole("region", { name: /Restore .* attempts and .* rehearsals/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: /Restore .* attempts and .* rehearsals/ })).toBeFocused();
  await expect(page.getByText(/Existing items are kept unchanged and skipped/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Restore these new items" })).toBeEnabled();
  await page.unroute("**/v1/journal/status");
  await page.getByRole("region", { name: /Restore .* attempts and .* rehearsals/ }).getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByRole("button", { name: "Create and save backup" })).toBeEnabled();
});

test("keyboard navigation reaches the safe action at both desktop sizes", async ({ page }) => {
  await page.route("**/v1/journal/status", route => route.fulfill({ json: damaged }));
  const screenshotDir = process.env.DATA_SAFETY_SCREENSHOT_DIR;
  if (screenshotDir) mkdirSync(screenshotDir, { recursive: true });
  for (const [width, height] of [[1360, 900], [1100, 760]]) {
    await page.setViewportSize({ width, height });
    await page.goto("/");
    const navLink = page.getByRole("navigation").getByRole("link", { name: "Your data" });
    await navLink.focus();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/settings#your-data$/);
    await expect(page.getByRole("heading", { name: "Your data", exact: true })).toBeFocused();
    const safe = page.getByRole("button", { name: "Export diagnostics only" });
    await safe.focus();
    await expect(safe).toBeFocused();
    if (screenshotDir) await page.screenshot({ path: path.join(screenshotDir, `after-damaged-${width}x${height}.png`) });
  }
});

for (const [label, buttonName] of [["Export diagnostics only", "Export diagnostics only"], ["Start a fresh journal", "Start a fresh journal"]]) {
  test(`recovery notice focuses ${label} after delayed journal status`, async ({ page }) => {
    let delay = false;
    let release: () => void = () => {};
    const gate = new Promise<void>(resolve => { release = resolve; });
    await page.route("**/v1/journal/status", async route => {
      if (delay) await gate;
      await route.fulfill({ json: damaged });
    });
    await page.goto("/");
    const notice = page.getByTestId("journal-recovery-notice");
    await expect(notice).toBeVisible();
    delay = true;
    await notice.getByRole("link", { name: label, exact: true }).click();
    await expect(page).toHaveURL(/\/settings#journal-/);
    await expect(page.getByRole("button", { name: buttonName, exact: true })).toHaveCount(0);
    release();
    await expect(page.getByRole("button", { name: buttonName, exact: true })).toBeFocused();
  });
}

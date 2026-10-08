import { guardBrowserContext, expect, test } from "../fixtures";
import { createRehearsal } from "../../src/lib/rehearsal/session";
import { readFile } from "node:fs/promises";

const session = () => createRehearsal("en", "B1", "Archive fixture", { provider: "ollama", model: "fixture", baseUrl: "", whisper: "tiny", feedbackLanguage: "en" });

test("populated v1 migrates without losing audio; global maintenance excludes another tab and archive supports Undo", async ({ page, context }) => {
  const saved = session(); saved.parts = [saved.parts[0]]; saved.parts[0].recorded = true;
  await page.goto("/");
  await page.evaluate(async item => {
    const db = await new Promise<IDBDatabase>((resolve, reject) => { const request = indexedDB.open("vostavo-rehearsals", 1);
      request.onupgradeneeded = () => { request.result.createObjectStore("sessions", { keyPath: "id" }); request.result.createObjectStore("recordings"); };
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
    await new Promise<void>((resolve, reject) => { const tx = db.transaction(["sessions", "recordings"], "readwrite"); tx.objectStore("sessions").add(item);
      tx.objectStore("recordings").add(new Blob(["Synthetic recording"], { type: "audio/wav" }), `${item.id}:0`);
      tx.oncomplete = () => resolve(); tx.onabort = () => reject(tx.error); }); db.close();
  }, saved);
  const migrated = await page.evaluate(async id => { const path = "/src/lib/rehearsal/storage.ts"; const store = await import(path);
    const value = await store.getRehearsal(id); return { id: value.id, audio: await (await store.loadPartRecording(value, 0)).text() }; }, saved.id);
  expect(migrated).toEqual({ id: saved.id, audio: "Synthetic recording" });
  await page.evaluate(() => { (window as unknown as { release: () => void }).release = () => {};
    void navigator.locks.request("vostavo:journal-maintenance", { mode: "exclusive" }, () => new Promise<void>(resolve => { (window as unknown as { release: () => void }).release = resolve; })); });
  const other = await context.newPage(); await other.goto("/settings");
  expect(await other.evaluate(async item => { const path = "/src/lib/rehearsal/storage.ts"; const store = await import(path);
    try { await store.saveRehearsal(item); return "incorrectly saved"; } catch (error) { return String(error); } }, saved)).toContain("another tab");
  await page.evaluate(() => (window as unknown as { release: () => void }).release());
  expect(await page.evaluate(async id => { const path = "/src/lib/rehearsal/storage.ts"; const store = await import(path);
    const original = await store.getRehearsal(id); await store.deleteRehearsal(original);
    const archived = await store.getRehearsal(id); const hidden = (await store.listRehearsals()).length === 0;
    const audio = await (await store.loadPartRecording(archived, 0)).text(); await store.undoRehearsalArchive(archived);
    return { hidden, audio, count: (await store.listRehearsals()).length }; }, saved.id)).toEqual({ hidden: true, audio: "Synthetic recording", count: 1 });
  expect(await page.evaluate(async id => { const path = "/src/lib/rehearsal/storage.ts", apiPath = "/src/lib/rehearsal/maintenance.ts";
    const store = await import(path); const api = await import(apiPath); const item = await store.getRehearsal(id);
    await store.deleteRehearsal(item); await api.rehearsalRemoval(await store.getRehearsal(id));
    return { session: await store.getRehearsal(id), audio: await store.loadPartRecording(item, 0) }; }, saved.id)).toEqual({ session: undefined, audio: undefined });
});

test("full learner ZIP transfers rehearsal audio to a fresh browser profile and skips existing identities on repeated restore", async ({ page, browser }, info) => {
  const saved = session(); saved.parts = [saved.parts[0]]; saved.parts[0].recorded = true;
  await page.goto("/settings");
  await page.evaluate(async item => { const path = "/src/lib/rehearsal/storage.ts"; const store = await import(path);
    await store.savePartRecording(item, 0, new Blob(["Synthetic cross-profile audio"], { type: "audio/wav" })); }, saved);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Create and save backup", exact: true }).click();
  const file = await download; const path = info.outputPath("learner-backup.zip"); await file.saveAs(path);
  await expect(page.getByTestId("journal-panel").getByRole("status")).toContainText("Backup download started");
  const bytes = await readFile(path);
  const target = await browser.newContext(); const verifyGuard = await guardBrowserContext(target); const second = await target.newPage(); await second.goto("/settings");
  await second.getByLabel("Open backup ZIP").setInputFiles({ name: "backup.zip", mimeType: "application/zip", buffer: bytes });
  await expect(second.getByText("Restore 0 attempts and 1 rehearsals?")).toBeVisible();
  await second.getByRole("button", { name: "Restore these new items", exact: true }).click();
  await expect(second.getByTestId("journal-panel").getByRole("status")).toContainText("Backup restored");
  expect(await second.evaluate(async id => { const path = "/src/lib/rehearsal/storage.ts"; const store = await import(path); const value = await store.getRehearsal(id);
    return { id: value.id, audio: await (await store.loadPartRecording(value, 0)).text(), phase: value.phase, job: value.parts[0].jobId }; }, saved.id)).toEqual({ id: saved.id, audio: "Synthetic cross-profile audio", phase: "review", job: undefined });
  await second.getByLabel("Open backup ZIP").setInputFiles({ name: "backup.zip", mimeType: "application/zip", buffer: bytes });
  await expect(second.getByText("Restore 0 attempts and 0 rehearsals?")).toBeVisible();
  await expect(second.getByText("1 existing items will be kept unchanged and skipped.")).toBeVisible();
  await second.getByRole("button", { name: "Restore these new items", exact: true }).click();
  await expect(second.getByTestId("journal-panel").getByRole("status")).toContainText("Backup restored");
  await target.close(); verifyGuard();
});

test("a maintenance lease refuses direct API uploads, submission and settings writes", async ({ request }) => {
  const backend = process.env.VOSTAVO_FIXTURE_BACKEND_URL!;
  const response = await request.post(`${backend}/v1/journal/begin`); expect(response.ok()).toBe(true); const { id } = await response.json();
  try {
    for (const path of ["uploads", "assessments", "support-bundles", "maintenance/cleanup"]) {
      const result = await request.post(`${backend}/v1/${path}`, { data: {} }); expect(result.status()).toBe(409);
      expect((await result.json()).detail.detail).toContain("recovery is in progress");
    }
    expect((await request.put(`${backend}/v1/runtime/settings`, { data: {} })).status()).toBe(409);
  } finally { expect((await request.post(`${backend}/v1/journal/${id}/abort`)).ok()).toBe(true); }
});

test("browser quota rollback compensates backend publication; a lost completion reply recovers after reload", async ({ page, browser }, info) => {
  const saved = session(); saved.parts = [saved.parts[0]]; saved.parts[0].recorded = true;
  await page.goto("/settings");
  await page.evaluate(async item => { const path = "/src/lib/rehearsal/storage.ts"; const store = await import(path);
    await store.savePartRecording(item, 0, new Blob(["Synthetic recovery audio"], { type: "audio/wav" })); }, saved);
  const downloading = page.waitForEvent("download"); await page.getByRole("button", { name: "Create and save backup", exact: true }).click();
  const zip = await downloading; const path = info.outputPath("recovery-backup.zip"); await zip.saveAs(path); const bytes = await readFile(path);
  const target = await browser.newContext(); const verifyGuard = await guardBrowserContext(target); const second = await target.newPage(); await second.goto("/settings");
  const upload = () => second.getByLabel("Open backup ZIP").setInputFiles({ name: "backup.zip", mimeType: "application/zip", buffer: bytes });
  await upload(); await expect(second.getByRole("button", { name: "Restore these new items", exact: true })).toBeVisible();
  await second.evaluate(() => {
    const original = IDBObjectStore.prototype.add;
    IDBObjectStore.prototype.add = function (...args) {
      const result = Reflect.apply(original, this, args);
      if (this.name === "sessions") { IDBObjectStore.prototype.add = original; queueMicrotask(() => this.transaction.abort()); }
      return result;
    };
  });
  await second.getByRole("button", { name: "Restore these new items", exact: true }).click();
  await expect(second.getByTestId("journal-panel").getByRole("alert")).toBeVisible();
  expect(await second.evaluate(async () => { const storagePath = "/src/lib/rehearsal/storage.ts", apiPath = "/src/lib/rehearsal/maintenance.ts";
    const store = await import(storagePath); const api = await import(apiPath); return { count: (await store.listRehearsals()).length, transaction: (await api.journalStatus()).transaction, local: await store.rehearsalMaintenance() }; })).toEqual({ count: 0, transaction: null, local: undefined });
  await upload(); await expect(second.getByRole("button", { name: "Restore these new items", exact: true })).toBeVisible();
  await second.route("**/v1/journal/*/complete", async route => { const accepted = await route.fetch(); expect(accepted.ok()).toBe(true);
    await route.fulfill({ status: 503, json: { detail: { detail: "Synthetic lost completion reply" } } }); });
  await second.getByRole("button", { name: "Restore these new items", exact: true }).click();
  await expect(second.getByTestId("journal-panel").getByRole("alert")).toContainText("lost completion reply");
  await second.unroute("**/v1/journal/*/complete"); await second.reload();
  await expect(second.getByRole("button", { name: "Finish recovery", exact: true })).toBeVisible();
  await second.getByRole("button", { name: "Finish recovery", exact: true }).click();
  await expect(second.getByTestId("journal-panel").getByRole("status")).toContainText("Journal recovery finished");
  expect(await second.evaluate(async id => { const path = "/src/lib/rehearsal/storage.ts"; const store = await import(path); const value = await store.getRehearsal(id);
    return { count: (await store.listRehearsals()).length, audio: await (await store.loadPartRecording(value, 0)).text(), local: await store.rehearsalMaintenance() }; }, saved.id)).toEqual({ count: 1, audio: "Synthetic recovery audio", local: undefined });
  await target.close(); verifyGuard();
});

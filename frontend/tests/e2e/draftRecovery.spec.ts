import { stubReadyReadiness } from "../readinessStub";
import { expect, test, type Page } from "../fixtures";

const wav = () => {
  const rate = 8000, seconds = 32, samples = rate * seconds;
  const bytes = Buffer.alloc(44 + samples * 2);
  bytes.write("RIFF", 0); bytes.writeUInt32LE(bytes.length - 8, 4); bytes.write("WAVEfmt ", 8);
  bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(1, 20); bytes.writeUInt16LE(1, 22);
  bytes.writeUInt32LE(rate, 24); bytes.writeUInt32LE(rate * 2, 28);
  bytes.writeUInt16LE(2, 32); bytes.writeUInt16LE(16, 34);
  bytes.write("data", 36); bytes.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) bytes.writeInt16LE(Math.round(2500 * Math.sin(2 * Math.PI * 220 * i / rate)), 44 + i * 2);
  return bytes;
};

const injectDraftBridge = async (page: Page) => page.addInitScript(() => {
  const key = "draft-playwright-v1";
  type Entry = { manifest: Record<string, any>; chunks: string[] };
  const read = (): Entry[] => JSON.parse(localStorage.getItem(key) || "[]");
  const write = (entries: Entry[]) => localStorage.setItem(key, JSON.stringify(entries));
  const pending = new Map<string, { draftId: string; revision: number; take: Record<string, any>; chunks: string[]; size: number }>();
  let sequence = 1;
  const id = () => (sequence++).toString(16).padStart(32, "0");
  const encode = (bytes: number[]) => {
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary);
  };
  const decode = (value: string) => Array.from(atob(value), char => char.charCodeAt(0));
  const store = {
    list: async () => ({ drafts: read().map(entry => entry.manifest), unreadableIds: [] }),
    recoveryIds: async () => read().map(entry => entry.manifest.draftId),
    saveSetup: async (setup: Record<string, any>, draftId?: string, expectedRevision?: number) => {
      const entries = read();
      const index = entries.findIndex(entry => entry.manifest.draftId === draftId);
      const old = entries[index]?.manifest;
      if (old && old.revision !== expectedRevision) throw Error("Draft changed in another window");
      if (!old && entries.some(entry => entry.manifest.speakerId === setup.speakerId.trim().toLowerCase())) throw Error("A draft already exists for this learner");
      const now = Date.now();
      const manifest = old ? { ...old, setup, revision: old.revision + 1, updatedAt: now }
        : { version: 1, draftId: id(), speakerId: setup.speakerId.trim().toLowerCase(), setup,
          createdAt: now, updatedAt: now, expiresAt: now + 7 * 86400000, revision: 1,
          status: "setup", take: null, submission: null };
      if (index >= 0) entries[index].manifest = manifest; else entries.push({ manifest, chunks: [] });
      write(entries); return manifest;
    },
    replaceSetup: async (setup: Record<string, any>, draftId: string, expectedRevision: number) => {
      const entries = read();
      const entry = entries.find(item => item.manifest.draftId === draftId);
      if (!entry || entry.manifest.revision !== expectedRevision) throw Error("Draft changed or expired");
      if (entry.manifest.speakerId !== setup.speakerId.trim().toLowerCase()) throw Error("Another learner owns this draft");
      const now = Date.now();
      entry.manifest = { ...entry.manifest, setup, revision: expectedRevision + 1, updatedAt: now,
        expiresAt: now + 7 * 86400000, status: "setup", take: null, submission: null };
      entry.chunks = [];
      write(entries);
      return entry.manifest;
    },
    detachRetry: async (draftId: string, expectedRevision: number, parentId: string) => {
      const entries = read();
      const entry = entries.find(item => item.manifest.draftId === draftId);
      if (!entry || entry.manifest.revision !== expectedRevision || entry.manifest.setup.retryOfSessionId !== parentId) throw Error("Draft changed in another window");
      entry.manifest = { ...entry.manifest, revision: expectedRevision + 1,
        setup: { ...entry.manifest.setup, retryOfSessionId: "", retryGoalEligible: false } };
      write(entries); return entry.manifest;
    },
    importLegacyBrief: async (setup: Record<string, any>, createdAt: number) => {
      const manifest = await store.saveSetup(setup);
      const entries = read(); const entry = entries.find(item => item.manifest.draftId === manifest.draftId)!;
      entry.manifest.createdAt = createdAt; entry.manifest.expiresAt = createdAt + 7 * 86400000;
      write(entries); return entry.manifest;
    },
    beginTake: async (draftId: string, revision: number, take: Record<string, any>) => {
      const entry = read().find(item => item.manifest.draftId === draftId);
      if (!entry || entry.manifest.revision !== revision) throw Error("Draft changed or expired");
      const transfer = id(); pending.set(transfer, { draftId, revision, take, chunks: [], size: 0 }); return transfer;
    },
    writeTakeChunk: async (transfer: string, bytes: number[]) => {
      const item = pending.get(transfer)!;
      item.chunks.push(encode(bytes)); item.size += bytes.length; return item.size;
    },
    abortTake: async (transfer: string) => { pending.delete(transfer); },
    abortTakeForDraft: async (draftId: string) => { for (const [transfer, item] of pending) if (item.draftId === draftId) pending.delete(transfer); },
    finishTake: async (transfer: string) => {
      const item = pending.get(transfer)!;
      const entries = read(); const entry = entries.find(value => value.manifest.draftId === item.draftId)!;
      if (item.size !== item.take.size || entry.manifest.revision !== item.revision) throw Error("Draft changed or incomplete");
      entry.manifest = { ...entry.manifest, revision: item.revision + 1, status: "take_ready",
        take: { ...item.take, mediaId: id() }, expiresAt: item.take.captureEndedAt + 7 * 86400000, updatedAt: Date.now() };
      entry.chunks = item.chunks; write(entries); pending.delete(transfer); return entry.manifest;
    },
    loadTakeChunk: async (draftId: string, mediaId: string, index: number) => {
      const entry = read().find(item => item.manifest.draftId === draftId && item.manifest.take?.mediaId === mediaId);
      if (!entry) throw Error("Saved take is damaged");
      return decode(entry.chunks[index]);
    },
    markSubmission: async () => { throw Error("No automatic submission receipt"); },
    discard: async (draftId: string) => write(read().filter(entry => entry.manifest.draftId !== draftId)),
    expire: async () => 0,
  };
  (window as any).__VOSTAVO_DESKTOP__ = {
    apiBaseUrl: "http://127.0.0.1:8800", deploymentMode: "local", launchMode: "packaged",
    packagingSafe: true, authMode: "guest", draftStore: store,
  };
});

const backend = async (page: Page, requests: { uploads: number; assessments: number }) => {
  await stubReadyReadiness(page);
  await page.route("**/v1/runtime", route => route.fulfill({ json: { configured: true, provider: "ollama", model: "test",
    base_url: "http://127.0.0.1:11434/v1", requires_api_key: false, has_api_key: false } }));
  await page.route("**/v1/runtime/settings", route => route.fulfill({ json: { ui_locale: "en", whisper_model: "small", active_connection_id: "", connections: [] } }));
  await page.route("**/v1/diagnostics", route => route.fulfill({ json: { items: [] } }));
  await page.route("**/v1/history", route => route.fulfill({ json: { items: [] } }));
  await page.route("**/v1/assessment-route", route => route.fulfill({ json: { version: 1, available: true,
    fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" },
    analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null } }));
  await page.route("**/v1/uploads/limits", route => route.fulfill({ json: { max_bytes: 100 * 1024 * 1024, available_bytes: 200 * 1024 * 1024 } }));
  await page.route("**/v1/uploads", route => { requests.uploads++; return route.fulfill({ json: {
    audio_id: "audio-1", stored_path: "/private/audio-1", sha1: "x", original_name: "take.wav" } }); });
  await page.route("**/v1/assessments", route => { requests.assessments++; return route.fulfill({ json: { assessment_id: "assessment-1", status: "queued" } }); });
  await page.route("**/v1/assessments/assessment-1", route => route.fulfill({ json: {
    assessment_id: "assessment-1", status: "queued", phase: "queued", progress: 0,
    error: null, report_path: null, payload: null, summary: null } }));
};

const save = async (page: Page) => {
  await page.goto("/session-setup");
  await page.getByTestId("setup.speaker_id").fill("Alex");
  await page.getByTestId("setup.recommended_start").click();
  await page.getByTestId("setup.continue").click();
  await page.getByTestId("speak.input_mode_upload").click();
  await page.getByTestId("speak.upload_input").setInputFiles({ name: "take.wav", mimeType: "audio/wav", buffer: wav() });
  await expect(page.getByText("Saved on this Mac.")).toHaveCount(0);
  await page.getByRole("button", { name: "Keep this selected file as the local take" }).click();
  await expect(page.getByTestId("speak.recording_status")).toContainText("Saved on this Mac", { timeout: 20000 });
};

for (const [label, viewport] of [["1360x900", { width: 1360, height: 900 }], ["1100x760", { width: 1100, height: 760 }]] as const) {
  test.describe(`draft recovery ${label}`, () => {
    test.use({ viewport });
    test("save, reload, resume, then explicit submit", async ({ page }) => {
      const requests = { uploads: 0, assessments: 0 };
      await injectDraftBridge(page); await backend(page, requests);
      await save(page);
      expect(requests).toEqual({ uploads: 0, assessments: 0 });
      await page.getByRole("link", { name: "Practice", exact: true }).click();
      await page.getByTestId("home.resume").click();
      await expect(page.getByTestId("speak.recording_status")).toContainText("Saved on this Mac");
      await page.reload();
      await expect(page.getByTestId("draft.entry")).toContainText("Alex");
      expect(requests).toEqual({ uploads: 0, assessments: 0 });
      await page.getByRole("button", { name: "Resume" }).click();
      await expect(page.getByTestId("speak.recording_status")).toContainText("Saved on this Mac");
      expect(await page.locator("audio[controls]").evaluate(async node => {
        const audio = node as HTMLAudioElement;
        const bytes = await fetch(audio.src).then(response => response.arrayBuffer());
        return (await new AudioContext().decodeAudioData(bytes)).duration;
      })).toBeGreaterThan(30);
      expect(requests).toEqual({ uploads: 0, assessments: 0 });
      await page.getByTestId("speak.submit").click();
      await expect.poll(() => requests.assessments).toBe(1);
      expect(requests.uploads).toBe(1);
    });
    test("discard deletes only the local draft without uploading", async ({ page }) => {
      const requests = { uploads: 0, assessments: 0 };
      await injectDraftBridge(page); await backend(page, requests);
      await save(page);
      await page.reload();
      await page.getByRole("button", { name: "Discard draft and take" }).click();
      await expect(page.getByText(/Delete the local draft for Alex/)).toBeVisible();
      await page.getByRole("button", { name: "Discard draft and take" }).click();
      await expect(page.getByTestId("draft.entry")).toHaveCount(0);
      await page.reload();
      await expect(page.getByTestId("draft.recovery")).toHaveCount(0);
      expect(requests).toEqual({ uploads: 0, assessments: 0 });
    });
  });
}

test("restore warns about an imported retry parent and keeps the unsent take", async ({ page }) => {
  const requests = { uploads: 0, assessments: 0 };
  await injectDraftBridge(page); await backend(page, requests);
  await save(page);
  await page.evaluate(() => {
    const key = "draft-playwright-v1";
    const entries = JSON.parse(localStorage.getItem(key)!);
    entries[0].manifest.setup.retryOfSessionId = "imported-parent";
    entries[0].manifest.setup.retryGoalEligible = true;
    entries[0].manifest.setup.coachingGoal = "Finish clearly";
    localStorage.setItem(key, JSON.stringify(entries));
  });
  const transaction = "c".repeat(32);
  await page.route("**/v1/journal/status", route => route.fulfill({ json: { transaction: null, archived: [], completed: [] } }));
  await page.route("**/v1/journal/begin-restore", route => route.fulfill({ json: { id: transaction } }));
  await page.route(`**/v1/journal/${transaction}/restore`, route => route.fulfill({ json: {
    id: transaction, attempts: 1, restored_attempt_ids: ["imported-parent"], skipped_attempts: [],
    skipped_rehearsals: [], missing: [], browser: { sessions: [], recordings: [] },
  } }));
  await page.route(`**/v1/journal/${transaction}/commit`, route => route.fulfill({ json: { phase: "committed" } }));
  await page.route(`**/v1/journal/${transaction}/complete`, route => route.fulfill({ json: { phase: "complete" } }));
  await page.route("**/v1/history/imported-parent", route => route.fulfill({ json: {
    payload: { restored_from: { restored_at_ms: Date.now() + 1000 } },
  } }));
  await page.reload();
  await page.getByRole("button", { name: "Continue without resuming" }).click();
  await page.getByRole("link", { name: "Settings", exact: true }).click();
  await expect(page.getByText(/Unfinished drafts and unsent recordings stay only on this Mac/).first()).toBeVisible();
  await page.getByLabel("Open backup ZIP").setInputFiles({ name: "backup.zip", mimeType: "application/zip", buffer: Buffer.from("synthetic") });
  const preview = page.getByRole("region", { name: /Restore 1 attempts/ });
  await expect(preview.getByRole("alert")).toContainText("1 local draft(s)");
  const commit = preview.getByRole("button", { name: "Restore these new items" });
  await expect(commit).toBeDisabled();
  await preview.getByLabel(/I understand that matching local drafts/).check();
  await commit.click();
  await expect(page.getByText("Backup restored.")).toBeVisible();
  await page.reload();
  await expect(page.getByText(/This draft refers to an attempt imported from a backup/)).toBeVisible();
  await page.getByRole("button", { name: "Resume" }).click();
  await expect(page.getByTestId("speak.recording_status")).toContainText("Saved on this Mac");
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("draft-playwright-v1")!)[0].manifest.setup.retryOfSessionId)).toBe("");
  expect(requests).toEqual({ uploads: 0, assessments: 0 });
});

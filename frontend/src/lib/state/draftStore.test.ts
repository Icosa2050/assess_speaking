import { createHash, webcrypto } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DesktopDraftStoreBridge, SavedDraft } from "@/lib/runtime/desktopBridge";
import { buildSessionDraft } from "./sessionDraft";
import { createDraftStoreAdapter, DraftStoreError } from "./draftStore";

const ID = "a".repeat(32);
const MEDIA = "b".repeat(32);
const setup = buildSessionDraft({ speakerId: "Alex", themeId: "trip", themeLabel: "Describe a trip", promptText: "Describe a trip." });
const base = (): SavedDraft => ({ version: 1, draftId: ID, speakerId: "alex", setup,
  createdAt: Date.now() - 1000, updatedAt: Date.now(), expiresAt: Date.now() + 86_400_000,
  revision: 1, status: "setup", take: null, submission: null });

const bridge = (): DesktopDraftStoreBridge => ({
  list: vi.fn().mockResolvedValue({ drafts: [], unreadableIds: [] }),
  recoveryIds: vi.fn().mockResolvedValue([]),
  saveSetup: vi.fn().mockResolvedValue(base()),
  importLegacyBrief: vi.fn().mockResolvedValue(base()),
  beginTake: vi.fn().mockResolvedValue("c".repeat(32)),
  writeTakeChunk: vi.fn().mockResolvedValue(0),
  abortTake: vi.fn().mockResolvedValue(undefined),
  abortTakeForDraft: vi.fn().mockResolvedValue(undefined),
  finishTake: vi.fn().mockResolvedValue(base()),
  loadTakeChunk: vi.fn().mockResolvedValue([]),
  markSubmission: vi.fn().mockResolvedValue(base()),
  discard: vi.fn().mockResolvedValue(undefined),
  expire: vi.fn().mockResolvedValue(0),
});

describe("desktop draft adapter", () => {
  beforeEach(() => vi.stubGlobal("crypto", webcrypto));
  afterEach(() => vi.unstubAllGlobals());

  it("uses bounded writes and returns success only after finishTake commits", async () => {
    const fake = bridge();
    const bytes = new Uint8Array(300_000).fill(7);
    const file = new File([bytes], "take.webm", { type: "audio/webm" });
    const digest = createHash("sha256").update(bytes).digest("hex");
    const saved = { ...base(), revision: 2, status: "take_ready" as const,
      take: { mediaId: MEDIA, originalFilename: file.name, mime: file.type, size: file.size,
        sha256: digest, durationSec: 8, captureEndedAt: Date.now(), inputMethod: "record" as const } };
    vi.mocked(fake.writeTakeChunk).mockImplementation(async (_id, part) => part.length === 262144 ? 262144 : 300000);
    let finish: (value: SavedDraft) => void = () => undefined;
    vi.mocked(fake.finishTake).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    const progress = vi.fn();
    let settled = false;
    const promise = createDraftStoreAdapter(fake).saveTake(base(), file, 8, "record", progress).then(value => { settled = true; return value; });
    await vi.waitFor(() => expect(fake.finishTake).toHaveBeenCalledOnce());
    expect(settled).toBe(false);
    expect(vi.mocked(fake.writeTakeChunk).mock.calls.map(call => call[1].length)).toEqual([262144, 37856]);
    finish(saved);
    expect(await promise).toEqual(saved);
    expect(progress).toHaveBeenLastCalledWith(100);
    expect(fake.abortTake).not.toHaveBeenCalled();
  });

  it("aborts a partial transfer and leaves the previous revision intact", async () => {
    const fake = bridge();
    vi.mocked(fake.writeTakeChunk).mockRejectedValue(new Error("disk full"));
    await expect(createDraftStoreAdapter(fake).saveTake(base(), new File(["audio"], "take.webm", { type: "audio/webm" }), 8, "record"))
      .rejects.toMatchObject({ kind: "storage" });
    expect(fake.abortTake).toHaveBeenCalledWith("c".repeat(32));
    expect(fake.finishTake).not.toHaveBeenCalled();
  });

  it("reports a revision conflict without writing a take", async () => {
    const fake = bridge();
    vi.mocked(fake.beginTake).mockRejectedValue(new Error("Draft changed in another window"));
    await expect(createDraftStoreAdapter(fake).saveTake(base(), new File(["audio"], "take.webm", { type: "audio/webm" }), 8, "record"))
      .rejects.toMatchObject({ kind: "conflict" });
    expect(fake.writeTakeChunk).not.toHaveBeenCalled();
  });

  it("lists unreadable entries, discards expired drafts, and aborts abandoned transfers", async () => {
    const fake = bridge();
    const expired = { ...base(), draftId: "d".repeat(32), expiresAt: Date.now() - 1 };
    vi.mocked(fake.list).mockResolvedValue({ drafts: [base(), expired], unreadableIds: ["e".repeat(32)] });
    const adapter = createDraftStoreAdapter(fake);
    const listed = await adapter.list();
    expect(listed.drafts).toHaveLength(1);
    expect(listed.unreadableIds).toEqual(["e".repeat(32)]);
    expect(fake.discard).toHaveBeenCalledWith(expired.draftId);
    await adapter.abortAbandoned(listed);
    expect(fake.abortTakeForDraft).toHaveBeenCalledWith(ID);
    expect(fake.abortTakeForDraft).toHaveBeenCalledWith("e".repeat(32));
  });

  it("fails closed when Keychain or storage is unavailable, while opaque IDs remain discardable", async () => {
    const fake = bridge();
    vi.mocked(fake.list).mockRejectedValue(new Error("Mac Keychain is unavailable"));
    vi.mocked(fake.recoveryIds).mockResolvedValue([ID]);
    const adapter = createDraftStoreAdapter(fake);
    await expect(adapter.list()).rejects.toMatchObject({ kind: "keychain" } satisfies Partial<DraftStoreError>);
    expect(await adapter.recoveryIds()).toEqual([ID]);
    await adapter.discard(ID);
    expect(fake.discard).toHaveBeenCalledWith(ID);
    vi.mocked(fake.list).mockRejectedValue(new Error("Disk unavailable"));
    await expect(adapter.list()).rejects.toMatchObject({ kind: "storage" });
  });

  it("has a clear unsupported browser path", async () => {
    const adapter = createDraftStoreAdapter(undefined);
    expect(adapter.supported).toBe(false);
    expect(await adapter.list()).toEqual({ drafts: [], unreadableIds: [] });
    await expect(adapter.saveSetup(setup)).rejects.toMatchObject({ kind: "unsupported" });
  });
});

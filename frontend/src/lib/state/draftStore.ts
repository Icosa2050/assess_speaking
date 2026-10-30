import { desktopBridge, type DesktopDraftStoreBridge, type DraftListing, type SavedDraft } from "@/lib/runtime/desktopBridge";
import type { SessionDraft } from "./sessionDraft";

const CHUNK_BYTES = 256 * 1024;
const MAX_TAKE_BYTES = 100 * 1024 * 1024;
const ID = /^[0-9a-f]{32}$/;

export type DraftStoreFailure = "unsupported" | "keychain" | "storage" | "conflict" | "unreadable";

export class DraftStoreError extends Error {
  constructor(readonly kind: DraftStoreFailure, message: string) {
    super(message);
  }
}

const failure = (error: unknown): DraftStoreError => {
  if (error instanceof DraftStoreError) return error;
  const message = error instanceof Error ? error.message : String(error);
  const kind: DraftStoreFailure = /keychain|key is unavailable|content key/i.test(message) ? "keychain"
    : /revision|another window|already exists|changed or expired/i.test(message) ? "conflict"
      : /damaged|unreadable|digest|saved take file/i.test(message) ? "unreadable" : "storage";
  return new DraftStoreError(kind, message);
};

export const normalizedLearner = (speakerId: string): string => speakerId.trim().toLowerCase();

export const validSavedDraft = (value: SavedDraft): boolean =>
  value.version === 1 && ID.test(value.draftId) && value.revision > 0 &&
  value.speakerId === normalizedLearner(value.setup.speakerId) &&
  Number.isFinite(value.expiresAt) && value.expiresAt > value.createdAt &&
  ["setup", "take_ready", "upload_uncertain", "uploaded", "assessment_uncertain", "assessing"].includes(value.status) &&
  (!value.take || (ID.test(value.take.mediaId) && value.take.size > 0 && value.take.size <= MAX_TAKE_BYTES &&
    /^[0-9a-f]{64}$/.test(value.take.sha256)));

const sha256 = async (bytes: ArrayBuffer): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), byte => byte.toString(16).padStart(2, "0")).join("");

export const createDraftStoreAdapter = (bridge: DesktopDraftStoreBridge | undefined = desktopBridge.readRuntime().draftStore) => {
  const unsupported = () => { throw new DraftStoreError("unsupported", "Local draft storage is unavailable in this browser"); };
  return {
    supported: Boolean(bridge),
    async list(): Promise<DraftListing> {
      if (!bridge) return { drafts: [], unreadableIds: [] };
      try {
        const listed = await bridge.list();
        const drafts: SavedDraft[] = [];
        const unreadableIds = [...listed.unreadableIds];
        for (const entry of listed.drafts) {
          if (!validSavedDraft(entry)) { unreadableIds.push(entry.draftId); continue; }
          if (entry.expiresAt <= Date.now()) { await bridge.discard(entry.draftId); continue; }
          drafts.push(entry);
        }
        return { drafts, unreadableIds: [...new Set(unreadableIds)] };
      } catch (error) { throw failure(error); }
    },
    async recoveryIds(): Promise<string[]> {
      if (!bridge) return [];
      try { return await bridge.recoveryIds(); } catch (error) { throw failure(error); }
    },
    async abortAbandoned(listing: DraftListing): Promise<void> {
      if (!bridge) return;
      try { await Promise.all([...listing.drafts.map(entry => entry.draftId), ...listing.unreadableIds].map(id => bridge.abortTakeForDraft(id))); }
      catch (error) { throw failure(error); }
    },
    async saveSetup(setup: SessionDraft, previous?: SavedDraft): Promise<SavedDraft> {
      if (!bridge) return unsupported();
      try { return await bridge.saveSetup(setup, previous?.draftId, previous?.revision); }
      catch (error) { throw failure(error); }
    },
    async importLegacyBrief(setup: SessionDraft, createdAt: number): Promise<SavedDraft> {
      if (!bridge) return unsupported();
      try { return await bridge.importLegacyBrief(setup, createdAt); }
      catch (error) { throw failure(error); }
    },
    async saveTake(draft: SavedDraft, file: File, durationSec: number, inputMethod: "record" | "upload", onProgress?: (percent: number) => void): Promise<SavedDraft> {
      if (!bridge) return unsupported();
      if (file.size <= 0 || file.size > MAX_TAKE_BYTES || !Number.isFinite(durationSec) || durationSec <= 0 || durationSec > 1200) {
        throw new DraftStoreError("storage", "Take size or duration is outside local storage limits");
      }
      let transfer: string | undefined;
      try {
        const digest = await sha256(await file.arrayBuffer());
        transfer = await bridge.beginTake(draft.draftId, draft.revision, {
          originalFilename: file.name, mime: file.type, size: file.size, sha256: digest,
          durationSec, captureEndedAt: Date.now(), inputMethod,
        });
        let written = 0;
        for (let offset = 0; offset < file.size; offset += CHUNK_BYTES) {
          const bytes = Array.from(new Uint8Array(await file.slice(offset, offset + CHUNK_BYTES).arrayBuffer()));
          written = await bridge.writeTakeChunk(transfer, bytes);
          if (written !== Math.min(offset + bytes.length, file.size)) throw new Error("Local take write count differs from file size");
          onProgress?.(Math.round(written * 100 / file.size));
        }
        const saved = await bridge.finishTake(transfer);
        transfer = undefined;
        if (saved.take?.sha256 !== digest || saved.take.size !== file.size) throw new Error("Saved take identity differs from the recording");
        return saved;
      } catch (error) {
        if (transfer) await bridge.abortTake(transfer).catch(() => undefined);
        throw failure(error);
      }
    },
    async loadTake(draft: SavedDraft): Promise<File> {
      if (!bridge) return unsupported();
      if (!validSavedDraft(draft) || !draft.take) throw new DraftStoreError("unreadable", "Saved take metadata is invalid");
      try {
        const chunks: Uint8Array<ArrayBuffer>[] = [];
        let size = 0;
        for (let index = 0; size < draft.take.size; index++) {
          const bytes = new Uint8Array(await bridge.loadTakeChunk(draft.draftId, draft.take.mediaId, index));
          if (!bytes.length || bytes.length > CHUNK_BYTES || size + bytes.length > draft.take.size) throw new Error("Saved take has an invalid chunk");
          chunks.push(bytes);
          size += bytes.length;
        }
        const file = new File(chunks, draft.take.originalFilename, { type: draft.take.mime, lastModified: draft.take.captureEndedAt });
        if (await sha256(await file.arrayBuffer()) !== draft.take.sha256) throw new Error("Saved take digest differs from metadata");
        return file;
      } catch (error) { throw failure(error); }
    },
    async discard(id: string): Promise<void> {
      if (!bridge) return unsupported();
      try { await bridge.discard(id); } catch (error) { throw failure(error); }
    },
  };
};

export type DraftStoreAdapter = ReturnType<typeof createDraftStoreAdapter>;

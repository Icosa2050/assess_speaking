import { desktopBridge, type DesktopDraftStoreBridge, type DraftListing, type DraftSubmission, type SavedDraft } from "@/lib/runtime/desktopBridge";
import type { SessionDraft } from "./sessionDraft";
import { Sha256Stream, sha256File } from "./sha256Stream";

const CHUNK_BYTES = 256 * 1024;
const MAX_TAKE_BYTES = 100 * 1024 * 1024;
const ID = /^[0-9a-f]{32}$/;
const TAKE_MIMES = new Set(["audio/webm", "audio/webm;codecs=opus", "audio/mp4", "audio/mp4;codecs=mp4a.40.2",
  "audio/wav", "audio/mpeg", "audio/ogg", "audio/ogg;codecs=opus", "audio/x-wav", "audio/flac", "audio/x-flac",
  "audio/aac", "audio/x-aac", "audio/aiff", "audio/x-aiff", "audio/x-m4a", "audio/opus"]);
const MIME_BY_EXTENSION: Record<string, string> = {
  wav: "audio/wav", mp3: "audio/mpeg", webm: "audio/webm", ogg: "audio/ogg", opus: "audio/opus",
  m4a: "audio/mp4", mp4: "audio/mp4", aac: "audio/aac", flac: "audio/flac", aiff: "audio/aiff", aif: "audio/aiff",
};
const MIME_ALIASES: Record<string, string> = {
  "audio/mp3": "audio/mpeg", "audio/wave": "audio/wav", "application/ogg": "audio/ogg", "video/webm": "audio/webm",
};
const safeTakeName = (name: string) => Boolean(name && new TextEncoder().encode(name).length <= 256 && !/[\\/\\\\\x00-\x1f\x7f]/.test(name));
const takeMime = (file: File): string => MIME_ALIASES[file.type] || file.type || MIME_BY_EXTENSION[file.name.split(".").pop()?.toLowerCase() || ""] || "";

export type DraftStoreFailure = "unsupported" | "keychain" | "device_only_unavailable" | "lost_key" | "storage" | "conflict" | "unreadable" | "uncertain" | "limit" | "format" | "history";

export class DraftStoreError extends Error {
  constructor(readonly kind: DraftStoreFailure, message: string) {
    super(message);
  }
}

const failure = (error: unknown): DraftStoreError => {
  if (error instanceof DraftStoreError) return error;
  const message = error instanceof Error ? error.message : String(error);
  const kind: DraftStoreFailure = /save may have changed/i.test(message) ? "uncertain"
    : /saved take cannot be opened on this mac/i.test(message) ? "lost_key"
    : /keychain.*code -34018/i.test(message) ? "device_only_unavailable"
    : /keychain|key is unavailable|content key/i.test(message) ? "keychain"
    : /three learner drafts|resolve unreadable drafts|draft storage limit|total encrypted/i.test(message) ? "limit"
    : /unsupported.*(mime|format)|invalid (saved )?take metadata/i.test(message) ? "format"
    : /revision|another window|already exists|changed or expired/i.test(message) ? "conflict"
      : /damaged|unreadable|digest|saved take file|failed authentication|integrity check|saved take is unavailable|file is unsafe|changed during verification|invalid chunk/i.test(message) ? "unreadable" : "storage";
  return new DraftStoreError(kind, message);
};

export const normalizedLearner = (speakerId: string): string => speakerId.trim().toLowerCase();

export const sameDraftAttempt = (saved: SavedDraft | null, setup: SessionDraft): saved is SavedDraft => Boolean(saved &&
  saved.speakerId === normalizedLearner(setup.speakerId) &&
  saved.setup.sessionId === setup.sessionId && saved.setup.retryOfSessionId === setup.retryOfSessionId &&
  saved.setup.learningLanguage === setup.learningLanguage && saved.setup.cefrLevel === setup.cefrLevel &&
  saved.setup.themeId === setup.themeId && saved.setup.promptId === setup.promptId &&
  saved.setup.promptText === setup.promptText && saved.setup.taskFamily === setup.taskFamily &&
  saved.setup.durationSec === setup.durationSec);

export const validSavedDraft = (value: SavedDraft): boolean =>
  value.version === 1 && ID.test(value.draftId) && value.revision > 0 &&
  value.speakerId === normalizedLearner(value.setup.speakerId) &&
  Number.isFinite(value.expiresAt) && value.expiresAt > value.createdAt &&
  ["setup", "take_ready", "upload_uncertain", "uploaded", "assessment_uncertain", "assessing"].includes(value.status) &&
  (!value.take || (ID.test(value.take.mediaId) && value.take.size > 0 && value.take.size <= MAX_TAKE_BYTES &&
    safeTakeName(value.take.originalFilename) && TAKE_MIMES.has(value.take.mime) &&
    Number.isFinite(value.take.durationSec) && value.take.durationSec > 0 && value.take.durationSec <= 1200 &&
    Number.isFinite(value.take.captureEndedAt) && ["record", "upload"].includes(value.take.inputMethod) &&
    /^[0-9a-f]{64}$/.test(value.take.sha256)));

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
      catch (error) {
        if (failure(error).kind === "uncertain") {
          try {
            await bridge.expire();
            const current = (await bridge.list()).drafts.find(entry => previous
              ? entry.draftId === previous.draftId && entry.revision > previous.revision
              : !entry.take && entry.status === "setup" && entry.speakerId === normalizedLearner(setup.speakerId));
            if (current && sameDraftAttempt(current, setup)) return current;
          } catch { /* The state remains uncertain. */ }
        }
        throw failure(error);
      }
    },
    async replaceSetup(setup: SessionDraft, previous: SavedDraft): Promise<SavedDraft> {
      if (!bridge) return unsupported();
      try { return await bridge.replaceSetup(setup, previous.draftId, previous.revision); }
      catch (error) {
        if (failure(error).kind === "uncertain") {
          try {
            await bridge.expire();
            const current = (await bridge.list()).drafts.find(entry => entry.draftId === previous.draftId);
            if (current && current.revision > previous.revision && !current.take && sameDraftAttempt(current, setup)) return current;
          } catch { /* The state remains uncertain. */ }
        }
        throw failure(error);
      }
    },
    async detachRetry(draft: SavedDraft): Promise<SavedDraft> {
      if (!bridge?.detachRetry) throw new DraftStoreError("unsupported", "This draft store cannot detach an imported retry link");
      if (!draft.setup.retryOfSessionId) return draft;
      try { return await bridge.detachRetry(draft.draftId, draft.revision, draft.setup.retryOfSessionId); }
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
      const mime = takeMime(file);
      if (!safeTakeName(file.name) || !TAKE_MIMES.has(mime)) {
        throw new DraftStoreError("format", "This audio format cannot be saved for local recovery");
      }
      let transfer: string | undefined;
      let digest = "";
      try {
        digest = await sha256File(file);
        transfer = await bridge.beginTake(draft.draftId, draft.revision, {
          originalFilename: file.name, mime, size: file.size, sha256: digest,
          durationSec, captureEndedAt: inputMethod === "record" ? file.lastModified : Date.now(), inputMethod,
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
        if (failure(error).kind === "uncertain") {
          try {
            await bridge.expire();
            const current = (await bridge.list()).drafts.find(entry => entry.draftId === draft.draftId);
            if (current && current.revision > draft.revision && current.take?.sha256 === digest &&
                current.take.size === file.size) return current;
          } catch { /* The take may have published; do not claim it is durable. */ }
        }
        throw failure(error);
      }
    },
    async loadTake(draft: SavedDraft): Promise<File> {
      if (!bridge) return unsupported();
      if (!validSavedDraft(draft) || !draft.take) throw new DraftStoreError("unreadable", "Saved take metadata is invalid");
      try {
        const chunks: Uint8Array<ArrayBuffer>[] = [];
        const hasher = new Sha256Stream();
        let size = 0;
        for (let index = 0; size < draft.take.size; index++) {
          const bytes = new Uint8Array(await bridge.loadTakeChunk(draft.draftId, draft.take.mediaId, index));
          if (!bytes.length || bytes.length > CHUNK_BYTES || size + bytes.length > draft.take.size) throw new Error("Saved take has an invalid chunk");
          chunks.push(bytes);
          hasher.update(bytes);
          size += bytes.length;
        }
        const file = new File(chunks, draft.take.originalFilename, { type: draft.take.mime, lastModified: draft.take.captureEndedAt });
        if (hasher.hex() !== draft.take.sha256) throw new Error("Saved take digest differs from metadata");
        return file;
      } catch (error) { throw failure(error); }
    },
    async markSubmission(draft: SavedDraft, status: SavedDraft["status"], submission: DraftSubmission): Promise<SavedDraft> {
      if (!bridge) return unsupported();
      try { return await bridge.markSubmission(draft.draftId, draft.revision, status, submission); }
      catch (error) {
        if (failure(error).kind === "uncertain") {
          try {
            const current = (await bridge.list()).drafts.find(entry => entry.draftId === draft.draftId);
            if (current && current.revision > draft.revision && current.status === status &&
                current.submission?.uploadRequestId === submission.uploadRequestId &&
                current.submission?.assessmentRequestId === submission.assessmentRequestId) return current;
          } catch { /* Keep the receipt uncertain. */ }
        }
        throw failure(error);
      }
    },
    async discard(id: string): Promise<void> {
      if (!bridge) return unsupported();
      try { await bridge.discard(id); } catch (error) { throw failure(error); }
    },
  };
};

export type DraftStoreAdapter = ReturnType<typeof createDraftStoreAdapter>;

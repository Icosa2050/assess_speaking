import { vi } from "vitest";
import type { DraftSubmission, SavedDraft } from "@/lib/runtime/desktopBridge";
import type { DraftStoreAdapter } from "@/lib/state/draftStore";
import { buildSessionDraft, type SessionDraft } from "@/lib/state/sessionDraft";

let nextId = 1;
const id = () => (nextId++).toString(16).padStart(32, "0");

export const savedDraftFixture = (speakerId = "Alex", sessionId = "saved-attempt"): SavedDraft => {
  const now = Date.now();
  return {
    version: 1, draftId: id(), speakerId: speakerId.toLowerCase(),
    setup: buildSessionDraft({ sessionId, speakerId, themeId: "trip", themeLabel: "Describe a trip", promptText: "Describe a trip." }),
    createdAt: now, updatedAt: now, expiresAt: now + 7 * 86400000, revision: 2, status: "take_ready",
    take: { mediaId: id(), originalFilename: "saved.wav", mime: "audio/wav", size: 5,
      sha256: "a".repeat(64), durationSec: 40, captureEndedAt: now, inputMethod: "upload" }, submission: null,
  };
};

export const draftAdapterFixture = (entries: SavedDraft[] = []) => {
  const adapter = {
    supported: true,
    list: vi.fn(async () => ({ drafts: entries, unreadableIds: [] as string[] })),
    recoveryIds: vi.fn(async () => entries.map(entry => entry.draftId)),
    abortAbandoned: vi.fn(async () => undefined),
    saveSetup: vi.fn(async (setup: SessionDraft, previous?: SavedDraft) => ({
      ...(previous || { version: 1, draftId: id(), speakerId: setup.speakerId.toLowerCase(), createdAt: Date.now(),
        take: null, submission: null, status: "setup" }),
      setup, updatedAt: Date.now(), expiresAt: Date.now() + 7 * 86400000,
      revision: (previous?.revision || 0) + 1,
    } as SavedDraft)),
    replaceSetup: vi.fn(async (setup: SessionDraft, previous: SavedDraft) => ({
      ...previous, setup, createdAt: Date.now(), updatedAt: Date.now(), expiresAt: Date.now() + 7 * 86400000,
      revision: previous.revision + 1, status: "setup" as const, take: null, submission: null,
    })),
    importLegacyBrief: vi.fn(async (setup: SessionDraft) => ({ ...savedDraftFixture(setup.speakerId, setup.sessionId), setup, take: null, status: "setup" } as SavedDraft)),
    saveTake: vi.fn(async (draft: SavedDraft, file: File) => ({ ...draft, revision: draft.revision + 1, status: "take_ready" as const,
      take: { ...savedDraftFixture().take!, originalFilename: file.name, size: file.size } })),
    loadTake: vi.fn(async () => new File(["audio"], "saved.wav", { type: "audio/wav" })),
    markSubmission: vi.fn(async (draft: SavedDraft, status: SavedDraft["status"], submission: DraftSubmission) => ({
      ...draft, status, submission, revision: draft.revision + 1,
    })),
    discard: vi.fn(async () => undefined),
  } satisfies DraftStoreAdapter;
  return adapter;
};

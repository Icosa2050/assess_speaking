import "@testing-library/jest-dom/vitest";
import { createHash, webcrypto } from "node:crypto";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, useLocation } from "react-router-dom";
import type { DesktopDraftStoreBridge, SavedDraft } from "@/lib/runtime/desktopBridge";
import { AppStoreProvider, useAppStore } from "./appStore";
import { DraftRecoveryProvider, useDraftRecovery } from "./DraftRecoveryProvider";
import { createDraftStoreAdapter } from "./draftStore";
import { buildSessionDraft } from "./sessionDraft";
import { persistRetryDraft, readLegacyRetryBrief, RETRY_DRAFT_STORAGE_KEY } from "./retryBriefStorage";

const bytes = new TextEncoder().encode("finalized browser audio");
const digest = createHash("sha256").update(bytes).digest("hex");
const draft = (): SavedDraft => ({
  version: 1, draftId: "a".repeat(32), speakerId: "alex",
  setup: buildSessionDraft({ speakerId: "Alex", themeId: "trip", themeLabel: "Describe a trip", promptText: "Tell us about a trip." }),
  createdAt: Date.now() - 1000, updatedAt: Date.now(), expiresAt: Date.now() + 86400000,
  revision: 2, status: "take_ready",
  take: { mediaId: "b".repeat(32), originalFilename: "take.webm", mime: "audio/webm", size: bytes.length,
    sha256: digest, durationSec: 8, captureEndedAt: Date.now(), inputMethod: "record" }, submission: null,
});
const fake = (entries = [draft()]): DesktopDraftStoreBridge => ({
  list: vi.fn().mockResolvedValue({ drafts: entries, unreadableIds: [] }),
  recoveryIds: vi.fn().mockResolvedValue([]), saveSetup: vi.fn(), importLegacyBrief: vi.fn(),
  beginTake: vi.fn(), writeTakeChunk: vi.fn(), abortTake: vi.fn(),
  abortTakeForDraft: vi.fn().mockResolvedValue(undefined), finishTake: vi.fn(),
  loadTakeChunk: vi.fn().mockResolvedValue(Array.from(bytes)), markSubmission: vi.fn(),
  discard: vi.fn().mockResolvedValue(undefined), expire: vi.fn(),
});
const Probe = () => {
  const speaker = useAppStore(state => state.draft.speakerId);
  const recovery = useDraftRecovery();
  const location = useLocation();
  return <div data-testid="probe">{speaker}|{recovery.recoveredFile?.name}|{location.pathname}</div>;
};
const show = (bridge: DesktopDraftStoreBridge) => render(
  <MemoryRouter initialEntries={["/speak"]}>
    <AppStoreProvider>
      <DraftRecoveryProvider adapter={createDraftStoreAdapter(bridge)}><Probe /></DraftRecoveryProvider>
    </AppStoreProvider>
  </MemoryRouter>,
);

describe("desktop draft recovery gate", () => {
  beforeEach(() => { vi.stubGlobal("crypto", webcrypto); window.localStorage.clear(); });
  afterEach(() => vi.unstubAllGlobals());

  it("waits for listing, resumes the verified file, and never sends it", async () => {
    const bridge = fake();
    let resolveList: (value: { drafts: SavedDraft[]; unreadableIds: string[] }) => void = () => undefined;
    vi.mocked(bridge.list).mockImplementation(() => new Promise(resolve => { resolveList = resolve; }));
    show(bridge);
    expect(screen.queryByTestId("probe")).not.toBeInTheDocument();
    resolveList({ drafts: [draft()], unreadableIds: [] });
    expect(await screen.findByTestId("draft.entry")).toHaveTextContent("Alex");
    expect(screen.queryByTestId("probe")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm|/speak"));
    expect(bridge.abortTakeForDraft).toHaveBeenCalledWith("a".repeat(32));
    expect(bridge.loadTakeChunk).toHaveBeenCalledWith("a".repeat(32), "b".repeat(32), 0);
    expect(bridge.markSubmission).not.toHaveBeenCalled();
  });

  it("requires an explicit confirmation before deleting the local draft", async () => {
    const bridge = fake();
    show(bridge);
    const discard = await screen.findByRole("button", { name: "Discard draft and take" });
    fireEvent.click(discard);
    expect(bridge.discard).not.toHaveBeenCalled();
    expect(screen.getByText(/Delete the local draft for Alex/)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith("a".repeat(32)));
    expect(await screen.findByTestId("probe")).toBeVisible();
  });

  it("offers unreadable IDs for discard when the Keychain cannot open", async () => {
    const bridge = fake([]);
    vi.mocked(bridge.list).mockRejectedValue(new Error("Mac Keychain is unavailable"));
    vi.mocked(bridge.recoveryIds).mockResolvedValue(["a".repeat(32)]);
    show(bridge);
    expect(await screen.findByTestId("draft.unreadable")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("Mac Keychain is unavailable");
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith("a".repeat(32)));
  });

  it("imports a valid #26 retry brief once with its original expiry", async () => {
    const bridge = fake([]);
    const brief = buildSessionDraft({ ...draft().setup, retryOfSessionId: "old-session", retryGoalEligible: true, coachingGoal: "Speak more clearly" });
    persistRetryDraft(brief);
    const stored = JSON.parse(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)!);
    stored.createdAt -= 3 * 86400000;
    window.localStorage.setItem(RETRY_DRAFT_STORAGE_KEY, JSON.stringify(stored));
    vi.mocked(bridge.importLegacyBrief).mockResolvedValue({ ...draft(), setup: brief, take: null, status: "setup",
      expiresAt: stored.createdAt + 7 * 86400000 });
    show(bridge);
    expect(await screen.findAllByTestId("draft.entry")).toHaveLength(1);
    expect(bridge.importLegacyBrief).toHaveBeenCalledWith(brief, stored.createdAt);
    expect(readLegacyRetryBrief()).toBeNull();
  });

  it("keeps a colliding legacy brief untouched until an explicit choice", async () => {
    const bridge = fake();
    const brief = buildSessionDraft({ ...draft().setup, retryOfSessionId: "old-session", retryGoalEligible: true });
    persistRetryDraft(brief);
    show(bridge);
    expect(await screen.findByTestId("draft.legacy")).toBeVisible();
    expect(bridge.importLegacyBrief).not.toHaveBeenCalled();
    expect(readLegacyRetryBrief()).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Discard older retry brief" }));
    expect(readLegacyRetryBrief()).toBeNull();
    expect(screen.getAllByTestId("draft.entry")).toHaveLength(1);
  });
});

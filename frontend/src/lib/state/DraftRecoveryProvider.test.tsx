import "@testing-library/jest-dom/vitest";
import { createHash, webcrypto } from "node:crypto";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, useLocation } from "react-router-dom";
import type { DesktopDraftStoreBridge, SavedDraft } from "@/lib/runtime/desktopBridge";
import { desktopBridge } from "@/lib/runtime/desktopBridge";
import { AppStoreProvider, useAppStore } from "./appStore";
import { DraftRecoveryProvider, useDraftRecovery } from "./DraftRecoveryProvider";
import { createDraftStoreAdapter, sameDraftAttempt } from "./draftStore";
import { buildSessionDraft } from "./sessionDraft";
import { persistRetryDraft, readLegacyRetryBrief, RETRY_DRAFT_STORAGE_KEY } from "./retryBriefStorage";
import { hasFreshRetryParent, hasRestoredRetryParent, rememberFreshRetryParents, rememberRestoredRetryParents } from "@/lib/rehearsal/draftConflicts";
import { apiClient, ApiClientError } from "@/lib/api/client";

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
const acceptedDraft = (): SavedDraft => {
  const saved = draft();
  saved.status = "assessing";
  saved.submission = { uploadRequestId: "upload-one", audioId: "aud_one", assessmentRequestId: "request-one",
    assessmentId: "assessment-1", requestFingerprint: "a".repeat(64),
    frozenRequestBody: JSON.stringify({ audio_id: "aud_one", request_id: "request-one", sharing_fingerprint: "a".repeat(64) }),
    lastConfirmedPhase: "assessment" };
  return saved;
};
const fake = (entries = [draft()]): DesktopDraftStoreBridge => ({
  list: vi.fn().mockResolvedValue({ drafts: entries, unreadableIds: [] }),
  recoveryIds: vi.fn().mockResolvedValue([]), saveSetup: vi.fn(), replaceSetup: vi.fn(), importLegacyBrief: vi.fn(),
  detachRetry: vi.fn(),
  beginTake: vi.fn(), writeTakeChunk: vi.fn(), abortTake: vi.fn(),
  abortTakeForDraft: vi.fn().mockResolvedValue(undefined), finishTake: vi.fn(),
  loadTakeChunk: vi.fn().mockResolvedValue(Array.from(bytes)), markSubmission: vi.fn(),
  discard: vi.fn().mockResolvedValue(undefined), expire: vi.fn(),
});
const Probe = () => {
  const speaker = useAppStore(state => state.draft.speakerId);
  const retry = useAppStore(state => `${state.draft.retryOfSessionId}|${state.draft.coachingGoal}|${state.draft.retryGoalEligible}`);
  const setRecordingJob = useAppStore(state => state.setRecordingJob);
  const updateReview = useAppStore(state => state.updateReview);
  const updateRecording = useAppStore(state => state.updateRecording);
  const assessmentState = useAppStore(state => state.recording.assessmentState);
  const currentDraft = useAppStore(state => state.draft);
  const recovery = useDraftRecovery();
  const location = useLocation();
  return <><div data-testid="probe">{speaker}|{recovery.recoveredFile?.name}|{location.pathname}</div><div data-testid="active-revision">{recovery.active?.revision}</div><div data-testid="retry-context">{retry}</div>
    <button onClick={() => void recovery.discardActive()}>Discard active probe</button>
    <button onClick={() => recovery.leaveActive()}>Leave active probe</button>
    <button onClick={() => void recovery.saveSetup(buildSessionDraft({ speakerId: "Bea", themeId: "trip", themeLabel: "Trip", promptText: "Trip" })).catch(() => undefined)}>Save setup probe</button>
    <button onClick={() => void recovery.saveTake(draft().setup, new File([bytes], "take.webm", { type: "audio/webm" }), 8, "upload", () => undefined).catch(() => undefined)}>Save take probe</button>
    <button onClick={recovery.manageDrafts}>Manage drafts probe</button>
    <button onClick={() => recovery.active && void recovery.saveSetup({ ...recovery.active.setup, coachingGoal: "Updated goal" }).catch(() => undefined)}>Edit active probe</button>
    <button onClick={() => recovery.active && void recovery.markSubmission(recovery.active, "upload_uncertain", {
      uploadRequestId: "upload-A", audioId: "", assessmentRequestId: "", assessmentId: "",
      requestFingerprint: "", frozenRequestBody: "", lastConfirmedPhase: "none",
    }).catch(() => undefined)}>Mark active receipt probe</button>
    <button onClick={() => void recovery.prepareNewAttempt("Alex")}>Prepare replacement probe</button>
    <button onClick={() => void recovery.reconcileActiveRetryParent()}>Reconcile retry probe</button>
    <button onClick={() => recovery.active && void recovery.saveSetup({ ...recovery.active.setup, sessionId: "next-attempt" }).catch(() => undefined)}>Save replacement probe</button>
    <button onClick={() => { setRecordingJob({ assessmentId: "assessment-1", status: "completed" }); updateReview({ reportId: "report-1" }); }}>Complete off Speak probe</button>
    <button onClick={() => updateRecording({ assessmentState: "queued" })}>Queue assessment probe</button>
    <button onClick={() => recovery.active && void recovery.replaceSetup({ ...recovery.active.setup, sessionId: "next-attempt" }, recovery.active).catch(() => undefined)}>Direct replace probe</button>
    <div data-testid="draft-error-kind">{recovery.error?.kind}</div>
    <div data-testid="draft-supported">{String(recovery.supported)}</div>
    <div data-testid="probe-assessment">{assessmentState}|{String(sameDraftAttempt(recovery.active, currentDraft))}</div></>;
};
const show = (bridge: DesktopDraftStoreBridge) => render(
  <MemoryRouter initialEntries={["/speak"]}>
    <AppStoreProvider>
      <DraftRecoveryProvider adapter={createDraftStoreAdapter(bridge)}><Probe /></DraftRecoveryProvider>
    </AppStoreProvider>
  </MemoryRouter>,
);

describe("desktop draft recovery gate", () => {
  beforeEach(() => { vi.stubGlobal("crypto", webcrypto); window.localStorage.clear();
    vi.spyOn(apiClient, "getHistoryDetail").mockResolvedValue({ payload: {} }); });
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  it("detaches an imported retry parent before loading the saved take", async () => {
    const ready = draft();
    ready.setup = { ...ready.setup, retryOfSessionId: "imported-parent", coachingGoal: "Finish clearly", retryGoalEligible: true };
    rememberRestoredRetryParents([{ draftId: ready.draftId, parentId: "imported-parent" }]);
    vi.mocked(apiClient.getHistoryDetail).mockResolvedValue({ payload: { restored_from: {} } });
    const bridge = fake([ready]);
    const detached = { ...ready, revision: ready.revision + 1, setup: { ...ready.setup, retryOfSessionId: "", retryGoalEligible: false } };
    vi.mocked(bridge.detachRetry!).mockResolvedValue(detached);
    show(bridge);
    expect(await screen.findByText(/refers to an attempt imported from a backup/)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(bridge.detachRetry).toHaveBeenCalledWith(ready.draftId, ready.revision, "imported-parent"));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    expect(screen.getByTestId("retry-context")).toHaveTextContent("|Finish clearly|false");
    expect(hasRestoredRetryParent(ready.draftId, "imported-parent")).toBe(false);
  });

  it.each(["assessing", "assessment_uncertain"] as const)("resumes a %s receipt with its frozen retry link when History says 404", async status => {
    const saved = acceptedDraft();
    saved.status = status;
    saved.setup = { ...saved.setup, retryOfSessionId: "prior", coachingGoal: "Finish clearly", retryGoalEligible: true };
    rememberFreshRetryParents([{ draftId: saved.draftId, parentId: "prior" }]);
    vi.spyOn(apiClient, "getAssessmentByRequest").mockRejectedValue(new ApiClientError(404, { code: "validation_error", detail: "Not found" }));
    const history = vi.spyOn(apiClient, "getHistoryDetail").mockRejectedValue(new ApiClientError(404, { code: "validation_error", detail: "Not found" }));
    const bridge = fake([saved]);
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    expect(screen.getByTestId("retry-context")).toHaveTextContent("prior|Finish clearly|true");
    expect(history).not.toHaveBeenCalled();
    expect(bridge.detachRetry).not.toHaveBeenCalled();
    expect(hasFreshRetryParent(saved.draftId, "prior")).toBe(false);
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Reconcile retry probe" })); });
    expect(bridge.detachRetry).not.toHaveBeenCalled();
    expect(history).not.toHaveBeenCalled();
  });

  it("uses report provenance after WebView storage is cleared, but keeps a later retry of that report", async () => {
    const old = draft();
    old.setup = { ...old.setup, retryOfSessionId: "imported-parent", coachingGoal: "Finish clearly", retryGoalEligible: true };
    const restoredAt = old.createdAt + 500;
    vi.mocked(apiClient.getHistoryDetail).mockResolvedValue({ payload: { restored_from: { restored_at_ms: restoredAt } } });
    rememberRestoredRetryParents([{ draftId: old.draftId, parentId: "imported-parent" }]);
    localStorage.clear();
    const bridge = fake([old]);
    vi.mocked(bridge.detachRetry!).mockResolvedValue({ ...old, revision: old.revision + 1,
      setup: { ...old.setup, retryOfSessionId: "", retryGoalEligible: false } });
    const first = show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(bridge.detachRetry).toHaveBeenCalledOnce());
    expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm");
    first.unmount();

    const later = { ...draft(), draftId: "c".repeat(32), createdAt: restoredAt + 1000,
      setup: { ...old.setup, sessionId: "later-attempt" } };
    const laterBridge = fake([later]);
    show(laterBridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    expect(laterBridge.detachRetry).not.toHaveBeenCalled();
    expect(screen.getByTestId("retry-context")).toHaveTextContent("imported-parent|Finish clearly|true");
  });

  it.each([
    ["HTTP 503", new ApiClientError(503, { code: "runtime_error", detail: "Starting" })],
    ["network failure", new Error("Network unavailable")],
  ])("opens the intact take after %s and can retry the history check", async (_label, failure) => {
    const ready = draft();
    ready.setup = { ...ready.setup, retryOfSessionId: "saved-parent", coachingGoal: "Finish clearly", retryGoalEligible: true };
    rememberFreshRetryParents([{ draftId: ready.draftId, parentId: "saved-parent" }]);
    vi.mocked(apiClient.getHistoryDetail).mockRejectedValueOnce(failure)
      .mockResolvedValue({ payload: {} });
    const bridge = fake([ready]);
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    expect(screen.getByRole("status")).toHaveTextContent("earlier attempt could not be checked");
    expect(bridge.loadTakeChunk).toHaveBeenCalled();
    expect(hasFreshRetryParent(ready.draftId, "saved-parent")).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Retry history check" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Retry history check" })).not.toBeInTheDocument());
    expect(hasFreshRetryParent(ready.draftId, "saved-parent")).toBe(false);
  });

  it("checks a retry parent after fresh start and detaches only when history is absent", async () => {
    const ready = draft();
    ready.setup = { ...ready.setup, retryOfSessionId: "quarantined-parent", coachingGoal: "Finish clearly", retryGoalEligible: true };
    rememberFreshRetryParents([{ draftId: ready.draftId, parentId: "quarantined-parent" }]);
    const lookup = vi.spyOn(apiClient, "getHistoryDetail").mockRejectedValue(new ApiClientError(404, { code: "runtime_error", detail: "Missing" }));
    try {
      const bridge = fake([ready]);
      vi.mocked(bridge.detachRetry!).mockResolvedValue({ ...ready, revision: ready.revision + 1,
        setup: { ...ready.setup, retryOfSessionId: "", retryGoalEligible: false } });
      show(bridge);
      fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
      await waitFor(() => expect(lookup).toHaveBeenCalledWith("quarantined-parent"));
      await waitFor(() => expect(bridge.detachRetry).toHaveBeenCalledOnce());
      expect(screen.getByTestId("retry-context")).toHaveTextContent("|Finish clearly|false");
      expect(hasFreshRetryParent(ready.draftId, "quarantined-parent")).toBe(false);
    } finally { lookup.mockRestore(); }
  });

  it("keeps a valid retry parent after fresh start", async () => {
    const ready = draft();
    ready.setup = { ...ready.setup, retryOfSessionId: "saved-parent", coachingGoal: "Finish clearly", retryGoalEligible: true };
    rememberFreshRetryParents([{ draftId: ready.draftId, parentId: "saved-parent" }]);
    const lookup = vi.spyOn(apiClient, "getHistoryDetail").mockResolvedValue({ payload: {} });
    try {
      const bridge = fake([ready]);
      show(bridge);
      fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
      await waitFor(() => expect(lookup).toHaveBeenCalledWith("saved-parent"));
      await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
      expect(bridge.detachRetry).not.toHaveBeenCalled();
      expect(screen.getByTestId("retry-context")).toHaveTextContent("saved-parent|Finish clearly|true");
      expect(hasFreshRetryParent(ready.draftId, "saved-parent")).toBe(false);
    } finally { lookup.mockRestore(); }
  });

  it("updates the active setup after fresh start without dropping its take", async () => {
    const ready = draft();
    ready.setup = { ...ready.setup, retryOfSessionId: "old-parent", coachingGoal: "Finish clearly", retryGoalEligible: true };
    const bridge = fake([ready]);
    vi.mocked(bridge.detachRetry!).mockResolvedValue({ ...ready, revision: ready.revision + 1,
      setup: { ...ready.setup, retryOfSessionId: "", retryGoalEligible: false } });
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    rememberFreshRetryParents([{ draftId: ready.draftId, parentId: "old-parent" }]);
    const lookup = vi.spyOn(apiClient, "getHistoryDetail").mockRejectedValue(new ApiClientError(404, { code: "runtime_error", detail: "Missing" }));
    try {
      fireEvent.click(screen.getByRole("button", { name: "Reconcile retry probe" }));
      await waitFor(() => expect(bridge.detachRetry).toHaveBeenCalledOnce());
      expect(screen.getByTestId("retry-context")).toHaveTextContent("|Finish clearly|false");
      expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm");
    } finally { lookup.mockRestore(); }
  });

  it("clears a matching take and retry brief after a completed receipt resolves on boot", async () => {
    const saved = draft();
    saved.setup = { ...saved.setup, retryOfSessionId: "prior", coachingGoal: "Finish clearly", retryGoalEligible: true };
    saved.status = "assessment_uncertain";
    saved.submission = { uploadRequestId: "upload-one", audioId: "aud_one", assessmentRequestId: "assessment-one",
      assessmentId: "", requestFingerprint: "a".repeat(64),
      frozenRequestBody: JSON.stringify({ audio_id: "aud_one", request_id: "assessment-one", sharing_fingerprint: "a".repeat(64) }),
      lastConfirmedPhase: "upload" };
    let resolveLookup: (value: {assessment_id: string; status: "completed"; report_id: string}) => void = () => undefined;
    vi.spyOn(apiClient, "getAssessmentByRequest").mockImplementation(() => new Promise(resolve => { resolveLookup = resolve; }));
    vi.spyOn(apiClient, "getHistoryDetail").mockResolvedValue({ payload: { report: { session_id: "report-one" } } });
    const bridge = fake([saved]);
    show(bridge);
    await screen.findByTestId("draft.entry");
    await waitFor(() => expect(apiClient.getAssessmentByRequest).toHaveBeenCalledWith("assessment-one", {
      audioId: "aud_one", speakerId: saved.setup.speakerId, sharingFingerprint: "a".repeat(64),
    }));
    persistRetryDraft(saved.setup);
    resolveLookup({ assessment_id: "asmt_one", status: "completed", report_id: "report-one" });
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith(saved.draftId));
    await waitFor(() => expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull());
    expect(screen.queryByTestId("draft.entry")).not.toBeInTheDocument();
  });

  it("keeps the only take when a completed job has no readable history report", async () => {
    const saved = draft();
    saved.status = "assessment_uncertain";
    saved.submission = { uploadRequestId: "upload-one", audioId: "aud_one", assessmentRequestId: "assessment-one",
      assessmentId: "asmt_one", requestFingerprint: "a".repeat(64),
      frozenRequestBody: JSON.stringify({ audio_id: "aud_one", request_id: "assessment-one", sharing_fingerprint: "a".repeat(64) }),
      lastConfirmedPhase: "assessment" };
    vi.spyOn(apiClient, "getAssessmentByRequest").mockResolvedValue({ assessment_id: "asmt_one", status: "completed", report_id: "missing" });
    vi.spyOn(apiClient, "getHistoryDetail").mockRejectedValue(new Error("history missing"));
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const bridge = fake([saved]);
    show(bridge);
    await waitFor(() => expect(apiClient.getHistoryDetail).toHaveBeenCalledWith("missing"));
    expect(bridge.discard).not.toHaveBeenCalled();
    expect(await screen.findByTestId("draft.entry")).toHaveTextContent("Alex");
  });

  it("resumes the latest revision when an upload lookup finishes after the click", async () => {
    const saved = draft();
    saved.status = "upload_uncertain";
    saved.submission = { uploadRequestId: "upload-one", audioId: "", assessmentRequestId: "", assessmentId: "",
      requestFingerprint: "", frozenRequestBody: "", lastConfirmedPhase: "none" };
    let resolveLookup: (value: { audio_id: string }) => void = () => undefined;
    vi.spyOn(apiClient, "getUploadByRequest").mockImplementation(() => new Promise(resolve => { resolveLookup = resolve; }));
    const bridge = fake([saved]);
    vi.mocked(bridge.markSubmission).mockImplementation(async (_id, revision, status, submission) =>
      ({ ...saved, revision: revision + 1, status: status as SavedDraft["status"], submission }));
    show(bridge);
    await screen.findByTestId("draft.entry");
    await waitFor(() => expect(apiClient.getUploadByRequest).toHaveBeenCalledWith("upload-one", {
      sha256: saved.take!.sha256, originalName: saved.take!.originalFilename,
    }));
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await act(async () => resolveLookup({ audio_id: "aud_one" }));
    await waitFor(() => expect(screen.getByTestId("active-revision")).toHaveTextContent("3"));
    expect(bridge.markSubmission).toHaveBeenCalledWith(saved.draftId, 2, "uploaded", expect.objectContaining({ audioId: "aud_one" }));
  });

  it("writes an in-flight A receipt to A's revision without touching newly activated B", async () => {
    const first = draft();
    const second = { ...draft(), draftId: "c".repeat(32), speakerId: "bea",
      setup: buildSessionDraft({ speakerId: "Bea", themeId: "trip", themeLabel: "Describe a trip", promptText: "Tell us about a trip." }) };
    const bridge = fake([first, second]);
    let finishMark: (value: SavedDraft) => void = () => undefined;
    vi.mocked(bridge.markSubmission).mockImplementation(() => new Promise(resolve => { finishMark = resolve; }));
    show(bridge);
    const firstEntry = (await screen.findAllByTestId("draft.entry")).find(entry => entry.textContent?.includes("Alex"))!;
    fireEvent.click(within(firstEntry).getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    fireEvent.click(screen.getByRole("button", { name: "Mark active receipt probe" }));
    await waitFor(() => expect(bridge.markSubmission).toHaveBeenCalledWith(first.draftId, first.revision,
      "upload_uncertain", expect.objectContaining({ uploadRequestId: "upload-A" })));
    fireEvent.click(screen.getByRole("button", { name: "Leave active probe" }));
    fireEvent.click(screen.getByRole("button", { name: "Manage drafts probe" }));
    const secondEntry = (await screen.findAllByTestId("draft.entry")).find(entry => entry.textContent?.includes("Bea"))!;
    fireEvent.click(within(secondEntry).getByRole("button", { name: "Resume" }));
    expect(bridge.markSubmission).toHaveBeenCalledTimes(1);
    await act(async () => finishMark({ ...first, revision: first.revision + 1, status: "upload_uncertain",
      submission: { uploadRequestId: "upload-A", audioId: "", assessmentRequestId: "", assessmentId: "",
        requestFingerprint: "", frozenRequestBody: "", lastConfirmedPhase: "none" } }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Bea|take.webm"));
    expect(bridge.markSubmission).toHaveBeenCalledTimes(1);
    expect(second.submission).toBeNull();
    expect(screen.getByTestId("active-revision")).toHaveTextContent(String(second.revision));
  });

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

  it("requires a new confirmation after a replacement setup save fails", async () => {
    const bridge = fake();
    const prior = draft();
    vi.mocked(bridge.replaceSetup!).mockRejectedValueOnce(new Error("disk unavailable"))
      .mockResolvedValue({ ...prior, setup: { ...prior.setup, sessionId: "next-attempt" }, revision: 3,
        take: null, status: "setup" });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Prepare replacement probe" })); });
    expect(confirm).toHaveBeenCalledOnce();
    expect(confirm.mock.calls[0][0]).toContain("Describe a trip");
    fireEvent.click(screen.getByRole("button", { name: "Save replacement probe" }));
    await waitFor(() => expect(bridge.replaceSetup).toHaveBeenCalledOnce());
    await waitFor(() => expect(screen.getByTestId("draft-error-kind")).toHaveTextContent("storage"));
    fireEvent.click(screen.getByRole("button", { name: "Save replacement probe" }));
    await act(async () => { await Promise.resolve(); });
    expect(bridge.replaceSetup).toHaveBeenCalledOnce();
    expect(bridge.discard).not.toHaveBeenCalled();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Prepare replacement probe" })); });
    fireEvent.click(screen.getByRole("button", { name: "Save replacement probe" }));
    await waitFor(() => expect(bridge.replaceSetup).toHaveBeenCalledTimes(2));
    confirm.mockRestore();
  });

  it("cleans up a completed report even when Speak is not mounted", async () => {
    const desktopStore = vi.spyOn(desktopBridge, "hasDraftStore").mockReturnValue(true);
    try {
      const ready = acceptedDraft();
      ready.setup = { ...ready.setup, retryOfSessionId: "prior", coachingGoal: "Finish clearly", retryGoalEligible: true };
      vi.spyOn(apiClient, "getAssessmentByRequest")
        .mockResolvedValueOnce({ assessment_id: "assessment-1", status: "queued", report_id: null })
        .mockResolvedValue({ assessment_id: "assessment-1", status: "completed", report_id: "report-1" });
      vi.spyOn(apiClient, "getHistoryDetail").mockResolvedValue({ payload: { report: { session_id: "report-1" } } });
      const bridge = fake([ready]);
      show(bridge);
      fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
      await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
      persistRetryDraft(ready.setup);
      fireEvent.click(screen.getByRole("button", { name: "Complete off Speak probe" }));
      await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith(ready.draftId));
      expect(apiClient.getHistoryDetail).toHaveBeenCalledWith("report-1");
      await waitFor(() => expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull());
      expect(screen.queryByTestId("draft.entry")).not.toBeInTheDocument();
    } finally { desktopStore.mockRestore(); }
  });

  it("retries failed completion cleanup within the same provider mount", async () => {
    const desktopStore = vi.spyOn(desktopBridge, "hasDraftStore").mockReturnValue(true);
    try {
      const ready = acceptedDraft();
      ready.setup = { ...ready.setup, retryOfSessionId: "prior", coachingGoal: "Finish clearly", retryGoalEligible: true };
      vi.spyOn(apiClient, "getAssessmentByRequest")
        .mockResolvedValueOnce({ assessment_id: "assessment-1", status: "queued", report_id: null })
        .mockResolvedValue({ assessment_id: "assessment-1", status: "completed", report_id: "report-1" });
      vi.spyOn(apiClient, "getHistoryDetail").mockResolvedValue({ payload: { report: { session_id: "report-1" } } });
      const bridge = fake([ready]);
      vi.mocked(bridge.discard).mockRejectedValueOnce(new Error("Mac storage is temporarily unavailable"));
      show(bridge);
      fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
      await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
      persistRetryDraft(ready.setup);
      fireEvent.click(screen.getByRole("button", { name: "Complete off Speak probe" }));
      await waitFor(() => expect(screen.getByTestId("draft-error-kind")).toHaveTextContent("storage"));
      expect(bridge.discard).toHaveBeenCalledTimes(1);
      expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).not.toBeNull();
      fireEvent.click(screen.getByRole("button", { name: "Manage drafts probe" }));
      fireEvent.click(screen.getByRole("button", { name: "Try loading local drafts again" }));
      await waitFor(() => expect(bridge.discard).toHaveBeenCalledTimes(2));
      await waitFor(() => expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull());
      expect(screen.getByTestId("draft-error-kind")).toBeEmptyDOMElement();
    } finally { desktopStore.mockRestore(); }
  });

  it("keeps the only take when foreground completion has a mismatched report", async () => {
    const desktopStore = vi.spyOn(desktopBridge, "hasDraftStore").mockReturnValue(true);
    try {
    const ready = acceptedDraft();
    ready.setup = { ...ready.setup, retryOfSessionId: "prior", coachingGoal: "Finish clearly", retryGoalEligible: true };
    vi.spyOn(apiClient, "getAssessmentByRequest")
      .mockResolvedValueOnce({ assessment_id: "assessment-1", status: "queued", report_id: null })
      .mockResolvedValue({ assessment_id: "assessment-1", status: "completed", report_id: "other-report" });
    const history = vi.spyOn(apiClient, "getHistoryDetail").mockResolvedValue({ payload: { report: { session_id: "other-report" } } });
    const bridge = fake([ready]);
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    persistRetryDraft(ready.setup);
    fireEvent.click(screen.getByRole("button", { name: "Complete off Speak probe" }));
    await waitFor(() => expect(apiClient.getAssessmentByRequest).toHaveBeenCalledTimes(2));
    expect(history).not.toHaveBeenCalled();
    expect(bridge.discard).not.toHaveBeenCalled();
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).not.toBeNull();
    expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm");
    } finally { desktopStore.mockRestore(); }
  });

  it("keeps the only saved take while its assessment is queued", async () => {
    const bridge = fake();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    try {
      show(bridge);
      fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
      await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
      fireEvent.click(screen.getByRole("button", { name: "Queue assessment probe" }));
      expect(screen.getByTestId("probe-assessment")).toHaveTextContent("queued|true");
      await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Prepare replacement probe" })); });
      fireEvent.click(screen.getByRole("button", { name: "Save replacement probe" }));
      await waitFor(() => expect(screen.getByTestId("draft-error-kind")).toHaveTextContent("conflict"));
      expect(bridge.replaceSetup).not.toHaveBeenCalled();
      fireEvent.click(screen.getByRole("button", { name: "Direct replace probe" }));
      await act(async () => { await Promise.resolve(); });
      expect(bridge.replaceSetup).not.toHaveBeenCalled();
      expect(bridge.discard).not.toHaveBeenCalled();
    } finally { confirm.mockRestore(); }
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

  it("does not re-open the gate when a new setup is saved after discarding the last draft", async () => {
    const bridge = fake();
    vi.mocked(bridge.saveSetup).mockResolvedValue({ ...draft(), draftId: "c".repeat(32), speakerId: "bea",
      setup: buildSessionDraft({ speakerId: "Bea", themeId: "trip", themeLabel: "Trip", promptText: "Trip" }),
      take: null, status: "setup" });
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Discard draft and take" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    fireEvent.click(await screen.findByRole("button", { name: "Save setup probe" }));
    await waitFor(() => expect(bridge.saveSetup).toHaveBeenCalledOnce());
    expect(screen.queryByTestId("draft.recovery")).not.toBeInTheDocument();
  });

  it("finishes an in-flight discard after the active draft is left", async () => {
    const bridge = fake();
    let finishDiscard: () => void = () => undefined;
    vi.mocked(bridge.discard).mockImplementation(() => new Promise(resolve => { finishDiscard = resolve; }));
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm"));
    fireEvent.click(screen.getByRole("button", { name: "Discard active probe" }));
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith("a".repeat(32)));
    fireEvent.click(screen.getByRole("button", { name: "Leave active probe" }));
    await act(async () => finishDiscard());
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex||/speak"));
    expect(screen.queryByTestId("draft.entry")).not.toBeInTheDocument();
  });

  it("serializes discard after an in-flight setup edit", async () => {
    const bridge = fake();
    let finishSave: (value: SavedDraft) => void = () => undefined;
    vi.mocked(bridge.saveSetup).mockImplementation(() => new Promise(resolve => { finishSave = resolve; }));
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await screen.findByTestId("probe");
    fireEvent.click(screen.getByRole("button", { name: "Edit active probe" }));
    await waitFor(() => expect(bridge.saveSetup).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole("button", { name: "Discard active probe" }));
    expect(bridge.discard).not.toHaveBeenCalled();
    await act(async () => finishSave({ ...draft(), revision: 3, setup: { ...draft().setup, coachingGoal: "Updated goal" } }));
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith("a".repeat(32)));
    expect(screen.getByTestId("draft-error-kind")).toBeEmptyDOMElement();
  });

  it("does not offer destructive discard for a temporary Keychain outage and can retry", async () => {
    const bridge = fake([]);
    vi.mocked(bridge.list).mockRejectedValueOnce(new Error("Mac Keychain is unavailable"))
      .mockResolvedValueOnce({ drafts: [draft()], unreadableIds: [] });
    vi.mocked(bridge.recoveryIds).mockResolvedValue(["a".repeat(32)]);
    show(bridge);
    expect(await screen.findByRole("alert")).toHaveTextContent("Mac Keychain is unavailable");
    expect(screen.queryByTestId("draft.unreadable")).not.toBeInTheDocument();
    expect(bridge.recoveryIds).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Try loading local drafts again" }));
    expect(await screen.findByTestId("draft.entry")).toBeVisible();
    expect(bridge.discard).not.toHaveBeenCalled();
  });

  it("shows a stable unavailable state when release Keychain entitlement is missing", async () => {
    const bridge = fake([]);
    vi.mocked(bridge.list).mockRejectedValue(new Error("Mac Keychain is unavailable; local save is disabled (code -34018)"));
    show(bridge);
    expect(await screen.findByRole("alert")).toHaveTextContent("Local drafts are unavailable on this Mac");
    expect(screen.getByTestId("draft-supported")).toHaveTextContent("false");
    expect(screen.getByTestId("probe")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Try loading local drafts again" })).not.toBeInTheDocument();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Save setup probe" })); });
    expect(bridge.saveSetup).not.toHaveBeenCalled();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Save take probe" })); });
    expect(bridge.beginTake).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Leave active probe" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Local drafts are unavailable on this Mac");
    expect(screen.getByTestId("draft-supported")).toHaveTextContent("false");
  });

  it("offers keyless discard of stored drafts when release Keychain entitlement is missing", async () => {
    const bridge = fake([]);
    vi.mocked(bridge.list).mockRejectedValue(new Error("Mac Keychain is unavailable; local save is disabled (code -34018)"));
    vi.mocked(bridge.recoveryIds).mockResolvedValue(["a".repeat(32)]);
    show(bridge);
    expect(await screen.findByTestId("draft.unreadable")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("Local drafts are unavailable on this Mac");
    expect(bridge.recoveryIds).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith("a".repeat(32)));
    expect(screen.getByTestId("draft-supported")).toHaveTextContent("false");
  });

  it("offers keyless discard only when the content key is confirmed lost", async () => {
    const bridge = fake([]);
    vi.mocked(bridge.list).mockRejectedValue(new Error("Saved take cannot be opened on this Mac"));
    vi.mocked(bridge.recoveryIds).mockResolvedValue(["a".repeat(32)]);
    show(bridge);
    expect(await screen.findByTestId("draft.unreadable")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("cannot be opened on this Mac");
    expect(bridge.recoveryIds).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith("a".repeat(32)));
  });

  it("bounds lost-key recovery ID discovery and still opens the gate", async () => {
    const bridge = fake([]);
    vi.mocked(bridge.list).mockRejectedValue(new Error("Saved take cannot be opened on this Mac"));
    vi.mocked(bridge.recoveryIds).mockImplementation(() => new Promise(() => undefined));
    vi.useFakeTimers();
    try {
      show(bridge);
      await act(async () => vi.advanceTimersByTimeAsync(15_000));
      expect(bridge.recoveryIds).toHaveBeenCalledOnce();
      expect(screen.getByRole("alert")).toHaveTextContent("cannot be opened on this Mac");
      expect(screen.getByRole("button", { name: "Continue without resuming" })).toBeVisible();
      expect(screen.queryByTestId("draft.unreadable")).not.toBeInTheDocument();
    } finally { vi.useRealTimers(); }
  });

  it("keeps a readable entry after a transient resume-time Keychain failure", async () => {
    const bridge = fake();
    vi.mocked(bridge.loadTakeChunk).mockRejectedValueOnce(new Error("Mac Keychain is unavailable"))
      .mockResolvedValue(Array.from(bytes));
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Mac Keychain is unavailable");
    expect(screen.getByTestId("draft.entry")).toBeVisible();
    expect(screen.queryByTestId("draft.unreadable")).not.toBeInTheDocument();
    expect(bridge.discard).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex|take.webm|/speak"));
  });

  it("moves a take with failed media authentication to confirmed unreadable discard", async () => {
    const bridge = fake();
    vi.mocked(bridge.loadTakeChunk).mockRejectedValue(new Error("Saved draft failed authentication"));
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    expect(await screen.findByTestId("draft.unreadable")).toBeVisible();
    expect(screen.queryByTestId("draft.entry")).not.toBeInTheDocument();
    expect(bridge.discard).not.toHaveBeenCalled();
  });

  it("offers genuinely unreadable entries for confirmed discard", async () => {
    const bridge = fake([]);
    vi.mocked(bridge.list).mockResolvedValue({ drafts: [], unreadableIds: ["a".repeat(32)] });
    show(bridge);
    expect(await screen.findByTestId("draft.unreadable")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and take" }));
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledWith("a".repeat(32)));
  });

  it("keeps readable drafts visible if abandoned-transfer cleanup fails", async () => {
    const bridge = fake();
    vi.mocked(bridge.abortTakeForDraft).mockRejectedValue(new Error("Could not cancel take save"));
    show(bridge);
    expect(await screen.findByTestId("draft.entry")).toHaveTextContent("Alex");
    expect(screen.getByRole("alert")).toHaveTextContent("Could not use local draft storage");
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    expect(await screen.findByTestId("probe")).toHaveTextContent("Alex|take.webm");
  });

  it("bounds an unresponsive listing and offers a real retry", async () => {
    const bridge = fake();
    vi.mocked(bridge.list).mockImplementationOnce(() => new Promise(() => undefined))
      .mockResolvedValueOnce({ drafts: [draft()], unreadableIds: [] });
    vi.useFakeTimers();
    try {
      show(bridge);
      expect(screen.queryByTestId("probe")).not.toBeInTheDocument();
      await act(async () => vi.advanceTimersByTimeAsync(15_000));
      expect(screen.getByRole("alert")).toHaveTextContent("Could not use local draft storage");
      fireEvent.click(screen.getByRole("button", { name: "Try loading local drafts again" }));
      await act(async () => { await Promise.resolve(); });
      expect(screen.getByTestId("draft.entry")).toBeVisible();
    } finally { vi.useRealTimers(); }
  });

  it("bounds abandoned-transfer recovery without hiding the readable draft", async () => {
    const bridge = fake();
    vi.mocked(bridge.abortTakeForDraft).mockImplementation(() => new Promise(() => undefined));
    vi.useFakeTimers();
    try {
      show(bridge);
      await act(async () => vi.advanceTimersByTimeAsync(15_000));
      expect(screen.getByTestId("draft.entry")).toHaveTextContent("Alex");
      expect(screen.getByRole("alert")).toHaveTextContent("Could not use local draft storage");
      expect(screen.getByRole("button", { name: "Continue without resuming" })).toBeVisible();
    } finally { vi.useRealTimers(); }
  });

  it("bounds legacy import while preserving the original brief", async () => {
    const bridge = fake([]);
    const brief = buildSessionDraft({ ...draft().setup, retryOfSessionId: "old-session", retryGoalEligible: true, coachingGoal: "Finish clearly" });
    persistRetryDraft(brief);
    vi.mocked(bridge.importLegacyBrief).mockImplementation(() => new Promise(() => undefined));
    vi.useFakeTimers();
    try {
      show(bridge);
      await act(async () => vi.advanceTimersByTimeAsync(15_000));
      expect(screen.getByTestId("draft.legacy")).toBeVisible();
      expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).not.toBeNull();
      expect(screen.getByRole("button", { name: "Continue without resuming" })).toBeVisible();
    } finally { vi.useRealTimers(); }
  });

  it("lets a fourth learner resolve the draft cap without reloading", async () => {
    const other = (id: string, speakerId: string): SavedDraft => ({ ...draft(), draftId: id,
      speakerId: speakerId.toLowerCase(), setup: buildSessionDraft({ speakerId, themeId: "trip", themeLabel: "Trip", promptText: "Trip" }) });
    const bridge = fake([draft(), other("c".repeat(32), "Cia"), other("d".repeat(32), "Dee")]);
    vi.mocked(bridge.saveSetup).mockRejectedValueOnce(new Error("Three learner drafts are already saved"))
      .mockResolvedValueOnce({ ...other("e".repeat(32), "Bea"), take: null, status: "setup" });
    show(bridge);
    fireEvent.click(await screen.findByRole("button", { name: "Continue without resuming" }));
    fireEvent.click(screen.getByRole("button", { name: "Save setup probe" }));
    await waitFor(() => expect(screen.getByTestId("draft-error-kind")).toHaveTextContent("limit"));
    fireEvent.click(screen.getByRole("button", { name: "Manage drafts probe" }));
    expect(screen.getAllByTestId("draft.entry")).toHaveLength(3);
    expect(screen.getByRole("alert")).toHaveTextContent("Local draft storage is full");
    expect(screen.queryByRole("button", { name: "Try loading local drafts again" })).not.toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "Discard draft and take" })[0]);
    fireEvent.click(screen.getAllByRole("button", { name: "Discard draft and take" })[0]);
    await waitFor(() => expect(bridge.discard).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole("button", { name: "Continue without resuming" }));
    fireEvent.click(screen.getByRole("button", { name: "Save setup probe" }));
    await waitFor(() => expect(bridge.saveSetup).toHaveBeenCalledTimes(2));
    expect(screen.queryByTestId("draft.recovery")).not.toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("Alex||/speak"));
    expect(screen.getByTestId("retry-context")).toHaveTextContent("old-session|Speak more clearly|true");
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

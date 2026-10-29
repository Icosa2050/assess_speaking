import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { commitLearnerRestore, exportJournalDiagnostics, JournalHttpError, reconcileFreshJournal, journalRecoveryStatus, recoverLearnerJournal, previewLearnerRestore, startFreshJournal } from "./maintenance";
import { clearRehearsalMaintenance, commitRehearsals, rehearsalMaintenance, stageRehearsals } from "./storage";
import { buildSessionDraft } from "@/lib/state/sessionDraft";
import { persistRetryDraft, RETRY_DRAFT_STORAGE_KEY } from "@/lib/state/retryBriefStorage";

vi.mock("./storage", () => ({
  withJournalLock: vi.fn(async action => action({})),
  clearRehearsalMaintenance: vi.fn(),
  rehearsalMaintenance: vi.fn(),
  stageRehearsals: vi.fn(),
  snapshotRehearsals: vi.fn(async () => ({sessions: [], recordings: []})),
  commitRehearsals: vi.fn(),
}));

const fetchMock = vi.fn();
const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), {status});

describe("damaged journal browser coordination", () => {
  afterEach(() => vi.unstubAllGlobals());
  beforeEach(() => { vi.clearAllMocks(); window.localStorage.clear(); vi.mocked(rehearsalMaintenance).mockResolvedValue(undefined); vi.stubGlobal("fetch", fetchMock); });

  it("forgets only the pending browser journal after confirmed fresh start", async () => {
    persistRetryDraft(buildSessionDraft({ speakerId: "learner", themeId: "trip", promptText: "Describe a trip",
      retryOfSessionId: "first", coachingGoal: "Add a closing sentence", retryGoalEligible: true }));
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).not.toBeNull();
    fetchMock.mockResolvedValueOnce(reply({phase: "fresh"}));
    await startFreshJournal("f".repeat(32));
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({confirmed: true, fresh_request: "f".repeat(32)});
    // No rollback flag: already committed sessions and audio must be retained.
    expect(clearRehearsalMaintenance).toHaveBeenCalledExactlyOnceWith({});
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull();
  });

  it("clears a saved retry after a successful learner restore", async () => {
    persistRetryDraft(buildSessionDraft({ speakerId: "learner", themeId: "trip", promptText: "Describe a trip",
      retryOfSessionId: "first", coachingGoal: "Add a closing sentence", retryGoalEligible: true }));
    fetchMock.mockImplementation(async () => reply({phase: "complete"}));
    await commitLearnerRestore({ id: "restore", attempts: 0, skipped_attempts: [], skipped_rehearsals: [], missing: [],
      browser: { sessions: [], recordings: [] } }, new AbortController().signal, vi.fn());
    expect(commitRehearsals).toHaveBeenCalledOnce();
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull();
  });

  it("retries interrupted fresh-start cleanup without rolling back saved sessions", async () => {
    const id = "a".repeat(32);
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id, phase: "committed", sessions: [], recordings: []});
    vi.mocked(clearRehearsalMaintenance).mockRejectedValueOnce(new Error("IndexedDB unavailable"));
    fetchMock.mockResolvedValueOnce(reply({phase: "fresh"}));
    await expect(startFreshJournal("f".repeat(32))).rejects.toThrow("IndexedDB unavailable");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({confirmed: true, browser_transaction: id, fresh_request: "f".repeat(32)});
    fetchMock.mockResolvedValueOnce(reply({transaction: null, completed: [id], recovery_error: null}));
    await recoverLearnerJournal();
    expect(clearRehearsalMaintenance).toHaveBeenNthCalledWith(1, {});
    expect(clearRehearsalMaintenance).toHaveBeenNthCalledWith(2, {});
  });

  it("finishes browser cleanup only for the matching fresh request and browser transaction", async () => {
    const id = "a".repeat(32);
    const requestId = "f".repeat(32);
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id, phase: "committed", sessions: [], recordings: []});
    persistRetryDraft(buildSessionDraft({speakerId: "learner", themeId: "trip", promptText: "Describe a trip",
      retryOfSessionId: "first", coachingGoal: "Add a closing sentence", retryGoalEligible: true}));
    fetchMock.mockResolvedValueOnce(reply({transaction: null, archived: [], completed: [id, requestId], recovery_error: null}));
    expect(await reconcileFreshJournal(requestId, id)).toMatchObject({applied: true, pendingRecovery: false});
    expect(clearRehearsalMaintenance).toHaveBeenCalledExactlyOnceWith({});
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toContain("/journal/status");
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull();
  });

  it("forgets a retry draft even when post-reset browser cleanup remains blocked", async () => {
    const id = "a".repeat(32);
    const requestId = "f".repeat(32);
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id, phase: "committed", sessions: [], recordings: []});
    persistRetryDraft(buildSessionDraft({speakerId: "learner", themeId: "trip", promptText: "Describe a trip",
      retryOfSessionId: "first", coachingGoal: "Add a closing sentence", retryGoalEligible: true}));
    vi.mocked(clearRehearsalMaintenance).mockRejectedValueOnce(new Error("IndexedDB blocked"));
    fetchMock.mockResolvedValueOnce(reply({transaction: null, archived: [], completed: [id, requestId], recovery_error: null}));
    expect(await reconcileFreshJournal(requestId, id)).toMatchObject({applied: true, cleanupError: "IndexedDB blocked"});
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull();
  });

  it("does not clear an unrelated pending restore after a rejected reset", async () => {
    const oldId = "a".repeat(32);
    const otherId = "b".repeat(32);
    const requestId = "f".repeat(32);
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id: otherId, phase: "staged", sessions: [], recordings: []});
    fetchMock.mockResolvedValueOnce(reply({transaction: {id: otherId, phase: "backend_staged"}, archived: [], completed: [], recovery_error: null}));
    expect(await reconcileFreshJournal(requestId, oldId)).toMatchObject({applied: false});
    expect(clearRehearsalMaintenance).not.toHaveBeenCalled();
    expect(rehearsalMaintenance).toHaveBeenCalledOnce();
  });

  it("preserves newer browser recovery even if this reset's receipt exists", async () => {
    const otherId = "b".repeat(32);
    const requestId = "f".repeat(32);
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id: otherId, phase: "staged", sessions: [], recordings: []});
    fetchMock.mockResolvedValueOnce(reply({transaction: {id: otherId, phase: "backend_staged"}, archived: [], completed: [requestId], recovery_error: null}));
    expect(await reconcileFreshJournal(requestId, "a".repeat(32))).toMatchObject({applied: true, pendingRecovery: true});
    expect(clearRehearsalMaintenance).not.toHaveBeenCalled();
  });

  it("shows a newer browser-only recovery as pending without clearing it", async () => {
    const otherId = "b".repeat(32);
    const requestId = "f".repeat(32);
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id: otherId, phase: "committed", sessions: [], recordings: []});
    fetchMock.mockResolvedValueOnce(reply({transaction: null, archived: [], completed: [requestId], recovery_error: null}));
    expect(await reconcileFreshJournal(requestId, "a".repeat(32))).toMatchObject({
      applied: true, pendingRecovery: true, status: {transaction: {id: otherId, phase: "browser_recovery"}},
    });
    expect(clearRehearsalMaintenance).not.toHaveBeenCalled();
  });

  it("retains browser recovery data when another backend transaction excludes fresh start", async () => {
    fetchMock.mockResolvedValueOnce(reply({detail: "Maintenance in progress"}, 409));
    const failure = await startFreshJournal("f".repeat(32)).catch(cause => cause);
    expect(failure).toBeInstanceOf(JournalHttpError);
    expect(failure).toMatchObject({message: "Maintenance in progress", status: 409});
    expect(clearRehearsalMaintenance).not.toHaveBeenCalled();
  });

  it("opens a backup despite an orphaned browser receipt without deleting saved data", async () => {
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id: "orphan", phase: "committed", sessions: [], recordings: []});
    fetchMock.mockResolvedValueOnce(reply({recovery_error: "damaged"}))
      .mockResolvedValueOnce(reply({id: "repair"}))
      .mockResolvedValueOnce(reply({id: "repair", browser: {sessions: [], recordings: []}}));
    await previewLearnerRestore(new File(["zip"], "backup.zip"), new AbortController().signal);
    expect(fetchMock.mock.calls[1][0]).toContain("/begin-restore");
    expect(clearRehearsalMaintenance).toHaveBeenCalledExactlyOnceWith({});
    expect(stageRehearsals).toHaveBeenCalledWith({id: "repair", phase: "prepared", sessions: [], recordings: []}, {});
  });

  it("does not let an orphaned browser receipt hide exits for a damaged purge", async () => {
    vi.stubGlobal("indexedDB", {databases: async () => [{name: "vostavo-rehearsals"}]});
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id: "orphan", phase: "committed", sessions: [], recordings: []});
    fetchMock.mockResolvedValueOnce(reply({transaction: null, recovery_error: "damaged"}));
    expect(await journalRecoveryStatus()).toEqual({transaction: null, recovery_error: "damaged"});
  });

  it("exports diagnostics without sending client state or private-content opt-ins", async () => {
    fetchMock.mockResolvedValueOnce(reply({bundle_id: "bundle_fixture"}));
    expect(await exportJournalDiagnostics()).toEqual({bundle_id: "bundle_fixture"});
    expect(fetchMock.mock.calls[0][0]).toContain("/journal/diagnostics");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({});
    expect(clearRehearsalMaintenance).not.toHaveBeenCalled();
  });
});

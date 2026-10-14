import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { exportJournalDiagnostics, journalRecoveryStatus, recoverLearnerJournal, previewLearnerRestore, startFreshJournal } from "./maintenance";
import { clearRehearsalMaintenance, rehearsalMaintenance, stageRehearsals } from "./storage";

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
  beforeEach(() => { vi.clearAllMocks(); vi.mocked(rehearsalMaintenance).mockResolvedValue(undefined); vi.stubGlobal("fetch", fetchMock); });

  it("forgets only the pending browser journal after confirmed fresh start", async () => {
    fetchMock.mockResolvedValueOnce(reply({phase: "fresh"}));
    await startFreshJournal();
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({confirmed: true});
    // No rollback flag: already committed sessions and audio must be retained.
    expect(clearRehearsalMaintenance).toHaveBeenCalledExactlyOnceWith({});
  });

  it("retries interrupted fresh-start cleanup without rolling back saved sessions", async () => {
    const id = "a".repeat(32);
    vi.mocked(rehearsalMaintenance).mockResolvedValue({id, phase: "committed", sessions: [], recordings: []});
    vi.mocked(clearRehearsalMaintenance).mockRejectedValueOnce(new Error("IndexedDB unavailable"));
    fetchMock.mockResolvedValueOnce(reply({phase: "fresh"}));
    await expect(startFreshJournal()).rejects.toThrow("IndexedDB unavailable");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({confirmed: true, browser_transaction: id});
    fetchMock.mockResolvedValueOnce(reply({transaction: null, completed: [id], recovery_error: null}));
    await recoverLearnerJournal();
    expect(clearRehearsalMaintenance).toHaveBeenNthCalledWith(1, {});
    expect(clearRehearsalMaintenance).toHaveBeenNthCalledWith(2, {});
  });

  it("retains browser recovery data when another backend transaction excludes fresh start", async () => {
    fetchMock.mockResolvedValueOnce(reply({detail: "Maintenance in progress"}, 409));
    await expect(startFreshJournal()).rejects.toThrow("Maintenance in progress");
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

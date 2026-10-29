import "@testing-library/jest-dom/vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTranslator } from "@/lib/i18n";
import { commitLearnerRestore, JournalHttpError, reconcileFreshJournal, journalRecoveryStatus, journalStatus, previewLearnerRestore, recoverLearnerJournal, startFreshJournal, type JournalStatus } from "@/lib/rehearsal/maintenance";
import { createRehearsal } from "@/lib/rehearsal/session";
import { listArchivedRehearsals } from "@/lib/rehearsal/storage";
import { renderWithProviders } from "@/test/renderWithProviders";
import { JournalRecoveryNotice } from "@/components/shell/JournalRecoveryNotice";
import controlStyles from "@/components/ui/controls.module.css";
import { JournalPanel } from "./JournalPanel";

vi.mock("@/lib/rehearsal/maintenance", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/rehearsal/maintenance")>(),
  journalRecoveryStatus: vi.fn(),
  journalStatus: vi.fn(),
  startFreshJournal: vi.fn(),
  reconcileFreshJournal: vi.fn(),
  recoverLearnerJournal: vi.fn(),
  previewLearnerRestore: vi.fn(),
  commitLearnerRestore: vi.fn(),
}));
vi.mock("@/lib/rehearsal/storage", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/rehearsal/storage")>(),
  listArchivedRehearsals: vi.fn(),
}));

const status = vi.mocked(journalRecoveryStatus);
const archived = vi.mocked(listArchivedRehearsals);
const t = createTranslator("en");
const clean = {transaction: null, archived: [], completed: [], recovery_error: null};
const damaged = {...clean, recovery_error: "Recovery files are unavailable. Check retained backups."};

describe("journal recovery visibility", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    status.mockResolvedValue(clean);
    vi.mocked(journalStatus).mockResolvedValue(damaged);
    vi.mocked(reconcileFreshJournal).mockResolvedValue({applied: false, status: damaged, pendingRecovery: false});
    archived.mockResolvedValue([]);
  });

  it("shows a recovery link for a damaged purge without a transaction", async () => {
    status.mockResolvedValue(damaged);
    renderWithProviders(<JournalRecoveryNotice locale="en" />);
    expect(await screen.findByTestId("journal-recovery-notice")).toBeVisible();
    expect(screen.getAllByRole("link").every(link => link.getAttribute("href")?.startsWith("/settings"))).toBe(true);
    status.mockResolvedValue(clean);
    fireEvent(window, new Event("focus"));
    await waitFor(() => expect(screen.queryByTestId("journal-recovery-notice")).not.toBeInTheDocument());
  });

  it("shows the backend repair guidance without ineffective recovery actions", async () => {
    status.mockResolvedValue(damaged);
    renderWithProviders(<JournalPanel locale="en" />);
    expect(await screen.findByRole("alert")).toHaveTextContent(damaged.recovery_error);
    expect(screen.queryByRole("button", {name: t("journal.recover")})).not.toBeInTheDocument();
    expect(screen.getByRole("button", {name: t("journal.backup")})).toBeDisabled();
  });

  it("enables and names all three damaged-journal exits", async () => {
    status.mockResolvedValue({...damaged, transaction: {id: "damaged", phase: "recovery_error"}});
    const { store } = renderWithProviders(<><JournalRecoveryNotice locale="en" /><JournalPanel locale="en" /></>, { appState: {
      draft: { speakerId: "learner", themeId: "trip", promptText: "Describe a trip",
        retryOfSessionId: "first", coachingGoal: "Add a closing sentence", retryGoalEligible: true },
    } });
    await screen.findByRole("alert");
    expect(screen.getByLabelText("Restore from a backup")).toBeEnabled();
    expect(screen.getByRole("button", {name: "Export diagnostics only"})).toBeEnabled();
    expect(screen.getByRole("button", {name: "Start a fresh journal"})).toBeEnabled();
    for (const name of ["Restore from a backup", "Export diagnostics only", "Start a fresh journal"]) {
      expect(screen.getByRole("link", {name})).toBeVisible();
    }
    expect(screen.queryByRole("button", {name: t("journal.recover")})).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {name: "Start a fresh journal"}));
    expect(screen.getByRole("button", {name: "Confirm: start fresh"})).toBeEnabled();
    expect(screen.getByText(/Saved history and media remain/)).toBeVisible();
    expect(startFreshJournal).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", {name: "Start a fresh journal?"})).toBeVisible();
    expect(document.activeElement).toBe(screen.getByRole("button", {name: "Cancel and keep current state"}));
    expect(screen.getByText(/Kept: saved reports, history, recordings, uploads, archived attempts, and existing backup ZIPs/)).toBeVisible();
    expect(screen.getByText(/A new recovery journal replaces the active damaged journal/)).toBeVisible();
    expect(screen.getByRole("dialog")).toHaveAttribute("aria-describedby", "fresh-dialog-consequences");
    fireEvent.click(screen.getByRole("button", {name: "Cancel and keep current state"}));
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("button", {name: "Start a fresh journal"})));
    expect(startFreshJournal).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", {name: "Start a fresh journal"}));
    status.mockResolvedValue(clean);
    fireEvent.click(screen.getByRole("button", {name: "Confirm: start fresh"}));
    await waitFor(() => expect(startFreshJournal).toHaveBeenCalledOnce());
    expect(await screen.findByText(t("journal.fresh_done"))).toBeVisible();
    expect(document.activeElement).toBe(screen.getByText(t("journal.fresh_done")));
    expect(store.getState().draft.coachingGoal).toBe("");
    expect(store.getState().draft.retryOfSessionId).toBe("");
  });

  it("disables the exits while a repair transaction is pending", async () => {
    status.mockResolvedValue({...damaged, transaction: {id: "repair", phase: "backend_staged"}});
    renderWithProviders(<JournalPanel locale="en" />);
    await screen.findByRole("alert");
    expect(screen.getByLabelText(t("journal.restore_backup"))).toBeDisabled();
    expect(screen.getByRole("button", {name: t("journal.diagnostics")})).toBeDisabled();
    expect(screen.getByRole("button", {name: t("journal.fresh")})).toBeDisabled();
    expect(screen.getByRole("button", {name: t("journal.rollback")})).toBeEnabled();
  });

  it("focuses the heading for a disabled fragment and retries the action when recovery enables it", async () => {
    status.mockResolvedValueOnce({...damaged, transaction: {id: "repair", phase: "backend_staged"}})
      .mockResolvedValue({...damaged, transaction: {id: "repair", phase: "backend_staged"}, recovery_retryable: true});
    renderWithProviders(<JournalPanel locale="en" />, {initialEntries: ["/settings#journal-fresh"]});
    const fresh = await screen.findByRole("button", {name: t("journal.fresh")});
    expect(fresh).toBeDisabled();
    expect(screen.getByRole("heading", {name: t("your_data.title")})).toHaveFocus();
    fireEvent.click(screen.getByRole("button", {name: t("journal.check_recovery")}));
    await waitFor(() => expect(fresh).toBeEnabled());
    await waitFor(() => expect(fresh).toHaveFocus());
  });

  it("keeps Check recovery focused when the fragment action remains disabled", async () => {
    status.mockResolvedValue({...damaged, transaction: {id: "repair", phase: "backend_staged"}});
    renderWithProviders(<JournalPanel locale="en" />, {initialEntries: ["/settings#journal-fresh"]});
    const fresh = await screen.findByRole("button", {name: t("journal.fresh")});
    expect(fresh).toBeDisabled();
    expect(screen.getByRole("heading", {name: t("your_data.title")})).toHaveFocus();
    const check = screen.getByRole("button", {name: t("journal.check_recovery")});
    fireEvent.click(check);
    await waitFor(() => expect(status).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(check).toBeEnabled());
    expect(check).toHaveFocus();
    expect(fresh).toBeDisabled();
  });

  it("announces a failed fresh start inside the dialog and permits retry", async () => {
    status.mockResolvedValue(damaged);
    vi.mocked(startFreshJournal).mockRejectedValueOnce(new Error("Journal disk is full"));
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.click(await screen.findByRole("button", { name: t("journal.fresh") }));
    fireEvent.click(screen.getByRole("button", { name: t("journal.confirm_fresh") }));
    await waitFor(() => expect(screen.getByRole("dialog").querySelector('[role="alert"]')).toHaveTextContent("Journal disk is full"));
    await waitFor(() => expect(document.activeElement).toHaveAttribute("id", "fresh-error"));
    expect(screen.getByRole("button", { name: "Cancel and keep current state" })).toBeEnabled();
    status.mockResolvedValue(clean);
    fireEvent.click(screen.getByRole("button", { name: t("journal.confirm_fresh") }));
    await waitFor(() => expect(startFreshJournal).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(t("journal.fresh_done"))).toBeVisible();
  });

  it("keeps the confirmation open when Escape is pressed during fresh start", async () => {
    status.mockResolvedValue(damaged);
    let finish!: () => void;
    vi.mocked(startFreshJournal).mockImplementation(() => new Promise<void>(resolve => { finish = resolve; }));
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.click(await screen.findByRole("button", { name: t("journal.fresh") }));
    fireEvent.click(screen.getByRole("button", { name: t("journal.confirm_fresh") }));
    const dialog = screen.getByRole("dialog");
    await waitFor(() => expect(screen.getByRole("button", { name: "Cancel and keep current state" })).toBeDisabled());
    fireEvent(dialog, new Event("cancel", { bubbles: false, cancelable: true }));
    expect(dialog).toBeInTheDocument();
    status.mockResolvedValue(clean);
    finish();
    expect(await screen.findByText(t("journal.fresh_done"))).toBeVisible();
  });

  it("reports a status refresh failure outside the closed dialog after fresh start succeeds", async () => {
    status.mockResolvedValueOnce(damaged).mockRejectedValueOnce(new Error("Status offline"));
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.click(await screen.findByRole("button", { name: t("journal.fresh") }));
    fireEvent.click(screen.getByRole("button", { name: t("journal.confirm_fresh") }));
    await waitFor(() => expect(startFreshJournal).toHaveBeenCalledOnce());
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("alert")).toHaveTextContent("A fresh journal was started");
    expect(screen.getByRole("alert")).toHaveTextContent("Status offline");
    expect(screen.getByTestId("journal-state")).not.toHaveTextContent("Journal damaged");
    expect(document.activeElement).toHaveAttribute("id", "journal-error");
  });

  it("reconciles a lost fresh-start reply when the backend confirms this request", async () => {
    status.mockResolvedValueOnce(damaged).mockResolvedValue(clean);
    vi.mocked(reconcileFreshJournal).mockResolvedValueOnce({applied: true, status: clean, pendingRecovery: false});
    vi.mocked(startFreshJournal).mockRejectedValueOnce(new Error("Response lost"));
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.click(await screen.findByRole("button", {name: t("journal.fresh")}));
    fireEvent.click(screen.getByRole("button", {name: t("journal.confirm_fresh")}));
    expect(await screen.findByText(t("journal.fresh_done"))).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByTestId("journal-state")).toHaveTextContent(t("your_data.state_healthy"));
    expect(reconcileFreshJournal).toHaveBeenCalledOnce();
    expect(vi.mocked(reconcileFreshJournal).mock.calls[0][0]).toMatch(/^[0-9a-f]{32}$/);
    expect(startFreshJournal).toHaveBeenCalledOnce();
  });

  it("reports a backend-applied fresh start when browser cleanup still fails", async () => {
    status.mockResolvedValueOnce(damaged);
    vi.mocked(reconcileFreshJournal).mockResolvedValueOnce({applied: true, status: clean, pendingRecovery: false, cleanupError: "IndexedDB blocked"});
    vi.mocked(startFreshJournal).mockRejectedValueOnce(new Error("IndexedDB blocked"));
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.click(await screen.findByRole("button", {name: t("journal.fresh")}));
    fireEvent.click(screen.getByRole("button", {name: t("journal.confirm_fresh")}));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("alert")).toHaveTextContent("A fresh journal was started");
    expect(screen.getByRole("alert")).toHaveTextContent("IndexedDB blocked");
    expect(screen.getByTestId("journal-state")).not.toHaveTextContent(t("your_data.state_damaged"));
    expect(reconcileFreshJournal).toHaveBeenCalledOnce();
  });

  it("keeps a definitive HTTP rejection inside the dialog without reconciling browser recovery", async () => {
    vi.mocked(startFreshJournal).mockRejectedValueOnce(new JournalHttpError("The journal is not damaged", 409));
    status.mockResolvedValueOnce(damaged);
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.click(await screen.findByRole("button", {name: t("journal.fresh")}));
    fireEvent.click(screen.getByRole("button", {name: t("journal.confirm_fresh")}));
    await waitFor(() => expect(screen.getByRole("dialog").querySelector('[role="alert"]')).toHaveTextContent("The journal is not damaged"));
    expect(reconcileFreshJournal).not.toHaveBeenCalled();
    expect(journalStatus).not.toHaveBeenCalled();
    expect(screen.queryByText(t("journal.fresh_done"))).not.toBeInTheDocument();
  });

  it("reports a healthy journal without claiming an unconfirmed reset", async () => {
    vi.mocked(reconcileFreshJournal).mockResolvedValueOnce({applied: false, status: clean, pendingRecovery: false});
    vi.mocked(startFreshJournal).mockRejectedValueOnce(new Error("Response lost"));
    status.mockResolvedValueOnce(damaged);
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.click(await screen.findByRole("button", {name: t("journal.fresh")}));
    fireEvent.click(screen.getByRole("button", {name: t("journal.confirm_fresh")}));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("alert")).toHaveTextContent("The journal is currently healthy");
    expect(screen.getByTestId("journal-state")).toHaveTextContent(t("your_data.state_healthy"));
    expect(screen.queryByText(t("journal.fresh_done"))).not.toBeInTheDocument();
  });

  it("keeps rollback retry reachable after an intact transaction hits a disk error", async () => {
    const retryable = {...damaged, recovery_retryable: true, transaction: {id: "a".repeat(32), phase: "rolling_back"}};
    status.mockResolvedValue(retryable);
    renderWithProviders(<JournalPanel locale="en" />);
    expect(await screen.findByRole("alert")).toHaveTextContent(retryable.recovery_error);
    expect(screen.getByRole("button", {name: t("journal.recover")})).toBeEnabled();
    expect(screen.getByRole("button", {name: t("journal.rollback")})).toBeEnabled();
    expect(screen.getByLabelText(t("journal.restore_backup"))).toBeEnabled();
    expect(screen.getByRole("button", {name: t("journal.fresh")})).toBeEnabled();
    expect(screen.getByRole("button", {name: t("journal.diagnostics")})).toBeEnabled();
    status.mockResolvedValue(clean);
    fireEvent.click(screen.getByRole("button", {name: t("journal.rollback")}));
    await waitFor(() => expect(recoverLearnerJournal).toHaveBeenCalledWith(true));
    expect(await screen.findByText(t("journal.rolled_back"))).toBeVisible();
    expect(screen.getByRole("button", {name: t("journal.backup")})).toBeEnabled();
  });

  it.each([false, true])("keeps an owned restore preview out of recovery prompts (damaged=%s)", async wasDamaged => {
    status.mockResolvedValue(wasDamaged ? damaged : clean);
    const view = renderWithProviders(<><JournalRecoveryNotice locale="en" /><JournalPanel locale="en" /></>);
    const input = await screen.findByLabelText(t(wasDamaged ? "journal.restore_backup" : "journal.open"));
    await waitFor(() => expect(status).toHaveBeenCalled());
    const staged = {...(wasDamaged ? damaged : clean), transaction: {id: "b".repeat(32), phase: "backend_staged"}};
    vi.mocked(previewLearnerRestore).mockImplementation(async () => {
      status.mockResolvedValue(staged);
      return {id: staged.transaction.id, attempts: 0, skipped_attempts: [], skipped_rehearsals: [], missing: [], browser: {sessions: [], recordings: []}};
    });
    fireEvent.change(input, {target: {files: [new File(["zip"], "backup.zip")]}});
    const confirmRestore = await screen.findByRole("button", {name: t("journal.confirm_restore")});
    await waitFor(() => expect(confirmRestore).toBeEnabled());
    expect(confirmRestore).toHaveClass(controlStyles.primary);
    expect(screen.getByRole("button", {name: t("journal.cancel")})).toHaveClass(controlStyles.secondary);
    expect(screen.queryByRole("button", {name: t("journal.recover")})).not.toBeInTheDocument();
    expect(screen.queryByRole("button", {name: t("journal.rollback")})).not.toBeInTheDocument();
    expect(input).toBeDisabled();
    fireEvent(window, new Event("focus"));
    await waitFor(() => expect(screen.queryByTestId("journal-recovery-notice")).not.toBeInTheDocument());
    // Leaving the preview makes the still-pending transaction recoverable again.
    view.rerender(<JournalRecoveryNotice locale="en" />);
    expect(await screen.findByTestId("journal-recovery-notice")).toBeVisible();
  });

  it.each(["backend_staged", "backend_committed", "browser_recovery"])("shows recovery immediately after a failed commit (%s)", async phase => {
    renderWithProviders(<><JournalRecoveryNotice locale="en" /><JournalPanel locale="en" /></>);
    const id = "c".repeat(32);
    vi.mocked(previewLearnerRestore).mockImplementation(async () => {
      status.mockResolvedValue({...clean, transaction: {id, phase: "backend_staged"}});
      return {id, attempts: 0, skipped_attempts: [], skipped_rehearsals: [], missing: [], browser: {sessions: [], recordings: []}};
    });
    vi.mocked(commitLearnerRestore).mockImplementation(async () => {
      status.mockResolvedValue({...clean, transaction: {id, phase}});
      throw new Error("Lost commit response");
    });
    fireEvent.change(screen.getByLabelText(t("journal.open")), {target: {files: [new File(["zip"], "backup.zip")]}});
    const restore = await screen.findByRole("button", {name: t("journal.confirm_restore")});
    await waitFor(() => expect(restore).toBeEnabled());
    fireEvent.click(restore);
    expect(await screen.findByRole("alert")).toHaveTextContent("Lost commit response");
    expect(screen.getByRole("button", {name: t("journal.recover")})).toBeEnabled();
    expect(screen.getByRole("button", {name: t("journal.rollback")})).toBeEnabled();
    expect(await screen.findByTestId("journal-recovery-notice")).toBeVisible();
    expect(screen.queryByRole("button", {name: t("journal.confirm_restore")})).not.toBeInTheDocument();
    status.mockResolvedValue(clean);
    fireEvent.click(screen.getByRole("button", {name: t("journal.recover")}));
    expect(await screen.findByText(t("journal.recovered"))).toBeVisible();
    expect(recoverLearnerJournal).toHaveBeenCalledWith();
  });

  it("starts a clean in-memory session after a successful restore", async () => {
    const { store } = renderWithProviders(<JournalPanel locale="en" />, { appState: { draft: {
      speakerId: "learner", themeId: "trip", promptText: "Describe a trip",
      retryOfSessionId: "first", coachingGoal: "Add a closing sentence", retryGoalEligible: true,
    } } });
    vi.mocked(previewLearnerRestore).mockResolvedValue({ id: "restore", attempts: 0,
      skipped_attempts: [], skipped_rehearsals: [], missing: [], browser: { sessions: [], recordings: [] } });
    fireEvent.change(await screen.findByLabelText(t("journal.open")), { target: { files: [new File(["zip"], "backup.zip")] } });
    fireEvent.click(await screen.findByRole("button", { name: t("journal.confirm_restore") }));
    await waitFor(() => expect(commitLearnerRestore).toHaveBeenCalledOnce());
    await waitFor(() => expect(store.getState().draft).toMatchObject({ coachingGoal: "", retryOfSessionId: "" }));
  });

  it("keeps restore success visible when the following status refresh fails", async () => {
    status.mockResolvedValueOnce(clean).mockResolvedValueOnce({...clean, transaction: {id: "restore", phase: "backend_staged"}})
      .mockRejectedValueOnce(new Error("Status offline"));
    vi.mocked(previewLearnerRestore).mockResolvedValue({id: "restore", attempts: 0,
      skipped_attempts: [], skipped_rehearsals: [], missing: [], browser: {sessions: [], recordings: []}});
    const {queryClient} = renderWithProviders(<JournalPanel locale="en" />);
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    fireEvent.change(await screen.findByLabelText(t("journal.open")), {target: {files: [new File(["zip"], "backup.zip")]}});
    const restore = await screen.findByRole("button", {name: t("journal.confirm_restore")});
    await waitFor(() => expect(restore).toBeEnabled());
    invalidate.mockClear();
    fireEvent.click(restore);
    await waitFor(() => expect(commitLearnerRestore).toHaveBeenCalledOnce());
    expect(await screen.findByText(t("journal.restored"))).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("The backup was restored");
    expect(screen.getByRole("alert")).toHaveTextContent("Status offline");
    expect(screen.getByTestId("journal-state")).not.toHaveTextContent(t("your_data.state_pending"));
    expect(screen.queryByRole("button", {name: t("journal.confirm_restore")})).not.toBeInTheDocument();
    await waitFor(() => expect(invalidate).toHaveBeenCalledOnce());
  });

  it("offers exits immediately when an explicit rollback retry fails", async () => {
    const transaction = {id: "d".repeat(32), phase: "backend_committed"};
    status.mockResolvedValue({...clean, transaction});
    vi.mocked(recoverLearnerJournal).mockImplementation(async () => {
      status.mockResolvedValue({...damaged, transaction, recovery_retryable: true});
      throw new Error("Recovery snapshot is unavailable");
    });
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.click(await screen.findByRole("button", {name: t("journal.rollback")}));
    expect(await screen.findByText("Recovery snapshot is unavailable")).toBeVisible();
    expect(screen.getByLabelText(t("journal.restore_backup"))).toBeEnabled();
    expect(screen.getByRole("button", {name: t("journal.fresh")})).toBeEnabled();
    expect(screen.getByRole("button", {name: t("journal.diagnostics")})).toBeEnabled();
    expect(screen.getByRole("button", {name: t("journal.recover")})).toBeEnabled();
  });

  const states: {name: string; value: JournalStatus; retry: boolean; exits: boolean; backup: boolean}[] = [
    {name: "healthy idle", value: clean, retry: false, exits: true, backup: true},
    {name: "healthy prepared", value: {...clean, transaction: {id: "a", phase: "prepared"}}, retry: true, exits: false, backup: false},
    {name: "healthy interrupted commit", value: {...clean, transaction: {id: "a", phase: "backend_committed"}}, retry: true, exits: false, backup: false},
    {name: "damaged without transaction", value: damaged, retry: false, exits: true, backup: false},
    {name: "damaged identity", value: {...damaged, transaction: {id: "damaged", phase: "recovery_error"}}, retry: false, exits: true, backup: false},
    {name: "failed prepared recovery", value: {...damaged, recovery_retryable: true, transaction: {id: "a", phase: "prepared"}}, retry: true, exits: true, backup: false},
    {name: "failed rollback", value: {...damaged, recovery_retryable: true, transaction: {id: "a", phase: "rolling_back"}}, retry: true, exits: true, backup: false},
    {name: "damaged completion receipt", value: {...damaged, recovery_retryable: true, transaction: {id: "a", phase: "complete"}}, retry: true, exits: true, backup: false},
    {name: "pending repair", value: {...damaged, transaction: {id: "a", phase: "backend_staged"}}, retry: true, exits: false, backup: false},
    {name: "failed repair", value: {...damaged, recovery_retryable: true, transaction: {id: "a", phase: "backend_staged"}}, retry: true, exits: true, backup: false},
  ];
  it.each(states)("matches the recovery control table: $name", async ({value, retry, exits, backup}) => {
    status.mockResolvedValue(value);
    renderWithProviders(<JournalPanel locale="en" />);
    await waitFor(() => {
    expect(screen.getByTestId("journal-state")).toHaveTextContent(value.recovery_error ? "Journal damaged" : value.transaction ? "Recovery pending" : "Journal healthy");
    expect(screen.getByLabelText(t(value.recovery_error ? "journal.restore_backup" : "journal.open"))).toHaveProperty("disabled", !exits);
    expect(screen.getByRole("button", {name: t("journal.backup")})).toHaveProperty("disabled", !backup);
    for (const key of ["journal.recover", "journal.rollback"]) {
      const button = screen.queryByRole("button", {name: t(key)});
      if (retry) expect(button).toBeEnabled();
      else expect(button).not.toBeInTheDocument();
    }
    for (const key of ["journal.diagnostics", "journal.fresh"]) {
      const button = screen.queryByRole("button", {name: t(key)});
      if (value.recovery_error) expect(button).toHaveProperty("disabled", !exits);
      else expect(button).not.toBeInTheDocument();
    }
    });
  });

  it("loads archived rehearsals immediately without requiring Refresh", async () => {
    const rehearsal = createRehearsal("en", "B1", "Archived fixture", {
      provider: "ollama", model: "fixture", baseUrl: "", whisper: "tiny", feedbackLanguage: "en",
    });
    archived.mockResolvedValue([{...rehearsal, archivedAt: "2026-10-09T00:00:00Z"}]);
    renderWithProviders(<JournalPanel locale="en" />);
    expect(await screen.findByRole("heading", {level: 3, name: t("journal.archived_rehearsals")})).toBeVisible();
    expect(screen.getByText(/Archived fixture/)).toBeVisible();
    expect(screen.getByRole("button", {name: t("journal.undo")})).toBeVisible();
    expect(archived).toHaveBeenCalledTimes(1);
  });

  it("keeps archived attempts in the heading outline", async () => {
    status.mockResolvedValue({...clean, archived: ["archived-attempt"]});
    renderWithProviders(<JournalPanel locale="en" />);
    expect(await screen.findByRole("heading", {level: 3, name: t("journal.archived_attempts")})).toBeVisible();
  });

  it("keeps archived rehearsals visible when backend status cannot be loaded", async () => {
    status.mockRejectedValue(new Error("Status unavailable"));
    const rehearsal = createRehearsal("en", "B1", "Retained fixture", {
      provider: "ollama", model: "fixture", baseUrl: "", whisper: "tiny", feedbackLanguage: "en",
    });
    archived.mockResolvedValue([{...rehearsal, archivedAt: "2026-10-09T00:00:00Z"}]);
    renderWithProviders(<JournalPanel locale="en" />);
    expect(await screen.findByText(/Retained fixture/)).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("Status unavailable");
    expect(screen.getByTestId("journal-state")).toHaveTextContent("Journal state unavailable");
  });

  it("clears the selected restore filename after cancelling a preview", async () => {
    vi.mocked(previewLearnerRestore).mockResolvedValue({ id: "restore", attempts: 0,
      skipped_attempts: [], skipped_rehearsals: [], missing: [], browser: { sessions: [], recordings: [] } });
    renderWithProviders(<JournalPanel locale="en" />);
    fireEvent.change(await screen.findByLabelText(t("journal.open")), { target: { files: [new File(["zip"], "backup.zip")] } });
    expect(await screen.findByText("backup.zip")).toBeVisible();
    fireEvent.click(await screen.findByRole("button", { name: t("journal.cancel") }));
    await waitFor(() => expect(screen.queryByText("backup.zip")).not.toBeInTheDocument());
  });
});

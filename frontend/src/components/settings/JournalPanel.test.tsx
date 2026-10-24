import "@testing-library/jest-dom/vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTranslator } from "@/lib/i18n";
import { commitLearnerRestore, journalRecoveryStatus, previewLearnerRestore, recoverLearnerJournal, startFreshJournal, type JournalStatus } from "@/lib/rehearsal/maintenance";
import { createRehearsal } from "@/lib/rehearsal/session";
import { listArchivedRehearsals } from "@/lib/rehearsal/storage";
import { renderWithProviders } from "@/test/renderWithProviders";
import { JournalRecoveryNotice } from "@/components/shell/JournalRecoveryNotice";
import { JournalPanel } from "./JournalPanel";

vi.mock("@/lib/rehearsal/maintenance", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/rehearsal/maintenance")>(),
  journalRecoveryStatus: vi.fn(),
  startFreshJournal: vi.fn(),
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
    renderWithProviders(<><JournalRecoveryNotice locale="en" /><JournalPanel locale="en" /></>);
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
    expect(screen.getByText(/Saved reports, history, recordings and uploads stay in place/)).toBeVisible();
    expect(startFreshJournal).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", {name: t("journal.cancel")}));
    expect(startFreshJournal).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", {name: "Start a fresh journal"}));
    status.mockResolvedValue(clean);
    fireEvent.click(screen.getByRole("button", {name: "Confirm: start fresh"}));
    await waitFor(() => expect(startFreshJournal).toHaveBeenCalledOnce());
    expect(await screen.findByText(t("journal.fresh_done"))).toBeVisible();
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
    await waitFor(() => expect(screen.getByRole("button", {name: t("journal.confirm_restore")})).toBeEnabled());
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
  });
});

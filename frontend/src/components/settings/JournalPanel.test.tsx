import "@testing-library/jest-dom/vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTranslator } from "@/lib/i18n";
import { journalRecoveryStatus, previewLearnerRestore, recoverLearnerJournal, startFreshJournal } from "@/lib/rehearsal/maintenance";
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
    const retryable = {...damaged, transaction: {id: "a".repeat(32), phase: "rolling_back"}};
    status.mockResolvedValue(retryable);
    renderWithProviders(<JournalPanel locale="en" />);
    expect(await screen.findByRole("alert")).toHaveTextContent(retryable.recovery_error);
    expect(screen.getByRole("button", {name: t("journal.recover")})).toBeEnabled();
    expect(screen.getByRole("button", {name: t("journal.rollback")})).toBeEnabled();
    expect(screen.getByLabelText(t("journal.restore_backup"))).toBeDisabled();
    expect(screen.getByRole("button", {name: t("journal.fresh")})).toBeDisabled();
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

  it("loads archived rehearsals immediately without requiring Refresh", async () => {
    const rehearsal = createRehearsal("en", "B1", "Archived fixture", {
      provider: "ollama", model: "fixture", baseUrl: "", whisper: "tiny", feedbackLanguage: "en",
    });
    archived.mockResolvedValue([{...rehearsal, archivedAt: "2026-10-09T00:00:00Z"}]);
    renderWithProviders(<JournalPanel locale="en" />);
    expect(await screen.findByText(t("journal.archived_rehearsals"))).toBeVisible();
    expect(screen.getByText(/Archived fixture/)).toBeVisible();
    expect(screen.getByRole("button", {name: t("journal.undo")})).toBeVisible();
    expect(archived).toHaveBeenCalledTimes(1);
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

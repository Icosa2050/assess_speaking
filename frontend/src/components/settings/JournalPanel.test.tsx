import "@testing-library/jest-dom/vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTranslator } from "@/lib/i18n";
import { journalRecoveryStatus } from "@/lib/rehearsal/maintenance";
import { createRehearsal } from "@/lib/rehearsal/session";
import { listArchivedRehearsals } from "@/lib/rehearsal/storage";
import { renderWithProviders } from "@/test/renderWithProviders";
import { JournalRecoveryNotice } from "@/components/shell/JournalRecoveryNotice";
import { JournalPanel } from "./JournalPanel";

vi.mock("@/lib/rehearsal/maintenance", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/rehearsal/maintenance")>(),
  journalRecoveryStatus: vi.fn(),
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
    expect(screen.getByRole("link")).toHaveAttribute("href", "/settings");
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

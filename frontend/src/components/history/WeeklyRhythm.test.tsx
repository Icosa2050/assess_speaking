import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { HistoryRow } from "@/lib/api/types";
import { WeeklyRhythm } from "./WeeklyRhythm";

const selected = { speaker_id: "alex", learning_language: "it" } as HistoryRow;
afterEach(() => { cleanup(); localStorage.clear(); vi.restoreAllMocks(); });
describe("weekly goal", () => {
  it("translates the Italian controls and keeps the chosen goal after remount", () => {
    const view = render(<WeeklyRhythm rows={[]} selected={selected} locale="it" />);
    expect(screen.getByText("Il ritmo di questa settimana")).toBeVisible();
    fireEvent.change(screen.getByRole("combobox", { name: "Obiettivo settimanale" }), { target: { value: "3" } });
    view.unmount();
    render(<WeeklyRhythm rows={[]} selected={selected} locale="it" />);
    expect(screen.getByRole("combobox")).toHaveValue("3");
    expect(screen.getByText("0 su 3")).toBeVisible();
    expect(screen.getByRole("option", { name: "1 giorno" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /^Questa settimana:/ })).toHaveAccessibleName("Questa settimana: 0 giorni. Obiettivo: 3 giorni.");
  });
  it("stays usable when browser storage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("unavailable"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("unavailable"); });
    render(<WeeklyRhythm rows={[]} selected={selected} locale="en" />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "2" } });
    expect(screen.getByRole("combobox")).toHaveValue("2");
  });
});

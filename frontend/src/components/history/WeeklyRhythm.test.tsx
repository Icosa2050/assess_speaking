import "@testing-library/jest-dom/vitest";
import { act, fireEvent, render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { HistoryRow } from "@/lib/api/types";
import { WeeklyRhythm } from "./WeeklyRhythm";

const selected = { speaker_id: "alex", learning_language: "it" } as HistoryRow;
afterEach(() => { cleanup(); localStorage.clear(); vi.restoreAllMocks(); vi.useRealTimers(); });
describe("weekly goal", () => {
  it.each(["en", "it"])("shows earned days, caps the ring, and rolls into a fresh week (%s)", locale => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 9, 11, 23, 59, 30)); // Sunday, local browser time.
    const row = (day: number, hour = 12) => ({ ...selected, timestamp: new Date(2026, 9, day, hour).toISOString() } as HistoryRow);
    const rows = [row(5), row(5, 13), row(8), row(11)];
    const view = render(<WeeklyRhythm rows={rows} selected={selected} locale={locale} />);
    expect(view.container.querySelectorAll('[data-completed="true"]')).toHaveLength(3);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "2" } });
    const ring = screen.getAllByRole("img")[0];
    expect(ring.style.getPropertyValue("--completion")).toBe("360deg");
    expect(ring).toHaveTextContent("3");
    // Calendar refresh updates the visible week without navigating or losing the goal.
    act(() => { vi.advanceTimersByTime(60_000); });
    expect(view.container.querySelectorAll('[data-completed="true"]')).toHaveLength(0);
    expect(ring.style.getPropertyValue("--completion")).toBe("0deg");
    expect(screen.getByRole("combobox")).toHaveValue("2");
    view.rerender(<WeeklyRhythm rows={[...rows, row(12, 0)]} selected={selected} locale={locale} />);
    expect(view.container.querySelectorAll('[data-completed="true"]')).toHaveLength(1);
    expect(ring.style.getPropertyValue("--completion")).toBe("180deg");
  });
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

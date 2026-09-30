import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { expect, it, vi } from "vitest";
import { AppShell } from "./AppShell";

const BrokenRoute = () => { throw new Error("Route failed"); };

it("recovers when navigating away from a failed route", async () => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    render(
      <MemoryRouter initialEntries={["/broken"]}>
        <Routes>
          <Route element={<AppShell appName="Practice" shellBody="Practice speaking"
            navAriaLabel="Navigation" navGroups={[{ id: "main", items: [{ href: "/healthy", label: "Healthy route" }] }]} />}>
            <Route path="/broken" element={<BrokenRoute />} />
            <Route path="/healthy" element={<p>Route is healthy</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByRole("button", { name: "Reload page" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("link", { name: "Healthy route" }));
    expect(await screen.findByText("Route is healthy")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reload page" })).not.toBeInTheDocument();
  } finally {
    errors.mockRestore();
  }
});

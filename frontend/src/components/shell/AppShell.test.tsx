import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
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

it("exposes grouped navigation, the current page, a disabled step and a skip target", () => {
  render(
    <MemoryRouter initialEntries={["/speak"]}>
      <Routes>
        <Route element={<AppShell appName="Practice" shellBody="Practice speaking"
          navAriaLabel="Main navigation" skipLabel="Skip to content" navGroups={[
            { id: "primary", items: [{ href: "/", label: "Practice" }, { href: "/history", label: "Progress" }] },
            { id: "context", label: "Current step", items: [{ href: "/speak", label: "Speak" }, { href: "/review", label: "Review", disabled: true }] },
            { id: "preferences", label: "Preferences", items: [{ href: "/guide", label: "How scores work" }] },
          ]} />}>
          <Route path="/speak" element={<p>Speak content</p>} />
          <Route path="/history" element={<p>Progress content</p>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
  const nav = screen.getByRole("navigation", { name: "Main navigation" });
  expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main-content");
  expect(screen.getByRole("main")).toHaveAttribute("id", "main-content");
  expect(within(nav).getByRole("link", { name: "Speak" })).toHaveAttribute("aria-current", "page");
  expect(within(nav).getByRole("link", { name: "Review" })).toHaveAttribute("aria-disabled", "true");
  expect(within(nav).getByRole("link", { name: "Review" })).not.toHaveAttribute("href");
  fireEvent.click(within(nav).getByRole("link", { name: "Progress" }));
  expect(within(nav).getByRole("link", { name: "Progress" })).toHaveAttribute("aria-current", "page");
  expect(screen.getByText("Progress content")).toBeInTheDocument();
});

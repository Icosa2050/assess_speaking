import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { BrowserRouter, MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
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
            { id: "context", label: "Current step", items: [{ href: "/speak", label: "Speak" }, { href: "/review", label: "Review", disabled: true, disabledReason: "Available after a speaking attempt" }] },
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
  expect(within(nav).getByRole("link", { name: "Review" })).toHaveAttribute("href", "/review");
  expect(within(nav).getByRole("link", { name: "Review" })).toHaveAttribute("tabindex", "0");
  expect(within(nav).getByRole("link", { name: "Review" })).toHaveAccessibleDescription("Available after a speaking attempt");
  fireEvent.click(within(nav).getByRole("link", { name: "Review" }));
  expect(screen.getByText("Speak content")).toBeInTheDocument();
  expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  fireEvent.click(within(nav).getByRole("link", { name: "Progress" }));
  expect(within(nav).getByRole("link", { name: "Progress" })).toHaveAttribute("aria-current", "page");
  expect(screen.getByText("Progress content")).toBeInTheDocument();
});

it.each(["/settings", "/settings#your-data", "/settings#journal-fresh"])("marks only one Preferences link current at %s", entry => {
  render(<MemoryRouter initialEntries={[entry]}><Routes><Route element={<AppShell appName="Practice" shellBody="Practice speaking" navAriaLabel="Navigation" navGroups={[
    { id: "preferences", label: "Preferences", items: [{ href: "/settings#your-data", label: "Your data" }, { href: "/settings", label: "Settings" }] },
  ]} />}><Route path="/settings" element={<p>Settings content</p>} /></Route></Routes></MemoryRouter>);
  const nav = screen.getByRole("navigation");
  const current = within(nav).getAllByRole("link").filter(link => link.getAttribute("aria-current") === "page");
  expect(current).toHaveLength(1);
  expect(current[0]).toHaveTextContent(entry === "/settings" ? "Settings" : "Your data");
});

it("focuses content without clearing route state or its contextual navigation", async () => {
  const originalUrl = window.location.href;
  const originalState = window.history.state;
  window.history.replaceState({ usr: { from: "/speak" }, key: "skip-link-test", idx: 0 }, "", "/");

  const ShellWithContext = () => {
    const location = useLocation();
    const from = location.state?.from;
    return (
      <Routes>
        <Route element={<AppShell appName="Practice" shellBody="Practice speaking"
          navAriaLabel="Main navigation" navGroups={[
            { id: "primary", items: [{ href: "/", label: "Practice" }] },
            ...(from === "/speak" ? [{ id: "context", label: "Current step", items: [{ href: "/speak", label: "Speak" }] }] : []),
          ]} />}>
          <Route path="/" element={<p data-testid="route-state">{from}</p>} />
        </Route>
      </Routes>
    );
  };

  const { unmount } = render(<BrowserRouter><ShellWithContext /></BrowserRouter>);
  const pushState = vi.spyOn(window.history, "pushState");
  try {
    const nav = screen.getByRole("navigation", { name: "Main navigation" });
    expect(within(nav).getByRole("group", { name: "Current step" })).toBeVisible();
    expect(screen.getByTestId("route-state")).toHaveTextContent("/speak");

    fireEvent.click(screen.getByRole("link", { name: "Skip to content" }));
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(screen.getByRole("main")).toHaveFocus();
    expect(screen.getByTestId("route-state")).toHaveTextContent("/speak");
    expect(within(nav).getByRole("group", { name: "Current step" })).toBeVisible();
    expect(window.location.hash).toBe("");
    expect(pushState).not.toHaveBeenCalled();
  } finally {
    pushState.mockRestore();
    unmount();
    window.history.replaceState(originalState, "", originalUrl);
  }
});

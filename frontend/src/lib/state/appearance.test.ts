import { beforeEach, describe, expect, it, vi } from "vitest";

import { createAppStore } from "./appStore";
import {
  APPEARANCE_STORAGE_KEY,
  applyAppearance,
  readAppearance,
  resolvedTheme,
} from "./appearance";

describe("appearance preference", () => {
  beforeEach(() => window.localStorage.clear());

  it("defaults to System, ignores stale values, and restores a saved choice", () => {
    expect(readAppearance()).toBe("system");
    window.localStorage.setItem(APPEARANCE_STORAGE_KEY, "unexpected");
    expect(readAppearance()).toBe("system");
    expect(resolvedTheme(readAppearance(), false)).toBe("light");
    expect(resolvedTheme(readAppearance(), true)).toBe("dark");
    const store = createAppStore();
    store.getState().setAppearance("dark");
    expect(window.localStorage.getItem(APPEARANCE_STORAGE_KEY)).toBe("dark");
    expect(createAppStore().getState().preferences.appearance).toBe("dark");
  });

  it("resolves System on OS changes and sets document color scheme", () => {
    expect(resolvedTheme("system", false)).toBe("light");
    expect(resolvedTheme("system", true)).toBe("dark");
    applyAppearance("system", true);
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.style.colorScheme).toBe("dark");
    applyAppearance("light", true);
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("retains the selected theme for the visit when local storage is blocked", () => {
    const setItem = vi.spyOn(window.localStorage, "setItem").mockImplementation(() => { throw new Error("blocked"); });
    const store = createAppStore();
    store.getState().setAppearance("dark");
    expect(store.getState().preferences.appearance).toBe("dark");
    setItem.mockRestore();
  });
});

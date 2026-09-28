import { afterEach, describe, expect, it } from "vitest";

import { renderWithProviders } from "./renderWithProviders";

describe("renderWithProviders", () => {
  const originalLocalStorage = Object.getOwnPropertyDescriptor(window, "localStorage");

  const installCompleteStorage = () => {
    const storage = new Map<string, string>();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        clear: () => {
          storage.clear();
        },
        getItem: (key: string) => storage.get(key) ?? null,
        key: (index: number) => Array.from(storage.keys())[index] ?? null,
        get length() {
          return storage.size;
        },
        removeItem: (key: string) => {
          storage.delete(key);
        },
        setItem: (key: string, value: string) => {
          storage.set(key, value);
        },
      },
    });
  };

  afterEach(() => {
    if (originalLocalStorage) {
      Object.defineProperty(window, "localStorage", originalLocalStorage);
    } else {
      Reflect.deleteProperty(window, "localStorage");
    }
  });

  it("clears existing localStorage state before rendering", () => {
    installCompleteStorage();
    window.localStorage.setItem("stale", "value");

    renderWithProviders(<div>ready</div>);

    expect(window.localStorage.getItem("stale")).toBeNull();
    expect(window.localStorage.length).toBe(0);
  });

  it("installs a complete storage shim when the environment storage is partial", () => {
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        clear: () => undefined,
        getItem: () => null,
        removeItem: () => undefined,
        setItem: () => undefined,
      },
    });

    renderWithProviders(<div>ready</div>);
    window.localStorage.setItem("alpha", "one");
    window.localStorage.setItem("beta", "two");

    expect(window.localStorage.length).toBe(2);
    expect(window.localStorage.key(0)).toBe("alpha");
    expect(window.localStorage.getItem("beta")).toBe("two");
  });
});

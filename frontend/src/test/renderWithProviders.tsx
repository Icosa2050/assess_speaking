import type { PropsWithChildren, ReactElement } from "react";

import { QueryClientProvider, type QueryClient } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";
import { MemoryRouter, type InitialEntry } from "react-router-dom";

import {
  AppStoreProvider,
  createAppStore,
  type AppStoreApi,
  type AppStoreSeed,
} from "@/lib/state/appStore";
import { DEFAULT_UI_LOCALE, type UiLocale } from "@/lib/state/sessionDraft";
import { createQueryClient } from "@/lib/query/queryClient";

type RenderWithProvidersOptions = Omit<RenderOptions, "wrapper"> & {
  initialEntries?: InitialEntry[];
  locale?: UiLocale;
  queryClient?: QueryClient;
  store?: AppStoreApi;
  /** Ignored when a custom store is provided. */
  appState?: AppStoreSeed;
};

const hasCompleteLocalStorage = (
  localStorage:
    | Storage
    | {
        clear?: () => void;
        getItem?: (key: string) => string | null;
        key?: (index: number) => string | null;
        length?: number;
        removeItem?: (key: string) => void;
        setItem?: (key: string, value: string) => void;
      }
    | undefined,
): localStorage is Storage =>
  Boolean(
    localStorage &&
      typeof localStorage.getItem === "function" &&
      typeof localStorage.setItem === "function" &&
      typeof localStorage.removeItem === "function" &&
      typeof localStorage.clear === "function" &&
      typeof localStorage.key === "function" &&
      typeof localStorage.length === "number",
  );

const installLocalStorageShim = () => {
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

export const renderWithProviders = (
  ui: ReactElement,
  {
    initialEntries = ["/"],
    locale = DEFAULT_UI_LOCALE,
    queryClient = createQueryClient(),
    store,
    appState,
    ...renderOptions
  }: RenderWithProvidersOptions = {},
) => {
  if (typeof window !== "undefined") {
    const currentLocalStorage = window.localStorage as Parameters<typeof hasCompleteLocalStorage>[0];

    if (hasCompleteLocalStorage(currentLocalStorage)) {
      currentLocalStorage.clear();
    } else {
      installLocalStorageShim();
    }
  }

  const resolvedStore =
    store ??
    createAppStore({
      ...appState,
      preferences: {
        ...appState?.preferences,
        uiLocale: locale,
      },
    });

  const Wrapper = ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={queryClient}>
      <AppStoreProvider store={resolvedStore}>
        <MemoryRouter initialEntries={initialEntries}>{children}</MemoryRouter>
      </AppStoreProvider>
    </QueryClientProvider>
  );

  return {
    queryClient,
    store: resolvedStore,
    ...render(ui, {
      wrapper: Wrapper,
      ...renderOptions,
    }),
  };
};

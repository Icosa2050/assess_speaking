export type Appearance = "system" | "light" | "dark";

export const APPEARANCE_STORAGE_KEY = "vostavo.appearance.v1";

// First run follows the operating system; an explicit choice remains persisted.
export const DEFAULT_APPEARANCE: Appearance = "system";

const isAppearance = (value: unknown): value is Appearance =>
  value === "system" || value === "light" || value === "dark";

export const readAppearance = (): Appearance => {
  try {
    const stored = window.localStorage.getItem(APPEARANCE_STORAGE_KEY);
    return isAppearance(stored) ? stored : DEFAULT_APPEARANCE;
  } catch {
    return DEFAULT_APPEARANCE;
  }
};

export const persistAppearance = (appearance: Appearance): void => {
  try {
    window.localStorage.setItem(APPEARANCE_STORAGE_KEY, appearance);
  } catch {
    // Keep the selected appearance for this visit if storage is unavailable.
  }
};

export const resolvedTheme = (appearance: Appearance, systemDark: boolean): "light" | "dark" =>
  appearance === "system" ? (systemDark ? "dark" : "light") : appearance;

export const applyAppearance = (appearance: Appearance, systemDark: boolean): void => {
  const theme = resolvedTheme(appearance, systemDark);
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
};

export const systemPrefersDark = (): boolean =>
  typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches;

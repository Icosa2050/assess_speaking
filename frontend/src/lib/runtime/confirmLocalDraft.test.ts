import { afterEach, expect, it, vi } from "vitest";
import { confirmLocalDraft } from "./confirmLocalDraft";

const desktopWindow = window as Window & { __VOSTAVO_DESKTOP__?: unknown };
afterEach(() => {
  delete desktopWindow.__VOSTAVO_DESKTOP__;
  vi.restoreAllMocks();
});

it("uses browser confirmation when no desktop bridge exists", async () => {
  const browserConfirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  expect(await confirmLocalDraft("Replace? ")).toBe(true);
  expect(browserConfirm).toHaveBeenCalledWith("Replace? ");
});

it("uses the native desktop confirmation and fails closed if it is unavailable", async () => {
  const browserConfirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  const nativeConfirm = vi.fn().mockResolvedValueOnce(true).mockResolvedValueOnce(false).mockRejectedValueOnce(new Error("dialog unavailable"));
  desktopWindow.__VOSTAVO_DESKTOP__ = { confirmDraft: nativeConfirm };
  expect(await confirmLocalDraft("Replace saved take? ")).toBe(true);
  expect(await confirmLocalDraft("Discard saved take? ")).toBe(false);
  expect(await confirmLocalDraft("Replace saved setup? ")).toBe(false);
  expect(nativeConfirm).toHaveBeenCalledTimes(3);
  desktopWindow.__VOSTAVO_DESKTOP__ = {};
  expect(await confirmLocalDraft("Replace? ")).toBe(false);
  expect(browserConfirm).not.toHaveBeenCalled();
});

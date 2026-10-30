import { desktopBridge } from "./desktopBridge";

export const confirmLocalDraft = (message: string): boolean | Promise<boolean> => {
  if (!desktopBridge.isDesktopRuntime()) return window.confirm(message);
  const nativeConfirm = desktopBridge.readRuntime().confirmDraft;
  if (!nativeConfirm) return false;
  try { return nativeConfirm(message).catch(() => false); }
  catch { return false; }
};

// Mark the exact saved draft affected by a journal operation. Backend report
// provenance remains authoritative if WebView storage is later cleared.
const KEY = "vostavo.restored-retry-parents.v2";
const FRESH_KEY = "vostavo.fresh-retry-parents.v2";
const ID = /^[A-Za-z0-9_-]{1,128}$/;
export type RetryParentRef = { draftId: string; parentId: string };

const read = (key: string): RetryParentRef[] => {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(value) ? value.filter((item): item is RetryParentRef =>
      !!item && typeof item === "object" && ID.test(item.draftId) && ID.test(item.parentId)).slice(-100) : [];
  } catch { return []; }
};
const write = (key: string, entries: RetryParentRef[]) => {
  try {
    if (entries.length) localStorage.setItem(key, JSON.stringify(entries.slice(-100)));
    else localStorage.removeItem(key);
  } catch { /* The backend restore timestamp still protects native drafts. */ }
};
const remember = (key: string, entries: RetryParentRef[]) => {
  const merged = new Map(read(key).map(entry => [entry.draftId, entry]));
  for (const entry of entries) if (ID.test(entry.draftId) && ID.test(entry.parentId)) merged.set(entry.draftId, entry);
  write(key, [...merged.values()]);
};
const has = (key: string, draftId: string, parentId: string) =>
  read(key).some(entry => entry.draftId === draftId && entry.parentId === parentId);

export const hasRestoredRetryParent = (draftId: string, parentId: string) => has(KEY, draftId, parentId);
export const hasFreshRetryParent = (draftId: string, parentId: string) => has(FRESH_KEY, draftId, parentId);
export const rememberRestoredRetryParents = (entries: RetryParentRef[]) => remember(KEY, entries);
export const rememberFreshRetryParents = (entries: RetryParentRef[]) => remember(FRESH_KEY, entries);
export const forgetRestoredRetryMarker = (draftId: string) => write(KEY, read(KEY).filter(entry => entry.draftId !== draftId));
export const forgetFreshRetryMarker = (draftId: string) => write(FRESH_KEY, read(FRESH_KEY).filter(entry => entry.draftId !== draftId));
export const forgetRetryMarkers = (draftId: string) => { forgetRestoredRetryMarker(draftId); forgetFreshRetryMarker(draftId); };

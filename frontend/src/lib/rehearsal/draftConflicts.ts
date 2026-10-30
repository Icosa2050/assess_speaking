// Restore IDs are kept in this browser profile so a recovered local take cannot
// silently become a retry of an unrelated imported attempt after a restart.
const KEY = "vostavo.restored-retry-parents.v1";
const FRESH_KEY = "vostavo.fresh-retry-parents.v1";
const ID = /^[A-Za-z0-9_-]{1,128}$/;

export function restoredRetryParents(): Set<string> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(KEY) || "[]");
    return new Set(Array.isArray(value) ? value.filter((id): id is string => typeof id === "string" && ID.test(id)).slice(-10000) : []);
  } catch { return new Set(); }
}

export function rememberRestoredRetryParents(ids: string[]): void {
  if (!ids.length) return;
  const merged = new Set([...restoredRetryParents(), ...ids.filter(id => ID.test(id))]);
  // Fail closed when the browser cannot persist the collision guard.
  localStorage.setItem(KEY, JSON.stringify([...merged].slice(-10000)));
}

export function freshRetryParents(): Set<string> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(FRESH_KEY) || "[]");
    return new Set(Array.isArray(value) ? value.filter((id): id is string => typeof id === "string" && ID.test(id)).slice(-3) : []);
  } catch { return new Set(); }
}

export function rememberFreshRetryParents(ids: string[]): void {
  const merged = new Set([...freshRetryParents(), ...ids.filter(id => ID.test(id))]);
  if (merged.size) localStorage.setItem(FRESH_KEY, JSON.stringify([...merged].slice(-3)));
}

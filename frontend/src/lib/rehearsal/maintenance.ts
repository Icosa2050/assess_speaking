import { buildApiUrl, desktopSessionHeaders } from "@/lib/runtime/environment";
import type { FrozenApiRoute } from "@/lib/api/types";
import {
  clearRehearsalMaintenance, commitRehearsals, rehearsalMaintenance,
  snapshotRehearsals, stageRehearsals, withJournalLock,
  type BrowserSnapshot, type BrowserRestore,
} from "./storage";

export type JournalStatus = { transaction: { id: string; phase: string } | null; archived: string[]; completed: string[] };
export type BackupResult = { id: string; filename: string; size_bytes: number; missing: string[] };
export type RestorePreview = { id: string; attempts: number; skipped_attempts: string[]; skipped_rehearsals: string[]; missing: string[]; browser: BrowserSnapshot };
const route = (path: string) => `/v1/journal/${path}` as FrozenApiRoute;
export async function journalRequest<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const binary = body instanceof Blob;
  const response = await fetch(buildApiUrl(route(path)), {
    method: body === undefined ? "GET" : "POST", signal: signal ?? AbortSignal.timeout(path.endsWith("/commit") ? 600000 : 180000),
    headers: { ...desktopSessionHeaders(), ...(binary ? {} : body === undefined ? {} : { "Content-Type": "application/json" }) },
    body: body === undefined ? undefined : binary ? body : JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail?.detail || (typeof error?.detail === "string" ? error.detail : "Journal operation failed. Retry in Settings."));
  }
  return response.json() as Promise<T>;
}
export const journalStatus = () => journalRequest<JournalStatus>("status");
export async function journalBlob(path: string, signal?: AbortSignal): Promise<Blob> {
  const response = await fetch(buildApiUrl(route(path)), { headers: desktopSessionHeaders(), signal });
  if (!response.ok) throw new Error("The backup or staged recording is unavailable. Retry from Settings.");
  return response.blob();
}
export async function createLearnerBackup(allowMissing: boolean, progress: (done: number, total: number) => void, signal: AbortSignal): Promise<BackupResult> {
  return withJournalLock(async access => {
    if (await rehearsalMaintenance()) throw new Error("Finish recovery before creating a backup.");
    const { id } = await journalRequest<{ id: string }>("begin", {}, signal);
    try {
      const snapshot = await snapshotRehearsals();
      const recordings: BrowserSnapshot["recordings"] = [];
      for (const [index, recording] of snapshot.recordings.entries()) {
        signal.throwIfAborted();
        const extension = recording.blob.type.includes("wav") ? ".wav" : recording.blob.type.includes("mp4") ? ".m4a" : recording.blob.type.includes("ogg") ? ".ogg" : ".webm";
        const { file } = await journalRequest<{ file: string }>(`${id}/media?suffix=${extension}`, recording.blob, signal);
        recordings.push({ key: recording.key, file, type: recording.blob.type });
        progress(index + 1, snapshot.recordings.length);
      }
      // Provider routing and worker IDs belong to this installation, not the backup.
      const sessions = snapshot.sessions.map(session => ({ ...session, runtime: { ...session.runtime, baseUrl: "" },
        parts: session.parts.map(({ jobId: _job, audioId: _audio, ...part }) => part) }));
      return await journalRequest<BackupResult>(`${id}/export`, { browser: { sessions, recordings }, allow_missing: allowMissing }, signal);
    } finally {
      await journalRequest(`${id}/abort`, {}).catch(() => { /* Recovery remains visible if cleanup loses its response. */ });
    }
  }, true);
}
export async function previewLearnerRestore(file: File, signal: AbortSignal): Promise<RestorePreview> {
  return withJournalLock(async access => {
    if (await rehearsalMaintenance()) throw new Error("Finish the previous recovery before opening another archive.");
    const { id } = await journalRequest<{ id: string }>("begin", {}, signal);
    try {
      const preview = await journalRequest<RestorePreview>(`${id}/restore`, file, signal);
      const current = await snapshotRehearsals();
      const used = new Set(current.sessions.map(item => item.id));
      current.recordings.forEach(item => used.add(item.key.slice(0, item.key.lastIndexOf(":"))));
      preview.skipped_rehearsals = preview.browser.sessions.filter(item => used.has(item.id)).map(item => item.id);
      preview.browser.sessions = preview.browser.sessions.filter(item => !used.has(item.id));
      preview.browser.recordings = preview.browser.recordings.filter(item => !used.has(item.key.slice(0, item.key.lastIndexOf(":"))));
      await stageRehearsals({ id, phase: "prepared", sessions: [], recordings: [] }, access);
      return preview;
    } catch (error) {
      await journalRequest(`${id}/abort`, {}).catch(() => { /* The persisted journal remains visible for recovery. */ });
      throw error;
    }
  }, true);
}
export async function commitLearnerRestore(preview: RestorePreview, signal: AbortSignal, progress: (done: number, total: number) => void, committing: () => void = () => {}): Promise<void> {
  await withJournalLock(async access => {
    const recordings: BrowserRestore["recordings"] = [];
    try {
      for (const [index, item] of preview.browser.recordings.entries()) {
        const blob = await journalBlob(`${preview.id}/media/${item.file.split("/")[1]}`, signal);
        recordings.push({ key: item.key, blob: new Blob([blob], { type: item.type }) });
        progress(index + 1, preview.browser.recordings.length);
      }
      const sessions = preview.browser.sessions.map(session => ({ ...session, revision: 1, restoredFrom: preview.id,
        preparationEndsAt: undefined, phase: (session.parts.every(part => part.recorded) ? "review" : "ready") as "review" | "ready",
        runtime: { ...session.runtime, baseUrl: "" }, parts: session.parts.map(({ jobId: _job, audioId: _audio, ...part }) => part) }));
      await stageRehearsals({ id: preview.id, phase: "staged", sessions, recordings }, access);
      signal.throwIfAborted();
      committing();
      await journalRequest(`${preview.id}/commit`, {});
      try { await commitRehearsals(access); }
      catch (error) {
        // A rejected IDB transaction definitely did not commit. Compensate the
        // backend now; preserve the journal if rollback itself cannot finish.
        await journalRequest(`${preview.id}/abort`, {});
        await clearRehearsalMaintenance(access, true);
        throw error;
      }
      await journalRequest(`${preview.id}/complete`, {});
      await clearRehearsalMaintenance(access);
    } catch (error) {
      // No blind rollback after a lost completion response: recovery reads the
      // durable backend receipt and the browser's atomic commit marker first.
      throw error;
    }
  }, true);
}
export async function recoverLearnerJournal(rollback = false): Promise<void> {
  await withJournalLock(async access => {
    const status = await journalStatus();
    const local = await rehearsalMaintenance();
    const remote = status.transaction;
    if (local && status.completed.includes(local.id)) {
      if (remote?.id === local.id) await journalRequest(`${remote.id}/complete`, {});
      await clearRehearsalMaintenance(access);
      return;
    }
    if (remote?.phase === "backend_committed" && local?.id === remote.id && local.phase !== "prepared" && !rollback) {
      await commitRehearsals(access);
      await journalRequest(`${remote.id}/complete`, {});
      await clearRehearsalMaintenance(access);
      return;
    }
    if (remote && !status.completed.includes(remote.id)) await journalRequest(`${remote.id}/abort`, {});
    if (local) await clearRehearsalMaintenance(access, true);
  }, true);
}

export async function attemptRemoval(sessionId: string, purge = false, fingerprint?: string): Promise<{ files: number; retained_files: number; size_bytes: number; fingerprint: string }> {
  return withJournalLock(async access => {
    const snapshot = await snapshotRehearsals();
    if (snapshot.sessions.some(session => session.parts.some(part => part.reportId === sessionId || part.retryOf === sessionId))) throw new Error("A saved rehearsal references this attempt. Keep it archived so that rehearsal can still open its report.");
    const browser_references = [...new Set(snapshot.sessions.flatMap(session => session.parts.flatMap(part => [part.reportId, part.retryOf].filter((id): id is string => !!id))))];
    return journalRequest(`attempts/${encodeURIComponent(sessionId)}/${purge ? "purge" : "preview-purge"}`, { fingerprint, browser_references });
  }, true);
}
export async function rehearsalRemoval(session: import("./session").Rehearsal): Promise<void> {
  const { purgeArchivedRehearsal } = await import("./storage");
  await withJournalLock(async access => {
    const { id } = await journalRequest<{ id: string }>("begin", {});
    try { await purgeArchivedRehearsal(session, access); }
    finally { await journalRequest(`${id}/abort`, {}); }
  }, true);
}

export async function journalRecoveryStatus(): Promise<JournalStatus> {
  const status = await journalStatus();
  // Do not create a rehearsal database merely by opening Home. Existing v1
  // stores migrate on their next actual read, preserving populated profiles.
  if (typeof indexedDB.databases !== "function") return status;
  const databases = await indexedDB.databases();
  if (!databases.some(database => database.name === "vostavo-rehearsals")) return status;
  const local = await rehearsalMaintenance();
  return local && !status.transaction ? { ...status, transaction: { id: local.id, phase: "browser_recovery" } } : status;
}

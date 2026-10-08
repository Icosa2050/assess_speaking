import type { Rehearsal } from "./session";

// Audio and its manifest commit together before the UI advances to another part.
// Native IndexedDB avoids base64 copies and localStorage's small synchronous quota.
const openDatabase = () => new Promise<IDBDatabase>((resolve, reject) => {
  let blocked = false;
  const request = indexedDB.open("vostavo-rehearsals", 2);
  request.onupgradeneeded = () => {
    if (!request.result.objectStoreNames.contains("sessions")) request.result.createObjectStore("sessions", { keyPath: "id" });
    if (!request.result.objectStoreNames.contains("recordings")) request.result.createObjectStore("recordings");
    if (!request.result.objectStoreNames.contains("maintenance")) request.result.createObjectStore("maintenance");
    request.result.onversionchange = () => request.result.close();
  };
  request.onsuccess = () => { if (blocked) request.result.close(); else { request.result.onversionchange = () => request.result.close(); resolve(request.result); } };
  request.onerror = () => reject(request.error);
  request.onblocked = () => { blocked = true; reject(new Error("Close other rehearsal tabs and retry saving.")); };
});

async function transact<T>(stores: string[], mode: IDBTransactionMode,
  action: (transaction: IDBTransaction) => IDBRequest<T>): Promise<T> {
  const db = await openDatabase();
  return new Promise<T>((resolve, reject) => {
    let transaction: IDBTransaction;
    try { transaction = db.transaction(stores, mode); }
    catch (cause) { db.close(); reject(cause); return; }
    let request: IDBRequest<T>;
    try { request = action(transaction); }
    catch (cause) { transaction.abort(); db.close(); reject(cause); return; }
    transaction.oncomplete = () => { db.close(); resolve(request.result); };
    transaction.onabort = () => { db.close(); reject(transaction.error ?? new Error("Browser storage transaction failed.")); };
    transaction.onerror = () => { /* onabort handles request and quota failures. */ };
  });
}
export const listRehearsals = async (): Promise<Rehearsal[]> => {
  const sessions = await transact<Rehearsal[]>(["sessions"], "readonly", tx => tx.objectStore("sessions").getAll());
  return sessions.filter(item => item.version === 1 && !item.archivedAt).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
};
export const recordingKey = (session: Rehearsal, index: number) => `${session.id}:${index}`;
export const getRehearsal = (id: string): Promise<Rehearsal | undefined> =>
  transact(["sessions"], "readonly", tx => tx.objectStore("sessions").get(id));
export const loadPartRecording = (session: Rehearsal, index: number): Promise<Blob | undefined> =>
  transact(["recordings"], "readonly", tx => tx.objectStore("recordings").get(recordingKey(session, index)));

// Optimistic revisions also protect against a stale tab saving after another tab
// released its analysis lock. Both stores roll back on quota or conflict failure.
async function updateSession(session: Rehearsal, recording?: { index: number; file: Blob }): Promise<Rehearsal> {
  if (await rehearsalMaintenance()) throw new Error("Finish journal recovery in Settings before saving rehearsals.");
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["sessions", "recordings"], "readwrite");
    const store = tx.objectStore("sessions");
    const request = store.get(session.id);
    const next = { ...session, revision: (session.revision || 0) + 1 };
    let failure: unknown;
    request.onsuccess = () => {
      try {
        const existing = request.result as Rehearsal | undefined;
        if ((existing?.revision || 0) !== (session.revision || 0)) {
          throw new Error("This rehearsal changed in another tab. Reopen it before continuing.");
        }
        {
          if (recording) tx.objectStore("recordings").put(recording.file, recordingKey(session, recording.index));
          store.put(next);
        }
      } catch (cause) { failure = cause; tx.abort(); }
    };
    tx.oncomplete = () => { db.close(); resolve(next); };
    tx.onabort = () => { db.close(); reject(failure || tx.error || new Error("Browser storage transaction failed.")); };
  });
}
export type JournalAccess = { readonly journalAccess: unique symbol };
const activeAccess = new WeakSet<object>();
export const withJournalLock = <T>(action: (access: JournalAccess) => Promise<T>, exclusive = false): Promise<T> => {
  if (!navigator.locks) return Promise.reject(new Error("This browser cannot coordinate journal maintenance. Use the desktop app."));
  return navigator.locks.request("vostavo:journal-maintenance", { mode: exclusive ? "exclusive" : "shared", ifAvailable: true }, async lock => {
    if (!lock) throw new Error("The journal is in use in another tab. Finish that work and retry.");
    const access = Object.freeze({}) as JournalAccess;
    activeAccess.add(access);
    try { return await action(access); }
    finally { activeAccess.delete(access); }
  });
};
export const saveRehearsal = (session: Rehearsal, access?: JournalAccess) => {
  if (access && !activeAccess.has(access)) return Promise.reject(new Error("Journal lock is not held."));
  return access ? updateSession(session) : withJournalLock(() => updateSession(session));
};
export const savePartRecording = (session: Rehearsal, index: number, file: Blob) => withJournalLock(() => updateSession(session, { index, file }));
// Archiving retains all audio and backend reports. Undo restores the same identity.
export const deleteRehearsal = (session: Rehearsal) => saveRehearsal({ ...session, archivedAt: new Date().toISOString() });
export const undoRehearsalArchive = (session: Rehearsal) => saveRehearsal({ ...session, archivedAt: undefined });
export const listArchivedRehearsals = async (): Promise<Rehearsal[]> => (await transact<Rehearsal[]>(["sessions"], "readonly", tx => tx.objectStore("sessions").getAll())).filter(item => !!item.archivedAt);

export type BrowserSnapshot = { sessions: Rehearsal[]; recordings: { key: string; file: string; type: string }[] };
export type BrowserRestore = { id: string; phase: "prepared" | "staged" | "committed"; sessions: Rehearsal[]; recordings: { key: string; blob: Blob }[] };
export const rehearsalMaintenance = () => transact<BrowserRestore | undefined>(["maintenance"], "readonly", tx => tx.objectStore("maintenance").get("restore"));
export async function snapshotRehearsals(): Promise<{ sessions: Rehearsal[]; recordings: { key: string; blob: Blob }[] }> {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["sessions", "recordings"], "readonly");
    const sessions = tx.objectStore("sessions").getAll();
    const keys = tx.objectStore("recordings").getAllKeys();
    const blobs = tx.objectStore("recordings").getAll();
    tx.oncomplete = () => { db.close(); resolve({ sessions: sessions.result, recordings: keys.result.map((key, i) => ({ key: String(key), blob: blobs.result[i] })) }); };
    tx.onabort = () => { db.close(); reject(tx.error); };
  });
}
export async function stageRehearsals(journal: BrowserRestore, access: JournalAccess): Promise<void> {
  if (!activeAccess.has(access)) throw new Error("Journal lock is not held.");
  const current = await snapshotRehearsals();
  if (current.recordings.some(recording => journal.recordings.some(item => item.key === recording.key))) throw new Error("A recording key already exists. Restore cannot overwrite it.");
  if (current.sessions.some(session => journal.sessions.some(item => item.id === session.id))) throw new Error("A rehearsal already exists. Restore adds new identities only.");
  const pending = await rehearsalMaintenance();
  if (pending && (pending.id !== journal.id || pending.phase !== "prepared")) throw new Error("Finish the previous restore first.");
  await transact(["maintenance"], "readwrite", tx => tx.objectStore("maintenance").put(journal, "restore"));
}
export async function commitRehearsals(access: JournalAccess): Promise<void> {
  if (!activeAccess.has(access)) throw new Error("Journal lock is not held.");
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["sessions", "recordings", "maintenance"], "readwrite");
    const request = tx.objectStore("maintenance").get("restore");
    request.onsuccess = () => {
      const journal = request.result as BrowserRestore | undefined;
      if (!journal) { tx.abort(); return; }
      if (journal.phase === "committed") return;
      if (journal.phase !== "staged") { tx.abort(); return; }
      journal.sessions.forEach(session => tx.objectStore("sessions").add(session));
      journal.recordings.forEach(recording => tx.objectStore("recordings").add(recording.blob, recording.key));
      tx.objectStore("maintenance").put({ ...journal, phase: "committed", recordings: [] }, "restore");
    };
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onabort = () => { db.close(); reject(tx.error ?? new Error("Restore failed. Browser storage remains unchanged.")); };
  });
}
export async function clearRehearsalMaintenance(access: JournalAccess, rollback = false): Promise<void> {
  if (!activeAccess.has(access)) throw new Error("Journal lock is not held.");
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["sessions", "recordings", "maintenance"], "readwrite");
    const request = tx.objectStore("maintenance").get("restore");
    request.onsuccess = () => {
      const journal = request.result as BrowserRestore | undefined;
      if (rollback && journal?.phase === "committed") journal.sessions.forEach(session => {
        tx.objectStore("sessions").delete(session.id);
        session.parts.forEach((_, index) => tx.objectStore("recordings").delete(recordingKey(session, index)));
      });
      tx.objectStore("maintenance").delete("restore");
    };
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onabort = () => { db.close(); reject(tx.error); };
  });
}

export async function purgeArchivedRehearsal(session: Rehearsal, access: JournalAccess): Promise<void> {
  if (!activeAccess.has(access)) throw new Error("Journal lock is not held.");
  if (!session.archivedAt || session.parts.some(part => part.jobId && !part.reportId)) throw new Error("Archive a stopped rehearsal without pending analysis before permanent removal.");
  if (await rehearsalMaintenance()) throw new Error("Finish recovery before permanently removing a rehearsal.");
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["sessions", "recordings"], "readwrite");
    let failure: Error | undefined;
    const request = tx.objectStore("sessions").get(session.id);
    request.onsuccess = () => {
      const current = request.result as Rehearsal | undefined;
      if (!current?.archivedAt || current.revision !== session.revision) { failure = new Error("The rehearsal changed. Refresh the archive before removal."); tx.abort(); return; }
      current.parts.forEach((_, index) => tx.objectStore("recordings").delete(recordingKey(current, index)));
      tx.objectStore("sessions").delete(current.id);
    };
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onabort = () => { db.close(); reject(failure || tx.error || new Error("Permanent removal failed; browser storage is unchanged.")); };
  });
}

import type { Rehearsal } from "./session";

// Audio and its manifest commit together before the UI advances to another part.
// Native IndexedDB avoids base64 copies and localStorage's small synchronous quota.
const openDatabase = () => new Promise<IDBDatabase>((resolve, reject) => {
  let blocked = false;
  const request = indexedDB.open("vostavo-rehearsals", 1);
  request.onupgradeneeded = () => {
    request.result.createObjectStore("sessions", { keyPath: "id" });
    request.result.createObjectStore("recordings");
  };
  request.onsuccess = () => { if (blocked) request.result.close(); else resolve(request.result); };
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
  return sessions.filter(item => item.version === 1).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
};
export const recordingKey = (session: Rehearsal, index: number) => `${session.id}:${index}`;
export const getRehearsal = (id: string): Promise<Rehearsal | undefined> =>
  transact(["sessions"], "readonly", tx => tx.objectStore("sessions").get(id));
export const loadPartRecording = (session: Rehearsal, index: number): Promise<Blob | undefined> =>
  transact(["recordings"], "readonly", tx => tx.objectStore("recordings").get(recordingKey(session, index)));

// Optimistic revisions also protect against a stale tab saving after another tab
// released its analysis lock. Both stores roll back on quota or conflict failure.
async function updateSession(session: Rehearsal, recording?: { index: number; file: Blob }, remove = false): Promise<Rehearsal> {
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
        if (remove) {
          session.parts.forEach((_, index) => tx.objectStore("recordings").delete(recordingKey(session, index)));
          store.delete(session.id);
        } else {
          if (recording) tx.objectStore("recordings").put(recording.file, recordingKey(session, recording.index));
          store.put(next);
        }
      } catch (cause) { failure = cause; tx.abort(); }
    };
    tx.oncomplete = () => { db.close(); resolve(next); };
    tx.onabort = () => { db.close(); reject(failure || tx.error || new Error("Browser storage transaction failed.")); };
  });
}
export const saveRehearsal = (session: Rehearsal) => updateSession(session);
export const savePartRecording = (session: Rehearsal, index: number, file: Blob) => updateSession(session, { index, file });
export const deleteRehearsal = (session: Rehearsal) => updateSession(session, undefined, true);

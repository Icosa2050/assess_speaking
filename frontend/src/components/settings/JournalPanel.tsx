import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";
import { isDesktopRuntime, readDesktopRuntimeBridge } from "@/lib/runtime/desktopBridge";
import { attemptRemoval, rehearsalRemoval, createLearnerBackup, previewLearnerRestore, commitLearnerRestore, recoverLearnerJournal, journalRecoveryStatus, journalStatus, journalRequest, journalBlob, type BackupResult, type RestorePreview, type JournalStatus } from "@/lib/rehearsal/maintenance";
import { listArchivedRehearsals, undoRehearsalArchive } from "@/lib/rehearsal/storage";
import type { Rehearsal } from "@/lib/rehearsal/session";

export function JournalPanel({ locale }: { locale: UiLocale }) {
  const t = createTranslator(locale);
  const cache = useQueryClient();
  const controller = useRef<AbortController | null>(null);
  const [committing, setCommitting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [allowMissing, setAllowMissing] = useState(false);
  const [backup, setBackup] = useState<BackupResult | null>(null);
  const [preview, setPreview] = useState<RestorePreview | null>(null);
  const [status, setStatus] = useState<JournalStatus | null>(null);
  const [removal, setRemoval] = useState<{ id: string; kind: "attempt" | "rehearsal"; files: number; retained: number; fingerprint?: string } | null>(null);
  const [archived, setArchived] = useState<Rehearsal[]>([]);
  const run = async (action: (signal: AbortSignal) => Promise<void>) => {
    setError(""); setMessage(""); setBusy(true);
    const abort = new AbortController(); controller.current = abort;
    try { await action(abort.signal); await cache.invalidateQueries(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { controller.current = null; setBusy(false); setCommitting(false); }
  };
  const progress = (done: number, total: number) => setMessage(t("journal.progress", { done, total }));
  const save = async (item: BackupResult) => {
    if (isDesktopRuntime()) {
      const native = readDesktopRuntimeBridge().saveLearnerBackup;
      if (!native) throw new Error(t("settings.support_native_unavailable"));
      const result = await native(item.id);
      setMessage(t(result.status === "saved" ? "journal.saved" : "journal.cancelled"));
    } else {
      const blob = await journalBlob(`backups/${item.id}`);
      const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
      anchor.href = url; anchor.download = item.filename; anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      setMessage(t("journal.download_started"));
    }
  };
  const refresh = async () => { setStatus(await journalRecoveryStatus()); setArchived(await listArchivedRehearsals()); };
  useEffect(() => { let active = true; void journalRecoveryStatus().then(value => { if (active) setStatus(value); }).catch(() => {}); return () => { active = false; controller.current?.abort(); }; }, []);
  return <section style={{ padding: "1.25rem", border: "1px solid #c4d4cd", borderRadius: 8, display: "grid", gap: "0.8rem" }} data-testid="journal-panel">
    <h2>{t("journal.title")}</h2><p>{t("journal.body")}</p><p>{t("journal.limit")}</p>
    {message && <p role="status">{message}</p>}{error && <p role="alert">{error}</p>}
    <label><input type="checkbox" checked={allowMissing} onChange={e => setAllowMissing(e.target.checked)} disabled={busy} /> {t("journal.allow_missing")}</label>
    <button disabled={busy || !!preview} onClick={() => void run(async signal => { const result = await createLearnerBackup(allowMissing, progress, signal); setBackup(result); await save(result); })}>{t("journal.backup")}</button>
    {backup && <button disabled={busy} onClick={() => void run(() => save(backup))}>{t("journal.save_again")}</button>}
    {!!backup?.missing.length && <p>{t("journal.missing", { count: backup.missing.length })}</p>}
    <label>{t("journal.open")} <input type="file" accept=".zip,application/zip" disabled={busy || !!preview} onChange={e => { const file = e.target.files?.[0]; e.target.value = ""; if (file) void run(async signal => { setPreview(await previewLearnerRestore(file, signal)); }); }} /></label>
    {preview && <div><p>{t("journal.preview", { attempts: preview.attempts, rehearsals: preview.browser.sessions.length })}</p><p>{t("journal.restore_note")}</p>
      {!!(preview.skipped_attempts.length + preview.skipped_rehearsals.length) && <p>{t("journal.skipped", { count: preview.skipped_attempts.length + preview.skipped_rehearsals.length })}</p>}
      {!!preview.missing.length && <p>{t("journal.missing", { count: preview.missing.length })}</p>}
      <button disabled={busy} onClick={() => void run(async signal => { try { await commitLearnerRestore(preview, signal, progress, () => setCommitting(true)); } finally { const current = await journalRecoveryStatus(); setStatus(current); if (!current.transaction) setPreview(null); } setPreview(null); setMessage(t("journal.restored")); await refresh(); })}>{t("journal.confirm_restore")}</button>
      <button disabled={busy} onClick={() => void run(async () => { await recoverLearnerJournal(true); setPreview(null); setMessage(t("journal.cancelled")); })}>{t("journal.cancel")}</button>
    </div>}
    {busy && !committing && <button onClick={() => controller.current?.abort()}>{t("journal.cancel")}</button>}
    <button disabled={busy} onClick={() => void run(refresh)}>{t("journal.check_recovery")}</button>
    {status?.transaction && <div><p>{t("journal.recovery_needed")}</p>
      <button disabled={busy} onClick={() => void run(async () => { await recoverLearnerJournal(); setPreview(null); await refresh(); setMessage(t("journal.recovered")); })}>{t("journal.recover")}</button>
      <button disabled={busy} onClick={() => void run(async () => { await recoverLearnerJournal(true); setPreview(null); await refresh(); setMessage(t("journal.rolled_back")); })}>{t("journal.rollback")}</button>
    </div>}
    {!!status?.archived.length && <div><h3>{t("journal.archived_attempts")}</h3>{status.archived.map(id => <p key={id}>{id} <button disabled={busy} onClick={() => void run(async () => { await journalRequest(`attempts/${encodeURIComponent(id)}/undo`, {}); await refresh(); })}>{t("journal.undo")}</button> <button disabled={busy} onClick={() => void run(async () => { const preview = await attemptRemoval(id); setRemoval({ id, kind: "attempt", files: preview.files, retained: preview.retained_files, fingerprint: preview.fingerprint }); })}>{t("journal.remove")}</button></p>)}</div>}
    {!!archived.length && <div><h3>{t("journal.archived_rehearsals")}</h3><p>{t("journal.rehearsal_archive_note")}</p>{archived.map(item => <p key={item.id}>{item.speaker} · {item.createdAt} <button disabled={busy} onClick={() => void run(async () => { await undoRehearsalArchive(item); await refresh(); })}>{t("journal.undo")}</button> <button disabled={busy} onClick={() => setRemoval({ id: item.id, kind: "rehearsal", files: item.parts.filter(part => part.recorded).length, retained: 0 })}>{t("journal.remove")}</button></p>)}</div>}
    {removal && <div><p>{t("journal.removal_preview", { files: removal.files, retained: removal.retained })}</p><p>{t("journal.removal_note")}</p>
      <button disabled={busy} onClick={() => void run(async () => {
        if (removal.kind === "attempt") await attemptRemoval(removal.id, true, removal.fingerprint);
        else { const session = archived.find(item => item.id === removal.id); if (!session) throw new Error("Refresh archived rehearsals first."); await rehearsalRemoval(session); }
        setRemoval(null); await refresh(); setMessage(t("journal.removed"));
      })}>{t("journal.confirm_remove")}</button>
      <button disabled={busy} onClick={() => setRemoval(null)}>{t("journal.cancel")}</button>
    </div>}
  </section>;
}

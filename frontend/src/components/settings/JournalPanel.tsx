import { apiClient } from "@/lib/api/client";
import type { SupportBundleCreateResponse } from "@/lib/api/types";
import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";
import { isDesktopRuntime, readDesktopRuntimeBridge } from "@/lib/runtime/desktopBridge";
import { ownJournalPreview, startFreshJournal, exportJournalDiagnostics, attemptRemoval, rehearsalRemoval, createLearnerBackup, previewLearnerRestore, commitLearnerRestore, recoverLearnerJournal, journalRecoveryStatus, journalStatus, journalRequest, journalBlob, type BackupResult, type RestorePreview, type JournalStatus } from "@/lib/rehearsal/maintenance";
import { listArchivedRehearsals, undoRehearsalArchive } from "@/lib/rehearsal/storage";
import type { Rehearsal } from "@/lib/rehearsal/session";
import { Button } from "@/components/ui/Button";
import { Card, Disclosure } from "@/components/ui/Card";
import { Field } from "@/components/ui/Field";
import { StatusBanner } from "@/components/ui/StatusBanner";

import styles from "./JournalPanel.module.css";

export function JournalPanel({ locale }: { locale: UiLocale }) {
  const t = createTranslator(locale);
  const cache = useQueryClient();
  const controller = useRef<AbortController | null>(null);
  const [confirmFresh, setConfirmFresh] = useState(false);
  const [diagnostics, setDiagnostics] = useState<SupportBundleCreateResponse | null>(null);
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
    finally { controller.current = null; setBusy(false); setCommitting(false); window.dispatchEvent(new Event("journal-recovery-changed")); }
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
  const saveDiagnostics = async (item: SupportBundleCreateResponse) => {
    if (isDesktopRuntime()) {
      const native = readDesktopRuntimeBridge().saveSupportBundle;
      if (!native) throw new Error(t("settings.support_native_unavailable"));
      const result = await native(item.bundle_id);
      setMessage(t(result.status === "saved" ? "settings.support_bundle_saved" : "settings.support_bundle_cancelled"));
    } else {
      const blob = await apiClient.downloadSupportBundle(item.bundle_id);
      const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
      anchor.href = url; anchor.download = item.filename; anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      setMessage(t("settings.support_bundle_download_started"));
    }
  };
  useEffect(() => preview ? ownJournalPreview(preview.id) : undefined, [preview?.id]);
  const pendingTransaction = !!status?.transaction && status.transaction.phase !== "recovery_error";
  const activeTransaction = pendingTransaction && !status?.recovery_retryable;
  const refresh = async () => { setStatus(await journalRecoveryStatus()); setArchived(await listArchivedRehearsals()); };
  const retryRecovery = async (rollback = false) => {
    try {
      if (rollback) await recoverLearnerJournal(true);
      else await recoverLearnerJournal();
    } finally { setPreview(null); await refresh(); }
  };
  useEffect(() => {
    let active = true;
    const failed = (cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : String(cause)); };
    void journalRecoveryStatus().then(value => { if (active) setStatus(value); }).catch(failed);
    void listArchivedRehearsals().then(value => { if (active) setArchived(value); }).catch(failed);
    return () => { active = false; controller.current?.abort(); };
  }, []);
  return <Card as="section" className={styles.panel} data-testid="journal-panel">
    <h2>{t("journal.title")}</h2><p>{t("journal.body")}</p><p>{t("journal.limit")}</p>
    {message && <p role="status">{message}</p>}{error && <StatusBanner title={error} tone="error" urgent />}
    {status?.recovery_error && <div className={styles.actionGroup}>
      <StatusBanner title={status.recovery_error} tone="error" urgent><p>{t("journal.damaged_guidance")}</p></StatusBanner>
      <p>{t("journal.diagnostics_note")}</p>
      <Button variant="primary" id="journal-diagnostics" disabled={busy || activeTransaction} onClick={() => void run(async () => {
        const result = await exportJournalDiagnostics(); setDiagnostics(result); await saveDiagnostics(result);
      })}>{t("journal.diagnostics")}</Button>
      <Button variant="quiet" id="journal-fresh" disabled={busy || activeTransaction} onClick={() => setConfirmFresh(true)}>{t("journal.fresh")}</Button>
      {confirmFresh && <div className={styles.actionGroup} role="group" aria-label={t("journal.confirm_fresh")}>
        <p>{t("journal.fresh_note")}</p>
        <Button variant="danger" disabled={busy || activeTransaction} onClick={() => void run(async () => {
          await startFreshJournal(); setConfirmFresh(false); await refresh(); setMessage(t("journal.fresh_done"));
        })}>{t("journal.confirm_fresh")}</Button>
        <Button variant="secondary" disabled={busy} onClick={() => setConfirmFresh(false)}>{t("journal.cancel")}</Button>
      </div>}
    </div>}
    {diagnostics && <Button disabled={busy} onClick={() => void run(() => saveDiagnostics(diagnostics))}>{t("settings.support_bundle_save_again")}</Button>}
    <label><input type="checkbox" checked={allowMissing} onChange={e => setAllowMissing(e.target.checked)} disabled={busy} /> {t("journal.allow_missing")}</label>
    <Button variant="primary" disabled={busy || !!preview || activeTransaction || !!status?.recovery_error} onClick={() => void run(async signal => { const result = await createLearnerBackup(allowMissing, progress, signal); setBackup(result); await save(result); })}>{t("journal.backup")}</Button>
    {backup && <Button disabled={busy} onClick={() => void run(() => save(backup))}>{t("journal.save_again")}</Button>}
    {!!backup?.missing.length && <p>{t("journal.missing", { count: backup.missing.length })}</p>}
    <Field label={t(status?.recovery_error ? "journal.restore_backup" : "journal.open")} id="journal-restore" type="file" accept=".zip,application/zip" disabled={busy || !!preview || activeTransaction} onChange={e => { const file = e.target.files?.[0]; e.target.value = ""; if (file) void run(async signal => { try { setPreview(await previewLearnerRestore(file, signal)); } finally { await refresh(); } }); }} />
    {preview && <div><p>{t("journal.preview", { attempts: preview.attempts, rehearsals: preview.browser.sessions.length })}</p><p>{t("journal.restore_note")}</p>
      {!!(preview.skipped_attempts.length + preview.skipped_rehearsals.length) && <p>{t("journal.skipped", { count: preview.skipped_attempts.length + preview.skipped_rehearsals.length })}</p>}
      {!!preview.missing.length && <p>{t("journal.missing", { count: preview.missing.length })}</p>}
      <Button variant="danger" disabled={busy} onClick={() => void run(async signal => { try { await commitLearnerRestore(preview, signal, progress, () => setCommitting(true)); } finally { setPreview(null); setStatus(await journalRecoveryStatus()); } setMessage(t("journal.restored")); await refresh(); })}>{t("journal.confirm_restore")}</Button>
      <Button disabled={busy} onClick={() => void run(async () => { await retryRecovery(true); setMessage(t("journal.cancelled")); })}>{t("journal.cancel")}</Button>
    </div>}
    {busy && !committing && <Button onClick={() => controller.current?.abort()}>{t("journal.cancel")}</Button>}
    <Button disabled={busy} onClick={() => void run(refresh)}>{t("journal.check_recovery")}</Button>
    {pendingTransaction && status?.transaction?.id !== preview?.id && <div><p>{t("journal.recovery_needed")}</p>
      <Button disabled={busy} onClick={() => void run(async () => { await retryRecovery(); setMessage(t("journal.recovered")); })}>{t("journal.recover")}</Button>
      <Button disabled={busy} onClick={() => void run(async () => { await retryRecovery(true); setMessage(t("journal.rolled_back")); })}>{t("journal.rollback")}</Button>
    </div>}
    {!!status?.archived.length && <Disclosure title={t("journal.archived_attempts")} open>{status.archived.map(id => <p key={id}>{id} <Button disabled={busy} onClick={() => void run(async () => { await journalRequest(`attempts/${encodeURIComponent(id)}/undo`, {}); await refresh(); })}>{t("journal.undo")}</Button> <Button variant="quiet" disabled={busy} onClick={() => void run(async () => { const preview = await attemptRemoval(id); setRemoval({ id, kind: "attempt", files: preview.files, retained: preview.retained_files, fingerprint: preview.fingerprint }); })}>{t("journal.remove")}</Button></p>)}</Disclosure>}
    {!!archived.length && <Disclosure title={t("journal.archived_rehearsals")} open><p>{t("journal.rehearsal_archive_note")}</p>{archived.map(item => <p key={item.id}>{item.speaker} · {item.createdAt} <Button disabled={busy} onClick={() => void run(async () => { await undoRehearsalArchive(item); await refresh(); })}>{t("journal.undo")}</Button> <Button variant="quiet" disabled={busy} onClick={() => setRemoval({ id: item.id, kind: "rehearsal", files: item.parts.filter(part => part.recorded).length, retained: 0 })}>{t("journal.remove")}</Button></p>)}</Disclosure>}
    {removal && <div><p>{t("journal.removal_preview", { files: removal.files, retained: removal.retained })}</p><p>{t("journal.removal_note")}</p>
      <Button variant="danger" disabled={busy} onClick={() => void run(async () => {
        if (removal.kind === "attempt") await attemptRemoval(removal.id, true, removal.fingerprint);
        else { const session = archived.find(item => item.id === removal.id); if (!session) throw new Error("Refresh archived rehearsals first."); await rehearsalRemoval(session); }
        setRemoval(null); await refresh(); setMessage(t("journal.removed"));
      })}>{t("journal.confirm_remove")}</Button>
      <Button disabled={busy} onClick={() => setRemoval(null)}>{t("journal.cancel")}</Button>
    </div>}
  </Card>;
}

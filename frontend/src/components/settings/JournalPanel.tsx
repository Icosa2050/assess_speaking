import { apiClient } from "@/lib/api/client";
import type { SupportBundleCreateResponse } from "@/lib/api/types";
import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";
import { useAppStore } from "@/lib/state/appStore";
import { isDesktopRuntime, readDesktopRuntimeBridge } from "@/lib/runtime/desktopBridge";
import { ownJournalPreview, startFreshJournal, reconcileFreshJournal, exportJournalDiagnostics, attemptRemoval, rehearsalRemoval, createLearnerBackup, previewLearnerRestore, commitLearnerRestore, recoverLearnerJournal, journalRecoveryStatus, journalStatus, journalRequest, journalBlob, type BackupResult, type RestorePreview, type JournalStatus } from "@/lib/rehearsal/maintenance";
import { listArchivedRehearsals, undoRehearsalArchive } from "@/lib/rehearsal/storage";
import type { Rehearsal } from "@/lib/rehearsal/session";
import { Button } from "@/components/ui/Button";
import { Card, Disclosure } from "@/components/ui/Card";
import { Field } from "@/components/ui/Field";
import { StatusBanner } from "@/components/ui/StatusBanner";

import styles from "./JournalPanel.module.css";

export function JournalPanel({ locale }: { locale: UiLocale }) {
  const t = createTranslator(locale);
  const location = useLocation();
  const beginNewSession = useAppStore((state) => state.beginNewSession);
  const cache = useQueryClient();
  const controller = useRef<AbortController | null>(null);
  const freshDialog = useRef<HTMLDialogElement>(null);
  const messageRef = useRef<HTMLParagraphElement>(null);
  const focusedFragment = useRef<string | null>(null);
  const [confirmFresh, setConfirmFresh] = useState(false);
  const [freshError, setFreshError] = useState("");
  const [diagnostics, setDiagnostics] = useState<SupportBundleCreateResponse | null>(null);
  const [committing, setCommitting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [allowMissing, setAllowMissing] = useState(false);
  const [restoreName, setRestoreName] = useState("");
  const [backup, setBackup] = useState<BackupResult | null>(null);
  const [preview, setPreview] = useState<RestorePreview | null>(null);
  const [status, setStatus] = useState<JournalStatus | null>(null);
  const [removal, setRemoval] = useState<{ id: string; kind: "attempt" | "rehearsal"; files: number; retained: number; fingerprint?: string } | null>(null);
  const [archived, setArchived] = useState<Rehearsal[]>([]);
  const run = async (action: (signal: AbortSignal) => Promise<void>, onError: (detail: string) => void | Promise<void> = setError) => {
    setError(""); setMessage(""); setBusy(true);
    const abort = new AbortController(); controller.current = abort;
    try { await action(abort.signal); await cache.invalidateQueries(); }
    catch (cause) { await onError(cause instanceof Error ? cause.message : String(cause)); }
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
  useEffect(() => { if (preview) document.getElementById("journal-preview-heading")?.focus(); }, [preview]);
  useEffect(() => { if (message && !busy) messageRef.current?.focus(); }, [message, busy]);
  useEffect(() => { if (freshError) document.getElementById("fresh-error")?.focus(); }, [freshError]);
  useEffect(() => { if (error && !confirmFresh) document.getElementById("journal-error")?.focus(); }, [error, confirmFresh]);
  useEffect(() => {
    const dialog = freshDialog.current;
    if (!dialog || !confirmFresh) return;
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    document.getElementById("fresh-cancel")?.focus();
    return () => { if (typeof dialog.close === "function" && dialog.open) dialog.close(); };
  }, [confirmFresh]);
  const cancelFresh = () => { setConfirmFresh(false); setFreshError(""); window.setTimeout(() => document.getElementById("journal-fresh")?.focus(), 0); };
  const pendingTransaction = !!status?.transaction && status.transaction.phase !== "recovery_error";
  const activeTransaction = pendingTransaction && !status?.recovery_retryable;
  useEffect(() => {
    if (focusedFragment.current === location.key) return;
    const target = ({ "#your-data": "your-data-title", "#journal-diagnostics": "journal-diagnostics", "#journal-restore": "journal-restore-action", "#journal-fresh": "journal-fresh" } as Record<string, string>)[location.hash];
    const element = target && document.getElementById(target);
    if (!element) return;
    element.scrollIntoView?.({ block: "start" });
    if (element instanceof HTMLButtonElement && element.disabled) {
      document.getElementById("your-data-title")?.focus();
      return;
    }
    element.focus();
    if (document.activeElement === element) focusedFragment.current = location.key;
  }, [location.hash, location.key, status, busy]);
  const refresh = async () => { setStatus(await journalRecoveryStatus()); setArchived(await listArchivedRehearsals()); };
  const retryRecovery = async (rollback = false) => {
    try {
      if (rollback) await recoverLearnerJournal(true);
      else await recoverLearnerJournal();
    } finally { setPreview(null); setRestoreName(""); await refresh(); }
  };
  useEffect(() => {
    let active = true;
    const failed = (cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : String(cause)); };
    void journalRecoveryStatus().then(value => { if (active) setStatus(value); }).catch(failed);
    void listArchivedRehearsals().then(value => { if (active) setArchived(value); }).catch(failed);
    return () => { active = false; controller.current?.abort(); };
  }, []);
  const stateLabel = status?.recovery_error ? t("your_data.state_damaged") : pendingTransaction ? t("your_data.state_pending") : status ? t("your_data.state_healthy") : error ? t("your_data.state_unavailable") : t("your_data.state_loading");
  return <Card as="section" id="your-data" className={styles.panel} data-testid="journal-panel" aria-labelledby="your-data-title">
    <header className={styles.header}><div><h2 id="your-data-title" tabIndex={-1}>{t("your_data.title")}</h2><p>{t("journal.body")}</p></div><span className={styles.state} data-testid="journal-state" aria-live="polite">{t("your_data.status_heading")}: {stateLabel}</span></header>
    {message && <p ref={messageRef} tabIndex={-1} role="status">{message}</p>}{error && <div id="journal-error" tabIndex={-1}><StatusBanner title={error} tone="error" urgent /></div>}
    {status?.recovery_error && <><StatusBanner title={status.recovery_error} tone="error" urgent /><p>{t("journal.damaged_guidance")}</p></>}
    {status?.recovery_error && <div className={`${styles.actionGroup} ${styles.safeCard}`}>
      <span className={styles.eyebrow}>{t("your_data.safe_to_try")}</span>
      <h3>{t("journal.diagnostics")}</h3>
      <p>{t("your_data.safe_note")}</p><p>{t("journal.diagnostics_note")}</p>
      <Button variant="primary" id="journal-diagnostics" disabled={busy || activeTransaction} onClick={() => void run(async () => {
        const result = await exportJournalDiagnostics(); setDiagnostics(result); await saveDiagnostics(result);
      })}>{t("journal.diagnostics")}</Button>
    </div>}
    {diagnostics && <Button disabled={busy} onClick={() => void run(() => saveDiagnostics(diagnostics))}>{t("settings.support_bundle_save_again")}</Button>}
    <div className={`${styles.actionGroup} ${styles.actionCard}`}>
      <span className={styles.eyebrow}>{t("your_data.needs_file")}</span><h3>{t("your_data.restore_heading")}</h3><p>{t("your_data.restore_intro")}</p>
      <Field className={styles.visuallyHidden} label={t(status?.recovery_error ? "journal.restore_backup" : "journal.open")} id="journal-restore" type="file" tabIndex={-1} accept=".zip,application/zip" disabled={busy || !!preview || activeTransaction} onChange={e => { const file = e.target.files?.[0]; e.target.value = ""; if (file) { setRestoreName(file.name); void run(async signal => { try { setPreview(await previewLearnerRestore(file, signal)); } catch (cause) { setRestoreName(""); throw cause; } finally { await refresh(); } }); } }} />
      <Button id="journal-restore-action" disabled={busy || !!preview || activeTransaction} onClick={() => document.getElementById("journal-restore")?.click()}>{t("your_data.choose_backup")}</Button>
      {restoreName && <p>{restoreName}</p>}
    </div>
    {preview && <div className={`${styles.actionGroup} ${styles.actionCard}`} role="region" aria-labelledby="journal-preview-heading"><h3 id="journal-preview-heading" tabIndex={-1}>{t("journal.preview", { attempts: preview.attempts, rehearsals: preview.browser.sessions.length })}</h3><p>{t("journal.restore_note")}</p>
      {!!(preview.skipped_attempts.length + preview.skipped_rehearsals.length) && <p>{t("journal.skipped", { count: preview.skipped_attempts.length + preview.skipped_rehearsals.length })}</p>}
      {!!preview.missing.length && <p>{t("journal.missing", { count: preview.missing.length })}</p>}
      <Button variant="primary" disabled={busy} onClick={() => { let restoreSucceeded = false; void run(async signal => {
        try { await commitLearnerRestore(preview, signal, progress, () => setCommitting(true)); restoreSucceeded = true; beginNewSession(); }
        finally { setPreview(null); setRestoreName(""); if (!restoreSucceeded) await refresh(); }
        setStatus(null); setMessage(t("journal.restored")); await refresh();
      }, detail => setError(restoreSucceeded ? t("your_data.restore_refresh_failed", { detail }) : detail)); }}>{t("journal.confirm_restore")}</Button>
      <Button disabled={busy} onClick={() => void run(async () => { await retryRecovery(true); setMessage(t("journal.cancelled")); })}>{t("journal.cancel")}</Button>
    </div>}
    {status?.recovery_error && <div className={`${styles.actionGroup} ${styles.dangerCard}`}>
      <span className={styles.eyebrow}>{t("your_data.changes_data")}</span><h3>{t("journal.fresh")}</h3>
      <p>{t("your_data.fresh_intro")}</p>
      <Button variant="danger" id="journal-fresh" disabled={busy || activeTransaction} onClick={() => { setFreshError(""); setConfirmFresh(true); }}>{t("journal.fresh")}</Button>
    </div>}
    {confirmFresh && <dialog ref={freshDialog} className={styles.dialog} aria-labelledby="fresh-dialog-title" aria-describedby="fresh-dialog-consequences" onCancel={event => { event.preventDefault(); if (!busy) cancelFresh(); }}>
      <h2 id="fresh-dialog-title">{t("your_data.fresh_heading")}</h2>
      <div id="fresh-dialog-consequences"><p>{t("your_data.fresh_kept")}</p><p>{t("your_data.fresh_replaced")}</p></div>
      {freshError && <div id="fresh-error" tabIndex={-1}><StatusBanner title={freshError} tone="error" urgent /></div>}
      <div className={styles.dialogActions}>
        <Button id="fresh-cancel" variant="secondary" disabled={busy} onClick={cancelFresh}>{t("your_data.fresh_cancel")}</Button>
        <Button variant="danger" disabled={busy || activeTransaction} onClick={() => {
          setFreshError("");
          const requestId = Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, "0")).join("");
          let browserTransaction: string | undefined;
          let resetSucceeded = false;
          void run(async () => {
            await startFreshJournal(requestId, id => { browserTransaction = id; }); resetSucceeded = true; beginNewSession(); setConfirmFresh(false); setStatus(null); setRestoreName("");
            await refresh(); setMessage(t("journal.fresh_done"));
          }, async detail => {
            if (resetSucceeded) { setError(t("your_data.fresh_refresh_failed", { detail })); return; }
            let result: Awaited<ReturnType<typeof reconcileFreshJournal>>;
            try { result = await reconcileFreshJournal(requestId, browserTransaction); }
            catch {
              try {
                const current = await journalStatus();
                if (current.completed.includes(requestId) && !current.recovery_error) {
                  beginNewSession(); setConfirmFresh(false); setStatus(current); setRestoreName("");
                  setError(current.transaction ? t("your_data.fresh_followup_pending") : t("your_data.fresh_refresh_failed", { detail }));
                  return;
                }
                if (!current.recovery_error && !current.transaction) {
                  setConfirmFresh(false); setStatus(current); setError(t("your_data.fresh_outcome_unverified", { detail })); return;
                }
                if (current.transaction) { setConfirmFresh(false); setStatus(current); setError(detail); return; }
              } catch { /* Keep the original error when status is unavailable. */ }
              setFreshError(detail); return;
            }
            if (!result.applied) {
              setStatus(result.status);
              if (result.status.transaction) { setConfirmFresh(false); setError(detail); }
              else if (!result.status.recovery_error) { setConfirmFresh(false); setError(t("your_data.fresh_outcome_unverified", { detail })); }
              else setFreshError(detail);
              return;
            }
            beginNewSession(); setConfirmFresh(false); setRestoreName("");
            if (result.pendingRecovery) { setStatus(result.status); setError(t("your_data.fresh_followup_pending")); return; }
            setStatus(null);
            if (result.cleanupError) { setError(t("your_data.fresh_refresh_failed", { detail: result.cleanupError })); return; }
            try { await refresh(); setMessage(t("journal.fresh_done")); }
            catch (cause) { setError(t("your_data.fresh_refresh_failed", { detail: cause instanceof Error ? cause.message : String(cause) })); }
          }); }}>{t("journal.confirm_fresh")}</Button>
      </div>
    </dialog>}
    <div className={`${styles.actionGroup} ${styles.actionCard}`}><h3>{t("your_data.backup_heading")}</h3><p>{t("journal.limit")}</p>
      <label><input type="checkbox" checked={allowMissing} onChange={e => setAllowMissing(e.target.checked)} disabled={busy} /> {t("journal.allow_missing")}</label>
      <Button variant="primary" disabled={busy || !!preview || activeTransaction || !!status?.recovery_error} onClick={() => void run(async signal => { const result = await createLearnerBackup(allowMissing, progress, signal); setBackup(result); await save(result); })}>{t("journal.backup")}</Button>
      {backup && <Button disabled={busy} onClick={() => void run(() => save(backup))}>{t("journal.save_again")}</Button>}
      {!!backup?.missing.length && <p>{t("journal.missing", { count: backup.missing.length })}</p>}
    </div>
    {busy && !committing && <Button onClick={() => controller.current?.abort()}>{t("journal.cancel")}</Button>}
    <Button disabled={busy} onClick={() => void run(refresh)}>{t("journal.check_recovery")}</Button>
    {pendingTransaction && status?.transaction?.id !== preview?.id && <div className={styles.actionGroup}><p>{t("journal.recovery_needed")}</p>
      <Button disabled={busy} onClick={() => void run(async () => { await retryRecovery(); setMessage(t("journal.recovered")); })}>{t("journal.recover")}</Button>
      <Button disabled={busy} onClick={() => void run(async () => { await retryRecovery(true); setMessage(t("journal.rolled_back")); })}>{t("journal.rollback")}</Button>
    </div>}
    {!!status?.archived.length && <Disclosure title={<h3>{t("journal.archived_attempts")}</h3>} open>{status.archived.map(id => <p key={id}>{id} <Button disabled={busy} onClick={() => void run(async () => { await journalRequest(`attempts/${encodeURIComponent(id)}/undo`, {}); await refresh(); })}>{t("journal.undo")}</Button> <Button variant="quiet" disabled={busy} onClick={() => void run(async () => { const preview = await attemptRemoval(id); setRemoval({ id, kind: "attempt", files: preview.files, retained: preview.retained_files, fingerprint: preview.fingerprint }); })}>{t("journal.remove")}</Button></p>)}</Disclosure>}
    {!!archived.length && <Disclosure title={<h3>{t("journal.archived_rehearsals")}</h3>} open><p>{t("journal.rehearsal_archive_note")}</p>{archived.map(item => <p key={item.id}>{item.speaker} · {item.createdAt} <Button disabled={busy} onClick={() => void run(async () => { await undoRehearsalArchive(item); await refresh(); })}>{t("journal.undo")}</Button> <Button variant="quiet" disabled={busy} onClick={() => setRemoval({ id: item.id, kind: "rehearsal", files: item.parts.filter(part => part.recorded).length, retained: 0 })}>{t("journal.remove")}</Button></p>)}</Disclosure>}
    {removal && <div className={styles.actionGroup}><p>{t("journal.removal_preview", { files: removal.files, retained: removal.retained })}</p><p>{t("journal.removal_note")}</p>
      <Button variant="danger" disabled={busy} onClick={() => void run(async () => {
        if (removal.kind === "attempt") await attemptRemoval(removal.id, true, removal.fingerprint);
        else { const session = archived.find(item => item.id === removal.id); if (!session) throw new Error("Refresh archived rehearsals first."); await rehearsalRemoval(session); }
        setRemoval(null); await refresh(); setMessage(t("journal.removed"));
      })}>{t("journal.confirm_remove")}</Button>
      <Button disabled={busy} onClick={() => setRemoval(null)}>{t("journal.cancel")}</Button>
    </div>}
  </Card>;
}

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { createTranslator } from "@/lib/i18n";
import { apiClient, ApiClientError } from "@/lib/api/client";
import type { JobStatus } from "@/lib/api/types";
import type { DraftSubmission, SavedDraft } from "@/lib/runtime/desktopBridge";
import { useAppStore, useAppStoreApi } from "./appStore";
import { createDraftStoreAdapter, DraftStoreError, normalizedLearner, sameDraftAttempt, type DraftStoreAdapter } from "./draftStore";
import { clearRetryDraft, readLegacyRetryBrief, readRetryDraft } from "./retryBriefStorage";
import { confirmLocalDraft } from "@/lib/runtime/confirmLocalDraft";
import type { SessionDraft } from "./sessionDraft";

type LegacyBrief = NonNullable<ReturnType<typeof readLegacyRetryBrief>>;
type BootResult = { drafts: SavedDraft[]; unreadableIds: string[]; legacy: LegacyBrief | null; error: DraftStoreError | null };
const BOOT_TIMEOUT_MS = 15_000;
const withinBootDeadline = async <T,>(work: () => Promise<T>, deadline: number): Promise<T> => {
  const remaining = deadline - Date.now();
  if (remaining <= 0) throw new Error("Draft recovery timed out");
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      work(),
      new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("Draft recovery timed out")), remaining); }),
    ]);
  } finally { if (timer) clearTimeout(timer); }
};
type RecoveryContext = {
  supported: boolean;
  active: SavedDraft | null;
  drafts: SavedDraft[];
  hasStoredEntries: boolean;
  recoveredFile: File | null;
  error: DraftStoreError | null;
  saveSetup: (setup: SessionDraft) => Promise<SavedDraft | null>;
  replaceSetup: (setup: SessionDraft, previous: SavedDraft) => Promise<SavedDraft>;
  saveTake: (setup: SessionDraft, file: File, duration: number, method: "record" | "upload", progress: (percent: number) => void) => Promise<SavedDraft>;
  markSubmission: (target: SavedDraft, status: SavedDraft["status"], submission: DraftSubmission) => Promise<SavedDraft>;
  prepareNewAttempt: (speakerId: string) => Promise<boolean>;
  discardActive: () => Promise<void>;
  discardDraft: (id: string) => Promise<boolean>;
  leaveActive: () => void;
  clearRecoveredFile: () => void;
  reloadActiveTake: () => Promise<void>;
  findLearner: (speakerId: string) => SavedDraft | undefined;
  manageDrafts: () => void;
};

const defaultContext: RecoveryContext = {
  supported: false, active: null, drafts: [], hasStoredEntries: false, recoveredFile: null, error: null,
  saveSetup: async () => null,
  replaceSetup: async () => { throw new Error("Local draft storage is unavailable"); },
  saveTake: async () => { throw new Error("Local draft storage is unavailable"); },
  markSubmission: async () => { throw new DraftStoreError("unsupported", "Local draft storage is unavailable"); },
  prepareNewAttempt: async () => true,
  discardActive: async () => undefined,
  discardDraft: async () => true,
  leaveActive: () => undefined,
  clearRecoveredFile: () => undefined,
  reloadActiveTake: async () => undefined,
  findLearner: () => undefined,
  manageDrafts: () => undefined,
};
const Context = createContext<RecoveryContext>(defaultContext);
export const useDraftRecovery = (): RecoveryContext => useContext(Context);

export const DraftRecoveryProvider = ({ children, adapter = createDraftStoreAdapter() }: { children: ReactNode; adapter?: DraftStoreAdapter }) => {
  const stableAdapter = useRef(adapter).current;
  adapter = stableAdapter;
  const appStore = useAppStoreApi();
  const navigate = useNavigate();
  const location = useLocation();
  const locale = useAppStore(state => state.preferences.uiLocale);
  const applySetup = useAppStore(state => state.applySetup);
  const clearRecording = useAppStore(state => state.clearRecording);
  const currentSetup = useAppStore(state => state.draft);
  const assessmentState = useAppStore(state => state.recording.assessmentState);
  const completedReportId = useAppStore(state => state.review.reportId);
  const translate = useMemo(() => createTranslator(locale), [locale]);
  const [booted, setBooted] = useState(!adapter.supported);
  const [choosing, setChoosing] = useState(false);
  const [managing, setManaging] = useState(false);
  const [drafts, setDrafts] = useState<SavedDraft[]>([]);
  const [receiptStatuses, setReceiptStatuses] = useState<Record<string, JobStatus>>({});
  const receiptStatusesRef = useRef<Record<string, JobStatus>>({});
  const bootReconciled = useRef(false);
  const draftsRef = useRef<SavedDraft[]>([]);
  const [unreadableIds, setUnreadableIds] = useState<string[]>([]);
  const [active, setActive] = useState<SavedDraft | null>(null);
  const activeRef = useRef<SavedDraft | null>(null);
  const [recoveredFile, setRecoveredFile] = useState<File | null>(null);
  const [legacy, setLegacy] = useState<LegacyBrief | null>(null);
  const [error, setError] = useState<DraftStoreError | null>(null);
  const [confirmDiscard, setConfirmDiscard] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [completionCleanupRetry, setCompletionCleanupRetry] = useState(0);
  const heading = useRef<HTMLHeadingElement>(null);
  const serial = useRef<Promise<unknown>>(Promise.resolve());
  const bootRun = useRef<Promise<BootResult> | null>(null);
  const confirmedReplacement = useRef<string | null>(null);
  const completionCleanupId = useRef<string | null>(null);
  const setCurrent = (value: SavedDraft | null) => { activeRef.current = value; setActive(value); };
  const enqueue = <T,>(work: () => Promise<T>): Promise<T> => {
    const result = serial.current.then(work, work);
    serial.current = result.catch(() => undefined);
    return result;
  };
  const replaceListing = (saved: SavedDraft) => {
    const previous = draftsRef.current.find(entry => entry.draftId === saved.draftId);
    if (previous?.submission?.assessmentRequestId !== saved.submission?.assessmentRequestId) {
      const next = { ...receiptStatusesRef.current };
      delete next[saved.draftId];
      receiptStatusesRef.current = next;
      setReceiptStatuses(next);
    }
    draftsRef.current = [saved, ...draftsRef.current.filter(d => d.draftId !== saved.draftId)];
    setDrafts(draftsRef.current);
  };
  const replacementWouldReleasePendingTake = (previous: SavedDraft | undefined): boolean => {
    if (!previous?.take) return false;
    const state = appStore.getState();
    return sameDraftAttempt(previous, state.draft) &&
      ["queued", "running", "uncertain"].includes(state.recording.assessmentState);
  };

  const loadBoot = async (): Promise<BootResult> => {
      const deadline = Date.now() + BOOT_TIMEOUT_MS;
      try {
        const listing = await withinBootDeadline(() => adapter.list(), deadline);
        let cleanupError: DraftStoreError | null = null;
        try { await withinBootDeadline(() => adapter.abortAbandoned(listing), deadline); }
        catch (error) { cleanupError = error as DraftStoreError; }
        const brief = readLegacyRetryBrief();
        if (brief && !listing.drafts.some(item => item.speakerId === normalizedLearner(brief.draft.speakerId)) && !listing.unreadableIds.length) {
          try {
            const imported = await withinBootDeadline(() => adapter.importLegacyBrief(brief.draft, brief.createdAt), deadline);
            clearRetryDraft();
            return { drafts: [imported, ...listing.drafts], unreadableIds: listing.unreadableIds, legacy: null, error: cleanupError };
          } catch (importError) {
            return { ...listing, legacy: brief, error: importError as DraftStoreError };
          }
        }
        return { ...listing, legacy: brief, error: cleanupError };
      } catch (bootError) {
        if ((bootError as DraftStoreError).kind === "lost_key") {
          try { return { drafts: [], unreadableIds: await withinBootDeadline(() => adapter.recoveryIds(), deadline), legacy: readLegacyRetryBrief(), error: bootError as DraftStoreError }; }
          catch { /* Keep the original key-loss error visible without offering unverifiable IDs. */ }
        }
        return { drafts: [], unreadableIds: [], legacy: readLegacyRetryBrief(), error: bootError as DraftStoreError };
      }
  };
  const applyBoot = (result: BootResult) => {
    draftsRef.current = result.drafts;
    setDrafts(result.drafts); setUnreadableIds(result.unreadableIds);
    setLegacy(result.legacy); setError(result.error);
    setBooted(true);
    setChoosing(Boolean(result.drafts.length || result.unreadableIds.length || result.legacy || result.error));
  };
  useEffect(() => {
    if (!adapter.supported) return;
    bootRun.current ??= loadBoot();
    let cancelled = false;
    void bootRun.current.then(result => {
      if (cancelled) return;
      applyBoot(result);
    });
    return () => { cancelled = true; };
  }, [adapter]);
  const retryBoot = () => {
    if (busy) return;
    setBusy(true);
    bootRun.current = loadBoot();
    void bootRun.current.then(result => {
      applyBoot(result);
      completionCleanupId.current = null;
      setCompletionCleanupRetry(value => value + 1);
      setBusy(false);
    });
  };

  useEffect(() => { if (booted && choosing) heading.current?.focus(); }, [booted, choosing, drafts.length, unreadableIds.length]);

  const saveSetup = (setup: SessionDraft): Promise<SavedDraft | null> => enqueue(async () => {
    if (!adapter.supported) return null;
    const activeAtStart = activeRef.current;
    const previous = activeRef.current?.speakerId === normalizedLearner(setup.speakerId)
      ? activeRef.current : draftsRef.current.find(item => item.speakerId === normalizedLearner(setup.speakerId));
    try {
      const replacing = previous && confirmedReplacement.current === previous.draftId && !sameDraftAttempt(previous, setup);
      if (replacing && replacementWouldReleasePendingTake(previous)) {
        throw new DraftStoreError("conflict", "Assessment is pending; keep the saved take until its result is known");
      }
      if (previous?.take && !sameDraftAttempt(previous, setup) && !replacing) {
        throw new Error("Confirm replacement of the saved take before starting another attempt");
      }
      const saved = replacing ? await adapter.replaceSetup(setup, previous) : await adapter.saveSetup(setup, previous);
      if (replacing) confirmedReplacement.current = null;
      replaceListing(saved);
      if (activeRef.current !== activeAtStart) throw new DraftStoreError("conflict", "The active draft changed while saving.");
      setCurrent(saved); setError(null);
      return saved;
    } catch (saveError) { confirmedReplacement.current = null; setError(saveError as DraftStoreError); throw saveError; }
  });
  const replaceSetup: RecoveryContext["replaceSetup"] = (setup, previous) => enqueue(async () => {
    try {
      if (replacementWouldReleasePendingTake(previous)) {
        throw new DraftStoreError("conflict", "Assessment is pending; keep the saved take until its result is known");
      }
      const saved = await adapter.replaceSetup(setup, previous);
      confirmedReplacement.current = null;
      setCurrent(saved); replaceListing(saved); setRecoveredFile(null); setError(null);
      return saved;
    } catch (replaceError) { confirmedReplacement.current = null; setError(replaceError as DraftStoreError); throw replaceError; }
  });
  const saveTake: RecoveryContext["saveTake"] = (setup, file, duration, method, progress) => enqueue(async () => {
    let current = activeRef.current;
    if (!sameDraftAttempt(current, setup)) {
      throw new Error("Session setup has not been saved for this attempt");
    }
    try {
      const saved = await adapter.saveTake(current, file, duration, method, progress);
      replaceListing(saved);
      if (activeRef.current?.draftId !== current.draftId || activeRef.current.revision !== current.revision ||
          !sameDraftAttempt(activeRef.current, setup)) throw new DraftStoreError("conflict", "The active draft changed while saving.");
      setCurrent(saved); setError(null);
      return saved;
    } catch (saveError) { setError(saveError as DraftStoreError); throw saveError; }
  });
  const markSubmission: RecoveryContext["markSubmission"] = (target, status, submission) => enqueue(async () => {
    const current = activeRef.current;
    if (!adapter.supported || !current || current.draftId !== target.draftId ||
        current.revision !== target.revision || !sameDraftAttempt(current, target.setup)) {
      throw new DraftStoreError("conflict", "The submitting draft changed; resume it before retrying.");
    }
    const saved = await adapter.markSubmission(target, status, submission);
    if (!saved || saved.draftId !== target.draftId || saved.revision <= target.revision || saved.status !== status ||
        saved.submission?.uploadRequestId !== submission.uploadRequestId ||
        saved.submission?.assessmentRequestId !== submission.assessmentRequestId ||
        !sameDraftAttempt(saved, target.setup)) {
      throw new DraftStoreError("conflict", "The submission receipt was not saved to its draft.");
    }
    replaceListing(saved);
    if (!activeRef.current || activeRef.current.draftId !== target.draftId ||
        !sameDraftAttempt(activeRef.current, target.setup)) {
      throw new DraftStoreError("conflict", "The submitting draft changed; resume it before retrying.");
    }
    setCurrent(saved);
    return saved;
  });
  useEffect(() => {
    if (!booted || !adapter.supported) return;
    const enteringHistoryOrReview = location.pathname.startsWith("/history") || location.pathname.startsWith("/review");
    if (bootReconciled.current && !enteringHistoryOrReview) return;
    bootReconciled.current = true;
    let cancelled = false;
    void enqueue(async () => {
      for (const entry of [...draftsRef.current]) {
        if (cancelled || !entry.submission) continue;
        const receipt = entry.submission;
        try {
          if (receipt.assessmentRequestId) {
            const found = await apiClient.getAssessmentByRequest(receipt.assessmentRequestId, {
              audioId: receipt.audioId, speakerId: entry.setup.speakerId, sharingFingerprint: receipt.requestFingerprint,
            });
            if (cancelled) return;
            if (receipt.assessmentId && receipt.assessmentId !== found.assessment_id) {
              throw new Error("Assessment receipt identity does not match the saved take");
            }
            receiptStatusesRef.current = { ...receiptStatusesRef.current, [entry.draftId]: found.status };
            setReceiptStatuses(receiptStatusesRef.current);
            if (found.status === "completed") {
              if (!found.report_id) throw new Error("Completed assessment report is unavailable");
              const detail = await apiClient.getHistoryDetail(found.report_id).catch(() => {
                throw new Error("Completed assessment report is unavailable");
              });
              const report = detail.payload.report as { session_id?: string } | undefined;
              if (report?.session_id !== found.report_id) throw new Error("Completed assessment report is unavailable");
              const latest = draftsRef.current.find(item => item.draftId === entry.draftId);
              if (latest?.revision !== entry.revision || latest.submission?.assessmentRequestId !== receipt.assessmentRequestId) continue;
              await adapter.discard(entry.draftId);
              draftsRef.current = draftsRef.current.filter(item => item.draftId !== entry.draftId);
              setDrafts(draftsRef.current);
              if (activeRef.current?.draftId === entry.draftId && sameDraftAttempt(activeRef.current, entry.setup)) { setCurrent(null); setRecoveredFile(null); }
              const brief = readRetryDraft();
              if (brief && sameDraftAttempt(entry, brief)) clearRetryDraft();
            } else if (entry.status === "assessment_uncertain" ||
                (["failed", "cancelled"].includes(found.status) && receipt.lastConfirmedPhase !== found.status)) {
              const saved = await adapter.markSubmission(entry, "assessing", {
                ...receipt, assessmentId: found.assessment_id,
                lastConfirmedPhase: found.status === "failed" || found.status === "cancelled" ? found.status : "assessment",
              });
              replaceListing(saved);
              if (activeRef.current?.draftId === entry.draftId && sameDraftAttempt(activeRef.current, entry.setup)) setCurrent(saved);
            }
          } else if (receipt.uploadRequestId && !receipt.audioId) {
            if (!entry.take) continue;
            const found = await apiClient.getUploadByRequest(receipt.uploadRequestId, {
              sha256: entry.take.sha256, originalName: entry.take.originalFilename,
            });
            if (cancelled) return;
            const saved = await adapter.markSubmission(entry, "uploaded", {
              ...receipt, audioId: found.audio_id, lastConfirmedPhase: "upload",
            });
            replaceListing(saved);
            if (activeRef.current?.draftId === entry.draftId && sameDraftAttempt(activeRef.current, entry.setup)) setCurrent(saved);
          }
        } catch (lookupError) {
          if (!(lookupError instanceof ApiClientError && lookupError.responseStatus === 404)) {
            console.warn("Draft receipt reconciliation failed; the saved take remains available.", lookupError);
          }
        }
      }
    });
    return () => { cancelled = true; };
  }, [booted, adapter, location.pathname]);
  const discardActive = () => enqueue(async () => {
    const id = activeRef.current?.draftId;
    if (!id) return;
    await adapter.discard(id);
    if (activeRef.current?.draftId === id) {
      setCurrent(null); setRecoveredFile(null);
    }
    draftsRef.current = draftsRef.current.filter(item => item.draftId !== id);
    setDrafts(draftsRef.current);
    setError(null);
  });
  useEffect(() => {
    if (!adapter.supported || assessmentState !== "completed" || !completedReportId ||
        !sameDraftAttempt(active, currentSetup) || completionCleanupId.current === active.draftId) return;
    const completed = active;
    completionCleanupId.current = completed.draftId;
    void enqueue(async () => {
      if (activeRef.current?.draftId !== completed.draftId) return;
      try {
        await adapter.discard(completed.draftId);
        draftsRef.current = draftsRef.current.filter(item => item.draftId !== completed.draftId);
        setDrafts(draftsRef.current);
        setCurrent(null); setRecoveredFile(null);
        const brief = readRetryDraft();
        if (brief && sameDraftAttempt(completed, brief)) clearRetryDraft();
        setError(null);
      } catch (cleanupError) {
        completionCleanupId.current = null;
        setError(cleanupError as DraftStoreError);
      }
    });
  }, [adapter, active, assessmentState, completedReportId, currentSetup, completionCleanupRetry]);
  const discard = async (id: string): Promise<boolean> => {
    setBusy(true);
    return enqueue(async () => { try {
      await adapter.discard(id);
      draftsRef.current = draftsRef.current.filter(item => item.draftId !== id);
      setDrafts(draftsRef.current);
      const remainingUnreadable = unreadableIds.filter(item => item !== id);
      setUnreadableIds(remainingUnreadable);
      if (activeRef.current?.draftId === id) { setCurrent(null); setRecoveredFile(null); clearRecording(); }
      setConfirmDiscard(null); setError(null);
      if (!draftsRef.current.length && !remainingUnreadable.length && !legacy) { setChoosing(false); setManaging(false); }
      return true;
    } catch (discardError) { setError(discardError as DraftStoreError); return false; }
    finally { setBusy(false); } });
  };
  const prepareNewAttempt = async (speakerId: string): Promise<boolean> => {
    const existing = draftsRef.current.find(item => item.speakerId === normalizedLearner(speakerId));
    if (existing && !await confirmLocalDraft(translate("draft.replace_setup_detail", {
      learner: existing.setup.speakerId,
      task: existing.setup.themeLabel || existing.setup.promptText,
      size: existing.take ? `${(existing.take.size / 1024 / 1024).toFixed(1)} MiB` : "0 MiB",
    }))) return false;
    confirmedReplacement.current = existing?.draftId ?? null;
    if (!existing) setCurrent(null);
    setRecoveredFile(null);
    return true;
  };
  const resume = (entry: SavedDraft) => {
    setBusy(true);
    void enqueue(async () => { try {
      const latest = draftsRef.current.find(item => item.draftId === entry.draftId);
      if (!latest) return;
      if (latest.expiresAt <= Date.now()) {
        await adapter.discard(latest.draftId);
        draftsRef.current = draftsRef.current.filter(item => item.draftId !== entry.draftId);
        setDrafts(draftsRef.current);
        if (!draftsRef.current.length && !unreadableIds.length && !legacy) { setChoosing(false); setManaging(false); }
        return;
      }
      const file = latest.take ? await adapter.loadTake(latest) : null;
      applySetup(latest.setup);
      const status = receiptStatusesRef.current[latest.draftId];
      if (status && latest.submission?.assessmentId) {
        appStore.getState().setRecordingJob({ assessmentId: latest.submission.assessmentId, status,
          phase: status, progress: ["completed", "failed", "cancelled"].includes(status) ? 1 : 0 });
      }
      setCurrent(latest);
      setRecoveredFile(file);
      setChoosing(false);
      setManaging(false);
      setError(null);
      navigate(latest.take || (latest.setup.retryOfSessionId && latest.setup.retryGoalEligible) ? "/speak" : "/session-setup", { replace: true });
    } catch (resumeError) {
      setError(resumeError as DraftStoreError);
      if (["unreadable", "lost_key"].includes((resumeError as DraftStoreError).kind)) {
        setUnreadableIds(previous => [...new Set([...previous, entry.draftId])]);
        draftsRef.current = draftsRef.current.filter(item => item.draftId !== entry.draftId);
        setDrafts(draftsRef.current);
      }
    } finally { setBusy(false); } });
  };
  const context: RecoveryContext = {
    supported: adapter.supported, active, drafts, hasStoredEntries: Boolean(drafts.length || unreadableIds.length), recoveredFile, error,
    saveSetup, replaceSetup, saveTake, markSubmission, prepareNewAttempt, discardActive, discardDraft: discard,
    leaveActive: () => { setCurrent(null); setRecoveredFile(null); setError(null); },
    clearRecoveredFile: () => setRecoveredFile(null),
    reloadActiveTake: () => enqueue(async () => {
      if (!activeRef.current?.take) return;
      try { setRecoveredFile(await adapter.loadTake(activeRef.current)); setError(null); }
      catch (loadError) { setError(loadError as DraftStoreError); }
    }),
    findLearner: speakerId => draftsRef.current.find(item => item.speakerId === normalizedLearner(speakerId)),
    manageDrafts: () => { setManaging(true); setChoosing(true); },
  };
  const pending = choosing && (drafts.length > 0 || unreadableIds.length > 0 || legacy || error || managing);
  const recoveryCard = (
      <Card as="section" data-testid="draft.recovery" style={{ display: "grid", gap: "var(--space-3)" }}>
        <h1 ref={heading} tabIndex={-1}>{translate("draft.resume_title")}</h1>
        <p>{translate("draft.privacy")}</p>
        {drafts.map(entry => <div key={entry.draftId} data-testid="draft.entry">
          <h2>{entry.setup.speakerId} · {entry.setup.themeLabel || entry.setup.promptText}</h2>
          <p>{translate(entry.take ? "draft.recorded_at" : "draft.edited_at", { date: new Date(entry.take?.captureEndedAt ?? entry.updatedAt).toLocaleString(locale) })}</p>
          <p>{translate(entry.take ? "draft.take_saved" : "draft.setup_saved", { date: new Date(entry.expiresAt).toLocaleString(locale) })}</p>
          {entry.expiresAt - Date.now() < 24 * 60 * 60 * 1000 && <p>{translate("draft.expires_soon")}</p>}
          {entry.take && <p>{!entry.submission ? translate("draft.not_sent") : receiptStatuses[entry.draftId] === "failed"
            ? translate("draft.assessment_failed") : receiptStatuses[entry.draftId] === "cancelled"
            ? translate("draft.assessment_cancelled") : receiptStatuses[entry.draftId] === "queued" || receiptStatuses[entry.draftId] === "running"
            ? translate("draft.assessment_pending") : entry.status === "uploaded"
            ? translate("draft.upload_confirmed") : translate("draft.submission_uncertain")}</p>}
          {confirmDiscard === entry.draftId ? <div role="group" aria-label={translate("draft.discard_confirm_title")}>
            <p>{translate("draft.discard_confirm", { learner: entry.setup.speakerId, task: entry.setup.themeLabel, size: entry.take ? `${(entry.take.size / 1024 / 1024).toFixed(1)} MiB` : "0 MiB" })}</p>
            <Button disabled={busy} onClick={() => void discard(entry.draftId)}>{translate("draft.discard_action")}</Button>
            <Button onClick={() => setConfirmDiscard(null)}>{translate("draft.cancel")}</Button>
          </div> : <div style={{ display: "flex", gap: "var(--space-2)" }}>
            <Button variant="primary" disabled={busy} onClick={() => void resume(entry)}>{translate("draft.resume_action")}</Button>
            <Button disabled={busy} onClick={() => setConfirmDiscard(entry.draftId)}>{translate("draft.discard_action")}</Button>
          </div>}
        </div>)}
        {unreadableIds.map(id => <div key={id} data-testid="draft.unreadable">
          <p>{translate("draft.unreadable")}</p>
          {confirmDiscard === id ? <div><p>{translate("draft.unreadable_discard_confirm")}</p>
            <Button disabled={busy} onClick={() => void discard(id)}>{translate("draft.discard_action")}</Button>
            <Button onClick={() => setConfirmDiscard(null)}>{translate("draft.cancel")}</Button></div>
            : <Button onClick={() => setConfirmDiscard(id)}>{translate("draft.discard_action")}</Button>}
        </div>)}
        {legacy && <div data-testid="draft.legacy">
          <p>{translate("draft.legacy_conflict", { learner: legacy.draft.speakerId, date: new Date(legacy.createdAt + 7 * 24 * 60 * 60 * 1000).toLocaleString(locale) })}</p>
          <Button onClick={() => { clearRetryDraft(); setLegacy(null); if (!drafts.length && !unreadableIds.length && !error) { setChoosing(false); setManaging(false); } }}>{translate("draft.discard_legacy")}</Button>
        </div>}
        {error && <p role="alert">{translate(error.kind === "lost_key" || error.kind === "unreadable" ? "draft.unreadable"
          : error.kind === "keychain" ? "draft.keychain_error" : error.kind === "limit" ? "draft.limit_error"
          : error.kind === "format" ? "draft.format_error" : error.kind === "uncertain" ? "draft.save_uncertain" : "draft.storage_error")}</p>}
        {error && !["limit", "format"].includes(error.kind) && <Button disabled={busy} onClick={retryBoot}>{translate("draft.retry_listing")}</Button>}
        <Button variant="quiet" onClick={() => { setChoosing(false); setManaging(false); }}>{translate("draft.continue_without_resume")}</Button>
      </Card>
  );
  return <Context.Provider value={context}>
    {!booted ? <p role="status">{translate("journey.loading")}</p> : pending && !managing ? recoveryCard : <>{managing && recoveryCard}{children}</>}
  </Context.Provider>;
};

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBanner } from "@/components/ui/StatusBanner";
import { createTranslator } from "@/lib/i18n";
import type { SavedDraft } from "@/lib/runtime/desktopBridge";
import { useAppStore, useAppStoreApi } from "./appStore";
import { createDraftStoreAdapter, DraftStoreError, normalizedLearner, sameDraftAttempt, type DraftStoreAdapter } from "./draftStore";
import { clearRetryDraft, readLegacyRetryBrief, readRetryDraft } from "./retryBriefStorage";
import { confirmLocalDraft } from "@/lib/runtime/confirmLocalDraft";
import { forgetRetryMarkers, hasFreshRetryParent, hasRestoredRetryParent } from "@/lib/rehearsal/draftConflicts";
import { apiClient, ApiClientError } from "@/lib/api/client";
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
  prepareNewAttempt: (speakerId: string) => Promise<boolean>;
  discardActive: () => Promise<void>;
  discardDraft: (id: string) => Promise<boolean>;
  leaveActive: () => void;
  clearRecoveredFile: () => void;
  reloadActiveTake: () => Promise<void>;
  findLearner: (speakerId: string) => SavedDraft | undefined;
  manageDrafts: () => void;
  reconcileActiveRetryParent: () => Promise<void>;
};

const defaultContext: RecoveryContext = {
  supported: false, active: null, drafts: [], hasStoredEntries: false, recoveredFile: null, error: null,
  saveSetup: async () => null,
  replaceSetup: async () => { throw new Error("Local draft storage is unavailable"); },
  saveTake: async () => { throw new Error("Local draft storage is unavailable"); },
  prepareNewAttempt: async () => true,
  discardActive: async () => undefined,
  discardDraft: async () => true,
  leaveActive: () => undefined,
  clearRecoveredFile: () => undefined,
  reloadActiveTake: async () => undefined,
  findLearner: () => undefined,
  manageDrafts: () => undefined,
  reconcileActiveRetryParent: async () => undefined,
};
const Context = createContext<RecoveryContext>(defaultContext);
export const useDraftRecovery = (): RecoveryContext => useContext(Context);

export const DraftRecoveryProvider = ({ children, adapter = createDraftStoreAdapter() }: { children: ReactNode; adapter?: DraftStoreAdapter }) => {
  const stableAdapter = useRef(adapter).current;
  adapter = stableAdapter;
  const appStore = useAppStoreApi();
  const navigate = useNavigate();
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
  const draftsRef = useRef<SavedDraft[]>([]);
  const [unreadableIds, setUnreadableIds] = useState<string[]>([]);
  const [active, setActive] = useState<SavedDraft | null>(null);
  const activeRef = useRef<SavedDraft | null>(null);
  const [recoveredFile, setRecoveredFile] = useState<File | null>(null);
  const [legacy, setLegacy] = useState<LegacyBrief | null>(null);
  const [error, setError] = useState<DraftStoreError | null>(null);
  const [historyPendingId, setHistoryPendingId] = useState<string | null>(null);
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
    const previous = activeRef.current?.speakerId === normalizedLearner(setup.speakerId)
      ? activeRef.current : draftsRef.current.find(item => item.speakerId === normalizedLearner(setup.speakerId));
    try {
      const previousId = previous?.draftId;
      const replacing = previous && confirmedReplacement.current === previousId && !sameDraftAttempt(previous, setup);
      if (replacing && replacementWouldReleasePendingTake(previous)) {
        throw new DraftStoreError("conflict", "Assessment is pending; keep the saved take until its result is known");
      }
      if (previous?.take && !sameDraftAttempt(previous, setup) && !replacing) {
        throw new Error("Confirm replacement of the saved take before starting another attempt");
      }
      const saved = replacing ? await adapter.replaceSetup(setup, previous) : await adapter.saveSetup(setup, previous);
      if (replacing) { confirmedReplacement.current = null; if (previousId) forgetRetryMarkers(previousId); }
      setCurrent(saved); replaceListing(saved); setError(null);
      return saved;
    } catch (saveError) { confirmedReplacement.current = null; setError(saveError as DraftStoreError); throw saveError; }
  });
  const replaceSetup: RecoveryContext["replaceSetup"] = (setup, previous) => enqueue(async () => {
    try {
      if (replacementWouldReleasePendingTake(previous)) {
        throw new DraftStoreError("conflict", "Assessment is pending; keep the saved take until its result is known");
      }
      const saved = await adapter.replaceSetup(setup, previous);
      forgetRetryMarkers(previous.draftId);
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
      setCurrent(saved); replaceListing(saved); setError(null);
      return saved;
    } catch (saveError) { setError(saveError as DraftStoreError); throw saveError; }
  });
  const discardActive = () => enqueue(async () => {
    const id = activeRef.current?.draftId;
    if (!id) return;
    await adapter.discard(id);
    forgetRetryMarkers(id);
    setHistoryPendingId(null);
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
        forgetRetryMarkers(completed.draftId);
        setHistoryPendingId(null);
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
      forgetRetryMarkers(id);
      if (activeRef.current?.draftId === id) setHistoryPendingId(null);
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
  const resolveRetryParent = async (entry: SavedDraft): Promise<SavedDraft> => {
    const parent = entry.setup.retryOfSessionId;
    if (!parent) return entry;
    let detached = false;
    try {
      const detail = await apiClient.getHistoryDetail(parent);
      const provenance = detail.payload.restored_from;
      const restoredAt = provenance && typeof provenance === "object" && !Array.isArray(provenance)
        ? (provenance as Record<string, unknown>).restored_at_ms : undefined;
      detached = typeof restoredAt === "number" && Number.isFinite(restoredAt) && entry.createdAt <= restoredAt ||
        Boolean(provenance && restoredAt === undefined && hasRestoredRetryParent(entry.draftId, parent));
    } catch (error) {
      if (error instanceof ApiClientError && error.responseStatus === 404) detached = true;
      else throw new DraftStoreError("history", "Could not verify the retry parent in journal history");
    }
    if (detached) {
      const saved = await adapter.detachRetry(entry);
      forgetRetryMarkers(entry.draftId);
      return saved;
    }
    forgetRetryMarkers(entry.draftId);
    return entry;
  };
  const resume = (entry: SavedDraft) => {
    setBusy(true);
    void enqueue(async () => { try {
      if (entry.expiresAt <= Date.now()) {
        await adapter.discard(entry.draftId);
        forgetRetryMarkers(entry.draftId);
        draftsRef.current = draftsRef.current.filter(item => item.draftId !== entry.draftId);
        setDrafts(draftsRef.current);
        if (!draftsRef.current.length && !unreadableIds.length && !legacy) { setChoosing(false); setManaging(false); }
        return;
      }
      let saved = entry;
      let historyPending = false;
      try { saved = await resolveRetryParent(entry); }
      catch (lookupError) {
        if (lookupError instanceof DraftStoreError && lookupError.kind === "history") historyPending = true;
        else throw lookupError;
      }
      if (saved !== entry) replaceListing(saved);
      const file = saved.take ? await adapter.loadTake(saved) : null;
      applySetup(saved.setup);
      setCurrent(saved);
      setRecoveredFile(file);
      setChoosing(false);
      setManaging(false);
      setError(null);
      setHistoryPendingId(historyPending ? saved.draftId : null);
      navigate(saved.take || (saved.setup.retryOfSessionId && saved.setup.retryGoalEligible) ? "/speak" : "/session-setup", { replace: true });
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
    saveSetup, replaceSetup, saveTake, prepareNewAttempt, discardActive, discardDraft: discard,
    leaveActive: () => { setCurrent(null); setRecoveredFile(null); setError(null); setHistoryPendingId(null); },
    clearRecoveredFile: () => setRecoveredFile(null),
    reloadActiveTake: () => enqueue(async () => {
      if (!activeRef.current?.take) return;
      try { setRecoveredFile(await adapter.loadTake(activeRef.current)); setError(null); }
      catch (loadError) { setError(loadError as DraftStoreError); }
    }),
    findLearner: speakerId => draftsRef.current.find(item => item.speakerId === normalizedLearner(speakerId)),
    manageDrafts: () => { setManaging(true); setChoosing(true); },
    reconcileActiveRetryParent: () => enqueue(async () => {
      const entry = activeRef.current;
      if (!entry?.setup.retryOfSessionId) return;
      let saved: SavedDraft;
      try { saved = await resolveRetryParent(entry); }
      catch (lookupError) {
        if (lookupError instanceof DraftStoreError && lookupError.kind === "history") setHistoryPendingId(entry.draftId);
        throw lookupError;
      }
      setHistoryPendingId(null);
      if (saved !== entry) {
        setCurrent(saved); replaceListing(saved);
        if (sameDraftAttempt(entry, appStore.getState().draft)) {
          appStore.getState().updateDraft({ retryOfSessionId: "", retryGoalEligible: false });
        }
      }
    }),
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
          {entry.take && <p>{entry.submission ? translate("draft.submission_uncertain") : translate("draft.not_sent")}</p>}
          {entry.setup.retryOfSessionId && hasRestoredRetryParent(entry.draftId, entry.setup.retryOfSessionId) && <p>{translate("draft.restored_parent_conflict")}</p>}
          {entry.setup.retryOfSessionId && hasFreshRetryParent(entry.draftId, entry.setup.retryOfSessionId) && <p>{translate("draft.fresh_parent_check")}</p>}
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
          : error.kind === "format" ? "draft.format_error" : error.kind === "uncertain" ? "draft.save_uncertain"
          : error.kind === "history" ? "draft.history_error" : "draft.storage_error")}</p>}
        {error && !["limit", "format"].includes(error.kind) && <Button disabled={busy} onClick={retryBoot}>{translate("draft.retry_listing")}</Button>}
        <Button variant="quiet" onClick={() => { setChoosing(false); setManaging(false); }}>{translate("draft.continue_without_resume")}</Button>
      </Card>
  );
  return <Context.Provider value={context}>
    {!booted ? <p role="status">{translate("journey.loading")}</p> : pending && !managing ? recoveryCard : <>
      {active && historyPendingId === active.draftId && <StatusBanner title={translate("draft.history_error")} tone="warning">
        <Button onClick={() => void context.reconcileActiveRetryParent().catch(() => undefined)}>{translate("draft.retry_history_check")}</Button>
      </StatusBanner>}
      {children}{managing && recoveryCard}</>}
  </Context.Provider>;
};

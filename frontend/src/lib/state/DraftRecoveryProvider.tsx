import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { createTranslator } from "@/lib/i18n";
import type { SavedDraft } from "@/lib/runtime/desktopBridge";
import { useAppStore } from "./appStore";
import { createDraftStoreAdapter, normalizedLearner, sameDraftAttempt, type DraftStoreAdapter, type DraftStoreError } from "./draftStore";
import { clearRetryDraft, readLegacyRetryBrief } from "./retryBriefStorage";
import type { SessionDraft } from "./sessionDraft";

type LegacyBrief = NonNullable<ReturnType<typeof readLegacyRetryBrief>>;
type BootResult = { drafts: SavedDraft[]; unreadableIds: string[]; legacy: LegacyBrief | null; error: DraftStoreError | null };
const BOOT_TIMEOUT_MS = 15_000;
const boundedList = async (adapter: DraftStoreAdapter) => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      adapter.list(),
      new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("Draft listing timed out")), BOOT_TIMEOUT_MS); }),
    ]);
  } finally { if (timer) clearTimeout(timer); }
};
type RecoveryContext = {
  supported: boolean;
  active: SavedDraft | null;
  drafts: SavedDraft[];
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
};

const defaultContext: RecoveryContext = {
  supported: false, active: null, drafts: [], recoveredFile: null, error: null,
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
};
const Context = createContext<RecoveryContext>(defaultContext);
export const useDraftRecovery = (): RecoveryContext => useContext(Context);

export const DraftRecoveryProvider = ({ children, adapter = createDraftStoreAdapter() }: { children: ReactNode; adapter?: DraftStoreAdapter }) => {
  const stableAdapter = useRef(adapter).current;
  adapter = stableAdapter;
  const navigate = useNavigate();
  const locale = useAppStore(state => state.preferences.uiLocale);
  const applySetup = useAppStore(state => state.applySetup);
  const clearRecording = useAppStore(state => state.clearRecording);
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
  const [confirmDiscard, setConfirmDiscard] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const serial = useRef<Promise<unknown>>(Promise.resolve());
  const bootRun = useRef<Promise<BootResult> | null>(null);
  const confirmedReplacement = useRef<string | null>(null);
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

  const loadBoot = async (): Promise<BootResult> => {
      try {
        const listing = await boundedList(adapter);
        let cleanupError: DraftStoreError | null = null;
        try { await adapter.abortAbandoned(listing); }
        catch (error) { cleanupError = error as DraftStoreError; }
        const brief = readLegacyRetryBrief();
        if (brief && !listing.drafts.some(item => item.speakerId === normalizedLearner(brief.draft.speakerId)) && !listing.unreadableIds.length) {
          try {
            const imported = await adapter.importLegacyBrief(brief.draft, brief.createdAt);
            clearRetryDraft();
            return { drafts: [imported, ...listing.drafts], unreadableIds: listing.unreadableIds, legacy: null, error: cleanupError };
          } catch (importError) {
            return { ...listing, legacy: brief, error: importError as DraftStoreError };
          }
        }
        return { ...listing, legacy: brief, error: cleanupError };
      } catch (bootError) {
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
    void bootRun.current.then(result => { applyBoot(result); setBusy(false); });
  };

  useEffect(() => { if (booted && choosing) heading.current?.focus(); }, [booted, choosing, drafts.length, unreadableIds.length]);

  const saveSetup = (setup: SessionDraft): Promise<SavedDraft | null> => enqueue(async () => {
    if (!adapter.supported) return null;
    const previous = activeRef.current?.speakerId === normalizedLearner(setup.speakerId)
      ? activeRef.current : draftsRef.current.find(item => item.speakerId === normalizedLearner(setup.speakerId));
    try {
      const replacing = previous && confirmedReplacement.current === previous.draftId && !sameDraftAttempt(previous, setup);
      if (previous?.take && !sameDraftAttempt(previous, setup) && !replacing) {
        throw new Error("Confirm replacement of the saved take before starting another attempt");
      }
      const saved = replacing ? await adapter.replaceSetup(setup, previous) : await adapter.saveSetup(setup, previous);
      if (replacing) confirmedReplacement.current = null;
      setCurrent(saved); replaceListing(saved); setError(null);
      return saved;
    } catch (saveError) { setError(saveError as DraftStoreError); throw saveError; }
  });
  const replaceSetup: RecoveryContext["replaceSetup"] = (setup, previous) => enqueue(async () => {
    try {
      const saved = await adapter.replaceSetup(setup, previous);
      confirmedReplacement.current = null;
      setCurrent(saved); replaceListing(saved); setRecoveredFile(null); setError(null);
      return saved;
    } catch (replaceError) { setError(replaceError as DraftStoreError); throw replaceError; }
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
    if (activeRef.current?.draftId === id) {
      setCurrent(null); setRecoveredFile(null);
    }
    draftsRef.current = draftsRef.current.filter(item => item.draftId !== id);
    setDrafts(draftsRef.current);
    setError(null);
  });
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
    if (existing && !window.confirm(translate("draft.replace_setup_confirm", { learner: existing.setup.speakerId }))) return false;
    confirmedReplacement.current = existing?.draftId ?? null;
    if (!existing) setCurrent(null);
    setRecoveredFile(null);
    return true;
  };
  const resume = (entry: SavedDraft) => {
    setBusy(true);
    void enqueue(async () => { try {
      if (entry.expiresAt <= Date.now()) {
        await adapter.discard(entry.draftId);
        draftsRef.current = draftsRef.current.filter(item => item.draftId !== entry.draftId);
        setDrafts(draftsRef.current);
        if (!draftsRef.current.length && !unreadableIds.length && !legacy) { setChoosing(false); setManaging(false); }
        return;
      }
      const file = entry.take ? await adapter.loadTake(entry) : null;
      applySetup(entry.setup);
      setCurrent(entry);
      setRecoveredFile(file);
      setChoosing(false);
      setManaging(false);
      setError(null);
      navigate(entry.take || (entry.setup.retryOfSessionId && entry.setup.retryGoalEligible) ? "/speak" : "/session-setup", { replace: true });
    } catch (resumeError) {
      setError(resumeError as DraftStoreError);
      setUnreadableIds(previous => [...new Set([...previous, entry.draftId])]);
      draftsRef.current = draftsRef.current.filter(item => item.draftId !== entry.draftId);
      setDrafts(draftsRef.current);
    } finally { setBusy(false); } });
  };
  const context: RecoveryContext = {
    supported: adapter.supported, active, drafts, recoveredFile, error,
    saveSetup, replaceSetup, saveTake, prepareNewAttempt, discardActive, discardDraft: discard,
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
          {entry.take && <p>{entry.submission ? translate("draft.submission_uncertain") : translate("draft.not_sent")}</p>}
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
        {error && <p role="alert">{translate(error.kind === "keychain" ? "draft.keychain_error" : "draft.storage_error")}</p>}
        {error && <Button disabled={busy} onClick={retryBoot}>{translate("draft.retry_listing")}</Button>}
        <Button variant="quiet" onClick={() => { setChoosing(false); setManaging(false); }}>{translate("draft.continue_without_resume")}</Button>
      </Card>
  );
  return <Context.Provider value={context}>
    {!booted ? <p role="status">{translate("journey.loading")}</p> : pending && !managing ? recoveryCard : <>{children}{managing && recoveryCard}</>}
  </Context.Provider>;
};

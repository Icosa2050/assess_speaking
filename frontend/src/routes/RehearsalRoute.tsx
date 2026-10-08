import { useSharingRoute, SharingSummary } from "@/lib/setup/sharing";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { RecorderPanel } from "@/components/speak/RecorderPanel";
import { RehearsalAudio } from "@/components/speak/RehearsalAudio";
import { apiClient, ApiClientError } from "@/lib/api/client";
import type { AssessmentStatusResponse, JsonRecord } from "@/lib/api/types";
import { createTranslator } from "@/lib/i18n";
import { useAppStore } from "@/lib/state/appStore";
import type { CefrLevel } from "@/lib/state/sessionDraft";
import { clockText, createRehearsal, retryPart, secondsRemaining, type Rehearsal, type RehearsalLanguage } from "@/lib/rehearsal/session";
import { type JournalAccess, withJournalLock, deleteRehearsal, getRehearsal, listRehearsals, loadPartRecording, savePartRecording, saveRehearsal } from "@/lib/rehearsal/storage";
import styles from "./RehearsalRoute.module.css";

const record = (value: unknown): JsonRecord => value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
const waitForPoll = (signal: AbortSignal) => new Promise<void>((resolve, reject) => {
  const abort = () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); };
  const timer = window.setTimeout(() => { signal.removeEventListener("abort", abort); resolve(); }, 1000);
  signal.addEventListener("abort", abort, { once: true });
  if (signal.aborted) abort();
});

export const RehearsalRoute = () => {
  const navigate = useNavigate();
  const setMicrophoneStatus = useAppStore(state => state.setMicrophoneStatus);
  const microphoneSetupPassed = useAppStore(state => state.microphoneSetupPassed);
  const microphoneDeviceId = useAppStore(state => state.microphoneDeviceId);
  const microphoneVoiceProcessing = useAppStore(state => state.microphoneVoiceProcessing);
  const locale = useAppStore(state => state.preferences.uiLocale);
  const draft = useAppStore(state => state.draft);
  const translate = createTranslator(locale);
  const t = (key: string, vars?: Record<string, string | number>) => translate(`rehearsal.${key}`, vars);
  const [language, setLanguage] = useState<RehearsalLanguage>(draft.learningLanguage === "it" ? "it" : "en");
  const [goal, setGoal] = useState<CefrLevel>(draft.cefrLevel);
  const speaker = draft.speakerId;
  const updateDraft = useAppStore(state => state.updateDraft);
  const [saved, setSaved] = useState<Rehearsal[]>([]);
  const [session, setSession] = useState<Rehearsal | null>(null);
  const [, setRemaining] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [recordingActive, setRecordingActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [reports, setReports] = useState<Record<string, JsonRecord>>({});
  const [deleteId, setDeleteId] = useState("");
  const mounted = useRef(true);
  const controller = useRef<AbortController | null>(null);
  const runtime = useQuery({ queryKey: ["runtime"], queryFn: () => apiClient.getRuntime() });
  const settings = useQuery({ queryKey: ["runtime", "settings"], queryFn: () => apiClient.getRuntimeSettings() });

  const sharingQuery = useSharingRoute();

  useEffect(() => {
    mounted.current = true;
    listRehearsals().then(items => { if (mounted.current) setSaved(items); }).catch(() => {
      if (mounted.current) setError(translate("rehearsal.storage_error"));
    });
    return () => { mounted.current = false; controller.current?.abort(); };
  }, []);
  useEffect(() => {
    if (!file) { setPreview(""); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (session?.phase !== "preparation" || !session.preparationEndsAt) return;
    const deadline = session.preparationEndsAt;
    const update = () => setRemaining(secondsRemaining(deadline));
    update(); const timer = window.setInterval(update, 250);
    return () => clearInterval(timer);
  }, [session?.phase, session?.preparationEndsAt]);

  const persist = async (next: Rehearsal, maintenanceHeld?: JournalAccess) => {
    const stored = await saveRehearsal(next, maintenanceHeld);
    next.revision = stored.revision;
    if (mounted.current) {
      setSession(structuredClone(next));
      setSaved(items => [structuredClone(next), ...items.filter(item => item.id !== next.id)].sort((a, b) => b.createdAt.localeCompare(a.createdAt)));
    }
  };
  const operate = async (operation: () => Promise<void>) => {
    setBusy(true); setError("");
    try { await operation(); }
    catch (cause) { if (mounted.current && !(cause instanceof DOMException && cause.name === "AbortError")) setError(cause instanceof DOMException && cause.name === "QuotaExceededError" ? t("storage_error") : cause instanceof Error ? cause.message : t("storage_error")); }
    finally { if (mounted.current) { setBusy(false); setStatus(""); } }
  };
  const open = async (savedSession: Rehearsal) => {
    const next = await getRehearsal(savedSession.id);
    if (!next) throw new Error(t("recording_missing"));
    setSaved(items => [next, ...items.filter(item => item.id !== next.id)]);
    setFile(null); setReports({}); setSession(next); setError(""); setDeleteId("");
    const results = await Promise.allSettled(next.parts.filter(part => part.reportId).map(async part => {
      const detail = await apiClient.getHistoryDetail(part.reportId!);
      return [part.reportId!, detail.payload] as const;
    }));
    if (!mounted.current) return;
    setReports(Object.fromEntries(results.flatMap(result => result.status === "fulfilled" ? [result.value] : [])));
    if (results.some(result => result.status === "rejected")) setError(t("report_missing"));
  };
  const start = () => operate(async () => {
    if (!runtime.data?.configured || !speaker.trim()) return;
    const next = createRehearsal(language, goal, speaker, { provider: runtime.data.provider || "", model: runtime.data.model || "",
      baseUrl: runtime.data.base_url || "", whisper: settings.data?.whisper_model || "large-v3", feedbackLanguage: locale });
    await persist(next); setReports({}); setFile(null);
  });
  const remaining = session?.preparationEndsAt ? secondsRemaining(session.preparationEndsAt) : 0;
  const index = session?.parts.findIndex(part => !part.recorded) ?? -1;
  const currentPart = session && index >= 0 ? session.parts[index] : undefined;
  const saveRecording = () => operate(async () => {
    if (!session || !file || index < 0) return;
    const next = { ...session, parts: session.parts.map((part, i) => i === index ? { ...part, recorded: true } : part) };
    if (next.parts.every(part => part.recorded)) next.phase = "review";
    const stored = await savePartRecording(next, index, file);
    next.revision = stored.revision;
    setFile(null); setSession(next); setSaved(items => [next, ...items.filter(item => item.id !== next.id)]);
  });
  const analyse = () => operate(async () => {
    if (!session) return;
    const abort = new AbortController(); controller.current = abort;
    if (!navigator.locks) throw new Error(t("locking_unavailable"));
    await withJournalLock(access => navigator.locks.request(`rehearsal:${session.id}`, { ifAvailable: true }, async lock => {
    if (!lock) throw new Error(t("locked"));
    let current = await getRehearsal(session.id);
    if (!current) throw new Error(t("recording_missing"));
    for (let i = 0; i < current.parts.length; i++) {
      abort.signal.throwIfAborted();
      const part = current.parts[i];
      if (part.reportId) continue;
      setStatus(t("analysing", { part: i + 1, total: current.parts.length }));
      if (!part.jobId) {
        const refreshed = await sharingQuery.refetch();
        if (!refreshed.data?.available || !sharingQuery.data?.available || refreshed.data.fingerprint !== sharingQuery.data.fingerprint) throw new Error(translate("sharing.changed"));
        current.runtime = { ...current.runtime, provider: refreshed.data.analysis.provider, model: refreshed.data.analysis.model, baseUrl: "", whisper: refreshed.data.audio.local ? refreshed.data.audio.model : current.runtime.whisper };
        if (!part.audioId) {
          const blob = await loadPartRecording(current, i);
          if (!blob) throw new Error(t("recording_missing"));
          const limits = await apiClient.getUploadLimits({ signal: abort.signal });
          if (blob.size > limits.available_bytes) throw new Error(translate("speak.upload_disk_error"));
          const upload = await apiClient.uploadAudio(new File([blob], `rehearsal-${i + 1}.${blob.type.includes("mp4") ? "m4a" : blob.type.includes("ogg") ? "ogg" : "webm"}`, { type: blob.type }), { signal: abort.signal });
          part.audioId = upload.audio_id; await persist(current, access);
        }
        abort.signal.throwIfAborted();
        let job;
        try { job = await apiClient.createAssessment({ sharing_fingerprint: refreshed.data.fingerprint, audio_id: part.audioId, whisper: refreshed.data.audio.local ? refreshed.data.audio.model : current.runtime.whisper,
          provider: refreshed.data.analysis.provider, llm_model: refreshed.data.analysis.model,
          expected_language: current.language, feedback_language: current.runtime.feedbackLanguage, speaker_id: current.speaker,
          task_family: part.taskFamily || "free_monologue", theme: part.prompt, target_duration_sec: part.durationSec, target_cefr: current.goal,
          prompt_id: part.id, prompt_text: part.prompt, retry_of_session_id: part.retryOf || "",
          label: `rehearsal-${current.id}-part-${i + 1}`, notes: "App-authored solo rehearsal; no partner interaction assessed." }); }
        catch (cause) {
          if (cause instanceof ApiClientError && [400, 404, 410, 422].includes(cause.responseStatus)) {
            part.audioId = undefined; await persist(current, access);
          }
          throw cause;
        }
        // Persist the accepted job even if navigation happened during creation.
        part.jobId = job.assessment_id; await persist(current, access);
      }
      let result: AssessmentStatusResponse;
      do {
        abort.signal.throwIfAborted();
        try { result = await apiClient.getAssessmentStatus(part.jobId, { signal: abort.signal }); }
        catch (cause) {
          if (cause instanceof ApiClientError && cause.responseStatus === 404) {
            part.jobId = undefined; part.audioId = undefined; await persist(current, access);
          }
          throw cause;
        }
        if (result.status === "queued" || result.status === "running") await waitForPoll(abort.signal);
      } while (result.status === "queued" || result.status === "running");
      if (result.status !== "completed" || !result.payload) {
        part.jobId = undefined; part.audioId = undefined; await persist(current, access);
        throw new Error(result.error?.detail || t("analysis_failed"));
      }
      const report = record(result.payload.report);
      if (typeof report.session_id !== "string" || !report.session_id) {
        part.jobId = undefined; part.audioId = undefined; await persist(current, access);
        throw new Error(t("analysis_failed"));
      }
      part.reportId = report.session_id; await persist(current, access);
      if (mounted.current) setReports(items => ({ ...items, [part.reportId!]: result.payload! }));
      current = structuredClone(current);
    }
    }));
  });

  if (!session) return <section className={styles.page} data-testid="rehearsal-screen">
    <header><p className={styles.eyebrow}>{t("eyebrow")}</p><h1>{t("title")}</h1><p>{t("body")}</p></header>
    {error && <p role="alert">{error}</p>}
    <div className={styles.card}>
      <label>{t("learner")}<input data-testid="rehearsal-speaker" value={speaker} onChange={event => updateDraft({ speakerId: event.target.value })} /></label>
      <label>{t("language")}<select data-testid="rehearsal-language" value={language} onChange={event => { setLanguage(event.target.value as RehearsalLanguage); updateDraft({ learningLanguage: event.target.value, learningLanguageLabel: event.target.value === "it" ? "Italiano" : "English" }); }}><option value="en">English</option><option value="it">Italiano</option></select></label>
      <label>{t("goal")}<select data-testid="rehearsal-goal" value={goal} onChange={event => { setGoal(event.target.value as CefrLevel); updateDraft({ cefrLevel: event.target.value as CefrLevel }); }}>{["B1", "B2", "C1"].map(level => <option key={level}>{level}</option>)}</select></label>
      <p>{t("outline")}</p><p>{t("scope")}</p>
      {!runtime.data?.configured && <Link to="/runtime-setup">{t("configure")}</Link>}
      {!microphoneSetupPassed && <p>{translate("speak.microphone_setup_required")} <Link to="/runtime-setup#runtime-setup-microphone">{translate("runtime_setup.setup_guide_check_microphone")}</Link></p>}
      <button data-testid="rehearsal-create" disabled={busy || !speaker.trim() || !runtime.data?.configured || !settings.data || !microphoneSetupPassed} onClick={start}>{t("create")}</button>
    </div>
    <h2>{t("saved")}</h2>
    {saved.length === 0 && <p>{t("empty")}</p>}
    {saved.map(item => <div className={styles.saved} key={item.id}>
      <button disabled={busy} onClick={() => operate(() => open(item))}>{item.language.toUpperCase()} · {item.goal} · {item.speaker} · {new Date(item.createdAt).toLocaleString(locale)} · {item.parts.filter(part => part.recorded).length}/{item.parts.length}</button>
      <button disabled={busy} onClick={() => deleteId === item.id ? operate(async () => { await deleteRehearsal(item); setSaved(items => items.filter(value => value.id !== item.id)); setDeleteId(""); }) : setDeleteId(item.id)}>{deleteId === item.id ? t("confirm_delete") : t("delete")}</button>
    </div>)}
  </section>;

  const recorded = session.parts.filter(part => part.recorded).length;
  return <section className={styles.page} data-testid="rehearsal-screen">
    <header><p className={styles.eyebrow}>{session.language.toUpperCase()} · {t("goal")} {session.goal} · {session.speaker}</p><h1>{t("title")}</h1>
      <p>{t("feedback_model", { provider: session.runtime.provider, model: session.runtime.model })}</p>
      <p>{t("saved_notice")}</p><progress value={recorded} max={session.parts.length} aria-label={t("parts_saved")} /> <span>{recorded}/{session.parts.length} {t("parts_saved")}</span>
    </header>
    {error && <p role="alert">{error}</p>}
    {busy && <p role="status">{status || t("saving")}</p>}
    {busy && status && <button data-testid="rehearsal-pause-analysis" onClick={() => controller.current?.abort()}>{t("pause_analysis")}</button>}
    {(session.phase === "ready" || session.phase === "preparation") && <div className={styles.card}>
      <h2>{t("prepare")}</h2><ol>{session.parts.map(part => <li key={part.id}>{part.prompt} <strong>{clockText(part.durationSec)}</strong></li>)}</ol>
      {session.phase === "ready" ? <button data-testid="rehearsal-prepare" disabled={busy} onClick={() => operate(() => persist({ ...session, phase: "preparation", preparationEndsAt: Date.now() + session.preparationSec * 1000 }))}>{t("start_preparation", { minutes: session.preparationSec / 60 })}</button>
        : <><p className={styles.clock} role="timer" data-testid="rehearsal-preparation-clock">{clockText(remaining)}</p><button data-testid="rehearsal-speak" disabled={busy} onClick={() => operate(() => persist({ ...session, phase: "speaking" }))}>{remaining === 0 ? t("begin_speaking") : t("finish_preparation")}</button></>}
    </div>}
    {session.phase === "speaking" && currentPart && <div className={styles.card}>
      <h2>{t("part", { number: index + 1, total: session.parts.length })} · {clockText(currentPart.durationSec)}</h2><p className={styles.prompt}>{currentPart.prompt}</p><p>{t("record_help")}</p>
      <fieldset disabled={busy}><RecorderPanel onMicrophoneStatusChange={setMicrophoneStatus} key={`${session.id}-${index}`} canRemove={false} inputMode="record" onRecordingActiveChange={setRecordingActive} allowUpload={false} maxSeconds={currentPart.durationSec}
        microphoneSetupPassed={microphoneSetupPassed} microphoneDeviceId={microphoneDeviceId} microphoneVoiceProcessing={microphoneVoiceProcessing}
        onSetupMicrophone={() => navigate("/runtime-setup#runtime-setup-microphone")}
        onInputModeChange={() => undefined} onFileSelected={setFile} onRemove={() => setFile(null)} previewUrl={preview}
        showReadyCheckpoint={Boolean(file)} statusMessage={t("record_help")} statusTone="info" translate={translate} /></fieldset>
      <button data-testid="rehearsal-save-part" disabled={busy || !file} onClick={saveRecording}>{t("save_part")}</button>
    </div>}
    {session.phase === "review" && <>
      <SharingSummary route={sharingQuery.data} locale={locale} />
      <div className={styles.card}><h2>{t("review")}</h2><p>{t("review_body")}</p>
        {session.parts.some(part => !part.reportId) && <button data-testid="rehearsal-analyse" disabled={busy || (!sharingQuery.data?.available && session.parts.some(part => !part.reportId && !part.jobId))} onClick={analyse}>{t("analyse")}</button>}
      </div>
      {session.parts.map((part, i) => {
        const report = record(reports[part.reportId || ""]?.report);
        const metrics = record(report.metrics); const coaching = record(report.coaching);
        const scores = record(report.scores); const warnings = Array.isArray(report.warnings) ? report.warnings : [];
        return <article className={styles.card} data-testid={`rehearsal-result-${i}`} key={part.id}>
          <h3>{t("part", { number: i + 1, total: session.parts.length })} · {clockText(part.durationSec)}</h3><p>{part.prompt}</p>
          <RehearsalAudio session={session} index={i} label={t("playback")} unavailable={t("audio_missing")} downloadLabel={translate("speak.download_recording")} />
          {part.reportId ? <>
            <dl className={styles.metrics}>{[["duration", metrics.duration_sec], ["wpm", metrics.wpm], ["score", record(report.eligibility).state && record(report.eligibility).state !== "assessable" ? null : scores.final]].map(([label, value]) => <div key={String(label)}><dt>{t(String(label))}</dt><dd>{typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) : "—"}</dd></div>)}</dl>
            {(!report.rubric || warnings.includes("coaching_unavailable")) && <p>{translate("review.general_practice_tips")}</p>}
            {warnings.includes("transcript_uncertain") && <p>{translate("review.transcript_uncertain")}</p>}
            <p>{String(coaching.coach_summary || t("report_missing"))}</p><strong>{String(coaching.next_focus || "")}</strong>
            <button data-testid={`rehearsal-retry-${i}`} disabled={busy} onClick={() => operate(async () => { const next = retryPart(session, i); await persist(next); setReports({}); setFile(null); })}>{t("retry")}</button>
          </> : <p>{part.jobId ? t("in_progress") : t("awaiting_analysis")}</p>}
        </article>;
      })}
    </>}
    <button data-testid="rehearsal-back" disabled={busy || Boolean(file) || recordingActive} onClick={() => { setSession(null); setError(""); }}>{t("back")}</button>
    {session.phase === "speaking" && <p>{t("leave_notice")}</p>}
    <Link to="/history">{t("history")}</Link>
  </section>;
};

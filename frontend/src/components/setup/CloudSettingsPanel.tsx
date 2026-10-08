import { CurrentSharingSummary } from "@/lib/setup/sharing";
import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { buildApiUrl, desktopSessionHeaders } from "@/lib/runtime/environment";
import type { RuntimeSettingsConnection } from "@/lib/api/types";
import type { UiLocale } from "@/lib/state/sessionDraft";

export type CloudSettings = {
  version: number; asr_provider: string; asr_connection_id: string; asr_model: string;
  openrouter_modes: Record<string, string>; fallback_connection_id: string; paid_fallback_enabled: boolean;
  monthly_budget_usd: number; max_output_tokens: number;
};
type CloudResponse = {settings: CloudSettings; spending: {spent_usd: number; reserved_usd: number; unresolved_requests: string[]}};
export async function cloudRequest<T>(path = "", method = "GET", body?: object): Promise<T> {
  const response = await fetch(buildApiUrl(`/v1/runtime/cloud${path}`), {
    method, headers: {...desktopSessionHeaders(), "X-Vostavo-Client": "desktop", "Content-Type": "application/json"},
    body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(30_000),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : value.detail?.detail || "Cloud settings could not be updated.");
  return value as T;
}
const words: Record<UiLocale, string[]> = {
  en: ["Cloud services", "Transcription", "Local Whisper", "Groq cloud transcription", "Groq account", "Speech model", "OpenRouter access", "Current settings", "Free models only", "Paid, with spending controls", "Allow paid fallback", "Fallback account", "Monthly OpenRouter app budget (USD)", "Save cloud settings", "Saved", "Connect OpenRouter", "Continue in your browser", "Cancel", "Groq is free only on its Free plan. Managed paid OpenRouter requests require a provider key limit. Disconnect removes the local key; revoke it separately in OpenRouter.", "Spent / reserved (USD)", "Select a saved connection", "Choose an OpenRouter model in the connection form after login.", "Verify uncertain spending in the provider dashboard before retrying paid requests.", "Provider-confirmed cost (USD)", "I checked the actual charge in the provider dashboard", "Reconcile spending", "Spending controls apply to OpenRouter connections set to Paid. Current settings retains the connection’s existing billing behavior."],
  it: ["Servizi cloud", "Trascrizione", "Whisper locale", "Trascrizione cloud Groq", "Account Groq", "Modello vocale", "Accesso OpenRouter", "Impostazioni attuali", "Solo modelli gratuiti", "A pagamento, con limiti di spesa", "Consenti alternativa a pagamento", "Account alternativo", "Budget mensile OpenRouter dell’app (USD)", "Salva impostazioni cloud", "Salvato", "Collega OpenRouter", "Continua nel browser", "Annulla", "Groq è gratuito solo con il piano Free. Le richieste OpenRouter con limiti di spesa richiedono un limite sulla chiave. La disconnessione rimuove la chiave locale; revocala anche in OpenRouter.", "Speso / riservato (USD)", "Scegli una connessione salvata", "Dopo l’accesso, scegli un modello OpenRouter nel modulo della connessione.", "Verifica le spese incerte nel pannello del servizio prima di riprovare.", "Costo confermato dal servizio (USD)", "Ho verificato l’addebito effettivo nel pannello del servizio", "Riconcilia spesa", "I limiti di spesa si applicano alle connessioni OpenRouter impostate a pagamento. Impostazioni attuali mantiene il comportamento di fatturazione esistente."],
  de: ["Cloud-Dienste", "Transkription", "Lokales Whisper", "Groq Cloud-Transkription", "Groq-Konto", "Sprachmodell", "OpenRouter-Zugriff", "Aktuelle Einstellungen", "Nur kostenlose Modelle", "Kostenpflichtig mit Ausgabenlimit", "Kostenpflichtige Alternative erlauben", "Alternatives Konto", "Monatliches OpenRouter-App-Budget (USD)", "Cloud-Einstellungen speichern", "Gespeichert", "OpenRouter verbinden", "Im Browser fortfahren", "Abbrechen", "Groq ist nur im Free-Tarif kostenlos. Verwaltete kostenpflichtige OpenRouter-Anfragen benötigen ein Schlüssellimit. Trennen entfernt den lokalen Schlüssel; widerrufe ihn zusätzlich bei OpenRouter.", "Ausgegeben / reserviert (USD)", "Gespeicherte Verbindung wählen", "Nach der Anmeldung ein OpenRouter-Modell im Verbindungsformular wählen.", "Unklare Ausgaben im Anbieter-Dashboard vor dem erneuten Versuch prüfen.", "Beim Anbieter bestätigte Kosten (USD)", "Ich habe die tatsächliche Gebühr im Anbieter-Dashboard geprüft", "Ausgaben abgleichen", "Ausgabenlimits gelten für OpenRouter-Verbindungen mit kostenpflichtigem Zugriff. Aktuelle Einstellungen behält die bestehende Abrechnung bei."],
  es: ["Servicios en la nube", "Transcripción", "Whisper local", "Transcripción Groq en la nube", "Cuenta Groq", "Modelo de voz", "Acceso OpenRouter", "Configuración actual", "Solo modelos gratuitos", "De pago, con límites de gasto", "Permitir alternativa de pago", "Cuenta alternativa", "Presupuesto mensual OpenRouter de la app (USD)", "Guardar configuración", "Guardado", "Conectar OpenRouter", "Continuar en el navegador", "Cancelar", "Groq es gratuito solo con el plan Free. Las solicitudes OpenRouter con límites de gasto requieren un límite en la clave. Desconectar elimina la clave local; revócala también en OpenRouter.", "Gastado / reservado (USD)", "Elegir conexión guardada", "Después de iniciar sesión, elige un modelo OpenRouter en el formulario.", "Verifica los gastos inciertos en el panel del proveedor antes de reintentar.", "Coste confirmado por el proveedor (USD)", "He comprobado el cargo real en el panel del proveedor", "Conciliar gastos", "Los límites de gasto se aplican a conexiones OpenRouter configuradas de pago. Configuración actual conserva la facturación existente."],
  fr: ["Services cloud", "Transcription", "Whisper local", "Transcription cloud Groq", "Compte Groq", "Modèle vocal", "Accès OpenRouter", "Paramètres actuels", "Modèles gratuits uniquement", "Payant, avec limites de dépenses", "Autoriser une alternative payante", "Compte alternatif", "Budget mensuel OpenRouter de l’app (USD)", "Enregistrer les paramètres", "Enregistré", "Connecter OpenRouter", "Continuer dans le navigateur", "Annuler", "Groq est gratuit uniquement avec le forfait Free. Les requêtes OpenRouter payantes gérées exigent une limite sur la clé. Déconnecter supprime la clé locale ; révoquez-la aussi dans OpenRouter.", "Dépensé / réservé (USD)", "Choisir une connexion enregistrée", "Après connexion, choisissez un modèle OpenRouter dans le formulaire.", "Vérifiez les dépenses incertaines dans le tableau de bord du fournisseur avant de réessayer.", "Coût confirmé par le fournisseur (USD)", "J’ai vérifié le montant facturé dans le tableau de bord du fournisseur", "Rapprocher les dépenses", "Les limites de dépenses s’appliquent aux connexions OpenRouter configurées comme payantes. Paramètres actuels conserve la facturation existante."],
};
const recoveryCopy = {
  en: ["Retry loading", "Loading cloud settings…", "This connection is available for this app session only. Reconnect after restarting.", "A selected account is unavailable. Choose a saved account before saving."],
  it: ["Riprova caricamento", "Caricamento impostazioni cloud…", "Questa connessione è disponibile solo in questa sessione. Ricollegala dopo il riavvio.", "Un account selezionato non è disponibile. Scegli un account salvato prima di salvare."],
  de: ["Erneut laden", "Cloud-Einstellungen werden geladen…", "Diese Verbindung ist nur in dieser Sitzung verfügbar. Nach dem Neustart erneut verbinden.", "Ein gewähltes Konto ist nicht verfügbar. Vor dem Speichern ein gespeichertes Konto wählen."],
  es: ["Reintentar carga", "Cargando configuración…", "Esta conexión está disponible solo en esta sesión. Vuelve a conectarla tras reiniciar.", "Una cuenta seleccionada no está disponible. Elige una cuenta guardada antes de guardar."],
  fr: ["Réessayer le chargement", "Chargement des paramètres…", "Cette connexion est disponible uniquement pour cette session. Reconnectez-la après le redémarrage.", "Un compte sélectionné est indisponible. Choisissez un compte enregistré avant de sauvegarder."],
};
const removeAccountLabels: Record<UiLocale, string> = {en:"Remove unavailable accounts", it:"Rimuovi account non disponibili", de:"Nicht verfügbare Konten entfernen", es:"Eliminar cuentas no disponibles", fr:"Retirer les comptes indisponibles"};
export function CloudSettingsPanel({locale, connections}: {locale: UiLocale; connections: RuntimeSettingsConnection[]}) {
  const t = words[locale]; const recovery = recoveryCopy[locale]; const client = useQueryClient();
  const query = useQuery({queryKey:["runtime","cloud"], queryFn: () => cloudRequest<CloudResponse>(), retry:false});
  const [settings, setSettings] = useState<CloudSettings | null>(null);
  const [message, setMessage] = useState(""); const [busy, setBusy] = useState(false);
  const [costs, setCosts] = useState<Record<string, string>>({});
  const [confirmed, setConfirmed] = useState<Record<string, boolean>>({});
  const [attempt, setAttempt] = useState(""); const [url, setUrl] = useState("");
  const initialized = useRef(false);
  useEffect(() => { if (query.data && !initialized.current) { setSettings(query.data.settings); initialized.current = true; } }, [query.data]);
  useEffect(() => {
    if (!attempt) return; let stopped=false; let timer: ReturnType<typeof setTimeout>;
    const poll=async()=>{ try {
      const result=await cloudRequest<{status:string;detail?:string}>(`/openrouter/attempts/${attempt}`);
      if(stopped)return;
      if(["waiting","processing"].includes(result.status)) {timer=setTimeout(()=>void poll(),1500);return;}
      setAttempt("");setUrl("");setMessage(result.status==="connected"?t[21]:result.detail||result.status);
      await client.invalidateQueries({queryKey:["runtime"]});
    } catch(error) {if(!stopped){setMessage(String(error));setAttempt("");}}}; void poll();
    return()=>{stopped=true;clearTimeout(timer);};
  },[attempt, client, locale]);
  const act=async(action:()=>Promise<void>)=>{setBusy(true);setMessage("");try{await action();}catch(error){setMessage(error instanceof Error?error.message:String(error));}finally{setBusy(false);}};
  const update=(patch:Partial<CloudSettings>)=>setSettings(current=>current?{...current,...patch}:current);
  const missingAccount = settings && [settings.asr_provider === "groq" ? settings.asr_connection_id : "", settings.fallback_connection_id, ...Object.keys(settings.openrouter_modes)].filter(Boolean).some(id => !connections.some(c => c.connection_id === id));
  return <section aria-label={t[0]} style={{display:"grid",gap:".8rem",padding:"1.25rem",background:"white",borderRadius:8}}>
    {query.isPending && <p role="status">{recovery[1]}</p>}
    {query.isError && <button type="button" onClick={()=>void query.refetch()}>{recovery[0]}</button>}
    {missingAccount && <><p role="status">{recovery[3]}</p><button type="button" disabled={busy} onClick={()=>setSettings(current=>{
      if(!current)return current;
      const exists=(id:string)=>connections.some(c=>c.connection_id===id);
      const speechAvailable=exists(current.asr_connection_id);
      const fallbackAvailable=exists(current.fallback_connection_id);
      return {...current, asr_provider:speechAvailable?current.asr_provider:"local", asr_connection_id:speechAvailable?current.asr_connection_id:"",
        fallback_connection_id:fallbackAvailable?current.fallback_connection_id:"", paid_fallback_enabled:fallbackAvailable&&current.paid_fallback_enabled,
        openrouter_modes:Object.fromEntries(Object.entries(current.openrouter_modes).filter(([id])=>exists(id)))};
    })}>{removeAccountLabels[locale]}</button></>}
    {connections.some(c=>c.provider_metadata.persistent === false) && <p>{recovery[2]}</p>}
    <h2>{t[0]}</h2><p>{t[18]}</p><p>{t[26]}</p>
    <button type="button" disabled={busy||Boolean(attempt)} onClick={()=>void act(async()=>{const result=await cloudRequest<{attempt_id:string;authorization_url:string}>("/openrouter/sign-in","POST");setAttempt(result.attempt_id);setUrl(result.authorization_url);await cloudRequest(`/openrouter/attempts/${result.attempt_id}/open-browser`,"POST");})}>{t[15]}</button>
    {url&&<a href={url} target="_blank" rel="noopener noreferrer">{t[16]}</a>}
    {attempt&&<button type="button" onClick={()=>void act(async()=>{await cloudRequest("/openrouter/cancel","POST");setAttempt("");setUrl("");})}>{t[17]}</button>}
    {settings&&<fieldset disabled={busy} style={{display:"grid",gap:".8rem",border:0,padding:0}}>
      <label>{t[1]} <select value={settings.asr_provider} onChange={e=>update({asr_provider:e.target.value})}><option value="local">{t[2]}</option><option value="groq">{t[3]}</option></select></label>
      {settings.asr_provider==="groq"&&<><label>{t[4]} <select value={settings.asr_connection_id} onChange={e=>update({asr_connection_id:e.target.value})}><option value="">{t[20]}</option>{connections.filter(c=>c.provider_key==="groq").map(c=><option key={c.connection_id} value={c.connection_id}>{c.label}</option>)}</select></label><label>{t[5]} <select value={settings.asr_model} onChange={e=>update({asr_model:e.target.value})}><option>whisper-large-v3</option><option>whisper-large-v3-turbo</option></select></label></>}
      {connections.filter(c=>c.provider_key==="openrouter").map(c=><label key={c.connection_id}>{t[6]}: {c.label} <select value={settings.openrouter_modes[c.connection_id]||""} onChange={e=>{const modes={...settings.openrouter_modes};if(e.target.value)modes[c.connection_id]=e.target.value;else delete modes[c.connection_id];update({openrouter_modes:modes});}}><option value="">{t[7]}</option><option value="free">{t[8]}</option><option value="paid">{t[9]}</option></select></label>)}
      <label>{t[12]} <input type="number" min="0.01" max="1000" step="0.01" value={settings.monthly_budget_usd} onChange={e=>update({monthly_budget_usd:Number(e.target.value)})}/></label>
      <label>{t[11]} <select value={settings.fallback_connection_id} onChange={e=>update({fallback_connection_id:e.target.value,paid_fallback_enabled:false})}><option value="">{t[20]}</option>{connections.filter(c=>c.provider_key==="openrouter"&&!c.is_default).map(c=><option key={c.connection_id} value={c.connection_id}>{c.label} — {c.model}</option>)}</select></label>
      <label><input type="checkbox" checked={settings.paid_fallback_enabled} disabled={!settings.fallback_connection_id} onChange={e=>update({paid_fallback_enabled:e.target.checked,openrouter_modes:e.target.checked?{...settings.openrouter_modes,[settings.fallback_connection_id]:"paid"}:settings.openrouter_modes})}/>{t[10]}</label>
      <button type="button" onClick={()=>void act(async()=>{await cloudRequest("","PUT",settings);await client.invalidateQueries({queryKey:["runtime"]});setMessage(t[14]);})}>{t[13]}</button>
    </fieldset>}
    {query.data&&<p>{t[19]}: {query.data.spending.spent_usd.toFixed(2)} / {query.data.spending.reserved_usd.toFixed(2)}</p>}
    {Boolean(query.data?.spending.unresolved_requests.length)&&<p>{t[22]}</p>}
    {query.data?.spending.unresolved_requests.map(id => <fieldset key={id} disabled={busy}>
      <legend>{id.slice(0, 8)}</legend>
      <label>{t[23]} <input type="number" min="0" step="0.000001" value={costs[id] ?? ""} onChange={e => setCosts(current => ({...current, [id]: e.target.value}))}/></label>
      <label><input type="checkbox" checked={confirmed[id] ?? false} onChange={e => setConfirmed(current => ({...current, [id]: e.target.checked}))}/>{t[24]}</label>
      <button type="button" disabled={!confirmed[id] || costs[id] === undefined || costs[id] === "" || !Number.isFinite(Number(costs[id])) || Number(costs[id]) < 0}
        onClick={() => void act(async () => {
          await cloudRequest(`/spending/${id}/reconcile`, "POST", {actual_cost_usd: Number(costs[id]), provider_cost_confirmed: true});
          await client.invalidateQueries({queryKey:["runtime", "cloud"]}); setMessage(t[14]);
        })}>{t[25]}</button>
    </fieldset>)}
    {(message||query.isError)&&<p role="status">{message||String(query.error)}</p>}
  <CurrentSharingSummary locale={locale} />
    </section>;
}

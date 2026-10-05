import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { RuntimeConnectionDraft } from "@/lib/api/types";
import { apiClient } from "@/lib/api/client";
import { isDesktopRuntime } from "@/lib/runtime/desktopBridge";
import { buildApiUrl, desktopSessionHeaders } from "@/lib/runtime/environment";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";

type Attempt = { status: string; detail?: string; connection_id?: string; persistent?: boolean };
export async function chatGPTRequest<T>(path: string, method = "GET", body?: object): Promise<T> {
  const response = await fetch(buildApiUrl(`/v1/runtime/chatgpt/${path}`), {
    method, headers: { ...desktopSessionHeaders(), "X-Vostavo-Client": "desktop", "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(60_000),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(result.detail?.detail || "ChatGPT connection failed."), { status: response.status });
  return result as T;
}
const button = { padding: ".7rem 1rem", minHeight: 44, cursor: "pointer", borderRadius: 8 };

export function ChatGPTConnection({ connectionId = "", locale, onConnected, onTest, onSave, testMessage, busy = false }: {
  connectionId?: string; locale: UiLocale; onConnected?: (id: string, notice?: string) => void;
  onTest: (draft: RuntimeConnectionDraft) => void; onSave: (draft: RuntimeConnectionDraft) => void; testMessage: string; busy?: boolean;
}) {
  const t = createTranslator(locale);
  const translate = useRef(t);
  translate.current = t;
  const queryClient = useQueryClient();
  const [id, setId] = useState(connectionId);
  const [attempt, setAttempt] = useState("");
  const [url, setUrl] = useState("");
  const [message, setMessage] = useState("");
  const [working, setWorking] = useState(false);
  const latestConnected = useRef(onConnected);
  latestConnected.current = onConnected;
  const settings = useQuery({ queryKey: ["runtime", "settings"], queryFn: () => apiClient.getRuntimeSettings() });
  const connection = settings.data?.connections.find(c => c.connection_id === id && c.provider_key === "chatgpt");
  const models = (connection?.provider_metadata.models || []) as { slug: string; display_name: string }[];
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["runtime"] });
  useEffect(() => { setId(connectionId); }, [connectionId]);
  useEffect(() => {
    let active = true;
    void chatGPTRequest<{ attempt_id?: string; authorization_url?: string }>("pending").then(result => {
      if (active && result.attempt_id) {
        setAttempt(result.attempt_id); setUrl(result.authorization_url || "");
      }
    }).catch(() => {});
    return () => { active = false; };
  }, []);
  useEffect(() => {
    if (!attempt) return;
    let stopped = false;
    const deadline = Date.now() + 10 * 60_000;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await chatGPTRequest<Attempt>(`attempts/${attempt}`);
        if (stopped) return;
        if (["waiting", "processing"].includes(result.status)) {
          timer = setTimeout(() => void poll(), 1500);
          return;
        }
        setAttempt(""); setUrl("");
        if (result.status === "connected" && result.connection_id) {
          setId(result.connection_id);
          setMessage([translate.current(result.persistent ? "chatgpt.connected" : "chatgpt.session_only"), result.detail].filter(Boolean).join(" "));
          await refresh();
          latestConnected.current?.(result.connection_id);
        } else setMessage(result.detail || translate.current("chatgpt.cancelled"));
      } catch (error) {
        if (!stopped) {
          setMessage(String(error instanceof Error ? error.message : error));
          if (Date.now() >= deadline || (error && typeof error === "object" && "status" in error && [400, 404].includes(Number(error.status)))) {
            setAttempt(""); setUrl("");
          } else timer = setTimeout(() => void poll(), 3000);
        }
      }
    };
    void poll();
    return () => {
      stopped = true; clearTimeout(timer);
      // Leaving settings stops polling; only the explicit Cancel button revokes sign-in.
    };
  }, [attempt]);
  const act = async (action: () => Promise<void>) => {
    setWorking(true); setMessage("");
    try { await action(); } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
    finally { setWorking(false); }
  };
  return <section aria-label="ChatGPT" style={{ display: "grid", gap: ".85rem", padding: "1rem", background: "white", borderRadius: 8 }}>
    <h2 style={{ margin: 0 }}>{t("chatgpt.title")}</h2>
    <p style={{ margin: 0 }}>{t("chatgpt.description")}</p>
    <p style={{ margin: 0 }}>{t("chatgpt.sharing")}</p>
    {!attempt ? <button type="button" style={button} disabled={working} onClick={() => void act(async () => {
      const result = await chatGPTRequest<{ attempt_id: string; authorization_url: string }>("sign-in", "POST", { connection_id: id });
      const target = new URL(result.authorization_url);
      if (target.protocol !== "https:" || target.hostname !== "auth.openai.com") throw new Error("Unexpected sign-in address.");
      setUrl(result.authorization_url); setAttempt(result.attempt_id); setMessage(t("chatgpt.waiting"));
    })}>{t(id ? "chatgpt.reconnect" : "chatgpt.sign_in")}</button> : <>
      {isDesktopRuntime() ? <button type="button" style={button} onClick={() => void act(async () => {
        await chatGPTRequest(`attempts/${attempt}/open-browser`, "POST"); setMessage(t("chatgpt.waiting"));
      })}>{t("chatgpt.open_browser")}</button> : <a href={url} target="_blank" rel="noopener noreferrer" style={button}>{t("chatgpt.open_browser")}</a>}
      <button type="button" style={button} onClick={() => void act(async () => {
        await chatGPTRequest(`attempts/${attempt}/cancel`, "POST"); setAttempt(""); setUrl(""); setMessage(t("chatgpt.cancelled"));
      })}>{t("chatgpt.cancel")}</button>
    </>}
    {connection ? <>
      <label>{t("settings.model")} <select aria-label={t("settings.model")} value={connection.model} disabled={working} onChange={event => {
        const model = event.target.value;
        void act(async () => { await chatGPTRequest(`connections/${id}/model`, "PUT", { model }); await refresh(); });
      }}>{models.map(m => <option key={m.slug} value={m.slug}>{m.display_name}</option>)}</select></label>
      <p>{t(connection.provider_metadata.persistent ? "chatgpt.saved" : "chatgpt.session_only")}</p>
      <button type="button" style={button} disabled={busy || working || !connection.has_api_key} onClick={() => { setMessage(""); onTest({ connection_id: connection.connection_id, provider_choice: "chatgpt", label: connection.label, model: connection.model, base_url: connection.base_url, api_key: "" }); }}>{t("settings.test_connection")}</button>
      <button type="button" style={button} disabled={busy || working} onClick={() => {
        setMessage(""); onSave({ connection_id: connection.connection_id, provider_choice: "chatgpt", label: connection.label, model: connection.model, base_url: connection.base_url, api_key: "" });
      }}>{t("settings.save")}</button>
      <button type="button" style={button} disabled={working || !!attempt} onClick={() => void act(async () => {
        const result = await chatGPTRequest<{ revocation_confirmed: boolean }>(`connections/${id}`, "DELETE");
        const notice = t(result.revocation_confirmed ? "chatgpt.disconnected" : "chatgpt.revocation_unconfirmed");
        setId(""); setMessage(notice); latestConnected.current?.("", notice); await refresh();
      })}>{t("chatgpt.disconnect")}</button>
    </> : null}
    <p role="status" style={{ margin: 0 }}>{message || testMessage}</p>
  </section>;
}

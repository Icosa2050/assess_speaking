import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { CloudSettingsPanel, type CloudSettings } from "./CloudSettingsPanel";
import { renderWithProviders } from "@/test/renderWithProviders";
import type { RuntimeSettingsConnection } from "@/lib/api/types";

const settings: CloudSettings = {version:1, asr_provider:"local",asr_connection_id:"",asr_model:"whisper-large-v3",openrouter_modes:{},fallback_connection_id:"",paid_fallback_enabled:false,monthly_budget_usd:5,max_output_tokens:4096};
const record=(id:string, provider:string):RuntimeSettingsConnection=>({connection_id:id,provider_key:provider,provider_choice:provider,provider_label:provider,label:id,model:"vendor/model:free",base_url:"",is_default:false,is_local:false,requires_api_key:true,has_api_key:true,secret_state:"present",last_test_status:"",last_tested_at:"",openrouter_http_referer:"",openrouter_app_title:"",provider_metadata:{}});
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
for(const locale of ["en","it"] as const) it(`saves independent cloud ASR and explicit paid fallback (${locale})`,async()=>{
  const saved: CloudSettings[]=[];
  vi.stubGlobal("fetch",vi.fn(async (_url:string,init?:RequestInit)=>{
    if(init?.method==="PUT")saved.push(JSON.parse(String(init.body)));
    return new Response(JSON.stringify({settings,spending:{spent_usd:0,reserved_usd:0,unresolved_requests:[]}}),{status:200});
  }));
  renderWithProviders(<CloudSettingsPanel locale={locale} connections={[record("groq-account","groq"),record("paid-account","openrouter")]}/>);
  const panel=screen.getByRole("region",{name:locale==="en"?"Cloud services":"Servizi cloud"});
  await waitFor(()=>expect(within(panel).getAllByRole("combobox")).toHaveLength(3));
  const [asr,mode,fallback]=within(panel).getAllByRole("combobox");
  expect(within(panel).getByRole("checkbox")).not.toBeChecked();
  fireEvent.change(asr,{target:{value:"groq"}});
  fireEvent.change(within(panel).getByLabelText(locale==="en"?"Groq account":"Account Groq"),{target:{value:"groq-account"}});
  fireEvent.change(mode,{target:{value:"free"}});
  fireEvent.change(fallback,{target:{value:"paid-account"}});
  fireEvent.click(within(panel).getByRole("checkbox"));
  fireEvent.click(within(panel).getByRole("button",{name:locale==="en"?"Save cloud settings":"Salva impostazioni cloud"}));
  await waitFor(()=>expect(saved).toHaveLength(1));
  expect(saved[0]).toMatchObject({asr_provider:"groq",asr_connection_id:"groq-account",paid_fallback_enabled:true,fallback_connection_id:"paid-account",openrouter_modes:{"paid-account":"paid"}});
  expect(String(JSON.stringify(saved))).not.toMatch(/api_key|access_token/);
});

it("retries a failed initial load", async()=>{
  let fail=true;
  vi.stubGlobal("fetch",vi.fn(async()=>new Response(JSON.stringify(fail?{detail:"fixture load failure"}:{settings,spending:{spent_usd:0,reserved_usd:0,unresolved_requests:[]}}),{status:fail?503:200})));
  renderWithProviders(<CloudSettingsPanel locale="en" connections={[]}/>);
  await screen.findByRole("button",{name:"Retry loading"});
  fail=false;
  fireEvent.click(screen.getByRole("button",{name:"Retry loading"}));
  await screen.findByRole("button",{name:"Save cloud settings"});
});

it("keeps the draft and unresolved charge after unsuccessful saves and reconciliation",async()=>{
  vi.stubGlobal("fetch",vi.fn(async(_url:string,init?:RequestInit)=>new Response(JSON.stringify(init?.method && init.method!=="GET"?{detail:"fixture failure"}:{settings,spending:{spent_usd:0,reserved_usd:.2,unresolved_requests:["pending-fixture"]}}),{status:init?.method && init.method!=="GET"?400:200})));
  renderWithProviders(<CloudSettingsPanel locale="en" connections={[]}/>);
  const budget=await screen.findByLabelText("Monthly OpenRouter app budget (USD)");
  fireEvent.change(budget,{target:{value:"7"}});
  fireEvent.click(screen.getByRole("button",{name:"Save cloud settings"}));
  await screen.findByText("fixture failure");
  expect(budget).toHaveValue(7);
  fireEvent.change(screen.getByLabelText("Provider-confirmed cost (USD)"),{target:{value:"0.1"}});
  fireEvent.click(screen.getByLabelText("I checked the actual charge in the provider dashboard"));
  fireEvent.click(screen.getByRole("button",{name:"Reconcile spending"}));
  await waitFor(()=>expect(screen.getByLabelText("Provider-confirmed cost (USD)")).toHaveValue(.1));
  expect(screen.getByText(/0.00 \/ 0.20/)).toBeInTheDocument();
});

for(const status of ["failed","cancelled","expired"]) it(`shows terminal ${status} and allows reconnect`,async()=>{
  vi.stubGlobal("fetch",vi.fn(async(url:string,init?:RequestInit)=>{
    const payload=String(url).endsWith("sign-in")?{attempt_id:"fixture",authorization_url:"https://openrouter.ai/auth?fixture"}:String(url).includes("attempts/")?{status}:{settings,spending:{spent_usd:0,reserved_usd:0,unresolved_requests:[]}};
    return new Response(JSON.stringify(payload),{status:200});
  }));
  renderWithProviders(<CloudSettingsPanel locale="en" connections={[]}/>);
  fireEvent.click(screen.getByRole("button",{name:"Connect OpenRouter"}));
  await screen.findByText(status);
  expect(screen.getByRole("button",{name:"Connect OpenRouter"})).toBeEnabled();
});

it("keeps a manual sign-in link after browser launch fails and can cancel",async()=>{
  vi.stubGlobal("fetch",vi.fn(async(url:string)=>{
    const path=String(url);
    const payload=path.endsWith("sign-in")?{attempt_id:"fixture",authorization_url:"https://openrouter.ai/auth?fixture"}:path.endsWith("open-browser")?{detail:"Open the provided sign-in link in your browser."}:path.includes("attempts/")?{status:"waiting"}:{settings,spending:{spent_usd:0,reserved_usd:0,unresolved_requests:[]}};
    return new Response(JSON.stringify(payload),{status:path.endsWith("open-browser")?400:200});
  }));
  renderWithProviders(<CloudSettingsPanel locale="en" connections={[]}/>);
  fireEvent.click(screen.getByRole("button",{name:"Connect OpenRouter"}));
  const link=await screen.findByRole("link",{name:"Continue in your browser"});
  expect(link).toHaveAttribute("href","https://openrouter.ai/auth?fixture");
  await screen.findByText("Open the provided sign-in link in your browser.");
  fireEvent.click(screen.getByRole("button",{name:"Cancel"}));
  await waitFor(()=>expect(screen.queryByRole("link")).not.toBeInTheDocument());
});

for(const [locale,title,session] of [
  ["en","Cloud services","This connection is available"],
  ["it","Servizi cloud","Questa connessione è disponibile"],
  ["de","Cloud-Dienste","Diese Verbindung ist nur"],
  ["es","Servicios en la nube","Esta conexión está disponible"],
  ["fr","Services cloud","Cette connexion est disponible"],
] as const) it(`explains session storage and missing accounts (${locale})`,async()=>{
  vi.stubGlobal("fetch",vi.fn(async()=>new Response(JSON.stringify({settings:{...settings,asr_provider:"groq",asr_connection_id:"deleted"},spending:{spent_usd:0,reserved_usd:0,unresolved_requests:[]}}))));
  const conn=record("session","openrouter");conn.provider_metadata={persistent:false};
  renderWithProviders(<CloudSettingsPanel locale={locale} connections={[conn]}/>);
  expect(screen.getByRole("region",{name:title})).toBeInTheDocument();
  expect(screen.getByText(new RegExp(session))).toBeInTheDocument();
  await screen.findByRole("button",{name:({en:"Remove unavailable accounts",it:"Rimuovi account non disponibili",de:"Nicht verfügbare Konten entfernen",es:"Eliminar cuentas no disponibles",fr:"Retirer les comptes indisponibles"})[locale]});
  const removeLabels={en:"Remove unavailable accounts",it:"Rimuovi account non disponibili",de:"Nicht verfügbare Konten entfernen",es:"Eliminar cuentas no disponibles",fr:"Retirer les comptes indisponibles"};
  fireEvent.click(await screen.findByRole("button",{name:removeLabels[locale]}));
  await waitFor(()=>expect(screen.queryByRole("button",{name:removeLabels[locale]})).not.toBeInTheDocument());
});

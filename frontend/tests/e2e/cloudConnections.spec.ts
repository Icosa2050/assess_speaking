import { test, expect } from "../fixtures";

for (const locale of ["en", "it"]) {
  test(`ChatGPT connect, account model, feedback probe and disconnect (${locale})`, async ({ page }) => {
    let connected = false;
    let finished = false;
    let model = "plan-one";
    let probes = 0;
    const record = () => ({ connection_id: "account", provider_key: "chatgpt", provider_choice: "chatgpt",
      provider_label: "ChatGPT", label: "ChatGPT", model, base_url: "https://api.openai.com/v1", is_default: true,
      is_local: false, requires_api_key: true, has_api_key: true, secret_state: "present", last_test_status: "",
      last_tested_at: "", openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {
        persistent: false, models: [{slug:"plan-one",display_name:"Plan one"}, {slug:"plan-two",display_name:"Plan two"}],
      } });
    await page.route("**/v1/runtime/settings", r => r.fulfill({json: { ui_locale: locale, whisper_model: "tiny", active_connection_id: connected ? "account" : "", connections: connected ? [record()] : [] }}));
    await page.route("**/v1/runtime", r => r.fulfill({json: {configured: connected, provider: "chatgpt", model, base_url: "https://api.openai.com/v1", requires_api_key: true, has_api_key: connected}}));
    await page.route("**/v1/runtime/chatgpt/sign-in", async r => {
      expect(r.request().headers()["x-vostavo-client"]).toBe("desktop");
      await r.fulfill({json: {attempt_id:"attempt",authorization_url:"https://auth.openai.com/api/accounts/authorize?state=fixture"}});
    });
    await page.route("**/v1/runtime/chatgpt/attempts/attempt", async r => {
      if (finished) connected = true;
      await r.fulfill({json: {status: finished ? "connected" : "waiting", connection_id:"account",persistent:false}});
    });
    await page.route("**/v1/runtime/chatgpt/attempts/attempt/cancel", r => r.fulfill({json:{status: "cancelled"}}));
    await page.route("**/v1/runtime/chatgpt/connections/account/model", async r => {
      model = r.request().postDataJSON().model;
      await r.fulfill({json:{model}});
    });
    await page.route("**/v1/runtime/chatgpt/connections/account", async r => {
      connected = false;
      await r.fulfill({json:{disconnected:true,revocation_confirmed:false}});
    });
    await page.route("**/v1/runtime/settings/test-connection", async r => {
      probes++;
      expect(r.request().postDataJSON().connection).toMatchObject({provider_choice:"chatgpt",connection_id:"account",model:"plan-two",api_key:""});
      await r.fulfill({json:{provider:"chatgpt",base_url:"https://api.openai.com/v1",service_base_url:"https://api.openai.com/v1",health_endpoint:"https://api.openai.com/v1/models",discovered_models:["plan-one","plan-two"],tested_at:"2026-10-04",content_preview:"Structured feedback verified"}});
    });
    await page.goto("/settings");
    await expect(page.getByTestId("settings.ui_locale")).toHaveValue(locale);
    await page.getByTestId("runtime_connection.provider").selectOption("chatgpt");
    await page.getByRole("button", {name: locale === "en" ? "Continue with ChatGPT" : "Continua con ChatGPT",exact:true}).click();
    await expect(page.getByRole("link", {name: locale === "en" ? "Continue in your browser" : "Continua nel browser"})).toHaveAttribute("href", /auth.openai.com/);
    finished = true;
    const panel = page.getByRole("region", {name:"ChatGPT", exact:true});
    await expect(panel.getByRole("combobox")).toBeVisible();
    await panel.getByRole("combobox").selectOption("plan-two");
    await expect(panel.getByRole("combobox")).toHaveValue("plan-two");
    await panel.getByRole("button", {name: locale === "en" ? "Test provider connection" : "Verifica la connessione al provider",exact:true}).click();
    await expect(panel.getByRole("status")).toContainText("Structured feedback verified");
    expect(probes).toBe(1);
    await panel.getByRole("button", {name: locale === "en" ? "Disconnect ChatGPT" : "Scollega ChatGPT",exact:true}).click();
    await expect(panel.getByRole("status")).toContainText(locale === "en" ? "Disconnected locally" : "Scollegato localmente");
    await expect(panel.getByRole("combobox")).toHaveCount(0);
  });
}

test("xAI uses API keys at its fixed endpoint; switching providers clears an entered key", async ({page}) => {
  await page.route("**/v1/runtime/settings", r => r.fulfill({json:{ui_locale:"en",whisper_model:"tiny",connections:[],active_connection_id:""}}));
  await page.goto("/settings");
  const select = page.getByTestId("runtime_connection.provider");
  await select.selectOption("openrouter");
  await page.getByTestId("runtime_connection.api_key").fill("do-not-share-across-providers");
  await select.selectOption("xai");
  await expect(page.getByTestId("runtime_connection.api_key")).toHaveValue("");
  await expect(page.getByTestId("runtime_connection.base_url")).toHaveValue("https://api.x.ai/v1");
  await expect(page.getByTestId("runtime_connection.base_url")).toHaveAttribute("readonly", "");
  await expect(page.getByRole("link",{name:"Get an API key at the xAI console"})).toBeVisible();
});

for (const locale of ["en", "it"]) {
  test(`Groq key setup, quota error and retry (${locale})`, async ({ page }) => {
    let probes = 0;
    let selectedModel = "openai/gpt-oss-120b";
    await page.route("**/v1/runtime/settings", r => r.fulfill({json:{ui_locale:locale,whisper_model:"tiny",connections:[],active_connection_id:""}}));
    await page.route("**/v1/runtime/settings/test-connection", async r => {
      probes++;
      expect(r.request().postDataJSON().connection).toMatchObject({provider_choice:"groq",model:selectedModel,base_url:"https://api.groq.com/openai/v1",api_key:"groq-fixture"});
      if (probes === 1) {
        await r.fulfill({status:502,json:{detail:{code:"runtime_error",detail:"Groq usage limit reached. Wait and retry."}}});
      } else {
        await r.fulfill({json:{provider:"groq",base_url:"https://api.groq.com/openai/v1",service_base_url:"https://api.groq.com/openai/v1",health_endpoint:"https://api.groq.com/openai/v1/models",discovered_models:["openai/gpt-oss-120b"],tested_at:"2026-10-04",content_preview:`Structured feedback verified: ${selectedModel}`}});
      }
    });
    await page.goto("/settings");
    const select = page.getByTestId("runtime_connection.provider");
    await select.selectOption("openrouter");
    await page.getByTestId("runtime_connection.api_key").fill("unrelated-key");
    await select.selectOption("groq");
    await expect(page.getByTestId("runtime_connection.api_key")).toHaveValue("");
    await expect(page.getByTestId("runtime_connection.base_url")).toHaveValue("https://api.groq.com/openai/v1");
    await expect(page.getByTestId("runtime_connection.base_url")).toHaveAttribute("readonly", "");
    await expect(page.getByTestId("runtime_connection.model")).toHaveValue("openai/gpt-oss-120b");
    await expect(page.locator('a[href="https://console.groq.com/keys"]')).toBeVisible();
    await page.getByTestId("runtime_connection.api_key").fill("groq-fixture");
    await page.getByTestId("runtime_connection.test_connection").click();
    await expect(page.getByTestId("runtime_connection.form_status")).toContainText("Groq usage limit reached");
    await expect(page.getByTestId("runtime_connection.api_key")).toHaveValue("groq-fixture");
    await page.getByTestId("runtime_connection.test_connection").click();
    await expect(page.getByTestId("runtime_connection.form_status")).toContainText(`Structured feedback verified: ${selectedModel}`);
    expect(probes).toBe(2);
    for (const model of ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]) {
      selectedModel = model;
      await page.getByTestId("runtime_connection.model").selectOption(model);
      await page.getByTestId("runtime_connection.test_connection").click();
      await expect.poll(() => probes).toBe(model === "openai/gpt-oss-20b" ? 3 : 4);
      await expect(page.getByTestId("runtime_connection.form_status")).toContainText(`Structured feedback verified: ${selectedModel}`);
    }
    await select.selectOption("xai");
    await expect(page.getByTestId("runtime_connection.api_key")).toHaveValue("");
  });
}

for (const locale of ["en", "it"] as const) {
  test(`cloud transcription, OpenRouter login, fallback and spending recovery (${locale})`, async ({page}) => {
    const records = [
      {connection_id:"groq",provider_key:"groq",label:"Speech account",model:"openai/gpt-oss-120b",is_default:false},
      {connection_id:"free",provider_key:"openrouter",label:"Free account",model:"vendor/model:free",is_default:true},
      {connection_id:"paid",provider_key:"openrouter",label:"Paid account",model:"vendor/model",is_default:false},
    ].map(c => ({...c,provider_choice:c.provider_key,provider_label:c.provider_key,base_url:"https://openrouter.ai/api/v1",is_local:false,requires_api_key:true,has_api_key:true,secret_state:"present",last_test_status:"",last_tested_at:"",openrouter_http_referer:"",openrouter_app_title:"",provider_metadata:{}}));
    let settings = {version:1,asr_provider:"local",asr_connection_id:"",asr_model:"whisper-large-v3",openrouter_modes:{free:"free"},fallback_connection_id:"",paid_fallback_enabled:false,monthly_budget_usd:5,max_output_tokens:4096};
    let spent = 0; let reserved = .1; let saved = 0; let reconciled = 0; let finished = false;
    await page.route("**/v1/runtime/settings", r => r.fulfill({json:{ui_locale:locale,whisper_model:"tiny",active_connection_id:"free",connections:records}}));
    await page.route("**/v1/runtime", r => r.fulfill({json:{configured:true,provider:"openrouter",model:"vendor/model:free",has_api_key:true}}));
    await page.route("**/v1/runtime/cloud**", async r => {
      const request = r.request(); const path = new URL(request.url()).pathname;
      expect(request.headers()["x-vostavo-client"]).toBe("desktop");
      expect(request.postData() || "").not.toMatch(/api_key|access_token|refresh_token/);
      if (path.endsWith("/sign-in")) return r.fulfill({json:{attempt_id:"fixture",authorization_url:"https://openrouter.ai/auth?state=fixture"}});
      if (path.endsWith("/open-browser")) return r.fulfill({json:{opened:true}});
      if (path.endsWith("/attempts/fixture")) return r.fulfill({json:{status:finished?"connected":"waiting"}});
      if (path.endsWith("/reconcile")) {
        expect(request.postDataJSON()).toEqual({actual_cost_usd:.02,provider_cost_confirmed:true});
        spent=.02;reserved=0;reconciled++;
        return r.fulfill({json:{spending:{spent_usd:spent,reserved_usd:reserved,unresolved_requests:[]}}});
      }
      if(request.method()==="PUT") {settings=request.postDataJSON();saved++;}
      return r.fulfill({json:{settings,spending:{spent_usd:spent,reserved_usd:reserved,unresolved_requests:reserved?["reservation-fixture"]:[]}}});
    });
    await page.goto("/settings");
    await expect(page.getByTestId("settings.ui_locale")).toHaveValue(locale);
    const panel=page.getByRole("region",{name:locale==="en"?"Cloud services":"Servizi cloud",exact:true});
    await panel.getByRole("combobox",{name:locale==="en"?"Transcription":"Trascrizione",exact:true}).selectOption("groq");
    await panel.getByRole("combobox",{name:locale==="en"?"Groq account":"Account Groq",exact:true}).selectOption("groq");
    await panel.getByRole("combobox",{name:locale==="en"?"Fallback account":"Account alternativo",exact:true}).selectOption("paid");
    await panel.getByRole("checkbox",{name:locale==="en"?"Allow paid fallback":"Consenti alternativa a pagamento",exact:true}).check();
    await panel.getByRole("button",{name:locale==="en"?"Save cloud settings":"Salva impostazioni cloud",exact:true}).click();
    await expect.poll(()=>saved).toBe(1);
    expect(settings).toMatchObject({asr_provider:"groq",asr_connection_id:"groq",paid_fallback_enabled:true,fallback_connection_id:"paid",openrouter_modes:{free:"free",paid:"paid"}});
    await panel.getByRole("spinbutton",{name:locale==="en"?"Provider-confirmed cost (USD)":"Costo confermato dal servizio (USD)",exact:true}).fill("0.02");
    const reconcile = panel.getByRole("button",{name:locale==="en"?"Reconcile spending":"Riconcilia spesa",exact:true});
    await expect(reconcile).toBeDisabled();
    await panel.getByRole("checkbox",{name:locale==="en"?"I checked the actual charge in the provider dashboard":"Ho verificato l’addebito effettivo nel pannello del servizio",exact:true}).check();
    await reconcile.click(); await expect.poll(()=>reconciled).toBe(1);
    await expect(reconcile).toHaveCount(0);
    await panel.getByRole("button",{name:locale==="en"?"Connect OpenRouter":"Collega OpenRouter",exact:true}).click();
    await expect(panel.getByRole("link")).toHaveAttribute("href",/^https:\/\/openrouter.ai\/auth/);
    finished=true;
    await expect(panel.getByRole("status").filter({ hasText: locale==="en"?"Choose an OpenRouter model":"scegli un modello OpenRouter" })).toContainText(locale==="en"?"Choose an OpenRouter model":"scegli un modello OpenRouter");
  });
}

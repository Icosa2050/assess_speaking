import { test, expect } from "@playwright/test";

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
    await page.route("**/v1/runtime/settings", r => r.fulfill({json:{ui_locale:locale,whisper_model:"tiny",connections:[],active_connection_id:""}}));
    await page.route("**/v1/runtime/settings/test-connection", async r => {
      probes++;
      expect(r.request().postDataJSON().connection).toMatchObject({provider_choice:"groq",model:"openai/gpt-oss-120b",base_url:"https://api.groq.com/openai/v1",api_key:"groq-fixture"});
      if (probes === 1) {
        await r.fulfill({status:502,json:{detail:{code:"runtime_error",detail:"Groq usage limit reached. Wait and retry."}}});
      } else {
        await r.fulfill({json:{provider:"groq",base_url:"https://api.groq.com/openai/v1",service_base_url:"https://api.groq.com/openai/v1",health_endpoint:"https://api.groq.com/openai/v1/models",discovered_models:["openai/gpt-oss-120b"],tested_at:"2026-10-04",content_preview:"Structured feedback verified"}});
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
    await expect(page.getByTestId("runtime_connection.form_status")).toContainText("Structured feedback verified");
    expect(probes).toBe(2);
    await select.selectOption("xai");
    await expect(page.getByTestId("runtime_connection.api_key")).toHaveValue("");
  });
}

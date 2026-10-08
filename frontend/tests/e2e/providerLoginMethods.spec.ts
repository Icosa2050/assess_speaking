import { expect, test } from "../fixtures";
import type { RuntimeSettingsConnection } from "../../src/lib/api/types";

// All credentials here are fixtures. Real keys never enter browser traces.
const methods = [
  { choice: "groq", provider: "groq", url: "https://api.groq.com/openai/v1", model: "openai/gpt-oss-120b", key: true },
  { choice: "xai", provider: "xai", url: "https://api.x.ai/v1", model: "fixture-grok", key: true },
  { choice: "openrouter", provider: "openrouter", url: "https://openrouter.ai/api/v1", model: "fixture/model", key: true },
  { choice: "ollama_cloud", provider: "ollama", url: "https://ollama.com/api", model: "fixture-cloud", key: true },
  { choice: "ollama_local", provider: "ollama", url: "http://localhost:11434", model: "fixture-local", key: false },
  { choice: "lmstudio_local", provider: "lmstudio", url: "http://localhost:1234/v1", model: "fixture-local", key: false },
  { choice: "openai_compatible", provider: "openai_compatible", url: "http://localhost:1234/v1", model: "fixture-compatible", key: true },
];

for (const locale of ["en", "it"]) {
  for (const method of methods) {
    test(`${method.choice}: save, reload, use saved credential and remove (${locale})`, async ({page}) => {
      let saved: RuntimeSettingsConnection | null = null;
      const probes: unknown[] = [];
      let deletes = 0;
      const state = () => ({ui_locale:locale,whisper_model:"tiny",connections:saved ? [saved] : [],active_connection_id:saved?.connection_id || ""});
      await page.route("**/v1/runtime", r => r.fulfill({json:{configured:Boolean(saved),provider:method.provider,model:method.model,base_url:method.url,requires_api_key:method.key,has_api_key:Boolean(saved)}}));
      await page.route("**/v1/runtime/settings", async r => {
        if (r.request().method() === "PUT") {
          const draft = r.request().postDataJSON().connection;
          expect(draft).toMatchObject({provider_choice:method.choice,model:method.model,api_key:method.key ? "fixture-api-key" : ""});
          saved = {connection_id:"fixture-connection",provider_key:method.provider,provider_choice:method.choice,
            provider_label:method.choice,label:"Fixture connection",model:draft.model,base_url:draft.base_url,
            is_default:true,is_local:!method.key,requires_api_key:method.key,has_api_key:method.key,
            secret_state:method.key ? "present" : "absent",last_test_status:"",last_tested_at:"",
            openrouter_http_referer:"",openrouter_app_title:"",provider_metadata:{persistent:true}};
        }
        await r.fulfill({json:state()});
      });
      await page.route("**/v1/runtime/settings/test-connection", async r => {
        const draft = r.request().postDataJSON().connection;
        probes.push(draft);
        await r.fulfill({json:{provider:method.provider,base_url:method.url,service_base_url:method.url,
          health_endpoint:method.url+"/models",discovered_models:[method.model],tested_at:"2026-10-04",
          content_preview:`Probe ${probes.length} succeeded`}});
      });
      await page.route("**/v1/runtime/settings/connections/fixture-connection", async r => {
        expect(r.request().method()).toBe("DELETE"); deletes++; saved = null;
        await r.fulfill({json:state()});
      });
      await page.goto("/settings");
      await page.getByTestId("runtime_connection.provider").selectOption(method.choice);
      if (method.choice === "groq") await page.getByTestId("runtime_connection.model").selectOption(method.model);
      else await page.getByTestId("runtime_connection.model").fill(method.model);
      if (method.choice === "openai_compatible") await page.getByTestId("runtime_connection.base_url").fill(method.url);
      if (method.key) await page.getByTestId("runtime_connection.api_key").fill("fixture-api-key");
      await page.getByTestId("runtime_connection.test_connection").click();
      await expect(page.getByTestId("runtime_connection.form_status")).toContainText("Probe 1 succeeded");
      expect(probes[0]).toMatchObject({api_key:method.key ? "fixture-api-key" : "",provider_choice:method.choice});
      await page.getByTestId("runtime_connection.save_connection").click();
      await expect(page.getByTestId("settings.connection_row_delete")).toHaveCount(1);
      await page.reload();
      await expect(page.getByTestId("runtime_connection.provider")).toHaveValue(method.choice);
      await expect(page.getByTestId("runtime_connection.model")).toHaveValue(method.model);
      await expect(page.getByTestId("runtime_connection.api_key")).toHaveValue("");
      await page.getByTestId("runtime_connection.test_connection").click();
      await expect(page.getByTestId("runtime_connection.form_status")).toContainText("Probe 2 succeeded");
      expect(probes[1]).toMatchObject({connection_id:"fixture-connection",api_key:"",provider_choice:method.choice});
      await page.getByTestId("settings.connection_row_delete").click();
      await expect(page.getByTestId("settings.connection_row_delete")).toHaveCount(0);
      expect(deletes).toBe(1);
    });
  }

  test(`ChatGPT cancel and denied consent can be retried (${locale})`, async ({page}) => {
    let attempt = 0;
    await page.route("**/v1/runtime/settings", r => r.fulfill({json:{ui_locale:locale,whisper_model:"tiny",connections:[],active_connection_id:""}}));
    await page.route("**/v1/runtime/chatgpt/pending", r => r.fulfill({json:{}}));
    await page.route("**/v1/runtime/chatgpt/sign-in", async r => {
      attempt++;
      await r.fulfill({json:{attempt_id:`fixture-${attempt}`,authorization_url:"https://auth.openai.com/api/accounts/authorize?state=fixture"}});
    });
    await page.route("**/v1/runtime/chatgpt/attempts/*", r => r.fulfill({json:{status:attempt === 2 ? "failed" : "waiting",detail:attempt === 2 ? "Consent denied for fixture account" : ""}}));
    await page.route("**/v1/runtime/chatgpt/attempts/*/cancel", r => r.fulfill({json:{status:"cancelled"}}));
    await page.goto("/settings");
    await page.getByTestId("runtime_connection.provider").selectOption("chatgpt");
    const panel = page.getByRole("region", {name:"ChatGPT",exact:true});
    const signIn = panel.getByRole("button", {name:locale === "en" ? "Continue with ChatGPT" : "Continua con ChatGPT",exact:true});
    await signIn.click();
    await panel.getByRole("button", {name:locale === "en" ? "Cancel sign-in" : "Annulla accesso",exact:true}).click();
    await expect(signIn).toBeVisible();
    await signIn.click();
    await expect(panel.getByRole("status")).toContainText("Consent denied for fixture account");
    await expect(signIn).toBeVisible();
    await expect(panel.getByRole("combobox")).toHaveCount(0);
    await signIn.click();
    await expect(panel.getByRole("link")).toHaveAttribute("href", /auth.openai.com/);
  });
}

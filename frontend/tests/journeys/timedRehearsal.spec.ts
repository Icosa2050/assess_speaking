import { assertCapturedDuration, calibrateMicrophone } from "./microphone";
import { expect, test } from "../fixtures";
import { playAndSeek } from "../live/workflowEvidence";

const backend = process.env.VOSTAVO_FIXTURE_BACKEND_URL!;
for (const language of ["en", "it"] as const) {
  test.describe(`timed rehearsal ${language}`, () => {
    test.use({ locale: language });
    test("preparation → three saved parts → reload → recovered analysis → targeted retry", async ({ page, request, context }, testInfo) => {
      await request.put(`${backend}/v1/runtime/settings`, { data: { ui_locale: language, whisper_model: "small",
        connection: { provider_choice: "ollama_local", label: "Rehearsal fixture inference", model: "journey-fixture", base_url: "http://127.0.0.1:11434/v1" } } });
      await calibrateMicrophone(page, "/rehearsal", true);
      await page.clock.install();
      await page.getByTestId("rehearsal-speaker").fill(`oral-${language}`);
      await page.getByTestId("rehearsal-language").selectOption(language);
      await page.getByTestId("rehearsal-goal").selectOption("B2");
      await page.getByTestId("rehearsal-create").click();
      await expect(page.getByTestId("rehearsal-prepare")).toBeEnabled();
      const staleTab = await context.newPage();
      await staleTab.goto("/rehearsal");
      await staleTab.getByRole("button", { name: /oral-/ }).click();
      await expect(staleTab.getByTestId("rehearsal-prepare")).toBeEnabled();
      await page.getByTestId("rehearsal-prepare").click();
      await staleTab.getByTestId("rehearsal-prepare").click();
      await expect(staleTab.getByRole("alert")).toContainText("changed in another tab");
      await staleTab.close();
      await page.clock.fastForward(301000);
      await expect(page.getByTestId("rehearsal-preparation-clock")).toHaveText("0:00");
      await page.getByTestId("rehearsal-speak").click();
      await expect(page.getByTestId("speak.input_mode_upload")).toHaveCount(0);
      for (const [index, seconds] of [180, 180, 240].entries()) {
        await page.getByTestId("speak.record_start").click();
        await expect(page.getByTestId("speak.record_stop")).toBeVisible();
        // Capture at least 30 seconds of decoded media before testing the UI cap.
        // This is not a fifteen-minute audio or model-quality test.
        await page.waitForTimeout(45_000);
        await page.clock.fastForward(seconds * 1000);
        await expect(page.getByTestId("speak.record_stop")).toHaveCount(0);
        await expect(page.getByTestId("rehearsal-save-part")).toBeEnabled();
        await assertCapturedDuration(page);
        if (index === 0) {
          await page.evaluate(() => {
            const original = IDBObjectStore.prototype.put;
            IDBObjectStore.prototype.put = function (...args: Parameters<IDBObjectStore["put"]>) {
              if (this.name === "recordings") {
                IDBObjectStore.prototype.put = original;
                throw new DOMException("Test browser quota", "QuotaExceededError");
              }
              return original.apply(this, args);
            };
          });
          await page.getByTestId("rehearsal-save-part").click();
          await expect(page.getByRole("alert")).toContainText(language === "en" ? "Free space" : "Libera spazio");
          await expect(page.getByTestId("rehearsal-save-part")).toBeEnabled();
          await page.evaluate(() => {
            const original = IDBObjectStore.prototype.put;
            IDBObjectStore.prototype.put = function (...args: Parameters<IDBObjectStore["put"]>) {
              const request = original.apply(this, args);
              if (this.name === "recordings") {
                IDBObjectStore.prototype.put = original;
                queueMicrotask(() => this.transaction.abort());
              }
              return request;
            };
          });
          await page.getByTestId("rehearsal-save-part").click();
          await expect(page.getByRole("alert")).toContainText("transaction failed");
          await expect(page.getByTestId("rehearsal-save-part")).toBeEnabled();
        }
        await page.getByTestId("rehearsal-save-part").click();
        if (index === 2) await expect(page.getByTestId("rehearsal-analyse")).toBeEnabled();
        if (index === 0) {
          await page.reload();
          await page.getByRole("button", { name: /oral-/ }).click();
          await calibrateMicrophone(page, "/rehearsal");
          await page.getByRole("button", { name: /oral-/ }).click();
          await expect(page.getByTestId("rehearsal-screen")).toContainText(language === "en" ? "Part 2 of 3" : "Parte 2 di 3");
        }
      }
      await page.reload();
      await page.getByRole("button", { name: /oral-/ }).click();
      await page.route("**/v1/uploads", route => route.fulfill({ status: 507, json: { detail: { code: "storage_error", detail: "Test disk full; recordings retained" } } }), { times: 1 });
      await page.getByTestId("rehearsal-analyse").click();
      await expect(page.getByRole("alert")).toContainText("Test disk full");
      await page.route("**/v1/assessments", route => route.fulfill({ status: 404, json: { detail: { code: "storage_error", detail: "Expired upload test" } } }), { times: 1 });
      await page.getByTestId("rehearsal-analyse").click();
      await expect(page.getByRole("alert")).toContainText("Expired upload test");
      const created: { id: string; body: Record<string, unknown> }[] = [];
      page.on("response", async response => {
        if (new URL(response.url()).pathname === "/v1/assessments" && response.request().method() === "POST" && response.ok())
          created.push({ id: (await response.json()).assessment_id, body: response.request().postDataJSON() });
      });
      await page.route("**/v1/assessments/*", route => route.fulfill({ status: 200, json: {
        assessment_id: route.request().url().split("/").pop(), status: "running", phase: "testing_resume", progress: .5,
      } }));
      const pendingStatus = page.waitForResponse(response => /\/v1\/assessments\/[^/]+$/.test(new URL(response.url()).pathname));
      await page.getByTestId("rehearsal-analyse").click();
      await pendingStatus;
      const competing = await context.newPage();
      await competing.goto("/rehearsal");
      await competing.getByRole("button", { name: /oral-/ }).click();
      await competing.getByTestId("rehearsal-analyse").click();
      await expect(competing.getByRole("alert")).toContainText(language === "en" ? "another tab" : "un’altra scheda");
      await competing.close();
      await page.getByTestId("rehearsal-pause-analysis").click();
      await expect(page.getByTestId("rehearsal-analyse")).toBeEnabled();
      await page.unroute("**/v1/assessments/*");
      await page.reload();
      await page.getByRole("button", { name: /oral-/ }).click();
      await page.getByTestId("rehearsal-analyse").click();
      await expect(page.getByTestId("rehearsal-retry-2")).toBeVisible({ timeout: 45000 });
      expect(created).toHaveLength(3);
      expect(created.map(item => item.body.target_duration_sec)).toEqual([180, 180, 240]);
      expect(created.map(item => item.body.task_family)).toEqual(["personal_experience", "opinion_monologue", "free_monologue"]);
      expect(created.every(item => item.body.expected_language === language && item.body.target_cefr === "B2")).toBe(true);
      const parent = await (await request.get(`${backend}/v1/assessments/${created[2].id}`)).json();
      await playAndSeek(page, `[data-testid="rehearsal-result-2"] audio`, 1);
      const audio = await request.get(`${backend}/v1/history/${parent.payload.report.session_id}/audio`, { headers: { Range: "bytes=0-63" } });
      expect(audio.status()).toBe(206);
      await page.screenshot({ path: testInfo.outputPath(`${language}-whole-rehearsal.png`), fullPage: true });
      await page.setViewportSize({ width: 390, height: 844 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.setViewportSize({ width: 1280, height: 900 });
      await page.getByTestId("rehearsal-retry-2").click();
      await page.getByTestId("rehearsal-prepare").click();
      await page.getByTestId("rehearsal-speak").click();
      await calibrateMicrophone(page, "/rehearsal");
      await page.getByRole("button", { name: /oral-/ }).first().click();
      await page.getByTestId("speak.record_start").click();
      await expect(page.getByTestId("speak.record_stop")).toBeVisible();
      await page.waitForTimeout(45_000);
      await page.getByTestId("speak.record_stop").click();
      await expect(page.getByTestId("rehearsal-save-part")).toBeEnabled();
      await assertCapturedDuration(page);
      await page.getByTestId("rehearsal-save-part").click();
      await page.getByTestId("rehearsal-analyse").click();
      await expect(page.getByTestId("rehearsal-retry-0")).toBeVisible({ timeout: 45000 });
      expect(created).toHaveLength(4);
      expect(created[3].body).toMatchObject({ retry_of_session_id: parent.payload.report.session_id,
        task_family: created[2].body.task_family, target_duration_sec: 240, prompt_id: created[2].body.prompt_id, prompt_text: created[2].body.prompt_text });
      const retry = await (await request.get(`${backend}/v1/assessments/${created[3].id}`)).json();
      expect(retry.payload.report.progress_delta.previous_session_id).toBe(parent.payload.report.session_id);
      await page.reload();
      await page.getByRole("button", { name: /oral-/ }).first().click();
      await expect(page.getByTestId("rehearsal-retry-0")).toBeVisible();
    });
  });
}

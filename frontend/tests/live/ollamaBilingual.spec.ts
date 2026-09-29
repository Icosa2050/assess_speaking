import { expect, test } from "@playwright/test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import type { AssessmentStatusResponse, JsonRecord } from "../../src/lib/api/types";

const backend = "http://127.0.0.1:8816";
const provider = process.env.LOCAL_E2E_PROVIDER === "lmstudio" ? "lmstudio" : "ollama";
const providerChoice = provider === "lmstudio" ? "lmstudio_local" : "ollama_local";
const providerName = provider === "lmstudio" ? "LM Studio" : "Ollama";
const ollama = process.env.LOCAL_E2E_BASE_URL || (provider === "lmstudio" ? "http://127.0.0.1:1234/v1" : "http://127.0.0.1:11434/v1");
const model = process.env.LOCAL_E2E_MODEL || process.env.OLLAMA_E2E_MODEL || (provider === "lmstudio" ? "vostavo-qwen2.5-3b" : "qwen3.5:4b");
const whisper = process.env.OLLAMA_E2E_WHISPER || "large-v3";
const root = fileURLToPath(new URL("../../..", import.meta.url));
const cases = [
  { goal: "B1", file: "travel_story.wav", en: "My last trip", it: "Il mio ultimo viaggio" },
  { goal: "B2", file: "remote_work.wav", en: "Advantages and disadvantages of remote work", it: "Vantaggi e svantaggi del lavoro da remoto" },
  { goal: "C1", file: "public_debate.wav", en: "Public debate and digital platforms", it: "Dibattito pubblico e piattaforme digitali" },
];

test(`${providerName} setup discovers the installed model and tests its connection`, async ({ page }) => {
  await page.goto("/runtime-setup");
  await page.getByTestId("runtime_connection.provider").selectOption(providerChoice);
  await page.getByTestId("runtime_connection.base_url").fill(ollama);
  await page.getByTestId("runtime_setup.detect_local_models").click();
  await expect(page.getByRole("combobox", { name: "Detected local models" })).toBeVisible();
  await page.getByRole("combobox", { name: "Detected local models" }).selectOption(model);
  await page.getByTestId("runtime_connection.model").fill(model);
  const resultPromise = page.waitForResponse(response => new URL(response.url()).pathname === "/v1/runtime/settings/test-connection");
  await page.getByTestId("runtime_connection.test_connection").click();
  const response = await resultPromise;
  expect(response.ok()).toBeTruthy();
  expect((await response.json()).discovered_models).toContain(model);
  await expect(page.getByTestId("runtime_connection.base_url")).toHaveValue(ollama);
});

for (const language of ["en", "it"] as const) {
  test.describe(language, () => {
    test.use({ locale: language });
    for (const item of cases) {
      test(`${language} ${item.goal}: real Whisper + ${providerName} feedback, saved history${item.goal === "B1" ? " and retry" : ""}`, async ({ page, request }, testInfo) => {
        expect(path.basename(testInfo.config.configFile || "")).toBe("playwright.ollama.config.ts");
        const modelsResponse = await request.get(`${ollama}/models`);
        expect(modelsResponse.ok(), `${providerName} must be running`).toBeTruthy();
        expect((await modelsResponse.json()).data).toEqual(expect.arrayContaining([expect.objectContaining({ id: model })]));
        const cached = await request.get(`${backend}/v1/runtime/whisper-models/${whisper}`);
        expect((await cached.json()).cached, "Download Whisper before running this offline-ASR test").toBe(true);
        const saved = await request.put(`${backend}/v1/runtime/settings`, { data: {
          ui_locale: language, whisper_model: whisper,
          connection: { provider_choice: providerChoice, label: `${providerName} bilingual live test`, model, base_url: ollama },
        } });
        expect(saved.ok()).toBeTruthy();
        await page.goto("/session-setup");
        await page.getByTestId("setup.speaker_id").fill(`live-${language}-${item.goal}`);
        await page.getByTestId("setup.customize_details").click();
        await page.getByTestId("setup.learning_language").selectOption(language);
        await page.getByTestId("setup.cefr").selectOption(item.goal);
        await page.getByTestId("setup.advanced_topic").click();
        await page.getByTestId("setup.custom_theme").fill(item[language]);
        await page.getByTestId("setup.duration").selectOption("90");
        await page.getByTestId("setup.continue").click();
        let parent = "";
        let firstPrompt = "";
        for (let attempt = 0; attempt < (item.goal === "B1" ? 2 : 1); attempt++) {
          await page.getByTestId("speak.input_mode_upload").click();
          await page.getByTestId("speak.upload_input").setInputFiles(path.join(root, "samples/cefr", language, item.goal, item.file));
          const createdPromise = page.waitForResponse(response => new URL(response.url()).pathname === "/v1/assessments" && response.request().method() === "POST");
          await page.getByTestId("speak.submit").click();
          const created = await createdPromise;
          expect(created.ok()).toBeTruthy();
          const submission = created.request().postDataJSON();
          expect(submission).toMatchObject({ provider, llm_model: model, whisper, expected_language: language, feedback_language: language, target_cefr: item.goal, retry_of_session_id: parent });
          if (attempt) expect(submission.prompt_text).toBe(firstPrompt);
          else firstPrompt = submission.prompt_text;
          const { assessment_id: id } = await created.json();
          let result: AssessmentStatusResponse | undefined;
          await expect.poll(async () => {
            const response = await request.get(`${backend}/v1/assessments/${id}`, { maxRetries: 2 });
            expect(response.ok()).toBeTruthy();
            result = await response.json();
            return result!.status;
          }, { timeout: 420_000, intervals: [1000, 3000, 5000] }).not.toMatch(/^(queued|running)$/);
          expect(result?.status, JSON.stringify(result?.error)).toBe("completed");
          const payload = result!.payload!;
          const report = payload.report as JsonRecord;
          await testInfo.attach(`report-${attempt + 1}`, { body: JSON.stringify(payload, null, 2), contentType: "application/json" });
          const transcript = String(payload.transcript_full || "");
          expect(transcript.split(/\s+/).length).toBeGreaterThan(10);
          const input = report.input as JsonRecord;
          expect(input).toMatchObject({ provider, llm_model: model, expected_language: language, detected_language: language });
          expect(input.coaching_prompt_version).toBe("coaching_multilingual_v2");
          expect(input.llm_inference_profile).toBe(provider === "lmstudio" ? "lmstudio_bounded_v1" : "ollama_json_no_thinking_v1");
          expect(input.dry_run).not.toBe(true);
          expect(input.fixture_inference).not.toBe(true);
          expect(input.scoring_model_version).not.toBe("journey-fixture-v1");
          expect(report.warnings || []).not.toContain("journey_fixture_inference");
          expect(Number((report.timings_ms as JsonRecord).llm)).toBeGreaterThan(0);
          expect(Number((report.timings_ms as JsonRecord).coaching)).toBeGreaterThan(0);
          const scores = report.scores as JsonRecord;
          expect(scores.mode, JSON.stringify({ warnings: report.warnings, errors: report.errors })).toBe("hybrid");
          expect(typeof scores.llm).toBe("number");
          expect(Number(scores.final)).toBeGreaterThanOrEqual(1);
          expect(Number(scores.final)).toBeLessThanOrEqual(5);
          const rubric = report.rubric as JsonRecord;
          for (const field of ["fluency", "cohesion", "accuracy", "range", "overall"]) {
            expect(Number.isInteger(rubric[field])).toBe(true);
            expect(Number(rubric[field])).toBeGreaterThanOrEqual(1);
            expect(Number(rubric[field])).toBeLessThanOrEqual(5);
          }
          const coaching = report.coaching as JsonRecord;
          for (const field of ["coach_summary", "next_focus", "next_exercise"]) expect(String(coaching[field] || "").trim()).not.toBe("");
          expect(coaching.top_3_priorities).toHaveLength(3);
          // Lightweight smoke for wrong-language coaching; reports remain available for human review.
          const feedback = `${coaching.coach_summary} ${coaching.next_focus} ${coaching.next_exercise}`.toLowerCase();
          const markers = language === "it" ? /\b(?:il|la|le|un|una|che|di|per|con|nel|nella|tuo|tua|hai|puoi|frasi)\b/g : /\b(?:the|your|you|and|with|use|try|to|for|this|that|was|were)\b/g;
          expect((feedback.match(markers) || []).length, "Feedback should use the chosen UI language").toBeGreaterThanOrEqual(3);
          expect(report.warnings || []).not.toContain("coaching_unavailable");
          await expect(page).toHaveURL(/\/review$/, { timeout: 30_000 });
          await expect(page.getByTestId("review-transcript")).toHaveValue(transcript);
          const session = String(report.session_id);
          const detail = await (await request.get(`${backend}/v1/history/${session}`)).json();
          expect(detail.payload.report.coaching).toEqual(coaching);
          expect(detail.payload.meta.practice).toMatchObject({ goal: item.goal, retry_of_session_id: parent, prompt_text: firstPrompt });
          parent = session;
          await page.getByTestId("review-action-view-history").click();
          await page.reload();
          await page.getByTestId("history-language-filter").selectOption(language);
          await expect(page.getByTestId("history-detail-caption")).toContainText(session);
          await expect(page.getByTestId("practice-progress").locator("svg circle")).toHaveCount(attempt + 1);
          const audio = page.locator(`audio[src$="/${session}/audio"]`);
          await audio.evaluate(async (el: HTMLAudioElement) => { await el.play(); });
          await expect.poll(() => audio.evaluate((el: HTMLAudioElement) => el.currentTime)).toBeGreaterThan(0);
          await audio.evaluate((el: HTMLAudioElement) => el.pause());
          if (item.goal === "B1" && attempt === 0) await page.getByTestId("practice-retry").click();
        }
        await page.screenshot({ path: testInfo.outputPath(`${language}-${item.goal}-live-history.png`), fullPage: true });
      });
    }
  });
}

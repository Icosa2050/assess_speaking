import { expect, test as base, type Page, type TestInfo } from "@playwright/test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { readFileSync, writeFileSync } from "node:fs";
import { capture, downloadRecording, playAndSeek, retainJson, sha256 } from "./workflowEvidence";
import type { AssessmentStatusResponse, JsonRecord } from "../../src/lib/api/types";

// Per-test Chromium capture sources allow English/Italian WAVs to exercise real MediaRecorder.
const test = base.extend<{ speechFile: string; speechPage: Page }>({
  speechFile: ["", { option: true }],
  speechPage: async ({ playwright, baseURL, locale, speechFile }, use, testInfo) => {
    const browser = await playwright.chromium.launch({ headless: true, args: [
      // Decode/play against Chromium's null output sink on headless Macs.
      // Playback time and seeking remain asserted; microphone capture is real.
      "--disable-audio-output",
      "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
      ...(speechFile ? [`--use-file-for-fake-audio-capture=${speechFile}%noloop`] : []),
    ] });
    const context = await browser.newContext({ baseURL, locale, permissions: ["microphone"], acceptDownloads: true });
    try { await use(await context.newPage()); }
    finally {
      // The config starts tracing; retain this custom context before closing it.
      const trace = testInfo.outputPath("browser-trace.zip");
      await context.tracing.stop({ path: trace });
      await testInfo.attach("browser-trace", { path: trace, contentType: "application/zip" });
      await browser.close();
    }
  },
});

const backend = `http://127.0.0.1:${process.env.VOSTAVO_LIVE_BACKEND_PORT || "8816"}`;
const provider = process.env.LOCAL_E2E_PROVIDER || "ollama";
if (!["ollama", "lmstudio", "openrouter"].includes(provider)) throw new Error(`Unsupported live provider: ${provider}`);
const providerChoice = provider === "openrouter" ? "openrouter" : `${provider}_local`;
const providerName = provider === "lmstudio" ? "LM Studio" : provider === "openrouter" ? "OpenRouter" : "Ollama";
const endpoint = process.env.LOCAL_E2E_BASE_URL || ({
  lmstudio: "http://127.0.0.1:1234/v1", ollama: "http://127.0.0.1:11434/v1", openrouter: "https://openrouter.ai/api/v1",
}[provider]!);
const model = process.env.LOCAL_E2E_MODEL || (provider === "lmstudio" ? "qwen2.5-3b-instruct" : provider === "openrouter" ? "mistralai/mistral-small-3.2-24b-instruct" : process.env.OLLAMA_E2E_MODEL || "qwen3.5:4b");
const alternateModel = process.env.LOCAL_E2E_ALTERNATE_MODEL || model;
const whisper = process.env.OLLAMA_E2E_WHISPER || "large-v3";
const alternateWhisper = process.env.LIVE_E2E_ALTERNATE_WHISPER || "tiny";
const allowGuardedFallback = process.env.LIVE_E2E_ALLOW_GUARDED_FALLBACK === "1";
const root = fileURLToPath(new URL("../../..", import.meta.url));

async function saveSettings(page: Page, language: string, selectedModel: string, selectedWhisper: string) {
  await page.locator('a[href="/settings"]').click();
  // Wait for async initialization before editing the existing connection.
  await expect(page.getByTestId("runtime_connection.model")).toHaveValue(model);
  await page.getByTestId("settings.ui_locale").selectOption(language);
  await page.getByTestId("settings.whisper_model").selectOption(selectedWhisper);
  await page.getByTestId("runtime_connection.model").fill(selectedModel);
  const pending = page.waitForResponse(r => new URL(r.url()).pathname === "/v1/runtime/settings" && r.request().method() === "PUT");
  await page.getByTestId("runtime_connection.save_connection").click();
  const saved = await pending;
  expect(saved.ok()).toBeTruthy();
  expect((await saved.json()).whisper_model).toBe(selectedWhisper);
}

async function resetModelSettings(page: Page) {
  // Every case has its own browser but shares the isolated provider's saved app data.
  await page.goto("/settings");
  await expect(page.getByTestId("settings.connection_id")).not.toHaveValue("__new__");
  await page.getByTestId("runtime_connection.model").fill(model);
  const pending = page.waitForResponse(r => new URL(r.url()).pathname === "/v1/runtime/settings" && r.request().method() === "PUT");
  await page.getByTestId("runtime_connection.save_connection").click();
  expect((await pending).ok()).toBeTruthy();
}

async function labelAttempt(page: Page, label: string) {
  const disclosure = page.getByTestId("speak.optional_context");
  if (await disclosure.count()) {
    await disclosure.locator("summary").click();
    await page.getByTestId("speak.label").fill(label);
  }
}

async function retainStatus(page: Page, testInfo: TestInfo, name: string, status: unknown) {
  await retainJson(testInfo, name, status);
  await capture(page, testInfo, name);
}

const cases = [
  { goal: "B1", file: "travel_story.wav", en: "My last trip", it: "Il mio ultimo viaggio" },
  { goal: "B2", file: "remote_work.wav", en: "Advantages and disadvantages of remote work", it: "Vantaggi e svantaggi del lavoro da remoto" },
  { goal: "C1", file: "public_debate.wav", en: "Public debate and digital platforms", it: "Dibattito pubblico e piattaforme digitali" },
];

test(`${providerName} setup discovers the installed model and saves its connection`, async ({ speechPage: page, request }, testInfo) => {
  if (provider === "openrouter") {
    // Bootstrap through the real backend outside traced Playwright requests. The
    // key stays out of browser inputs/artifacts and the OS keychain; normal UI
    // connection testing/saving below then uses the backend's process-only store.
    const key = process.env.OPENROUTER_API_KEY;
    expect(Boolean(key), "OpenRouter needs its own assessment credential").toBe(true);
    const response = await fetch(`${backend}/v1/runtime/settings`, { method: "PUT",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ui_locale: "en", whisper_model: whisper,
        connection: { provider_choice: "openrouter", model, base_url: endpoint, api_key: key, label: "Isolated live OpenRouter" } }),
    });
    expect(response.ok, `Cloud credential bootstrap HTTP ${response.status}`).toBe(true);
  }
  await page.goto("/runtime-setup");
  if (provider === "openrouter") {
    await expect(page.getByTestId("runtime_connection.provider")).toHaveValue("openrouter");
    await expect(page.getByTestId("runtime_connection.model")).toHaveValue(model);
  }
  await page.getByTestId("runtime_connection.provider").selectOption(providerChoice);
  await page.getByTestId("runtime_connection.base_url").fill(endpoint);
  if (provider !== "openrouter") {
    await page.getByTestId("runtime_setup.detect_local_models").click();
    // Local connection checks allow 30 seconds for model cold starts.
    await expect(page.getByRole("combobox", { name: "Detected local models" })).toBeVisible({ timeout: 35_000 });
    await page.getByRole("combobox", { name: "Detected local models" }).selectOption(model);
  }
  await page.getByTestId("runtime_connection.model").fill(model);
  await page.getByRole("combobox", { name: "Whisper model", exact: true }).selectOption(whisper);
  const resultPromise = page.waitForResponse(response => new URL(response.url()).pathname === "/v1/runtime/settings/test-connection");
  await page.getByTestId("runtime_connection.test_connection").click();
  const response = await resultPromise;
  expect(response.ok()).toBeTruthy();
  expect((await response.json()).discovered_models).toContain(model);
  const savePromise = page.waitForResponse(r => new URL(r.url()).pathname === "/v1/runtime/settings" && r.request().method() === "PUT");
  await page.getByTestId("runtime_connection.save_connection").click();
  expect((await savePromise).ok()).toBeTruthy();
  const health = await request.get(`${backend}/v1/health`);
  expect(health.ok()).toBeTruthy();
  await retainJson(testInfo, "health", await health.json());
  await capture(page, testInfo, "provider-setup");
});

for (const language of ["en", "it"] as const) {
  test.describe(language, () => {
    test.use({ locale: language });
    for (const item of cases) {
      test.describe(item.goal, () => {
      const sample = path.join(root, "samples/cefr", language, item.goal, item.file);
      test.use({ speechFile: sample });
      test(`${language} ${item.goal}: real Whisper + ${providerName} feedback${allowGuardedFallback ? " or guarded validation fallback" : ""}, saved history${item.goal === "B1" ? " and retry" : ""}`, async ({ speechPage: page, request }, testInfo) => {
        expect(path.basename(testInfo.config.configFile || "")).toBe("playwright.ollama.config.ts");
        const modelsResponse = await request.get(`${endpoint}/models`);
        expect(modelsResponse.ok(), `${providerName} must be running`).toBeTruthy();
        expect((await modelsResponse.json()).data).toEqual(expect.arrayContaining([expect.objectContaining({ id: model })]));
        const cached = await request.get(`${backend}/v1/runtime/whisper-models/${whisper}`);
        expect((await cached.json()).cached, "Download Whisper before running this offline-ASR test").toBe(true);
        await resetModelSettings(page);
        await saveSettings(page, language, model, whisper);
        await capture(page, testInfo, "analysis-settings");
        await page.goto("/session-setup");
        await page.getByTestId("setup.speaker_id").fill(`live-${language}-${item.goal}`);
        await page.getByTestId("setup.customize_details").click();
        await page.getByTestId("setup.learning_language").selectOption(language);
        await page.getByTestId("setup.cefr").selectOption(item.goal);
        await page.getByTestId("setup.advanced_topic").click();
        await page.getByTestId("setup.custom_theme").fill(item[language]);
        await page.getByTestId("setup.duration").selectOption("90");
        await page.getByTestId("setup.continue").click();
        await capture(page, testInfo, "session-setup");
        let parent = "";
        let firstPayload: JsonRecord | undefined;
        let firstSession = "";
        let firstPrompt = "";
        for (let attempt = 0; attempt < (item.goal === "B1" ? 2 : 1); attempt++) {
          const selectedWhisper = attempt ? alternateWhisper : whisper;
          const selectedModel = attempt ? alternateModel : model;
          if (attempt) {
            // Change runtime through the UI while preserving the retry draft in memory.
            await saveSettings(page, language, selectedModel, selectedWhisper);
            await capture(page, testInfo, "changed-analysis-settings");
            await page.locator('a[href="/speak"]').click();
            await page.getByTestId("speak.input_mode_record").click();
            await page.getByTestId("speak.record_start").click();
            await expect(page.getByTestId("speak.record_stop")).toBeVisible();
            // This is real elapsed MediaRecorder input, supplied by the tracked WAV, not a beep.
            const sampleDuration = (readFileSync(sample).length - 44) / 32000;
            await page.waitForTimeout((sampleDuration + 1) * 1000);
            await page.getByTestId("speak.record_stop").click();
          } else {
            await page.getByTestId("speak.input_mode_upload").click();
            await page.getByTestId("speak.upload_input").setInputFiles(sample);
          }
          await expect(page.getByTestId("speak.submit")).toBeEnabled();
          await labelAttempt(page, `${provider}-${language}-${item.goal}-${attempt ? "recorded" : "upload"}`);
          const downloaded = await downloadRecording(page, testInfo, `attempt-${attempt + 1}-input`, attempt ? undefined : sample);
          await capture(page, testInfo, `attempt-${attempt + 1}-speak`);
          const createdPromise = page.waitForResponse(response => new URL(response.url()).pathname === "/v1/assessments" && response.request().method() === "POST");
          await page.getByTestId("speak.submit").click();
          const created = await createdPromise;
          expect(created.ok()).toBeTruthy();
          const submission = created.request().postDataJSON();
          expect(submission).toMatchObject({ provider, llm_model: selectedModel, whisper: selectedWhisper, expected_language: language, feedback_language: language, target_cefr: item.goal, retry_of_session_id: parent });
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
          await retainJson(testInfo, `attempt-${attempt + 1}-submission`, submission);
          await retainJson(testInfo, `attempt-${attempt + 1}-status`, result);
          expect(result?.status, JSON.stringify(result?.error)).toBe("completed");
          const payload = result!.payload!;
          const report = payload.report as JsonRecord;
          await testInfo.attach(`report-${attempt + 1}`, { body: JSON.stringify(payload, null, 2), contentType: "application/json" });
          const transcript = String(payload.transcript_full || "");
          expect(transcript.split(/\s+/).length).toBeGreaterThan(10);
          if (attempt) {
            const tokens = (text: string) => new Set(text.toLowerCase().match(/\p{L}+/gu) || []);
            const original = tokens(String(firstPayload!.transcript_full));
            const recorded = tokens(transcript);
            const overlap = [...original].filter(token => recorded.has(token)).length / original.size;
            expect(overlap, "Recorded speech should preserve the supplied WAV content, rather than synthetic beeps").toBeGreaterThan(.6);
          }
          const input = report.input as JsonRecord;
          expect(input).toMatchObject({ provider, llm_model: selectedModel, whisper_model: selectedWhisper, expected_language: language, detected_language: language });
          expect(input.coaching_prompt_version).toBe("coaching_multilingual_v6");
          const reportWarnings = (report.warnings as string[] || []);
          const uncertain = reportWarnings.includes("transcript_uncertain");
          const rubricFallback = uncertain || (allowGuardedFallback && reportWarnings.includes("llm_invalid_schema"));
          const coachingFallback = reportWarnings.includes("coaching_unavailable");
          if (report.rubric && provider !== "openrouter") expect(input.llm_inference_profile).toBe(provider === "lmstudio" ? "lmstudio_bounded_v1" : "ollama_schema_no_thinking_v2");
          expect(input.dry_run).not.toBe(true);
          expect(input.fixture_inference).not.toBe(true);
          expect(input.scoring_model_version).not.toBe("journey-fixture-v1");
          expect(reportWarnings).not.toContain("journey_fixture_inference");
          // This explicit functional mode never turns transport failures into passes.
          expect(reportWarnings).not.toContain("llm_unavailable");
          const scores = report.scores as JsonRecord;
          expect(Number(scores.final)).toBeGreaterThanOrEqual(1);
          expect(Number(scores.final)).toBeLessThanOrEqual(5);
          if (rubricFallback) {
            expect(report.requires_human_review).toBe(true);
            expect(report.rubric).toBeNull();
            expect(scores.llm).toBeNull();
            expect(scores.mode).toBe("deterministic_only");
            expect(Number((report.timings_ms as JsonRecord).coaching)).toBe(0);
            if (uncertain) {
              expect((input.transcript_quality as JsonRecord).status).toBe("uncertain");
              if ((input.transcript_quality as JsonRecord).trigger === "quoted_asr_evidence_uncertain") {
                expect(Number((report.timings_ms as JsonRecord).llm)).toBeGreaterThan(0);
              } else {
                expect(Number((report.timings_ms as JsonRecord).llm)).toBe(0);
              }
            } else {
              expect((report.errors as string[]).length).toBeGreaterThan(0);
              expect(Number((report.timings_ms as JsonRecord).llm)).toBeGreaterThan(0);
            }
          } else {
            expect(Number((report.timings_ms as JsonRecord).llm)).toBeGreaterThan(0);
            expect(Number((report.timings_ms as JsonRecord).coaching)).toBeGreaterThan(0);
            expect(scores.mode, JSON.stringify({ warnings: report.warnings, errors: report.errors })).toBe("hybrid");
            expect(typeof scores.llm).toBe("number");
            const rubric = report.rubric as JsonRecord;
            for (const field of ["fluency", "cohesion", "accuracy", "range", "overall"]) {
              expect(Number.isInteger(rubric[field])).toBe(true);
              expect(Number(rubric[field])).toBeGreaterThanOrEqual(1);
              expect(Number(rubric[field])).toBeLessThanOrEqual(5);
            }
          }
          await retainJson(testInfo, `attempt-${attempt + 1}-output-contract`, {
            acceptance_scope: allowGuardedFallback ? "workflow_only_with_guarded_validation_fallback" : "strict_model_output_and_workflow",
            output_contract_accepted: !rubricFallback && !coachingFallback,
            rubric_fallback: rubricFallback, coaching_fallback: coachingFallback,
            transcript_uncertain: uncertain, warnings: reportWarnings,
            linguistic_quality: "Requires separate human review",
          });
          const coaching = report.coaching as JsonRecord;
          for (const field of ["coach_summary", "next_focus", "next_exercise"]) expect(String(coaching[field] || "").trim()).not.toBe("");
          expect(coaching.top_3_priorities).toHaveLength(3);
          // Lightweight smoke for wrong-language coaching; reports remain available for human review.
          const feedback = `${coaching.coach_summary} ${coaching.next_focus} ${coaching.next_exercise}`.toLowerCase();
          const markers = language === "it" ? /\b(?:il|la|le|un|una|che|di|per|con|nel|nella|tuo|tua|hai|puoi|frasi)\b/g : /\b(?:the|your|you|and|with|use|try|to|for|this|that|was|were)\b/g;
          expect((feedback.match(markers) || []).length, "Feedback should use the chosen UI language").toBeGreaterThanOrEqual(3);
          if (!allowGuardedFallback) expect(reportWarnings).not.toContain("coaching_unavailable");
          if (!rubricFallback && !coachingFallback) {
            expect(coaching.retry_duration_sec).toBe(90);
            expect(String(coaching.next_attempt_instruction)).toMatch(/90|1[.,]5/);
          }
          if (attempt) expect(report.progress_delta).toBeNull();
          await expect(page).toHaveURL(/\/review$/, { timeout: 30_000 });
          await expect(page.getByTestId("review-transcript")).toHaveValue(transcript);
          if (rubricFallback || coachingFallback) await expect(page.getByTestId("review-general-practice-tips")).toBeVisible();
          if (uncertain) await expect(page.getByTestId("review-transcript-uncertain")).toBeVisible();
          await capture(page, testInfo, `attempt-${attempt + 1}-review`);
          const session = String(report.session_id);
          if (!attempt) { firstPayload = payload; firstSession = session; }
          const detail = await (await request.get(`${backend}/v1/history/${session}`)).json();
          expect(detail.payload.report.coaching).toEqual(coaching);
          expect(detail.payload.meta.practice).toMatchObject({ goal: item.goal, retry_of_session_id: parent, prompt_text: firstPrompt });
          expect(detail.payload.meta.practice).toMatchObject({ provider, model: selectedModel, whisper_model: selectedWhisper });
          const history = (await (await request.get(`${backend}/v1/history`)).json()).items;
          const row = history.find((entry: { session_id: string }) => entry.session_id === session);
          expect(row.practice).toMatchObject({ model: selectedModel, whisper_model: selectedWhisper });
          expect(row.elapsed_wpm).toBeCloseTo(Number(row.word_count) * 60 / Number((report.metrics as JsonRecord).duration_sec), 1);
          if (attempt) {
            expect(detail.payload.meta.practice.whisper_model).not.toBe(((firstPayload!.meta as JsonRecord).practice as JsonRecord).whisper_model);
            const oldDetail = await (await request.get(`${backend}/v1/history/${firstSession}`)).json();
            expect(oldDetail.payload.report).toEqual(firstPayload!.report);
          }
          parent = session;
          await page.getByTestId("review-action-view-history").click();
          await page.reload();
          await page.getByTestId("history-language-filter").selectOption(language);
          await expect(page.getByTestId("history-detail-caption")).toContainText(session);
          // A changed analysis/model is a distinct comparison cohort even for a linked retry.
          await expect(page.getByTestId("practice-progress").locator("svg circle")).toHaveCount(attempt && selectedWhisper !== whisper ? 1 : attempt + 1);
          const playbackDuration = await playAndSeek(page, `audio[src$="/${session}/audio"]`);
          expect(playbackDuration).toBeCloseTo(Number((report.metrics as JsonRecord).duration_sec), 0);
          const range = await request.get(`${backend}/v1/history/${session}/audio`, { headers: { Range: "bytes=0-63" } });
          expect(range.status()).toBe(206);
          expect((await range.body()).length).toBe(64);
          const recording = await request.get(`${backend}/v1/history/${session}/audio`);
          expect(recording.ok()).toBeTruthy();
          const retainedAudio = testInfo.outputPath(`attempt-${attempt + 1}-saved-audio${attempt ? ".webm" : ".wav"}`);
          const audioBytes = await recording.body();
          writeFileSync(retainedAudio, audioBytes);
          expect(sha256(audioBytes)).toBe(downloaded.sha256);
          await testInfo.attach(`attempt-${attempt + 1}-saved-audio`, { path: retainedAudio, contentType: attempt ? "audio/webm" : "audio/wav" });
          await retainStatus(page, testInfo, `attempt-${attempt + 1}-history`, detail);
          await retainJson(testInfo, `attempt-${attempt + 1}-evidence`, { session, provider, model: selectedModel,
            whisper: selectedWhisper, input: downloaded, source: sample, source_sha256: sha256(readFileSync(sample)),
            saved_audio_sha256: sha256(audioBytes), playbackDuration, range_bytes: 64 });
          if (item.goal === "B1" && attempt === 0) await page.getByTestId("practice-retry").click();
        }
        await page.screenshot({ path: testInfo.outputPath(`${language}-${item.goal}-live-history.png`), fullPage: true });
      });
      });
    }
  });
}

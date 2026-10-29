import { expect, test } from "../fixtures";
import { readyReadiness } from "../readinessStub";

test("unreachable local tutor shows rescue and refreshes to Ready before Speak", async ({ page }) => {
  let reachable = false;
  let calls = 0;
  await page.route("**/v1/readiness", (route) => {
    calls += 1;
    const checks = readyReadiness.checks.map((check) => check.key === "ai_tutor" && !reachable
      ? { ...check, state: "unavailable", reason: "local_service_unreachable" } : check);
    return route.fulfill({ json: { ready: reachable, checks } });
  });
  await page.route("**/v1/runtime", (route) => route.fulfill({ json: {
    configured: true, provider: "ollama", model: "fixture-model",
    base_url: "http://127.0.0.1:11434/v1", requires_api_key: false, has_api_key: false,
  } }));

  await page.goto("/");
  await page.getByTestId("home.setup_guide_button").click();
  const tutor = page.getByTestId("runtime_setup.setup_guide.ai_tutor");
  await expect(tutor).toHaveAttribute("data-status", "setup");
  await expect(tutor).toContainText("The local tutor service cannot be reached.");
  await expect(tutor).toContainText("Start the local tutor app and check its address");
  await expect(page.getByRole("button", { name: "Choose a practice task" })).toHaveCount(0);

  reachable = true;
  await page.getByRole("button", { name: "Refresh setup status" }).first().click();
  await expect(tutor).toHaveAttribute("data-status", "ready");
  await expect(page.getByRole("button", { name: "Choose a practice task" })).toBeEnabled();
  expect(calls).toBeGreaterThanOrEqual(2);

  await page.getByRole("button", { name: "Choose a practice task" }).click();
  await page.getByTestId("setup.speaker_id").fill("readiness-journey");
  await page.getByTestId("setup.recommended_start").click();
  await expect(page.getByRole("button", { name: "Start speaking" })).toBeVisible();
  await page.getByRole("button", { name: "Start speaking" }).click();
  await expect(page.getByTestId("speak.recording_panel")).toBeVisible();
  await expect(page.getByTestId("speak.record_start")).toBeDisabled();
});

test("ready Groq speech shows cloud processing copy and cloud settings action", async ({ page }) => {
  const checks = readyReadiness.checks.map((check) => check.key === "speech_recognition"
    ? { ...check, detail_args: { provider: "groq", model: "whisper-large-v3" } } : check);
  await page.route("**/v1/readiness", (route) => route.fulfill({ json: { ready: true, checks } }));
  await page.route("**/v1/runtime/settings", (route) => route.fulfill({ json: {
    ui_locale: "en", whisper_model: "small", active_connection_id: "",
    asr_provider: "groq", connections: [],
  } }));

  await page.goto("/runtime-setup");
  const speech = page.getByTestId("runtime_setup.setup_guide.speech_recognition");
  await expect(speech).toHaveAttribute("data-status", "ready");
  await expect(speech).toContainText("Groq cloud transcription is selected and its saved key is present.");
  await expect(speech).not.toContainText("on this device");
  await expect(page.getByTestId("runtime_setup.setup_guide.speech_recognition.action")).toHaveText("Manage cloud transcription");
});

test("Groq readiness opens cloud rescue when saved settings are unavailable", async ({ page }) => {
  const checks = readyReadiness.checks.map((check) => check.key === "speech_recognition"
    ? { ...check, detail_args: { provider: "groq", model: "whisper-large-v3" } } : check);
  await page.route("**/v1/readiness", (route) => route.fulfill({ json: { ready: true, checks } }));
  await page.route("**/v1/runtime/settings", (route) => route.fulfill({ status: 503, json: { detail: "offline" } }));
  await page.addInitScript(() => {
    Element.prototype.scrollIntoView = function () { this.setAttribute("data-scrolled", "true"); };
  });

  await page.goto("/runtime-setup");
  const speech = page.getByTestId("runtime_setup.setup_guide.speech_recognition");
  await expect(speech).toHaveAttribute("data-status", "ready");
  await expect(speech).toContainText("Groq cloud transcription is selected");
  await expect(page.getByText("Saved cloud connections could not be loaded. Retry saved settings before changing Groq transcription.")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#runtime-setup-cloud details")).toHaveAttribute("open", "");
  await speech.getByRole("button", { name: "Manage cloud transcription" }).click();
  await expect(page.locator("#runtime-setup-cloud")).toHaveAttribute("data-scrolled", "true");
  await expect(page.locator("#runtime-setup-whisper")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Retry saved settings" })).toBeVisible();
});

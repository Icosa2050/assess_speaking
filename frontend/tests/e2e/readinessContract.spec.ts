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

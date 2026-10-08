import { expect, test } from "../fixtures";

test("nickname survives navigation and starting another practice, including edits and clearing", async ({ page }) => {
  await page.route("**/v1/runtime", (route) => route.fulfill({ json: {
    configured: true, provider: "ollama", model: "nickname-fixture",
    base_url: "http://127.0.0.1:11434/v1", requires_api_key: false, has_api_key: false,
  } }));
  await page.route("**/v1/runtime/settings", (route) => route.fulfill({ json: {
    ui_locale: "en", whisper_model: "small", active_connection_id: "", connections: [],
  } }));
  await page.route("**/v1/diagnostics", (route) => route.fulfill({ json: { items: [] } }));
  await page.route("**/v1/history", (route) => route.fulfill({ json: { items: [] } }));

  await page.goto("/session-setup");
  const nickname = page.getByTestId("setup.speaker_id");
  await nickname.fill("maria");
  await page.getByRole("link", { name: "Practice Home", exact: true }).click();
  await page.getByRole("link", { name: "Session Setup", exact: true }).click();
  await expect(nickname).toHaveValue("maria");

  await nickname.fill("alex");
  await page.getByRole("link", { name: "Practice Home", exact: true }).click();
  await page.getByTestId("home.start_new").click();
  await expect(nickname).toHaveValue("alex");

  await nickname.fill("");
  await page.getByRole("link", { name: "Practice Home", exact: true }).click();
  await page.getByRole("link", { name: "Session Setup", exact: true }).click();
  await expect(nickname).toHaveValue("");
});

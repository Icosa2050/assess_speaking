import { expect, test, type Page, type Route } from "@playwright/test";

const runtimeSettings = {
  ui_locale: "en",
  whisper_model: "large-v3",
  active_connection_id: "conn-library-guide",
  connections: [
    {
      connection_id: "conn-library-guide",
      provider_key: "ollama",
      provider_choice: "ollama_local",
      provider_label: "Ollama local",
      label: "Library guide local",
      model: "llama3.2:3b",
      base_url: "http://127.0.0.1:11434/v1",
      is_default: true,
      is_local: true,
      requires_api_key: false,
      has_api_key: false,
      secret_state: "absent",
      last_test_status: "passed",
      last_tested_at: "2026-07-21T09:00:00Z",
      openrouter_http_referer: "",
      openrouter_app_title: "",
      provider_metadata: {},
    },
  ],
};

const sampleItems = [
  {
    sample_id: "en_B1_travel_story",
    language: "en",
    cefr: "B1",
    title: "TRAVEL STORY",
    path: "/samples/cefr/en/B1/travel_story.wav",
  },
  {
    sample_id: "it_C1_public_debate",
    language: "it",
    cefr: "C1",
    title: "public debate",
    path: "/samples/cefr/it/C1/public_debate.wav",
  },
];

const fulfillJson = (route: Route, json: unknown) =>
  route.fulfill({
    contentType: "application/json",
    json,
  });

const installLibraryGuideBackend = async (page: Page) => {
  await page.route("**/v1/diagnostics", (route) => fulfillJson(route, { items: [] }));
  await page.route("**/v1/runtime", (route) =>
    fulfillJson(route, {
      configured: true,
      provider: "ollama",
      model: "llama3.2:3b",
      base_url: "http://127.0.0.1:11434/v1",
      requires_api_key: false,
      has_api_key: false,
    }),
  );
  await page.route("**/v1/runtime/settings", (route) => fulfillJson(route, runtimeSettings));
  await page.route("**/v1/history", (route) => fulfillJson(route, { items: [] }));
  await page.route("**/v1/samples", (route) => fulfillJson(route, { items: sampleItems }));
};

test.describe("library and guide browser flow", () => {
  test("keeps Guide reachable and carries a prepared sample into Session Setup and Speak", async ({
    page,
  }) => {
    await installLibraryGuideBackend(page);

    await page.goto("/");

    await page.getByTestId("home.open_guide").click();
    await expect(page).toHaveURL(/\/guide$/);
    await expect(page.getByTestId("guide.practice_support_intro")).toBeVisible();
    await expect(page.getByRole("link", { name: "Validation gates" })).toBeVisible();

    await page.getByTestId("home.open_library").click();
    await expect(page).toHaveURL(/\/library$/);
    await expect(page.getByRole("heading", { name: "Pick your next exercise" })).toBeVisible();
    await expect(page.getByText("Travel Story")).toBeVisible();

    await page.getByTestId("library.sample_prepare").first().click();
    await expect(page).toHaveURL(/\/session-setup$/);
    await expect(page.getByTestId("setup.layout")).toBeVisible();
    await expect(page.getByDisplayValue("Travel Story")).toBeVisible();

    await page.getByTestId("setup.speaker_id").fill("playwright-library-guide");
    await page.getByTestId("setup.continue").click();

    await expect(page).toHaveURL(/\/speak$/);
    await expect(page.getByTestId("speak.session_summary")).toContainText(
      "Speaker playwright-library-guide",
    );
    await expect(page.getByTestId("speak.session_summary")).toContainText("Travel Story");
  });
});

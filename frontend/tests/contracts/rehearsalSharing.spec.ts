import { expect, test } from "../fixtures";
import { createRehearsal } from "../../src/lib/rehearsal/session";

test("Rehearsal preserves completed work when sharing changes and reload resumes only remaining parts", async ({ page, request }) => {
  const backend = process.env.VOSTAVO_FIXTURE_BACKEND_URL!;
  const connection = (model: string) => ({ provider_choice: "ollama_local", label: model, model, base_url: "http://127.0.0.1:11434/v1" });
  expect((await request.put(`${backend}/v1/runtime/settings`, { data: { ui_locale: "en", whisper_model: "small", connection: connection("journey-fixture") } })).ok()).toBe(true);
  const session = createRehearsal("en", "B2", "sharing-fixture", { provider: "ollama", model: "journey-fixture", baseUrl: "", whisper: "small", feedbackLanguage: "en" });
  session.phase = "review"; session.parts.forEach(part => { part.recorded = true; });
  await page.goto("/rehearsal");
  // Seed an already-recorded rehearsal, not calibration or microphone state.
  // Separate native recording journeys cover capture; this probes recovery.
  await page.evaluate(async saved => {
    const rate = 16000, count = rate * 31, bytes = new ArrayBuffer(44 + count * 2), view = new DataView(bytes);
    const label = (offset: number, text: string) => [...text].forEach((letter, i) => view.setUint8(offset + i, letter.charCodeAt(0)));
    label(0, "RIFF"); view.setUint32(4, bytes.byteLength - 8, true); label(8, "WAVEfmt "); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); label(36, "data"); view.setUint32(40, count * 2, true);
    for (let i = 0; i < count; i++) view.setInt16(44 + i * 2, Math.round(3000 * Math.sin(2 * Math.PI * 440 * i / rate)), true);
    const db = await new Promise<IDBDatabase>((resolve, reject) => { const open = indexedDB.open("vostavo-rehearsals", 1);
      open.onupgradeneeded = () => { open.result.createObjectStore("sessions", { keyPath: "id" }); open.result.createObjectStore("recordings"); };
      open.onsuccess = () => resolve(open.result); open.onerror = () => reject(open.error); });
    await new Promise<void>((resolve, reject) => { const tx = db.transaction(["sessions", "recordings"], "readwrite");
      tx.objectStore("sessions").put(saved); saved.parts.forEach((_, index) => tx.objectStore("recordings").put(new Blob([bytes], { type: "audio/wav" }), `${saved.id}:${index}`));
      tx.oncomplete = () => resolve(); tx.onabort = () => reject(tx.error); }); db.close();
  }, session);
  await page.reload(); await page.getByRole("button", { name: /sharing-fixture/ }).click();
  const created: { id: string; model: string }[] = [];
  page.on("response", async response => { if (new URL(response.url()).pathname === "/v1/assessments" && response.request().method() === "POST" && response.ok()) {
    created.push({ id: (await response.json()).assessment_id, model: response.request().postDataJSON().llm_model });
  } });
  let changed = false;
  await page.route("**/v1/assessments/*", async route => {
    const response = await route.fetch();
    if (!changed && (await response.json()).status === "completed") {
      changed = true;
      expect((await request.put(`${backend}/v1/runtime/settings`, { data: { connection: connection("journey-fixture-next") } })).ok()).toBe(true);
    }
    await route.fulfill({ response });
  });
  await page.getByTestId("rehearsal-analyse").click();
  await expect(page.getByRole("alert")).toContainText("Sharing destinations changed");
  expect(created).toHaveLength(1);
  await expect(page.getByTestId("rehearsal-result-0")).toBeVisible();
  await page.unroute("**/v1/assessments/*");
  await page.reload(); await page.getByRole("button", { name: /sharing-fixture/ }).click();
  await expect(page.getByTestId("sharing-summary")).toContainText("journey-fixture-next");
  await page.getByTestId("rehearsal-analyse").click();
  await expect(page.getByTestId("rehearsal-retry-2")).toBeVisible({ timeout: 45_000 });
  expect(created.map(item => item.model)).toEqual(["journey-fixture", "journey-fixture-next", "journey-fixture-next"]);
  const original = await (await request.get(`${backend}/v1/assessments/${created[0].id}`)).json();
  expect(original.payload.report.input.llm_model).toBe("journey-fixture");
});

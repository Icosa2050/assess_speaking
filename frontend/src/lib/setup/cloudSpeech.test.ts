import { describe, expect, it } from "vitest";

import type { RuntimeCloudSettingsResponse, RuntimeSettingsResponse } from "@/lib/api/types";
import { hasSelectedGroqConnection } from "./cloudSpeech";

const runtimeSettings: RuntimeSettingsResponse = {
  ui_locale: "en", whisper_model: "small", active_connection_id: "", asr_provider: "groq",
  connections: [{
    connection_id: "groq-1", provider_key: "groq", provider_choice: "groq", provider_label: "Groq",
    label: "Speech", model: "whisper-large-v3", base_url: "https://api.groq.com", is_default: false,
    is_local: false, requires_api_key: true, has_api_key: true, secret_state: "present",
    last_test_status: "", last_tested_at: "", openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {},
  }],
};

const cloudSettings = (id: string): RuntimeCloudSettingsResponse => ({
  settings: { asr_provider: "groq", asr_connection_id: id },
});

describe("cloud speech connection", () => {
  it("requires the selected Groq connection and its credential", () => {
    expect(hasSelectedGroqConnection(runtimeSettings, cloudSettings("groq-1"))).toBe(true);
    expect(hasSelectedGroqConnection(runtimeSettings, cloudSettings("deleted"))).toBe(false);
    expect(hasSelectedGroqConnection({ ...runtimeSettings, connections: [{ ...runtimeSettings.connections[0], has_api_key: false }] }, cloudSettings("groq-1"))).toBe(false);
  });
});

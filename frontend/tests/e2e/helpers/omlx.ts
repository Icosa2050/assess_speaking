import { existsSync, readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

export const readOmlxConfig = () => {
  const settingsPath = path.join(os.homedir(), ".omlx", "settings.json");
  const settings = existsSync(settingsPath)
    ? JSON.parse(readFileSync(settingsPath, "utf8")) as {
        server?: { host?: string; port?: number };
        auth?: { api_key?: string };
      }
    : {};
  const configuredHost = settings.server?.host || "127.0.0.1";
  const host = ["0.0.0.0", "::"].includes(configuredHost) ? "127.0.0.1" : configuredHost;
  const urlHost = host.includes(":") ? `[${host}]` : host;
  const localBaseUrl = `http://${urlHost}:${settings.server?.port || 8000}/v1`;
  const baseUrl = (process.env.OMLX_BASE_URL || localBaseUrl).replace(/\/+$/, "");
  const model = process.env.OMLX_MODEL?.trim() || "";
  // Never send the local server's key to an explicitly configured different server.
  const apiKey = process.env.OMLX_API_KEY ?? (baseUrl === localBaseUrl ? settings.auth?.api_key || "" : "");
  return { baseUrl, model, apiKey };
};

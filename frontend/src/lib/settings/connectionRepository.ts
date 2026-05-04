import type { RuntimeSettingsConnection } from "@/lib/api/types";

export type SavedConnectionSecretState = "absent" | "present" | "missing";

export interface SavedConnectionRecord {
  connectionId: string;
  label: string;
  providerKey: string;
  providerLabel: string;
  model: string;
  baseUrl: string;
  isDefault: boolean;
  hasApiKey: boolean;
  secretState: SavedConnectionSecretState;
  lastTestStatus: string;
  lastTestedAt: string;
}

const trimString = (value: unknown): string => String(value || "").trim();

export const toSavedConnectionRecord = (
  connection: RuntimeSettingsConnection,
): SavedConnectionRecord => ({
  connectionId: trimString(connection.connection_id),
  label: trimString(connection.label) || trimString(connection.connection_id),
  providerKey: trimString(connection.provider_key),
  providerLabel: trimString(connection.provider_label) || trimString(connection.provider_key),
  model: trimString(connection.model),
  baseUrl: trimString(connection.base_url),
  isDefault: Boolean(connection.is_default),
  hasApiKey: Boolean(connection.has_api_key),
  secretState:
    connection.secret_state === "present" || connection.secret_state === "missing"
      ? connection.secret_state
      : "absent",
  lastTestStatus: trimString(connection.last_test_status),
  lastTestedAt: trimString(connection.last_tested_at),
});

export const toSavedConnectionRecords = (
  connections: RuntimeSettingsConnection[],
): SavedConnectionRecord[] =>
  connections
    .map((connection) => toSavedConnectionRecord(connection))
    .sort((left, right) => {
      if (left.isDefault && !right.isDefault) {
        return -1;
      }
      if (!left.isDefault && right.isDefault) {
        return 1;
      }
      return left.label.localeCompare(right.label);
    });

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { RecorderPanel } from "./RecorderPanel";

const props = { canRemove: false, inputMode: "record" as const, onInputModeChange: vi.fn(), onRemove: vi.fn(),
  previewUrl: "", showReadyCheckpoint: false, statusMessage: "ready", statusTone: "info" as const, translate: (key: string) => key };
afterEach(() => vi.unstubAllGlobals());
function microphone() {
  const stop = vi.fn();
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop }] }) } });
  return stop;
}
it("releases the acquired microphone if constructing MediaRecorder fails", async () => {
  const stop = microphone();
  vi.stubGlobal("MediaRecorder", class { static isTypeSupported() { return true; } constructor() { throw new DOMException("Unsupported", "NotSupportedError"); } });
  render(<RecorderPanel {...props} onFileSelected={vi.fn()} />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  await waitFor(() => expect(stop).toHaveBeenCalledOnce());
});
it("does not present a recorder error's partial chunks as a saved answer", async () => {
  const stop = microphone(); const selected = vi.fn();
  let recorder: FakeRecorder;
  class FakeRecorder {
    static isTypeSupported() { return true; }
    state = "inactive"; mimeType = "audio/webm";
    onerror?: (event: object) => void; onstop?: () => void; ondataavailable?: (event: { data: Blob }) => void;
    constructor() { recorder = this; }
    start() { this.state = "recording"; }
    stop() { this.state = "inactive"; this.onstop?.(); }
  }
  vi.stubGlobal("MediaRecorder", FakeRecorder);
  render(<RecorderPanel {...props} onFileSelected={selected} />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  await screen.findByTestId("speak.record_stop");
  act(() => { recorder.onerror?.({}); recorder.ondataavailable?.({ data: new Blob(["partial"]) }); recorder.stop(); });
  expect(stop).toHaveBeenCalled();
  expect(selected.mock.calls.some(([file]) => file instanceof File)).toBe(false);
  expect(screen.getByTestId("speak.recording_visualizer").getAttribute("data-recording-state")).toBe("error");
});

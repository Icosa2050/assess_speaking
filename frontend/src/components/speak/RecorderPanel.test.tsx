import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { RecorderPanel } from "./RecorderPanel";

const props = { microphoneSetupPassed: true, canRemove: false, inputMode: "record" as const, onInputModeChange: vi.fn(), onRemove: vi.fn(),
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

it("does not claim clear input merely because a recording was saved", async () => {
  microphone();
  const status = vi.fn();
  let recorder!: FakeRecorder;
  class FakeRecorder {
    static isTypeSupported() { return true; }
    state = "inactive"; mimeType = "audio/webm";
    onstop?: () => void; ondataavailable?: (event: { data: Blob }) => void;
    constructor() { recorder = this; }
    start() { this.state = "recording"; }
    requestData() { this.ondataavailable?.({ data: new Blob(["recorded audio"]) }); }
    stop() { this.state = "inactive"; this.onstop?.(); }
  }
  vi.stubGlobal("MediaRecorder", FakeRecorder);
  render(<RecorderPanel {...props} onFileSelected={vi.fn()} onMicrophoneStatusChange={status} />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  await screen.findByTestId("speak.record_stop");
  expect(status).not.toHaveBeenCalledWith("ready");
  fireEvent.click(screen.getByTestId("speak.record_stop"));
  expect(recorder.state).toBe("inactive");
  expect(status).not.toHaveBeenCalledWith("ready");
});

it("requires setup confirmation before asking for recording permission", () => {
  microphone(); const setup = vi.fn();
  render(<RecorderPanel {...props} microphoneSetupPassed={false} onFileSelected={vi.fn()} onSetupMicrophone={setup} />);
  expect((screen.getByTestId("speak.record_start") as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
  fireEvent.click(screen.getByTestId("speak.microphone_setup"));
  expect(setup).toHaveBeenCalledOnce();
});

it("preserves a take but reports its actual short duration when the wall timer exceeds 30 seconds", async () => {
  const capture = await import("@/lib/setup/audioCapture");
  const probe = vi.spyOn(capture, "capturedAudioSeconds").mockResolvedValue(24.5);
  const now = vi.spyOn(Date, "now").mockReturnValue(100000);
  try {
    microphone(); const selected = vi.fn();
    class FakeRecorder {
      static isTypeSupported() { return true; }
      state = "inactive"; mimeType = "audio/webm";
      onstop?: () => void; ondataavailable?: (event: { data: Blob }) => void;
      start() { this.state = "recording"; }
      requestData() { this.ondataavailable?.({ data: new Blob(["synthetic audio"]) }); }
      stop() { this.state = "inactive"; this.onstop?.(); }
    }
    vi.stubGlobal("MediaRecorder", FakeRecorder);
    render(<RecorderPanel {...props} onFileSelected={selected} />);
    fireEvent.click(screen.getByTestId("speak.record_start")); await screen.findByTestId("speak.record_stop");
    now.mockReturnValue(135000); fireEvent.click(screen.getByTestId("speak.record_stop"));
    await waitFor(() => expect(selected).toHaveBeenCalledWith(expect.any(File), 24.5));
    expect(screen.getByTestId("speak.recording_status").textContent).toBe("speak.recording_too_short");
  } finally { probe.mockRestore(); now.mockRestore(); }
});

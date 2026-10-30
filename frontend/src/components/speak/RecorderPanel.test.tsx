import "@testing-library/jest-dom/vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { RecorderPanel } from "./RecorderPanel";

const props = { microphoneSetupPassed: true, canRemove: false, inputMode: "record" as const, onInputModeChange: vi.fn(), onRemove: vi.fn(),
  previewUrl: "", showReadyCheckpoint: false, statusMessage: "ready", statusTone: "info" as const, translate: (key: string) => key };
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
it("shows a finalized short take as ready in memory while retaining its warning", () => {
  render(<RecorderPanel {...props} onFileSelected={vi.fn()} previewUrl="blob:short-take"
    statusMessage="Too short to submit" statusTone="warning" />);
  expect(screen.getByTestId("speak.recording_visualizer")).toHaveAttribute("data-recording-state", "ready");
  expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Too short to submit");
});
it("waits for native replacement confirmation before switching input modes", async () => {
  let finish: (approved: boolean) => void = () => undefined;
  const onMode = vi.fn();
  render(<RecorderPanel {...props} onFileSelected={vi.fn()} onInputModeChange={onMode}
    onBeforeReplace={() => new Promise(resolve => { finish = resolve; })} />);
  fireEvent.click(screen.getByTestId("speak.input_mode_upload"));
  expect(onMode).not.toHaveBeenCalled();
  await act(async () => finish(false));
  expect(onMode).not.toHaveBeenCalled();
  fireEvent.click(screen.getByTestId("speak.input_mode_upload"));
  await act(async () => finish(true));
  expect(onMode).toHaveBeenCalledWith("upload");
});
it("resumes the meter in the Record gesture before an asynchronous confirmation", async () => {
  const resume = vi.fn().mockResolvedValue(undefined);
  const close = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("AudioContext", class { resume = resume; close = close; });
  microphone();
  vi.stubGlobal("MediaRecorder", class { static isTypeSupported() { return true; } });
  let finish: (allowed: boolean) => void = () => undefined;
  render(<RecorderPanel {...props} onFileSelected={vi.fn()}
    onBeforeReplace={() => new Promise(resolve => { finish = resolve; })} />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  fireEvent.click(screen.getByTestId("speak.record_start"));
  expect(resume).toHaveBeenCalledOnce();
  expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
  await act(async () => finish(false));
  expect(close).toHaveBeenCalledOnce();
  expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
});
it("does not request the microphone after switching to Upload during preparation", async () => {
  const close = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("AudioContext", class { resume = vi.fn().mockResolvedValue(undefined); close = close; });
  microphone();
  vi.stubGlobal("MediaRecorder", class { static isTypeSupported() { return true; } });
  let finish: (prepared: boolean) => void = () => undefined;
  const onMode = vi.fn();
  const { rerender } = render(<RecorderPanel {...props} onFileSelected={vi.fn()} onInputModeChange={onMode}
    onPrepareRecording={() => new Promise(resolve => { finish = resolve; })} />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  fireEvent.click(screen.getByTestId("speak.input_mode_upload"));
  expect(onMode).toHaveBeenCalledWith("upload");
  rerender(<RecorderPanel {...props} inputMode="upload" onFileSelected={vi.fn()} onInputModeChange={onMode} />);
  await act(async () => finish(true));
  expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
  expect(close).toHaveBeenCalledOnce();
});
it("does not request the microphone after switching to Upload during native confirmation", async () => {
  const close = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("AudioContext", class { resume = vi.fn().mockResolvedValue(undefined); close = close; });
  microphone();
  vi.stubGlobal("MediaRecorder", class { static isTypeSupported() { return true; } });
  let finish: (allowed: boolean) => void = () => undefined;
  const beforeReplace = vi.fn().mockImplementationOnce(() => new Promise(resolve => { finish = resolve; })).mockReturnValue(true);
  const onMode = vi.fn();
  const { rerender } = render(<RecorderPanel {...props} onFileSelected={vi.fn()} onInputModeChange={onMode}
    onBeforeReplace={beforeReplace} />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  fireEvent.click(screen.getByTestId("speak.input_mode_upload"));
  expect(onMode).toHaveBeenCalledWith("upload");
  rerender(<RecorderPanel {...props} inputMode="upload" onFileSelected={vi.fn()} onInputModeChange={onMode} />);
  await act(async () => finish(true));
  expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
  expect(close).toHaveBeenCalledOnce();
});
it("does not request the microphone after unmounting during preparation", async () => {
  const close = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("AudioContext", class { resume = vi.fn().mockResolvedValue(undefined); close = close; });
  microphone();
  vi.stubGlobal("MediaRecorder", class { static isTypeSupported() { return true; } });
  let finish: (prepared: boolean) => void = () => undefined;
  const { unmount } = render(<RecorderPanel {...props} onFileSelected={vi.fn()}
    onPrepareRecording={() => new Promise(resolve => { finish = resolve; })} />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  unmount();
  await act(async () => finish(true));
  expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
  expect(close).toHaveBeenCalledOnce();
});
it("does not report silence when the meter remains suspended after confirmation", async () => {
  const status = vi.fn();
  const node = () => ({ connect: vi.fn(), disconnect: vi.fn() });
  vi.stubGlobal("AudioContext", class {
    state = "suspended"; destination = {};
    resume = vi.fn().mockResolvedValue(undefined); close = vi.fn().mockResolvedValue(undefined);
    createAnalyser() { return { ...node(), fftSize: 2048, getFloatTimeDomainData: vi.fn() }; }
    createMediaStreamSource() { return node(); }
    createGain() { return { ...node(), gain: { value: 0 } }; }
  });
  microphone();
  class FakeRecorder {
    static isTypeSupported() { return true; }
    state = "inactive"; mimeType = "audio/webm";
    onstop?: () => void; ondataavailable?: (event: { data: Blob }) => void;
    start() { this.state = "recording"; }
    requestData() { this.ondataavailable?.({ data: new Blob(["audio"]) }); }
    stop() { this.state = "inactive"; this.onstop?.(); }
  }
  vi.stubGlobal("MediaRecorder", FakeRecorder);
  vi.useFakeTimers();
  render(<RecorderPanel {...props} onFileSelected={vi.fn()} onMicrophoneStatusChange={status}
    onBeforeReplace={() => Promise.resolve(true)} />);
  await act(async () => { fireEvent.click(screen.getByTestId("speak.record_start")); await Promise.resolve(); });
  expect(screen.getByTestId("speak.record_stop")).toBeEnabled();
  await act(async () => { await vi.advanceTimersByTimeAsync(11_000); });
  expect(screen.getByTestId("speak.signal_warning")).toHaveTextContent("speak.meter_unavailable");
  fireEvent.click(screen.getByTestId("speak.record_stop"));
  await act(async () => { await Promise.resolve(); });
  expect(status).not.toHaveBeenCalledWith("silent");
});
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

it("shows a microphone error over a previously saved status", async () => {
  const started = vi.fn();
  Object.defineProperty(navigator, "mediaDevices", { configurable: true,
    value: { getUserMedia: vi.fn().mockRejectedValue(new DOMException("denied", "NotAllowedError")) } });
  vi.stubGlobal("MediaRecorder", class { static isTypeSupported() { return true; } });
  render(<RecorderPanel {...props} onFileSelected={vi.fn()} onRecordingStarted={started} statusMessage="Saved on this Mac" preferStatusMessage />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("speak.recording_permission_denied"));
  expect(screen.getByTestId("speak.recording_status")).not.toHaveTextContent("Saved on this Mac");
  expect(started).not.toHaveBeenCalled();
});

it("does not claim clear input merely because a recording was saved", async () => {
  microphone();
  const status = vi.fn();
  const started = vi.fn();
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
  render(<RecorderPanel {...props} onFileSelected={vi.fn()} onMicrophoneStatusChange={status} onRecordingStarted={started} />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  await screen.findByTestId("speak.record_stop");
  expect(started).toHaveBeenCalledOnce();
  expect(status).not.toHaveBeenCalledWith("ready");
  fireEvent.click(screen.getByTestId("speak.record_stop"));
  expect(recorder.state).toBe("inactive");
  expect(status).not.toHaveBeenCalledWith("ready");
});

it("waits for the durable save callback before showing a saved status", async () => {
  microphone();
  let recorder!: FakeRecorder;
  class FakeRecorder {
    static isTypeSupported() { return true; }
    state = "inactive"; mimeType = "audio/webm";
    onstop?: () => void; ondataavailable?: (event: { data: Blob }) => void;
    constructor() { recorder = this; }
    start() { this.state = "recording"; }
    requestData() { this.ondataavailable?.({ data: new Blob(["finalized audio"]) }); }
    stop() { this.state = "inactive"; this.onstop?.(); }
  }
  vi.stubGlobal("MediaRecorder", FakeRecorder);
  let finish: () => void = () => undefined;
  const selected = vi.fn(() => new Promise<void>(resolve => { finish = resolve; }));
  const { rerender } = render(<RecorderPanel {...props} onFileSelected={selected} statusMessage="Take ready in memory" />);
  fireEvent.click(screen.getByTestId("speak.record_start"));
  await screen.findByTestId("speak.record_stop");
  fireEvent.click(screen.getByTestId("speak.record_stop"));
  await waitFor(() => expect(selected).toHaveBeenCalledOnce());
  expect(screen.getByTestId("speak.recording_status").textContent).not.toContain("Saved on this Mac");
  await act(async () => { finish(); await Promise.resolve(); });
  rerender(<RecorderPanel {...props} onFileSelected={selected} statusMessage="Saved on this Mac" preferStatusMessage />);
  expect(screen.getByTestId("speak.recording_status").textContent).toContain("Saved on this Mac");
  expect(recorder.state).toBe("inactive");
});

it("keeps the Stop time when MediaRecorder finalizes later", async () => {
  const now = vi.spyOn(Date, "now").mockReturnValue(100_000);
  try {
    microphone();
    const selected = vi.fn();
    let recorder!: FakeRecorder;
    class FakeRecorder {
      static isTypeSupported() { return true; }
      state = "inactive"; mimeType = "audio/webm";
      onstop?: () => void; ondataavailable?: (event: { data: Blob }) => void;
      constructor() { recorder = this; }
      start() { this.state = "recording"; }
      requestData() { this.ondataavailable?.({ data: new Blob(["finalized audio"]) }); }
      stop() { this.state = "inactive"; }
    }
    vi.stubGlobal("MediaRecorder", FakeRecorder);
    render(<RecorderPanel {...props} onFileSelected={selected} />);
    fireEvent.click(screen.getByTestId("speak.record_start"));
    await screen.findByTestId("speak.record_stop");
    now.mockReturnValue(135_000);
    fireEvent.click(screen.getByTestId("speak.record_stop"));
    now.mockReturnValue(150_000);
    act(() => recorder.onstop?.());
    await waitFor(() => expect(selected).toHaveBeenCalledOnce());
    expect((selected.mock.calls[0][0] as File).lastModified).toBe(135_000);
  } finally { now.mockRestore(); }
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

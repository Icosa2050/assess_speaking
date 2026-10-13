import "@testing-library/jest-dom/vitest";
import { act, fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/test/renderWithProviders";
import { MicrophoneTestPanel, useMicrophoneTest } from "./MicrophoneTestPanel";

const getUserMedia = vi.fn();
const track = { stop: vi.fn(), addEventListener: vi.fn(), getSettings: () => ({ deviceId: "tested-mic" }) };
const stream = { getTracks: () => [track], getAudioTracks: () => [track] };
const close = vi.fn().mockResolvedValue(undefined);
const revoke = vi.fn();
let amplitude = 0;
let encoderStalled = false;
const Test = () => <MicrophoneTestPanel test={useMicrophoneTest()} translate={key => key} />;
const click = () => fireEvent.click(screen.getByTestId("microphone-test-start"));
const tick = async (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
const confirm = () => {
  fireEvent.ended(screen.getByTestId("microphone-test-playback"));
  fireEvent.click(screen.getByTestId("microphone-test-confirm"));
};

beforeEach(() => {
  vi.useFakeTimers(); vi.clearAllMocks(); amplitude = 0; encoderStalled = false;
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
  Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:test-sample") });
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revoke });
  getUserMedia.mockResolvedValue(stream);
  vi.stubGlobal("AudioContext", class {
    state = "running";
    get currentTime() { return Date.now() / 1000; }
    close = close;
    resume = vi.fn().mockResolvedValue(undefined);
    destination = {};
    createGain = () => ({ gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() });
    createMediaStreamSource = () => ({ connect: vi.fn(), disconnect: vi.fn() });
    createAnalyser = () => ({ connect: vi.fn(), disconnect: vi.fn(), fftSize: 2048, getFloatTimeDomainData: (samples: Float32Array) => samples.fill(amplitude) });
  });
  vi.stubGlobal("MediaRecorder", class {
    static isTypeSupported() { return true; }
    state = "inactive"; mimeType = "audio/webm";
    onstop?: () => void; onerror?: () => void; ondataavailable?: (event: { data: Blob }) => void;
    start() { this.state = "recording"; }
    stop() {
      this.state = "inactive";
      if (encoderStalled) return;
      this.ondataavailable?.({ data: new Blob(["synthetic sample"]) });
      this.onstop?.();
    }
  });
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

it("requires a five-second sample, full playback and explicit confirmation, retaining readiness after navigation", async () => {
  const { store, unmount } = renderWithProviders(<Test />);
  expect(getUserMedia).not.toHaveBeenCalled(); amplitude = 0.04;
  click(); await tick(4900);
  expect(getUserMedia).toHaveBeenCalledWith({ audio: { autoGainControl: false, echoCancellation: true, noiseSuppression: true } });
  expect(store.getState().microphoneSetupPassed).toBe(false);
  expect(screen.queryByTestId("microphone-test-playback")).not.toBeInTheDocument();
  await tick(200);
  expect(store.getState().microphoneStatus).toBe("needs_review");
  expect(screen.getByTestId("microphone-test-confirm")).toBeDisabled();
  expect(track.stop).toHaveBeenCalledOnce(); expect(close).toHaveBeenCalledOnce();
  confirm();
  expect(store.getState().microphoneSetupPassed).toBe(true);
  expect(store.getState().microphoneDeviceId).toBe("tested-mic");
  unmount(); expect(revoke).toHaveBeenCalledWith("blob:test-sample");
  store.getState().beginNewSession(); renderWithProviders(<Test />, { store });
  expect(screen.getByTestId("microphone-test-status")).toHaveTextContent("microphone_test_ready");
  expect(getUserMedia).toHaveBeenCalledOnce();
});

it.each([[0, "silent"], [0.005, "quiet"], [0.99, "clipping"]])("rejects unsuitable input %s as %s even after playback", async (input, status) => {
  amplitude = input as number;
  const { store } = renderWithProviders(<Test />); click(); await tick(5100);
  expect(store.getState().microphoneStatus).toBe(status);
  confirm(); expect(store.getState().microphoneSetupPassed).toBe(false);
  expect(screen.getByTestId("microphone-test-confirm")).toBeDisabled();
});

it("allows the user to reject crackling that a level meter cannot detect", async () => {
  amplitude = 0.04;
  const { store } = renderWithProviders(<Test />); click(); await tick(5100);
  fireEvent.click(screen.getByTestId("microphone-test-reject"));
  expect(store.getState().microphoneStatus).toBe("distorted");
  confirm(); expect(store.getState().microphoneSetupPassed).toBe(false);
});

it("invalidates confirmation and uses the changed processing setting for the next sample", async () => {
  amplitude = 0.04;
  const { store } = renderWithProviders(<Test />); click(); await tick(5100); confirm();
  fireEvent.click(screen.getByTestId("microphone-processing"));
  expect(store.getState().microphoneSetupPassed).toBe(false);
  expect(screen.queryByTestId("microphone-test-playback")).not.toBeInTheDocument();
  expect(revoke).toHaveBeenCalledOnce();
  click(); await tick(0);
  expect(getUserMedia).toHaveBeenLastCalledWith({ audio: { deviceId: { exact: "tested-mic" }, autoGainControl: false, echoCancellation: false, noiseSuppression: false } });
});

it("does not cancel the current sample when permission exposes device labels", async () => {
  const devices = Object.assign(new EventTarget(), { getUserMedia });
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: devices });
  amplitude = 0.04;
  const { store } = renderWithProviders(<Test />); click(); await tick(100);
  act(() => devices.dispatchEvent(new Event("devicechange")));
  await tick(5000);
  expect(store.getState().microphoneStatus).toBe("needs_review");
  confirm(); expect(store.getState().microphoneSetupPassed).toBe(true);
});

it("does not misreport a stalled audio engine as silence", async () => {
  vi.spyOn(AudioContext.prototype, "currentTime", "get").mockReturnValue(0);
  const { store } = renderWithProviders(<Test />); click(); await tick(5100);
  expect(store.getState().microphoneStatus).toBe("error"); expect(track.stop).toHaveBeenCalledOnce();
});
it("bounds an encoder that never returns a final sample", async () => {
  encoderStalled = true; amplitude = 0.04;
  const { store } = renderWithProviders(<Test />); click(); await tick(7100);
  expect(store.getState().microphoneStatus).toBe("error"); expect(track.stop).toHaveBeenCalledOnce();
});

it.each([["NotAllowedError", "denied"], ["NotFoundError", "missing"], ["NotReadableError", "busy"]])(
  "reports %s and allows a complete retry", async (name, result) => {
    getUserMedia.mockRejectedValueOnce(new DOMException("test", name));
    const { store } = renderWithProviders(<Test />); click(); await tick(0);
    expect(store.getState().microphoneStatus).toBe(result);
    amplitude = 0.04; click(); await tick(5100); confirm();
    expect(store.getState().microphoneSetupPassed).toBe(true);
  },
);
it.each(["cancel", "unmount", "timeout"])("releases a late permission grant after %s", async action => {
  let resolve!: (value: typeof stream) => void;
  getUserMedia.mockImplementationOnce(() => new Promise(r => { resolve = r; }));
  const { store, unmount } = renderWithProviders(<Test />); click(); await tick(0);
  if (action === "cancel") click();
  if (action === "unmount") unmount();
  if (action === "timeout") await tick(15_000);
  await act(async () => { resolve(stream); });
  expect(track.stop).toHaveBeenCalledOnce(); expect(close).toHaveBeenCalledOnce();
  expect(store.getState().microphoneStatus).toBe(action === "timeout" ? "timeout" : "unknown");
});
it("stops listening when leaving setup", async () => {
  const { unmount } = renderWithProviders(<Test />); click(); await tick(0); unmount();
  expect(track.stop).toHaveBeenCalledOnce(); expect(close).toHaveBeenCalledOnce();
});
it("provides an actionable result when browser capture is unsupported", async () => {
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: undefined });
  const { store } = renderWithProviders(<Test />); click(); await tick(0);
  expect(store.getState().microphoneStatus).toBe("unsupported"); expect(getUserMedia).not.toHaveBeenCalled();
});
it("invalidates a previous check when permission is revoked or the input device changes", async () => {
  const permission = Object.assign(new EventTarget(), { state: "granted" });
  const devices = Object.assign(new EventTarget(), { getUserMedia });
  vi.stubGlobal("navigator", { mediaDevices: devices, permissions: { query: vi.fn().mockResolvedValue(permission) } });
  const { store, unmount } = renderWithProviders(<Test />, { appState: { microphoneStatus: "ready", microphoneSetupPassed: true } });
  await tick(0); expect(store.getState().microphoneSetupPassed).toBe(true);
  act(() => { permission.state = "denied"; permission.dispatchEvent(new Event("change")); });
  expect(store.getState().microphoneSetupPassed).toBe(false);
  act(() => { store.getState().setMicrophoneSetupPassed(true); devices.dispatchEvent(new Event("devicechange")); });
  expect(store.getState().microphoneSetupPassed).toBe(false);
  unmount(); store.getState().setMicrophoneSetupPassed(true); devices.dispatchEvent(new Event("devicechange"));
  expect(store.getState().microphoneSetupPassed).toBe(true);
});

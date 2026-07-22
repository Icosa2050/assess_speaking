# Signed macOS DMG Design

**Date:** 2026-07-22  
**Status:** Approved direction  
**Target:** Vostavo 0.1.0 for Apple Silicon

## Goal

Produce a Developer ID-signed, Apple-notarized, stapled DMG that installs a runnable `Vostavo.app`. The installed app must start its own packaged backend without Python, Homebrew, FFmpeg, the repository checkout, or a pre-existing virtual environment.

The packaged backend includes the Python interpreter, application modules, Python dependencies, and native libraries. Whisper model weights remain an on-demand download into the user's cache so the first DMG stays reasonably sized. Local and remote LLM services remain configurable providers rather than bundled services.

## Release Boundary

- Architecture: `arm64` (`aarch64-apple-darwin`).
- Minimum macOS version: 13.4, currently imposed by the packaged ONNX Runtime binary.
- Main bundle identifier: `com.vostavo.desktop`.
- Backend helper identifier: `com.vostavo.desktop.backend`.
- Distribution artifact: `Vostavo-0.1.0-arm64.dmg`.
- Installer layout: `Vostavo.app` plus an `/Applications` shortcut.
- App Sandbox: disabled for this direct-download release.
- Hardened Runtime: enabled for all executable components.

A universal build is a later release lane. The current CTranslate2, ONNX Runtime, PyAV, Parselmouth, NumPy, and related native wheels are arm64-only; a universal release needs a separately resolved and tested x86_64 backend.

## Bundle Architecture

The final application uses a standard nested-code layout:

```text
Vostavo.app/
  Contents/
    Info.plist
    MacOS/
      vostavo-desktop
    Helpers/
      Vostavo Backend.app/
        Contents/
          Info.plist
          MacOS/
            vostavo-backend
          Frameworks/
            Python and packaged native libraries
          Resources/
            packaged Python modules and runtime data
    Resources/
      frontend assets and app icon
```

PyInstaller builds the backend as an `onedir`, windowless helper app. This preserves fast repeat launches and gives Python packages the directory and symlink layout they expect. The complete helper app is copied with a symlink-preserving tool into `Contents/Helpers`, which Apple reserves for helper apps and tools.

The onefile alternative is retained only as a fallback if the onedir helper cannot pass runtime and notarization gates. Onefile is not the default because it extracts the large Python runtime on every launch, delays startup, and is harder to inspect when native imports fail.

## Reproducible Backend Build

The packaging environment is separate from the development `.venv`:

1. Use the installed uv-managed CPython 3.12.11 arm64 interpreter, whose deployment target is macOS 11.
2. Create a clean packaging virtual environment under a gitignored build directory.
3. Install an exact, production-only requirements file plus pinned PyInstaller tooling.
4. Exclude pytest, Playwright, coverage, and other development-only packages.
5. Freeze `scripts/run_backend.py` from a checked-in PyInstaller spec.
6. Collect required package metadata, CTranslate2, ONNX Runtime, PyAV, Parselmouth, faster-whisper, Hugging Face, and application data files.
7. Preserve `locales/`, `samples/cefr/`, and `assessment_runtime/data/` at the paths expected by the packaged modules.
8. Record interpreter, package, architecture, deployment-target, and artifact checksums in the build output.

`multiprocessing.freeze_support()` must run before the backend entry point. This is required because assessment jobs use the `spawn` multiprocessing context and re-enter the frozen executable.

## Self-Contained ASR

Native-file-versus-chunked semantics remain provider capabilities:

- `auto` first uses the provider's native-file path.
- `native` requires native-file transcription.
- `chunked` explicitly requests chunking.
- `auto` falls back to chunking only when the provider advertises that capability and native transcription fails in a fallback-safe way.

Chunk creation must no longer invoke `ffmpeg` from `PATH`. It will decode and resample through packaged PyAV, then write 16 kHz mono WAV chunks through Python's standard library. This removes the Homebrew dependency while retaining broad input-format support through the same native media libraries already needed by faster-whisper.

Diagnostics must report the packaged media path as available and must not instruct DMG users to install Homebrew FFmpeg.

## Desktop Lifecycle

Development and installed launches share one Rust-owned lifecycle with different backend resolvers:

1. An explicit `VOSTAVO_DESKTOP_API_BASE_URL` remains the highest-priority test/development override.
2. An installed app resolves `Contents/Helpers/Vostavo Backend.app/Contents/MacOS/vostavo-backend` relative to its own executable.
3. If the helper is absent, development builds retain the repository bootstrap path.
4. The desktop shell reserves an available loopback port and starts the helper on `127.0.0.1`.
5. It passes explicit app-data, cache, and report paths plus `VOSTAVO_LAUNCH_MODE=packaged`, `VOSTAVO_DEPLOYMENT_MODE=local`, and `VOSTAVO_AUTH_MODE=guest`.
6. It polls `/v1/health` with a bounded startup timeout before showing the main window.
7. It injects a bridge with `launchMode=packaged` and `packagingSafe=true`.
8. It retains the child process handle for the app lifetime.
9. On normal exit it sends `SIGTERM`, waits briefly for FastAPI lifespan cleanup, and uses a hard kill only as a bounded fallback.

The helper binds only to loopback. Mutable files are written under the user's Vostavo Application Support and Cache directories, never inside the signed app bundle.

Backend stdout and stderr are captured in the Vostavo logs directory. A startup failure must leave an actionable log rather than silently falling back to repository Python.

## macOS Metadata

The main app bundle includes:

- `NSMicrophoneUsageDescription` with localized user-facing text.
- A 512-pixel source icon converted to a complete `.icns` icon set.
- Minimum system version 13.4.
- Version and bundle identifiers consistent across Tauri, PyInstaller, and release filenames.
- A CSP that permits only the selected loopback backend origin required by the desktop bridge.

No App Sandbox, JIT, unsigned-executable-memory, or disabled-library-validation entitlement is added unless a signed runtime test proves it necessary. All nested libraries should instead carry the same Developer ID team signature.

## Signing And Notarization

Signing is manual and inside-out; `codesign --deep` is not used for signing:

1. Sign the deepest Mach-O libraries and Python extension modules.
2. Sign framework bundles after their contents.
3. Sign the backend executable and `Vostavo Backend.app` with Hardened Runtime and a secure timestamp.
4. Sign the Tauri executable and outer `Vostavo.app` last.
5. Verify the app with `codesign --verify --deep --strict --verbose=4`.
6. Verify Gatekeeper execution assessment with `spctl`.
7. Create the compressed DMG from a clean staging directory.
8. Sign the DMG with the same Developer ID Application identity.
9. Submit the DMG with `xcrun notarytool submit --keychain-profile vostavo-notary --wait`.
10. Staple and validate the notarization ticket.
11. Re-run `codesign`, `spctl`, `stapler validate`, and checksum verification against the final artifact.

The notarization credential is stored only in the user's login Keychain through `notarytool store-credentials`; it is never committed or written into build scripts.

External Apple prerequisites are:

- Acceptance of the current Apple Developer Program License Agreement.
- A valid `Developer ID Application` certificate and private key in the login Keychain.
- A working `vostavo-notary` Keychain profile.

## Build And Release Entry Point

One zsh script orchestrates a clean arm64 release and fails closed:

```text
build frontend -> build frozen helper -> smoke helper -> build Tauri app
-> inject helper -> verify bundle layout -> sign app -> create/sign DMG
-> notarize -> staple -> verify -> emit checksums and release report
```

The script supports an explicit ad-hoc mode for development evidence, but release mode refuses to continue when the Developer ID identity or notarization profile is missing. It never silently emits an artifact labelled signed or notarized when those steps were skipped.

## Completion Evidence

The goal is complete only when all of the following evidence exists for the final DMG:

- Clean frontend production build and focused desktop Rust tests pass.
- Clean production-only Python packaging environment is reproducible.
- Frozen helper starts from a temporary directory with `cwd=/`.
- `/v1/health` reports healthy and runtime metadata reports `launch_mode=packaged` and `packaging_safe=true`.
- Packaged sample/library data endpoints work.
- A packaged real-audio assessment exercises faster-whisper, PyAV, CTranslate2, multiprocessing, and result persistence.
- `lsof` for the installed app and helper shows no open files under the source checkout or development `.venv`.
- The DMG mounts and copies `Vostavo.app` into `/Applications` or a clean Applications-equivalent test directory.
- The installed app launches through LaunchServices, displays the production UI, requests microphone access with the Vostavo usage text, and shuts down its owned backend.
- `codesign --verify --deep --strict`, Gatekeeper `spctl`, notarization submission, and `stapler validate` all succeed.
- A quarantine-marked copy launches without bypassing Gatekeeper.
- SHA-256, app version, architecture, minimum macOS version, Team ID, notarization submission ID, and verification results are recorded in a release report.

An ad-hoc signed DMG is useful intermediate evidence but does not satisfy the signed-and-notarized release goal.

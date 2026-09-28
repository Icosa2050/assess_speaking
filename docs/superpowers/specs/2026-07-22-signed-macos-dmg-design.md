# Signed macOS DMG Design

**Date:** 2026-07-22 (revised 2026-07-23)  
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
- Backend port: a fixed default loopback port with a bounded fallback search, so the shipped CSP can name a concrete origin. See [Desktop Lifecycle](#desktop-lifecycle) and [macOS Metadata](#macos-metadata).

### Version single source of truth

`0.1.0` currently lives in both `frontend/src-tauri/tauri.conf.json` and `frontend/src-tauri/Cargo.toml`. The build script reads the version from `tauri.conf.json`, propagates it to the PyInstaller build and the release filename, and asserts that `Cargo.toml`, the frozen helper's `Info.plist`, and the artifact name all agree. A mismatch fails the release closed rather than shipping inconsistent identifiers.

A universal build is a later release lane. The current CTranslate2, ONNX Runtime, PyAV, Parselmouth, NumPy, and related native wheels are arm64-only; a universal release needs a separately resolved and tested x86_64 backend.

The `arm64`-only minimum of macOS 13.4 is imposed today by the `onnxruntime==1.23.0` wheel's deployment target. The build report records the exact wheel tag so a future dependency bump can revisit the floor deliberately rather than by accident.

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

## Clean And Repeatable Backend Build

PyInstaller output and code signatures are not byte-for-byte reproducible, so this build is *clean and repeatable with a recorded manifest* rather than bit-reproducible. The packaging environment is separate from the development `.venv`:

1. Use the installed uv-managed CPython 3.12.11 arm64 interpreter, whose deployment target is macOS 11.
2. Create a clean packaging virtual environment under a gitignored build directory.
3. Install from a checked-in, production-only lock file at `packaging/requirements-macos-arm64.txt` plus pinned PyInstaller tooling. This file is derived from `requirements.txt` with the development-only packages removed, and a `check_quality` gate asserts that every module imported by the packaged backend resolves from it so the two do not drift.
4. Exclude pytest, Playwright, coverage, and other development-only packages. Confirm `keyring` and `redis` are genuinely reached by the packaged runtime; if Keychain access is not needed at guest launch, exclude `keyring` rather than inherit its own prompt-and-signing story.
5. Freeze `scripts/run_backend.py` from a checked-in PyInstaller spec.
6. Collect required package metadata, CTranslate2, ONNX Runtime, PyAV, Parselmouth, faster-whisper, Hugging Face, and application data files. Collect `certifi`'s `cacert.pem` explicitly — the on-demand Whisper weight download fails with `CERTIFICATE_VERIFY_FAILED` when the CA bundle is missing from the frozen tree.
7. Preserve `locales/`, `samples/cefr/`, and `assessment_runtime/data/` at the paths expected by the packaged modules.
8. Record interpreter, package, architecture, deployment-target, ONNX Runtime wheel tag, and artifact checksums in the build output.

`multiprocessing.freeze_support()` must run before the backend entry point. This is required because assessment jobs use the `spawn` multiprocessing context (`app_backend/jobs.py`) and re-enter the frozen executable.

## Self-Contained ASR

Native-file-versus-chunked semantics remain provider capabilities:

- `auto` first uses the provider's native-file path.
- `native` requires native-file transcription.
- `chunked` explicitly requests chunking.
- `auto` falls back to chunking only when the provider advertises that capability and native transcription fails in a fallback-safe way.

Chunk creation must no longer invoke `ffmpeg` from `PATH`. It will decode and resample through packaged PyAV, then write 16 kHz mono WAV chunks through Python's standard library. This removes the Homebrew dependency while retaining broad input-format support through the same native media libraries already needed by faster-whisper.

The removal touches more than the chunker. Every current `ffmpeg` reference must be reconciled in the same change:

- `assessment_runtime/asr.py` — replace the `subprocess.run(["ffmpeg", ...])` chunker and delete the hardcoded `brew install ffmpeg` guidance string (which is also an unlocalized English literal, against the project's no-hardcoded-strings rule).
- `app_core/diagnostics.py` — stop probing `shutil.which("ffmpeg")`; report the packaged media path as available instead, keyed on PyAV import success.
- `app_backend/jobs.py` — the error classifier currently maps any message containing the substring `"ffmpeg"` to `MISSING_FFMPEG`. Once decoding raises `av.error.*`, that substring no longer matches and real decode failures silently degrade to a generic error. Reclassify decode failures on the PyAV exception types, not on a message substring.
- `app_backend/contracts.py`, `frontend/src/lib/api/types.ts`, `frontend/src/lib/api/client.ts` — decide the fate of the `missing_ffmpeg` error code. It should be retired or renamed to a media-decode code; whichever is chosen, backend contract and frontend union must move together.
- `locales/{en,de,es,fr,it}.json` — the `diagnostics.ffmpeg_*` keys and any replacement copy change in all five locales as part of the same slice, not as a follow-up.

Diagnostics must report the packaged media path as available and must not instruct DMG users to install Homebrew FFmpeg.

## Desktop Lifecycle

Development and installed launches share one Rust-owned lifecycle with different backend resolvers:

1. An explicit `VOSTAVO_DESKTOP_API_BASE_URL` remains the highest-priority test/development override.
2. An installed app resolves `Contents/Helpers/Vostavo Backend.app/Contents/MacOS/vostavo-backend` relative to its own executable.
3. If the helper is absent, development builds retain the repository bootstrap path.
4. The desktop shell binds a fixed default loopback port and, only if it is taken, walks a small bounded range; the first free port in that range is the backend origin. The range is chosen so the shipped CSP can enumerate it. The helper starts on `127.0.0.1`.
5. It generates a fresh random per-launch bearer token and passes it to the helper. The helper requires that token on every request; requests without it are rejected. This closes the gap where any local process — or any browser page able to reach `127.0.0.1:<port>` — could otherwise drive the assessment API and read stored reports.
6. It passes explicit app-data, cache, and report paths plus `VOSTAVO_LAUNCH_MODE=packaged`, `VOSTAVO_DEPLOYMENT_MODE=local`, and `VOSTAVO_AUTH_MODE=guest`.
7. It polls `/v1/health` with a bounded startup timeout before showing the main window.
8. It injects a bridge with `launchMode=packaged`, `packagingSafe=true`, and the per-launch token so the frontend can authenticate.
9. It retains the child process handle for the app lifetime.
10. On normal exit it sends `SIGTERM`, waits briefly for FastAPI lifespan cleanup, and uses a hard kill only as a bounded fallback.

**Ownership of the process differs by mode, and that is an accepted cost.** Today `main.rs` shells out to `scripts/bootstrap_backend.py`, and Python (`app_backend/lifecycle.py`) owns the process; only a URL is read back. In packaged mode the Rust shell owns the child directly. To keep the packaged path from being exercised only by a full DMG build, the Rust shell owns the child in development too — spawning the `.venv` interpreter against `scripts/run_backend.py` — so one spawn/health-poll/shutdown code path carries both modes and the repository bootstrap script becomes a legacy fallback rather than the primary dev path.

The helper binds only to loopback. Mutable files are written under the user's Vostavo Application Support and Cache directories, never inside the signed app bundle.

Backend stdout and stderr are captured in the Vostavo logs directory. A startup failure must surface a native error dialog that names the log path — not a silent process exit — and must never silently fall back to repository Python from an installed app. The current `expect(...)` on bridge initialization in `main()` is replaced by explicit handling that shows this dialog before exiting.

## macOS Metadata

The main app bundle includes:

- `NSMicrophoneUsageDescription` with localized user-facing text.
- The `com.apple.security.device.audio-input` entitlement. Under Hardened Runtime a non-sandboxed app still needs this entitlement to reach the microphone; the usage-description string alone yields a prompt-less denial inside the WKWebView. This entitlement is required, not speculative.
- A 512-pixel source icon converted to a complete `.icns` icon set.
- Minimum system version 13.4.
- Version and bundle identifiers consistent across Tauri, PyInstaller, and release filenames (see [Version single source of truth](#version-single-source-of-truth)).
- A CSP whose `connect-src` and `media-src` name the concrete loopback origin(s) the backend can occupy — the fixed default port plus the bounded fallback range from [Desktop Lifecycle](#desktop-lifecycle), and nothing else. The current dev-only `:4173` Vite origins are stripped from the shipped configuration. Because Tauri bakes the CSP at build time, the port set must be fixed and enumerable rather than a wide-open `127.0.0.1:*`.

Beyond the required microphone entitlement above, no App Sandbox, JIT, unsigned-executable-memory, or disabled-library-validation entitlement is added unless a signed runtime test proves it necessary. All nested libraries should instead carry the same Developer ID team signature.

## Signing And Notarization

Signing is manual and inside-out; `codesign --deep` is not used for signing. Tauri's own bundler and signing stay disabled (`bundle.active` remains `false` and no `signingIdentity` is set in `tauri.conf.json`) so the build never signs a bundle that the pipeline later mutates by injecting the helper — the script owns every signature. Every signature — libraries, frameworks, executables, both app bundles, and the DMG — carries Hardened Runtime (where applicable) and a secure `--timestamp`; notarization rejects any Mach-O whose signature lacks one.

1. Sign the deepest Mach-O libraries and Python extension modules, each with `--timestamp`.
2. Sign framework bundles after their contents, each with `--timestamp`.
3. Sign the backend executable and `Vostavo Backend.app` with Hardened Runtime, the required entitlements, and `--timestamp`.
4. Sign the Tauri executable and outer `Vostavo.app` last, with Hardened Runtime, entitlements, and `--timestamp`.
5. Verify the app with `codesign --verify --deep --strict --verbose=4`.
6. Verify Gatekeeper execution assessment on the app with `spctl -a -t exec -vvv`.
7. **Notarize and staple the app first.** Submit `Vostavo.app` (zipped) with `notarytool`, then `stapler staple Vostavo.app`. A ticket stapled only to the DMG does not travel with the app once it is dragged to `/Applications`, so first launch would require an online Gatekeeper check and fail offline — unacceptable for a local-first app.
8. Create the compressed DMG from a clean staging directory built around the already-stapled app.
9. Sign the DMG with the same Developer ID Application identity and `--timestamp`.
10. Submit the DMG with `xcrun notarytool submit --keychain-profile vostavo-notary --wait`, then `stapler staple` and validate the DMG ticket.
11. Re-run `codesign --verify`, `spctl -a -t open --context context:primary-signature -vvv` for the DMG (and `-a -t exec` for the app), `stapler validate` on both app and DMG, and checksum verification against the final artifact.

The notarization credential is stored only in the user's login Keychain through `notarytool store-credentials`; it is never committed or written into build scripts.

External Apple prerequisites are:

- Acceptance of the current Apple Developer Program License Agreement.
- A valid `Developer ID Application` certificate and private key in the login Keychain.
- A working `vostavo-notary` Keychain profile.

## Build And Release Entry Point

One zsh script orchestrates a clean arm64 release and fails closed:

```text
assert versions agree -> build frontend -> build frozen helper -> smoke helper
-> build Tauri app (bundler/signing off) -> inject helper -> verify bundle layout
-> sign app -> notarize+staple app -> create/sign DMG -> notarize+staple DMG
-> verify (codesign/spctl/stapler on app and DMG) -> emit checksums and release report
```

The script supports an explicit ad-hoc mode for development evidence, but release mode refuses to continue when the Developer ID identity or notarization profile is missing, or when the version assertion fails. It never silently emits an artifact labelled signed or notarized when those steps were skipped.

## Completion Evidence

The goal is complete only when all of the following evidence exists for the final DMG:

- Clean frontend production build and focused desktop Rust tests pass, including a unit test for helper-path resolution relative to the executable.
- Clean production-only Python packaging environment builds repeatably from `packaging/requirements-macos-arm64.txt`, and the import-vs-lock drift check passes.
- An automated test runs chunked ASR with `ffmpeg` scrubbed from `PATH` and succeeds through packaged PyAV, guarding the Homebrew removal against regression.
- Frozen helper starts from a temporary directory with `cwd=/`.
- `/v1/health` reports healthy and runtime metadata reports `launch_mode=packaged` and `packaging_safe=true`.
- Requests without the per-launch token are rejected; requests carrying it succeed.
- Packaged sample/library data endpoints work.
- A packaged real-audio assessment exercises faster-whisper, PyAV, CTranslate2, multiprocessing, and result persistence.
- A **cold-cache** run on a clean machine (empty `~/.cache`) downloads the Whisper weights over TLS and transcribes, proving the bundled `certifi` CA path works.
- `lsof` for the installed app and helper shows no open files under the source checkout or development `.venv`.
- The DMG mounts and copies `Vostavo.app` into `/Applications` or a clean Applications-equivalent test directory.
- The app also launches **directly from the mounted DMG under App Translocation** (randomized read-only bundle path) without writing inside the bundle or falling back to a repo-relative path.
- The installed app launches through LaunchServices, displays the production UI, requests microphone access with the Vostavo usage text, and shuts down its owned backend.
- A forced startup failure surfaces the native error dialog naming the log path rather than a silent exit.
- `codesign --verify --deep --strict`, Gatekeeper `spctl` (app and DMG variants), notarization submission, and `stapler validate` on both the app and the DMG all succeed.
- A quarantine-marked copy launches without bypassing Gatekeeper.
- SHA-256, app version, architecture, minimum macOS version, ONNX Runtime wheel tag, Team ID, notarization submission ID, and verification results are recorded in a release report.

An ad-hoc signed DMG is useful intermediate evidence but does not satisfy the signed-and-notarized release goal.

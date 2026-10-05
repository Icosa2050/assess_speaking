# Review: macOS delivery plan (docs/plans/2026-10-05-macos-delivery.md)

**Verdict:** The plan points in the right direction, but it isn't ready to implement yet. Five problems will break the packaged app or make the tests prove less than they claim:

1. The Tauri origin is probably rejected by the backend.
2. Startup uses port scanning, which has a race.
3. The helper's on-disk layout under signing isn't decided.
4. Release builds still contain the repo fallback.
5. Nothing proves the app is self-contained except PATH scrubbing.

All five are fixable without adding much scope. Several items below simplify the plan.

I didn't run any tools. Points I couldn't confirm from the code shown are marked **verify**.

---

## 1. Blocking issues

**B1. The origin guard will probably reject the packaged webview.** `local_origin_guard` in `app_backend/app.py` returns 403 for any `Origin` that doesn't match `LOCAL_GUEST_ORIGIN_REGEX`. Packaged Tauri on macOS serves the UI from `tauri://localhost`. Dev mode serves it from `http://localhost:<vite>`, so dev testing never hits this case.
- Add `tauri://localhost` to the allowed origins, but only when a session token is configured.
- Add a Host-header allowlist (`127.0.0.1:<port>`) to block DNS rebinding.
- Check `tauri.conf.json` CSP: `connect-src` and `media-src` need `http://127.0.0.1:*`.
- **Verify** in WKWebView that `fetch` and `<audio>` from `tauri://` to `http://127.0.0.1` aren't blocked as mixed content. Chromium tests can't show this.

**B2. Replace port-range scanning with a port-0 handshake.** If Rust picks a free port and the child binds it later, another process can take it in between. That race is the reason the plan needs the "verify token ownership" step at all. Instead:
- The helper binds `127.0.0.1:0` itself and runs `uvicorn.Server(...).run(sockets=[sock])`.
- It writes `VOSTAVO_READY port=N` to the **original** stdout (`sys.__stdout__`), or to a pipe fd that Rust passes in.
- This must happen before `_configure_backend_logging` replaces `sys.stdout`. Today that redirect would swallow the handshake.
- Rust then polls authenticated `/v1/health` on that port.
- This removes the port range, the fallback tests and most of the foreign-server cases.

**B3. Decide where the helper lives inside the bundle, and sign it the way release will.**
- A bare PyInstaller onedir (executable plus `_internal/` with data files) placed in `Contents/MacOS` or `Contents/Resources` is a known cause of codesign and notarization failures.
- Recommendation: build the helper as a PyInstaller `BUNDLE` (`VostavoBackend.app`, `LSBackgroundOnly=true`). PyInstaller 6 then splits it correctly into `Frameworks/` and `Resources/` with symlinks. Nest it at `Contents/Helpers/VostavoBackend.app`.
- It is still a self-contained onedir, which meets your requirement.
- Insert it after `tauri build --bundles app` using `ditto`. Don't use Tauri's `bundle.macOS.files`: **verify** whether it preserves symlinks before relying on it.
- Sign internal ad-hoc builds with `--options runtime` and the release entitlements. Otherwise hardened-runtime failures (libffi/ctypes, library validation, onnxruntime/ctranslate2 loading) appear only when release mode runs for the first time, and you can't run release mode now.
- **Verify** whether ad-hoc signing plus library validation needs `com.apple.security.cs.disable-library-validation` in internal mode only. If it does, write the difference down in the evidence report.

**B4. Release builds still contain the repo and Python fallback.** In `main.rs`, `repo_root()` bakes `CARGO_MANIFEST_DIR` into the binary at compile time. `resolve_python_executable` reads `PYTHON_BIN`, `.venv` and `python3` from PATH. `desktop_bridge_from_env` lets any inherited env var point the app at an arbitrary server. "Installed mode fails visibly" isn't enough, because a launch from the build machine would quietly find the repo.
- Put `bootstrap_from_repo_launcher`, `resolve_python_executable` and the env API override behind `#[cfg(debug_assertions)]` or a `dev-launcher` Cargo feature, so release binaries don't contain them.
- The artifact tests launch the helper directly, so they don't need the override in release builds.

**B5. Scrubbing PATH doesn't prove the app is self-contained.** dyld load paths, Homebrew's `/opt/homebrew/lib`, `~/.cache/huggingface`, the repo itself and the python.org framework can all still be reached. Add two automated gates:
- **Linkage audit:** run `otool -L` on every Mach-O. Fail on any path other than `@rpath`, `@loader_path`, `@executable_path`, `/usr/lib` or `/System`. Run `lipo -archs` and require arm64. Collect `vtool -show-build` (or `otool -l` `LC_BUILD_VERSION`) from every binary; the highest `minos` becomes `LSMinimumSystemVersion`.
- **Deny-read run:** run the CLI helper tests under `sandbox-exec` with `(allow default)` plus `(deny file-read* (subpath …))` for:
  - the repo
  - `/opt/homebrew` and `/usr/local`
  - `/Library/Frameworks/Python.framework`
  - `~/.cache`

  `sandbox-exec` is deprecated but still works, and it's cheap. This is a real check that nothing falls back to the repo or system installs.

---

## 2. Missing dependencies and decisions

**Python build environment**
- Pin a python.org 3.12 arm64/universal2 build. Don't use Homebrew Python: it links Homebrew OpenSSL and libraries and raises the minimum OS.
- Pin PyInstaller **and** `pyinstaller-hooks-contrib`.
- Use a hashed lock (uv or pip-tools).
- **Split `requirements.txt`.** It currently mixes in pytest, playwright, pytest-playwright and coverage, so it isn't a production-only lock. Also decide whether `redis` belongs there at all.
- After the build, assert that a forbidden-module list is absent from the helper's TOC: `pytest`, `playwright`, `coverage`, `tkinter`, `IPython`.

**PyInstaller collection that will be needed**
- `collect_data_files('faster_whisper')` for the Silero VAD `.onnx` asset.
- `copy_metadata` for any package whose version or metadata the code reads at runtime, plus `keyring`'s entry-point backends.
- uvicorn's string-imported loop, protocol and lifespan modules.
- The provider and ASR modules the plan already names.

**Certificates and the model cache**
- In the frozen entry point, set `SSL_CERT_FILE` to `certifi.where()`. Otherwise the stdlib `ssl` module looks for the build Python's OpenSSL directory.
- Point `HF_HOME` (or faster-whisper's `download_root`) at the app cache. Otherwise model downloads land in `~/.cache/huggingface` and the "fresh cache" test isn't isolated.

**Rust crates**
- A native dialog: `tauri-plugin-dialog` or `rfd`.
- `getrandom` or `rand` for the token.
- A small blocking HTTP client (`ureq`) for the health probe.
- `libc`/`nix` for `setpgid`/`killpg`.
- `tauri-plugin-single-instance`.
- Use `RunEvent::Exit` handling, which means switching from `.run(ctx)` to `.build(ctx)?.run(|_, ev| …)`.

**Toolchain**
- Xcode Command Line Tools, checked with `xcode-select -p` and `xcrun notarytool --version`.
- A pinned `@tauri-apps/cli`.
- Set `MACOSX_DEPLOYMENT_TARGET` for the Rust build so it matches the minimum OS derived from the helper.

**Single source of truth for the version.** `create_app` hard-codes `version="0.1.0"`. Generate a build-info module for the helper from `tauri.conf.json` or `Cargo.toml`, and have the version check compare against that.

**Localized microphone strings** need `Contents/Resources/<lang>.lproj/InfoPlist.strings` plus `CFBundleLocalizations` in the Info.plist. Tauri doesn't generate these. Add them after the Tauri build and before signing.

---

## 3. Unsafe assumptions

- **Subprocess and `sys.executable` usage is wider than the FFmpeg chunker.** In the frozen app, `sys.executable` is the helper itself, so any `[sys.executable, "-m", …]` call starts the server again. Before step 1, grep for `ffmpeg`, `ffprobe`, `subprocess`, `shutil.which`, `sys.executable`, `Path(__file__)` and `PROJECT_ROOT`.
- **The frozen entry point needs subcommands:** `serve`, `worker` and `self-test`, with `freeze_support()` as its first statement.
- **Resource paths.** `_sample_items` depends on `PROJECT_ROOT / "samples"`. Add one `resource_root()` helper that handles `sys._MEIPASS` or the bundle's Resources directory. Also filter samples by audio extension, because `is_file()` includes `.DS_Store`.
- **Recording format in WKWebView.** Safari/WKWebView `MediaRecorder` produces `audio/mp4` (AAC), not webm/opus. The upload path and frontend MIME selection must handle that. Chromium tests won't catch it.
- **Headers can't be set on every request type.** `<audio>`, `EventSource`, `<a download>` and WebSocket can't send custom headers, not just audio. Inventory all of them.
- **Simpler alternative to query tokens:** fetch media with the header and play it from a `blob:` URL. That removes the query-token exception and the log-redaction work. Clips are short, so memory is fine. If you keep query tokens:
  - set `access_log=False` explicitly in uvicorn;
  - send `Referrer-Policy: no-referrer`;
  - compare tokens with `hmac.compare_digest`.
- **Token handling.**
  - Pass the token to the helper on **stdin**, not argv: `ps` shows arguments.
  - Never write the token into the backend state file.
  - The packaged helper must **fail closed**: refuse to start without a token (`--require-session-token`). "Keep dev behavior when no token" must never apply to the frozen build.
  - Let OPTIONS preflight through the token check.
  - Require the token on `/v1/health` too.
  - Turn off `/docs` and `/openapi.json` in packaged mode.
- **Orphaned processes.**
  - Start the helper in its own process group.
  - On shutdown, send SIGTERM to the helper, wait a bounded time, then `killpg(SIGKILL)` to catch assessment workers.
  - Add a **stdin-EOF watchdog** in the helper so it exits if the app crashes or is force-quit. The stdin pipe used for the token covers this for free.
- **Child output pipes.** Send the child's stdout/stderr to a log file, or keep reading the pipes. A full undrained pipe will hang the helper.
- **Child environment.** Start the child with `env_clear()` plus an allowlist (`HOME`, `TMPDIR`, `LANG`, `USER` and the explicit `VOSTAVO_*` vars). Make sure the config never loads a `.env` from cwd.
- **Don't block the main thread during startup.** First launch of a nested PyInstaller app with ctranslate2 and onnxruntime can take tens of seconds while Gatekeeper scans it. Show a loading window, start the helper from a background thread in `setup`, and use a generous timeout (60–90 s). The current `expect(...)` panics silently in a GUI app.
- **Two instances share app data.** Launching the app twice starts two helpers against the same app-data directory, both running `execute_cleanup(ALL_SAFE)` and `write_backend_state`. Use the single-instance plugin or a lock file. Also **verify** that `ALL_SAFE` never deletes the model cache, or the warm-cache test is meaningless.
- **Rust should choose the paths.** Rust should set app-data, cache and log directories (`~/Library/{Application Support,Caches,Logs}/<bundle id>`) and pass them as `--app-data-dir`, `--cache-dir` and `--log-dir`. Then it can name the log location even when the helper dies before reporting it. Check that these match the existing platformdirs defaults, and handle migration if they don't.
- **Ad-hoc identity changes every rebuild.** Keychain (`keyring`) and TCC microphone grants are tied to the code identity, so expect access prompts and stale grants after each rebuild. The tests use a fake `HOME`, so `keyring` must fail gracefully there.
- **The internal DMG won't open easily elsewhere.** On another Mac, a quarantined ad-hoc app is blocked; recent macOS removed the right-click → Open bypass. Document "Open Anyway" in System Settings or `xattr -dr com.apple.quarantine`. Name the file `…-internal-adhoc.dmg` so it can't be confused with a release.
- **Expected `spctl` result.** `spctl` will **reject** ad-hoc artifacts. Record that as the expected result, not as a failure.

---

## 4. Practical improvements

- **Helper `self-test` subcommand.** It imports every dynamic provider and ASR module, decodes a bundled sample with PyAV, loads ctranslate2, opens the onnxruntime session, checks certifi and prints JSON. Run it under the deny-read sandbox. Missing hidden imports then fail in seconds instead of at the end of the full test.
- **PyAV chunking details:**
  - Use `AudioResampler(format="s16", layout="mono", rate=16000)` and flush it with `resample(None)`.
  - Cut chunks at exact sample counts and compute word offsets from cumulative samples, which is more accurate than ffmpeg's segment muxer.
  - Reject inputs with no audio stream.
  - Enforce a maximum decoded duration, so a tiny file can't expand to hours of audio.
  - Lower `av.logging` verbosity.
  - Also update any diagnostics that check `shutil.which("ffmpeg")`.
- **Artifact tests:**
  - Mount with `hdiutil attach -readonly -nobrowse -mountpoint <tmp>` and copy with `ditto` (never `cp -R`).
  - Also run one launch straight from the read-only mount; users do this, and it causes App Translocation.
  - After the whole test run, run `codesign --verify --deep --strict` again to prove nothing at runtime changed the bundle.
- **Signing script:**
  - Sign every `.so`/`.dylib` inside-out with `--options runtime`, then the helper executable, then the helper app, then the outer app. No `--deep`.
  - Use `--timestamp` in release mode and `--timestamp=none` for ad-hoc.
  - Use `ditto -c -k --keepParent` for the notarization zip.
  - The fail-closed check should match a `Developer ID Application` identity with the expected Team ID, plus the keychain profile name.
- **Optional, needs your OK:** sign with a self-signed code-signing certificate in a temporary keychain. That exercises the identity-based signing path without any Apple credentials. Notarization still stays untested.
- **Evidence report contents:**
  - lock hash
  - `otool`/`vtool`/`lipo` audit
  - `codesign -dvvv` and an entitlements dump
  - Python, PyInstaller, Rust and Tauri versions
  - test results
  - SHA-256
  - a prominent "INTERNAL — ad-hoc — not Gatekeeper-accepted" line

---

## 5. Scope and order

Build it in four milestones, each with a hard gate:

1. **Frozen helper:** PyAV chunking, entry point with subcommands, linkage audit, `self-test` and the CLI API, ASR and worker tests under the deny-read sandbox. Most of the risk is here.
2. **Lifecycle and token:** port-0 handshake, stdin token and watchdog, process group, single instance, loading window and dialog, release builds without the repo fallback.
3. **Bundle, ad-hoc signing with hardened runtime, and DMG,** plus the artifact tests.
4. **Release-mode script:** it fails closed now (and that is tested); the identity and notarytool paths are documented as unexercised.

To keep the scope realistic:
- Drop port scanning and port-fallback tests (B2).
- Prefer blob URLs over query tokens.
- Make the cold model download test opt-in.
- Keep native mic/TCC and second-Mac Gatekeeper checks as listed manual release steps, as the plan already does.

Also, these MCP servers need authorization before they can be used; this session can't run the OAuth flow: figma, intercom, atlassian, datadog, linear, notion and slack. For claude.ai connectors, authorize them in your claude.ai connector settings. For the others, use `claude mcp` or `/mcp` in an interactive session.

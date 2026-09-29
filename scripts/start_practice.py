#!/usr/bin/env python3
"""Start the complete local practice UI; keep this process open while practising."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import signal
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wait_ready(url: str, process: subprocess.Popen, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before it was ready ({process.returncode}).")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"Server did not become ready: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening the browser.")
    parser.add_argument("--port", type=int, default=4173, help="Local practice UI port (default 4173).")
    parser.add_argument("--backend-port", type=int, default=8802, help="Stable local API port (default 8802).")
    parser.add_argument("--app-data-dir", help="Optional separate practice workspace.")
    args = parser.parse_args()
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node 24 is required. Run `nvm install 24` and `nvm use 24`.")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if int(version.lstrip("v").split(".")[0]) != 24:
        raise RuntimeError(f"Found Node {version}. Run `nvm use 24` and try again.")
    vite = ROOT / "frontend/node_modules/vite/bin/vite.js"
    if not vite.is_file():
        raise RuntimeError("Frontend dependencies are missing. Run `npm --prefix frontend ci` first.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is missing. Install it with `brew install ffmpeg`.")
    # Verify the requested UI port is free before building or starting anything.
    import socket
    for port in (args.port, args.backend_port):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is already in use. If Vostavo is open, return to its window. Otherwise stop the other server or choose different --port / --backend-port values.") from exc
    backend_port = args.backend_port
    env = {**os.environ, "NODE_ENV": "production", "VITE_LOCAL_API_BASE_URL": f"http://127.0.0.1:{backend_port}"}
    env.setdefault("LLM_TIMEOUT_SEC", "180")
    command = [sys.executable, str(ROOT / "scripts/run_backend.py"), "--port", str(backend_port)]
    if args.app_data_dir:
        command.extend(["--app-data-dir", str(Path(args.app_data_dir).resolve())])
    processes: list[subprocess.Popen] = []
    output = tempfile.TemporaryDirectory(prefix="vostavo-ui-")
    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_on_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop_on_signal)
    try:
        subprocess.run([node, str(vite), "build", "--outDir", output.name], cwd=ROOT / "frontend", env=env, check=True)
        backend = subprocess.Popen(command, cwd=ROOT, env=env)
        processes.append(backend)
        wait_ready(f"http://127.0.0.1:{backend_port}/v1/health", backend)
        frontend = subprocess.Popen([node, str(vite), "preview", "--outDir", output.name, "--host", "127.0.0.1", "--port", str(args.port), "--strictPort"], cwd=ROOT / "frontend", env=env)
        processes.append(frontend)
        url = f"http://127.0.0.1:{args.port}"
        wait_ready(url, frontend)
        print(json.dumps({"practice_url": url, "backend_url": env["VITE_LOCAL_API_BASE_URL"]}), flush=True)
        print("Ready. Keep this window open while practising. Press Ctrl+C to stop.", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        raise RuntimeError("A practice server stopped. Restart the launcher to continue.")
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        output.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Cannot start practice: {error}", file=sys.stderr)
        raise SystemExit(1)

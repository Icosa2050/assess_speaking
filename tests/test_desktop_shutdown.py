"""Real localhost owner-lifecycle checks, opt-in when listeners are permitted."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import pytest


@pytest.mark.skipif(os.environ.get('VOSTAVO_TEST_LISTENERS') != '1', reason='Opt-in localhost process test')
@pytest.mark.parametrize('shutdown', ['owner_eof', 'sigterm'])
def test_owned_backend_removes_readiness_after_graceful_shutdown(tmp_path, shutdown):
    ready = tmp_path / 'ready.json'
    root = Path(__file__).resolve().parents[1]
    token = 'a' * 64
    with (tmp_path / 'startup.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, str(root / 'scripts/run_backend.py'), '--desktop-owned', '--port', '0', '--ready-file', str(ready), '--app-data-dir', str(tmp_path / 'state'), '--cache-dir', str(tmp_path / 'cache')], stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            process.stdin.write((token + '\n' + 'b' * 64 + '\n').encode())
            process.stdin.flush()
            deadline = time.monotonic() + 20
            healthy = False
            while time.monotonic() < deadline and process.poll() is None:
                if ready.exists():
                    port = json.loads(ready.read_text())['port']
                    try:
                        request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/health', headers={'X-Vostavo-Session': token})
                        with urllib.request.urlopen(request, timeout=1) as response:
                            healthy = response.status == 200
                        if healthy:
                            break
                    except OSError:
                        pass
                time.sleep(.1)
            assert healthy, (tmp_path / 'startup.log').read_text()
            if shutdown == 'owner_eof':
                process.stdin.close()
            else:
                process.terminate()
            assert process.wait(timeout=10) == 0
            assert not ready.exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()

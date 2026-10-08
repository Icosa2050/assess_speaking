"""Bound a verification process and its descendants, retaining live diagnostics."""
import os
import signal
import subprocess
import sys


def run(seconds, command):
    process = subprocess.Popen(command, start_new_session=os.name == 'posix')
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        print(f'Check exceeded {seconds:g} seconds; terminating its process group.', file=sys.stderr, flush=True)
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
        return 124


if __name__ == '__main__':
    seconds = float(sys.argv[1])
    if seconds <= 0 or not sys.argv[2:]:
        raise SystemExit('Provide a positive timeout and command.')
    raise SystemExit(run(seconds, sys.argv[2:]))

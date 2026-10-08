"""Own a browser fixture's lifetime; remove its root only after backend exit."""
import argparse
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile


def owned_root(root):
    if root.is_symlink() or root.resolve().parent != Path(tempfile.gettempdir()).resolve() or not re.fullmatch(r'vostavo-(default|connections|journeys)-[A-Za-z0-9_]+', root.name):
        raise ValueError('Refusing cleanup of an unowned fixture root')
    return root


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); parser.add_argument('--evidence', type=Path, required=True); parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(); root = owned_root(args.root)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('Missing backend command')
    child = subprocess.Popen([sys.executable, *command])
    def stop(_signal, _frame):
        if child.poll() is None:
            try: child.terminate()
            except ProcessLookupError: pass
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        return child.wait()
    finally:
        args.evidence.mkdir(parents=True, exist_ok=True)
        for name in ('guards.jsonl', 'dispatch.jsonl'):
            if (root / name).is_file(): shutil.copyfile(root / name, args.evidence / name)
        shutil.rmtree(owned_root(root), ignore_errors=False)


if __name__ == '__main__':
    raise SystemExit(main())

"""Cross-process exclusion for local journal mutations and CLI assessment runs."""
from contextlib import contextmanager
from pathlib import Path
import os


class JournalMaintenanceError(RuntimeError):
    pass


@contextmanager
def journal_guard(root: Path, *, exclusive=False):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.maintenance.lock').open('a+b') as handle:
        acquired = False
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0); handle.write(b'\0'); handle.flush(); handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, PermissionError) as exc:
            raise JournalMaintenanceError('The journal is in use by another process. Finish its work and retry.') from exc
        try:
            if not exclusive and (root / 'maintenance/transaction.json').exists():
                raise JournalMaintenanceError('Storage maintenance is in progress. Finish journal recovery in Settings first.')
            yield
        finally:
            if acquired:
                if os.name == 'nt':
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

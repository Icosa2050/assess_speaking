"""Authenticated local maintenance API; direct writers share its exclusion lease."""
import re

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app_backend.journal import Journal, JournalError, LIMIT, read_json


def install_journal_routes(app, config):
    journal = Journal(config, app.state.job_manager)
    app.state.journal = journal

    @app.exception_handler(JournalError)
    async def journal_error(request, exc):
        return JSONResponse(status_code=409, content={'detail': {'code': 'storage_error', 'detail': str(exc)}})

    previous_disk_handler = app.exception_handlers.get(OSError)

    @app.exception_handler(OSError)
    async def journal_disk_error(request, exc):
        if not request.url.path.startswith('/v1/journal/'):
            if previous_disk_handler is not None:
                return await previous_disk_handler(request, exc)
            raise exc
        return JSONResponse(status_code=507, content={'detail': {'code': 'storage_error', 'detail': 'Storage operation failed. Check disk space and permissions, then finish recovery in Settings.'}})

    @app.middleware('http')
    async def maintenance_exclusion(request: Request, call_next):
        # Reader endpoints remain available. Local settings are also excluded so
        # opt-in support exports cannot race private file maintenance.
        mutation = request.method in {'POST', 'PUT', 'PATCH', 'DELETE'} and not request.url.path.startswith('/v1/journal/')
        guard = None
        if mutation:
            try:
                guard = await run_in_threadpool(journal.begin_mutation)
            except JournalError as exc:
                return await journal_error(request, exc)
        try:
            return await call_next(request)
        finally:
            if mutation:
                await run_in_threadpool(journal.end_mutation, guard)

    @app.get('/v1/journal/status', tags=['local-support'])
    def status():
        state = journal.state()
        return {'transaction': state, 'archived': list(read_json(journal.directory / 'trash.json', {})), 'completed': read_json(journal.directory / 'completed.json', []), 'recovery_error': journal.recovery_error}

    @app.post('/v1/journal/begin', tags=['local-support'])
    def begin():
        return journal.begin()

    @app.post('/v1/journal/{transaction}/abort', tags=['local-support'])
    def abort(transaction: str):
        return journal.abort(transaction)

    @app.post('/v1/journal/{transaction}/complete', tags=['local-support'])
    def complete(transaction: str):
        return journal.complete(transaction)

    async def receive(request, path):
        size = 0
        try:
            with path.open('xb') as handle:
                path.chmod(0o600)
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > LIMIT:
                        raise HTTPException(413, 'Archive or recording exceeds 512 MB.')
                    import shutil
                    if shutil.disk_usage(path.parent).free < len(chunk) + 64 * 1024 * 1024:
                        raise OSError('Not enough space for this archive.')
                    await run_in_threadpool(handle.write, chunk)
            return path
        except BaseException:  # quality: allow[broad-except] disconnect and cancellation must remove incomplete files
            path.unlink(missing_ok=True)
            raise

    @app.post('/v1/journal/{transaction}/media', tags=['local-support'])
    async def media(transaction: str, request: Request):
        work = journal.work(transaction)
        if journal.check(transaction)['phase'] != 'prepared':
            raise JournalError('This transaction is no longer collecting recordings.')
        # Serial frontend transport; unique temporary names also tolerate direct callers.
        from uuid import uuid4
        path = work / f'incoming-{uuid4().hex}'
        try:
            await receive(request, path)
            name = await run_in_threadpool(journal.add_media, transaction, path, request.query_params.get('suffix', '.bin'))
            return {'file': name}
        finally:
            path.unlink(missing_ok=True)

    @app.post('/v1/journal/{transaction}/export', tags=['local-support'])
    def export(transaction: str, body: dict):
        return journal.export(transaction, body.get('browser'), allow_missing=body.get('allow_missing') is True)

    @app.get('/v1/journal/backups/{backup_id}', tags=['local-support'])
    def download(backup_id: str):
        if not re.fullmatch('[0-9a-f]{32}', backup_id):
            raise JournalError('Invalid backup identifier.')
        path = journal.directory / f'backup_{backup_id}.zip'
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, 'Backup is unavailable.')
        return FileResponse(path, media_type='application/zip', filename=f'Vostavo-backup-{backup_id}.zip')

    @app.post('/v1/journal/{transaction}/restore', tags=['local-support'])
    async def restore(transaction: str, request: Request):
        path = journal.work(transaction) / 'incoming.zip'
        await receive(request, path)
        try:
            return await run_in_threadpool(journal.stage, transaction, path)
        finally:
            path.unlink(missing_ok=True)

    @app.get('/v1/journal/{transaction}/browser', tags=['local-support'])
    def browser(transaction: str):
        from app_backend.journal import read_json
        return read_json(journal.work(transaction) / 'manifest.json')['browser']

    @app.get('/v1/journal/{transaction}/media/{filename}', tags=['local-support'])
    def staged_media(transaction: str, filename: str):
        from app_backend.journal import MEDIA
        if not MEDIA.fullmatch(f'media/{filename}'):
            raise JournalError('Invalid recording identifier.')
        return FileResponse(journal.work(transaction) / 'restore/media' / filename)

    @app.post('/v1/journal/{transaction}/commit', tags=['local-support'])
    def commit(transaction: str):
        return journal.commit(transaction)

    @app.post('/v1/journal/attempts/{session_id}/{action}', tags=['local-support'])
    def archive(session_id: str, action: str, body: dict = None):
        if action not in {'archive', 'undo', 'preview-purge', 'purge'}:
            raise JournalError('Choose archive, Undo, removal preview or confirmed removal.')
        transaction = journal.begin()['id']
        try:
            if action == 'preview-purge':
                validate_references(session_id, body)
                preview = journal.purge_preview(session_id)
                return {key: value for key, value in {**preview, 'files': len(preview['files'])}.items() if key not in {'rewrites', 'cache_dirs'}}
            if action == 'purge':
                validate_references(session_id, body)
                return journal.purge_attempt(session_id, fingerprint=(body or {}).get("fingerprint", ""))
            return journal.archive_attempt(session_id, undo=action == 'undo')
        finally:
            journal.abort(transaction)

    def validate_references(session_id, body):
        references = (body or {}).get('browser_references')
        if not isinstance(references, list) or len(references) > 100000 or any(not isinstance(item, str) or len(item) > 128 for item in references):
            raise JournalError('A current browser rehearsal reference snapshot is required for removal.')
        if session_id in references:
            raise JournalError('A saved rehearsal references this attempt. Keep it archived.')

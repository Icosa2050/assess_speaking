"""Local account-management endpoints; OAuth secrets never cross this API."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app_core.chatgpt_auth import ChatGPTAuth, ChatGPTAuthError, access_token, account_models
from app_core.services import save_provider_connection, delete_provider_connection
from app_core.state import ProviderConnection
from app_core.preference_lock import PREFERENCES_LOCK, synchronized_preferences


class SignInRequest(BaseModel):
    connection_id: str = ''


class ModelRequest(BaseModel):
    model: str


def install_chatgpt_routes(app, runtime_config, load_state):
    manager = ChatGPTAuth(runtime_config.app_data.root)
    app.state.chatgpt_auth = manager

    def local_client(request: Request):
        if request.url.hostname not in ('localhost', '127.0.0.1', '::1') or request.headers.get('X-Vostavo-Client') != 'desktop':
            raise HTTPException(403, detail={'code': 'configuration_error', 'detail': 'Use the local Vostavo app to manage ChatGPT access.'})

    router = APIRouter(prefix='/v1/runtime/chatgpt', tags=['local-runtime'], dependencies=[Depends(local_client)])

    @router.get('/pending')
    def pending():
        import time
        with manager.lock:
            for attempt_id, item in manager.pending.items():
                if item['status'] in ('waiting', 'processing') and item['expires'] > time.time():
                    return {'attempt_id': attempt_id, 'authorization_url': item['authorization_url']}
        return {}

    @router.post('/sign-in')
    def sign_in(body: SignInRequest):
        with manager.lock:
            if app.state.job_manager.has_active_provider('chatgpt'):
                raise ChatGPTAuthError('Finish or cancel the current ChatGPT assessment before signing in again.')
            return manager.start(body.connection_id)

    @router.post('/attempts/{attempt_id}/open-browser')
    def open_browser(attempt_id: str):
        if not manager.open_browser(attempt_id):
            raise ChatGPTAuthError('Could not open the browser. Open the sign-in link in your browser manually.')
        return {'opened': True}

    @synchronized_preferences
    def publish(account, models, persistent):
        state = load_state(runtime_config)
        previous = next((c for c in state.prefs.connections if c.connection_id == account['connection_id']), None)
        model = previous.default_model if previous and any(m['slug'] == previous.default_model for m in models) else models[0]['slug']
        connection = ProviderConnection(
            connection_id=account['connection_id'], provider_kind='chatgpt', label=previous.label if previous else 'ChatGPT ' + account['connection_id'][:6],
            base_url='https://api.openai.com/v1', default_model=model, auth_mode='bearer',
            secret_ref=account['secret_ref'], is_local=False,
            provider_metadata={'models': models, 'persistent': persistent, 'auth': 'chatgpt-plan'},
        )
        save_provider_connection(state, connection, persist_draft=False)

    manager.on_connected = publish
    for account in manager.records['accounts'].values():
        if account.get('pending_publish') and account.get('models'):
            publish(account, account['models'], account.get('persistent', False))
            account['pending_publish'] = False
    manager._save()

    @router.get('/attempts/{attempt_id}')
    def status(attempt_id: str):
        return manager.status(attempt_id)

    @router.post('/attempts/{attempt_id}/cancel')
    def cancel(attempt_id: str):
        manager.cancel(attempt_id)
        return manager.status(attempt_id)

    @router.put('/connections/{connection_id}/model')
    def model(connection_id: str, body: ModelRequest):
        state = load_state(runtime_config)
        connection = next((c for c in state.prefs.connections if c.connection_id == connection_id and c.provider_kind == 'chatgpt'), None)
        if connection is None:
            raise ChatGPTAuthError('ChatGPT connection was not found.')
        models = account_models(access_token(connection.secret_ref))
        if body.model not in {m['slug'] for m in models}:
            raise ChatGPTAuthError('Choose a model available to this ChatGPT account.')
        with PREFERENCES_LOCK:
            current = load_state(runtime_config)
            latest = next((c for c in current.prefs.connections if c.connection_id == connection_id and c.provider_kind == 'chatgpt' and c.secret_ref == connection.secret_ref), None)
            if latest is None:
                raise ChatGPTAuthError('This account was disconnected or changed. Reconnect before choosing a model.')
            latest.default_model = body.model
            latest.provider_metadata['models'] = models
            save_provider_connection(current, latest, persist_draft=False)
        return {'model': body.model, 'models': models}

    @router.delete('/connections/{connection_id}')
    def disconnect(connection_id: str):
        # Stop work before clearing credentials. The worker must not send another transcript.
        def remove_local():
            app.state.job_manager.cancel_provider('chatgpt')
            with PREFERENCES_LOCK:
                state = load_state(runtime_config)
                delete_provider_connection(state, connection_id, persist_draft=False)
        confirmed = manager.disconnect(connection_id, before_clear=remove_local)
        return {'disconnected': True, 'revocation_confirmed': confirmed}

    app.include_router(router)

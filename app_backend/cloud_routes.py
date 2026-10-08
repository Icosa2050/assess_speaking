"""Local cloud preferences and browser authorization management."""
from typing import Literal
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Request

from app_core.cloud_policy import CloudPolicyError, CloudSettings, SpendingLedger, read_settings, write_settings
from app_core.openrouter_auth import OpenRouterAuth
from app_core.preference_lock import synchronized_preferences
from app_core.secret_store import delete_secret
from app_core.services import save_provider_connection, set_default_provider_connection
from app_core.runtime_resolver import active_connection, resolve_connection_runtime
from app_core.runtime_providers import requires_api_key
from app_core.state import ProviderConnection
from uuid import uuid4


class SpendingReconciliation(BaseModel):
    actual_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    provider_cost_confirmed: Literal[True]


def install_cloud_routes(app, config, load_state):
    root = config.app_data.root
    def local(request: Request):
        if request.url.hostname not in {'localhost', '127.0.0.1', '::1'} or request.headers.get('X-Vostavo-Client') != 'desktop':
            raise HTTPException(403, detail='Use the local Vostavo app.')
    router = APIRouter(prefix='/v1/runtime/cloud', dependencies=[Depends(local)])

    @router.get('')
    def settings():
        return {'settings': read_settings(root).model_dump(), 'spending': SpendingLedger(root).summary()}

    @router.post('/spending/{request_id}/reconcile')
    def reconcile(request_id: str, request: SpendingReconciliation):
        try:
            SpendingLedger(root).reconcile(request_id, request.actual_cost_usd, source='user_confirmed_provider_cost')
        except CloudPolicyError as exc:
            raise HTTPException(400, detail=str(exc)) from None
        return {'spending': SpendingLedger(root).summary()}

    @router.put('')
    @synchronized_preferences
    def save(settings: CloudSettings):
        state = load_state(config)
        connections = {item.connection_id: item for item in state.prefs.connections}
        if settings.asr_provider == 'groq':
            connection = connections.get(settings.asr_connection_id)
            if not connection or connection.provider_kind != 'groq':
                raise HTTPException(400, detail='Choose a saved Groq connection for transcription.')
        for identity, mode in settings.openrouter_modes.items():
            connection = connections.get(identity)
            if not connection or connection.provider_kind != 'openrouter':
                raise HTTPException(400, detail='Choose a saved OpenRouter connection.')
            if mode == 'free' and (not connection.default_model.endswith(':free') or connection.default_model.startswith('openrouter/')):
                raise HTTPException(400, detail='Free-only access needs an explicitly selected :free model.')
        if settings.paid_fallback_enabled:
            from app_core.runtime_resolver import active_connection
            active = active_connection(state.prefs)
            fallback = connections.get(settings.fallback_connection_id)
            if not fallback or fallback.provider_kind != 'openrouter' or active and active.connection_id == fallback.connection_id:
                raise HTTPException(400, detail='Select a distinct saved OpenRouter fallback connection.')
        write_settings(root, settings)
        return {'settings': settings.model_dump(), 'spending': SpendingLedger(root).summary()}

    @synchronized_preferences
    def publish(key):
        state = load_state(config)
        previous = active_connection(state.prefs)
        runtime = resolve_connection_runtime(previous) if previous else None
        preserve_previous = runtime and runtime.model and (not requires_api_key(runtime.provider) or runtime.api_key)
        identity = uuid4().hex
        connection = ProviderConnection(connection_id=identity, provider_kind='openrouter', label='OpenRouter',
            base_url='https://openrouter.ai/api/v1', default_model='', auth_mode='bearer', secret_ref='cloud-' + identity,
            provider_metadata={'auth': 'openrouter-pkce', 'requires_model_selection': True})
        try:
            save_provider_connection(state, connection, api_key=key, persist_draft=False)
            if preserve_previous:
                set_default_provider_connection(state, previous.connection_id, persist_draft=False)
        except Exception:  # quality: allow[broad-except] failed publication cleans up only this new credential
            delete_secret(connection.secret_ref)
            raise

    manager = OpenRouterAuth(publish)
    app.state.openrouter_auth = manager

    @router.post('/openrouter/sign-in')
    def start():
        return manager.start()

    @router.get('/openrouter/attempts/{attempt_id}')
    def status(attempt_id: str):
        return manager.status(attempt_id)

    @router.post('/openrouter/attempts/{attempt_id}/open-browser')
    def open_browser(attempt_id: str):
        if not manager.open_browser(attempt_id):
            raise HTTPException(400, detail='Open the provided sign-in link in your browser.')
        return {'opened': True}

    @router.post('/openrouter/cancel')
    def cancel():
        manager.close()
        return {'status': 'cancelled'}

    app.include_router(router)

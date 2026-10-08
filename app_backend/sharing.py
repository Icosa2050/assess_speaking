"""Versioned, secret-free acceptance identity for actual assessment destinations."""
from __future__ import annotations
import hashlib
import ipaddress
import json
from urllib.parse import urlsplit
from app_core.runtime_resolver import active_connection
from app_core.runtime_providers import normalize_provider, runtime_base_url, SUPPORTED_PROVIDERS, SETUP_PROVIDER_CHOICES, provider_kind_from_choice


def destination(provider, model, connection_id='', base_url='', mode=''):
    if provider not in SUPPORTED_PROVIDERS and provider not in SETUP_PROVIDER_CHOICES:
        raise ValueError("Unknown provider")
    provider = provider_kind_from_choice(provider)
    fixed = {'chatgpt': 'https://api.openai.com/v1', 'groq': 'https://api.groq.com/openai/v1', 'xai': 'https://api.x.ai/v1', 'openrouter': 'https://openrouter.ai/api/v1'}
    endpoint = fixed.get(provider) or runtime_base_url(provider, base_url)
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Choose a provider endpoint without credentials, query or fragment.')
    host = parsed.hostname.lower()
    local = host == 'localhost'
    try:
        local = local or ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if provider not in {'ollama', 'lmstudio', 'openai_compatible', 'openrouter', 'groq', 'xai', 'chatgpt'} or not model or str(model).lower() in {'auto', 'openrouter/auto'}:
        raise ValueError('Choose a known provider and explicit model before sharing.')
    return {'provider': provider, 'connection_id': connection_id, 'model': model, 'host': host,
            'local': local, 'mode': mode}, endpoint.rstrip('/')


def assessment_route(state, settings, selection):
    selected = active_connection(state.prefs)
    raw_provider = selection.get('provider') or (selected.provider_kind if selected else state.prefs.provider)
    provider = provider_kind_from_choice(raw_provider)
    model = selection.get('llm_model') or (selected.default_model if selected else '')
    base = selection.get('llm_base_url') or (selected.base_url if selected else '')
    whisper = selection.get('whisper') or state.prefs.whisper_model
    route = {'version': 1, 'available': False, 'audio': None, 'analysis': None, 'fallback': None, 'fingerprint': ''}
    endpoints = {}
    try:
        if raw_provider not in SUPPORTED_PROVIDERS and raw_provider not in SETUP_PROVIDER_CHOICES:
            raise ValueError("Unknown provider")
        if selected and (provider != normalize_provider(selected.provider_kind) or model != selected.default_model or (selection.get('llm_base_url') and runtime_base_url(provider, base) != runtime_base_url(provider, selected.base_url))):
            raise ValueError('The selected analysis connection changed. Refresh its sharing route.')
        if settings.asr_provider == 'local':
            route['audio'] = {'provider': 'local', 'model': whisper, 'connection_id': '', 'host': '', 'local': True, 'mode': ''}
        else:
            speech = next((c for c in state.prefs.connections if c.connection_id == settings.asr_connection_id and c.provider_kind == 'groq'), None)
            if not speech or not speech.secret_ref:
                raise ValueError('Select a saved Groq transcription account.')
            route['audio'], endpoints['audio'] = destination('groq', settings.asr_model, speech.connection_id)
        if not selected and provider not in {'ollama', 'lmstudio', 'openai_compatible'}:
            raise ValueError('Select a saved analysis account.')
        route['analysis'], endpoints['analysis'] = destination(provider, model, selected.connection_id if selected else '', base, settings.openrouter_modes.get(selected.connection_id, '') if selected else '')
        if settings.paid_fallback_enabled:
            alternate = next((c for c in state.prefs.connections if c.connection_id == settings.fallback_connection_id and c.provider_kind == 'openrouter'), None)
            if not alternate or not alternate.secret_ref:
                raise ValueError('The enabled fallback account is unavailable.')
            route['fallback'], endpoints['fallback'] = destination(alternate.provider_kind, alternate.default_model, alternate.connection_id, alternate.base_url, settings.openrouter_modes.get(alternate.connection_id, ''))
        canonical = {key: route[key] for key in ('version', 'audio', 'analysis', 'fallback')}
        canonical['endpoints'] = endpoints
        route['fingerprint'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        route['available'] = True
    except (ValueError, TypeError):
        # Return partial known destinations without reflecting secret endpoint text.
        route['available'] = False
    return route

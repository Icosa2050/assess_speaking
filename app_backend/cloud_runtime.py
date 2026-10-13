"""Backend-owned cloud orchestration; workers exchange stage results, not keys."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re

from app_core.cloud_policy import CloudPolicyError, locked_file, read_settings
from app_core.runtime_resolver import active_connection


class CloudRuntime:
    def __init__(self, root, state, key_reader, *, audio_path=None, cache_root=None, settings=None):
        self.root = Path(root)
        self.settings = (settings or read_settings(self.root)).model_copy(deep=True)
        self.primary = active_connection(state.prefs)
        import copy
        self.connections = {item.connection_id: copy.deepcopy(item) for item in state.prefs.connections}
        self.primary = self.connections.get(self.primary.connection_id) if self.primary else None
        self.key_reader = key_reader
        self.audio_path = audio_path
        self.cache_root = cache_root
        self.fallback = False
        self.cancelled = lambda: False

    def _key(self, connection):
        if connection.provider_kind == 'chatgpt':
            from app_core.chatgpt_auth import access_token
            return access_token(connection.secret_ref)
        key = self.key_reader(connection)
        if not key:
            raise CloudPolicyError('Reconnect the selected provider; its saved key is unavailable.')
        return key

    def handle(self, message):
        if self.cancelled():
            raise CloudPolicyError('Cloud assessment was cancelled.')
        if message['operation'] == 'reject_reply':
            name, digest = message.get('reply_cache_id', ''), message.get('reply_text_sha256', '')
            if not self.cache_root or not re.fullmatch(r'reply-[0-9a-f]{64}', name) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                return {'invalidated': False}
            from assessment_runtime.checkpoints import StageCache
            return {'invalidated': StageCache(Path(self.cache_root) / 'cloud-replies').reject_reply(name, digest)}
        if message['operation'] == 'asr':
            connection = self.connections.get(self.settings.asr_connection_id)
            if not connection or connection.provider_kind != 'groq':
                raise CloudPolicyError('Select a saved Groq transcription connection.')
            from assessment_runtime.groq_asr import transcribe
            from assessment_runtime.checkpoints import StageCache
            cache = StageCache(Path(self.cache_root) / 'cloud-asr') if self.cache_root else None
            def cached_chunk(name, identity, produce):
                with locked_file(cache.root / (name + '.lock')):
                    return cache.run(name, identity, produce)
            return transcribe(Path(self.audio_path), api_key=self._key(connection), model=self.settings.asr_model,
                              language=message.get('language'), cached=cached_chunk if cache else None, cancelled=self.cancelled)
        if message['operation'] != 'completion' or not self.primary:
            raise CloudPolicyError('Unsupported cloud operation or missing saved connection.')
        if message['provider'] != self.primary.provider_kind or message['model'] != self.primary.default_model:
            raise CloudPolicyError('The assessment provider snapshot does not match its saved connection.')
        primary = self.connections.get(self.settings.fallback_connection_id) if self.fallback else self.primary
        message = {**message, 'timeout': min(float(message.get('timeout', 120)), 120)}
        try:
            return self._cached_complete(primary, message)
        except Exception as exc:  # quality: allow[broad-except] boundary considers only explicitly typed transient failures
            from app_core.openrouter_cloud import OpenRouterError
            from assessment_runtime.responses_client import ResponsesError
            # Do not replay ambiguous paid requests, auth failures, schema failures or unknown timeouts.
            status = getattr(exc, 'status', None)
            eligible = (isinstance(exc, OpenRouterError) and status in {429, 503} and self.settings.openrouter_modes.get(primary.connection_id) == 'free') or (isinstance(exc, ResponsesError) and status == 429 and primary.provider_kind in {'groq', 'chatgpt'})
            if not eligible or not self.settings.paid_fallback_enabled or self.fallback or self.cancelled():
                raise
            alternate = self.connections.get(self.settings.fallback_connection_id)
            if not alternate or alternate.provider_kind != 'openrouter' or alternate.connection_id == primary.connection_id:
                raise CloudPolicyError('The authorized paid fallback connection is unavailable.') from None
            self.fallback = True
            result = self._cached_complete(alternate, message)
            return {**result, 'fallback_reason': 'free_provider_quota_or_service'}

    def _cached_complete(self, connection, message):
        if not self.cache_root or connection is None:
            return self._complete(connection, message)
        from assessment_runtime.checkpoints import StageCache
        identity = {'connection_id': connection.connection_id, 'provider': connection.provider_kind,
                    'model': connection.default_model, 'base_url': connection.base_url,
                    'mode': self.settings.openrouter_modes.get(connection.connection_id),
                    'prompt': message['prompt'], 'schema': message.get('schema'),
                    'max_output_tokens': self.settings.max_output_tokens, 'version': 1}
        name = 'reply-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        produced = False
        def dispatch():
            nonlocal produced
            if self.cancelled():
                raise CloudPolicyError('Cloud assessment was cancelled.')
            produced = True
            return self._complete(connection, message)
        root = Path(self.cache_root) / 'cloud-replies'
        # A resumed worker waits for an older backend request of this stage,
        # rather than dispatching the same paid prompt while it is in flight.
        with locked_file(root / (name + '.lock')):
            value = StageCache(root).run(name, identity, dispatch)
        result = value if produced else {**value, 'cache_reused': True, 'original_cost_usd': value.get('cost_usd'), 'cost_usd': 0}
        return {**result, 'reply_cache_id': name, 'reply_text_sha256': hashlib.sha256(str(value.get('text') or '').encode()).hexdigest()}

    def _complete(self, connection, message):
        if connection is None:
            raise CloudPolicyError('The selected connection was deleted. Select another explicitly.')
        if connection.provider_kind in {'ollama', 'lmstudio', 'openai_compatible'}:
            from assessment_runtime.llm_client import _chat_completion
            key = self.key_reader(connection)
            if connection.auth_mode == 'bearer' and not key:
                raise CloudPolicyError('Reconnect the selected analysis connection.')
            extra = {'response_format': {'type': 'json_schema', 'json_schema': message['schema']}} if message.get('schema') else None
            return {'text': _chat_completion(connection.provider_kind, connection.default_model, message['prompt'], message['timeout'], None,
                                             api_key=key, base_url=connection.base_url, extra_payload=extra, require_json_object=True),
                    'provider': connection.provider_kind, 'model': connection.default_model, 'cost_usd': None}
        key = self._key(connection)
        if connection.provider_kind == 'openrouter':
            if connection.connection_id not in self.settings.openrouter_modes:
                # Existing manual connections retain their explicit selected route. They never enable managed fallback.
                from assessment_runtime.llm_client import _chat_completion
                extra = {'response_format': {'type': 'json_schema', 'json_schema': message['schema']}} if message.get('schema') else None
                return {'text': _chat_completion('openrouter', connection.default_model, message['prompt'], message['timeout'], key,
                                                api_key=key, base_url='https://openrouter.ai/api/v1', extra_payload=extra),
                        'provider': 'openrouter', 'model': connection.default_model, 'cost_usd': None}
            from app_core.openrouter_cloud import complete
            return complete(root=self.root, connection=connection, api_key=key,
                            mode=self.settings.openrouter_modes.get(connection.connection_id, 'free'), settings=self.settings,
                            prompt=message['prompt'], schema=message.get('schema'), timeout=message['timeout'])
        from assessment_runtime.responses_client import complete
        return {'text': complete(provider=connection.provider_kind, model=connection.default_model, prompt=message['prompt'],
                                 api_key=key, timeout_sec=message['timeout'], schema=message.get('schema')),
                'provider': connection.provider_kind, 'model': connection.default_model, 'cost_usd': None}

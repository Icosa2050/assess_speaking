"""Policy-enforced OpenRouter calls, with live price evidence and durable reservations."""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import json
from pathlib import Path

import httpx

from app_core.cloud_policy import CloudPolicyError, SpendingLedger, money

BASE = 'https://openrouter.ai/api/v1'


class OpenRouterError(CloudPolicyError):
    def __init__(self, detail, status=None, retry_after=None):
        super().__init__(detail)
        self.status = status
        self.retry_after = retry_after


def request_json(client, method, url, **kwargs):
    try:
        response = client.request(method, url, **kwargs)
        if response.status_code >= 400:
            raise OpenRouterError('OpenRouter authentication, quota or service request failed.', response.status_code, response.headers.get('retry-after'))
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError('Unexpected response')
        return data
    except (httpx.HTTPError, ValueError):
        raise OpenRouterError('OpenRouter request failed; its remote outcome may be unknown.') from None


def complete(*, root: Path, connection, api_key: str, mode: str, settings, prompt: str, schema: dict | None, timeout: float) -> dict:
    model = connection.default_model
    if not model or not api_key or mode not in {'free', 'paid'}:
        raise OpenRouterError('Select an explicit OpenRouter access mode, model and saved key.')
    if mode == 'free' and (not model.endswith(':free') or model.startswith('openrouter/')):
        raise OpenRouterError('Free-only access requires a pinned :free model; paid models and automatic routers are disabled.')
    with httpx.Client(timeout=min(timeout, 120), follow_redirects=False) as client:
        catalog = request_json(client, 'GET', BASE + '/models', timeout=min(timeout, 20))
        item = next((item for item in catalog.get('data', []) if item.get('id') == model), None)
        if not item or not {'prompt', 'completion'} <= set(item.get('pricing') or {}):
            raise OpenRouterError('The selected model price could not be verified.')
        prices = item['pricing']
        if mode == 'free' and any(money(value) != 0 for value in prices.values()):
            raise OpenRouterError('The selected route is no longer free.')
        if 'response_format' not in item.get('supported_parameters', []) and schema:
            raise OpenRouterError('The selected model does not advertise structured-output support.')
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': settings.max_output_tokens,
                   'provider': {'allow_fallbacks': False, 'require_parameters': True}, 'usage': {'include': True}}
        payload['provider']['max_price'] = {'prompt': float(money(prices['prompt']) * 1_000_000),
                                             'completion': float(money(prices['completion']) * 1_000_000),
                                             'request': float(money(prices.get('request', 0))), 'image': 0}
        if schema:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': schema}
        reservation = None
        ledger = SpendingLedger(root)
        if mode == 'paid':
            limit = request_json(client, 'GET', BASE + '/key', headers={'Authorization': 'Bearer ' + api_key}, timeout=min(timeout, 20)).get('data', {})
            if limit.get('limit') is None or limit.get('limit_remaining') is None or money(limit['limit']) <= 0:
                raise OpenRouterError('Set a provider-side spending limit on this key before enabling managed paid requests.')
            # UTF-8 bytes conservatively bound token count without a provider-specific tokenizer.
            inputs = len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) + 1024
            estimate = (money(prices['prompt']) * inputs + money(prices['completion']) * settings.max_output_tokens + money(prices.get('request', 0))) * Decimal('1.1')
            if any(money(value) for name, value in prices.items() if name not in {'prompt', 'completion', 'request', 'input_cache_read'}) or money(prices.get('input_cache_read', 0)) > money(prices['prompt']):
                raise OpenRouterError('This paid model has additional pricing that is not supported by the budget estimator.')
            if money(limit['limit_remaining']) < estimate:
                raise OpenRouterError('Provider-side remaining key budget is insufficient.')
            reservation = ledger.reserve(estimate, settings.monthly_budget_usd, connection.connection_id)
        try:
            result = request_json(client, 'POST', BASE + '/chat/completions', json=payload, headers={'Authorization': 'Bearer ' + api_key})
        except OpenRouterError as exc:
            if reservation and exc.status and 400 <= exc.status < 500 and exc.status != 408:
                # These are explicit pre-generation rejections. HTTP 200 errors,
                # timeouts and server failures still have an unknown billing outcome.
                ledger.reconcile(reservation, 0, source=f'provider_rejected_http_{exc.status}')
            raise
        if reservation:
            ledger.reconcile(reservation, (result.get('usage') or {}).get('cost'))
        from assessment_runtime.llm_client import _extract_assistant_message_text
        return {'text': _extract_assistant_message_text(result), 'provider': 'openrouter', 'model': model,
                'route': result.get('provider'), 'cost_usd': (result.get('usage') or {}).get('cost'), 'reservation_id': reservation}


@contextmanager
def probe_policy(root, connection_id, model, api_key):
    from app_core.cloud_policy import read_settings
    from app_core.state import ProviderConnection
    from assessment_runtime.cloud_transport import completion
    from assessment_runtime.llm_client import LLMClientError
    settings = read_settings(root)
    mode = settings.openrouter_modes.get(connection_id)
    if not mode:
        yield
        return
    connection = ProviderConnection(connection_id=connection_id, provider_kind='openrouter', default_model=model)
    def dispatch(message):
        try:
            return complete(root=root, connection=connection, api_key=api_key, mode=mode, settings=settings,
                            prompt=message['prompt'], schema=message.get('schema'), timeout=message['timeout'])
        except CloudPolicyError as exc:
            raise LLMClientError(str(exc)) from None
    token = completion.set(dispatch)
    try:
        yield
    finally:
        completion.reset(token)

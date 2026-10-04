"""Provider Responses adapters using the official client and bounded stream assembly."""
from __future__ import annotations

import time
import httpx
import httpx2
from openai import OpenAI, OpenAIError, DefaultHttpxClient


class ResponsesError(RuntimeError):
    pass


def complete(*, provider: str, model: str, prompt: str, api_key: str,
             timeout_sec: float, schema: dict | None = None) -> str:
    if not api_key or not model:
        raise ResponsesError('Connect your account and choose an available model.')
    url = 'https://api.openai.com/v1' if provider == 'chatgpt' else 'https://api.x.ai/v1'
    payload = dict(model=model, input=[{'role': 'user', 'content': prompt}], store=False, stream=True)
    if schema:
        payload['text'] = {'format': {'type': 'json_schema', **schema}}
    if provider == 'xai':
        payload['max_output_tokens'] = 4096
    deadline = time.monotonic() + timeout_sec
    chunks: list[str] = []
    received = 0
    try:
        with OpenAI(api_key=api_key, base_url=url, max_retries=0, timeout=timeout_sec, http_client=DefaultHttpxClient(follow_redirects=False)) as client:
            with client.responses.create(**payload) as stream:
                for event in stream:
                    if time.monotonic() > deadline:
                        raise ResponsesError('The provider took too long. Your recording is retained; retry later.')
                    if event.type in ('error', 'response.failed', 'response.incomplete', 'response.refusal.delta', 'response.refusal.done'):
                        raise ResponsesError('The provider could not complete feedback. Check account access or allowance and retry.')
                    if event.type == 'response.output_text.delta':
                        received += len(event.delta.encode('utf-8'))
                        if received > 1_000_000:
                            raise ResponsesError('The provider response exceeded the feedback size limit.')
                        chunks.append(event.delta)
                    elif event.type == 'response.completed':
                        if getattr(event.response, 'status', 'completed') != 'completed':
                            raise ResponsesError('The provider response was incomplete.')
                        text = ''.join(chunks).strip()
                        if not text:
                            raise ResponsesError('The provider returned no feedback text.')
                        return text
    except (httpx.HTTPError, httpx2.HTTPError):
        raise ResponsesError('The provider connection was interrupted. Retry feedback later.') from None
    except OpenAIError as exc:
        status = getattr(exc, 'status_code', None)
        if status in (401, 403):
            raise ResponsesError('Account authorization failed. Reconnect your account.') from None
        if status in (402, 429):
            raise ResponsesError('Provider allowance or credits are exhausted. Check your provider account.') from None
        raise ResponsesError('The provider request failed. Check the connection and selected model.') from None
    raise ResponsesError('The connection ended before feedback completed. Please retry.')


def models(provider: str, api_key: str, timeout_sec: float) -> dict:
    """Use fixed cloud endpoints and keep upstream bodies out of diagnostics."""
    if not api_key:
        raise ResponsesError('Connect this provider account first.')
    url = 'https://api.openai.com/v1' if provider == 'chatgpt' else 'https://api.x.ai/v1'
    try:
        with httpx.Client(timeout=timeout_sec, follow_redirects=False) as client:
            response = client.get(url + '/models', headers={'Authorization': 'Bearer ' + api_key})
            if not 200 <= response.status_code < 300:
                raise ResponsesError('Could not load models. Check account authorization and API access.')
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError('Unexpected catalog')
            if provider == 'chatgpt':
                return {'data': [{'id': item['slug']} for item in data.get('models', [])
                                 if item.get('visibility') == 'list' and item.get('slug')]}
            return data
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        raise ResponsesError('Could not load models. Check the provider connection.') from None

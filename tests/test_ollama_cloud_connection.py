"""Exercise Ollama Cloud URL construction below the provider helper mocks."""
import pytest

from app_core.runtime_providers import runtime_base_url, service_base_url
from app_core.services import test_runtime_connection as probe_connection
from assessment_runtime import llm_client


@pytest.mark.parametrize('base', ['https://ollama.com', 'https://ollama.com/api',
                                 'https://ollama.com/v1', 'https://ollama.com/api/v1/'])
def test_cloud_discovery_and_completion_use_distinct_api_roots(monkeypatch, base):
    calls = []
    def get(url, headers, timeout):
        calls.append(url)
        assert headers['Authorization'] == 'Bearer fixture-key'
        return {'models': [{'name': 'gpt-oss:120b'}]}
    def post(url, payload, headers, timeout):
        calls.append(url)
        assert headers['Authorization'] == 'Bearer fixture-key'
        assert payload['model'] == 'gpt-oss:120b'
        assert payload['max_tokens'] == 256
        return {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}
    monkeypatch.setattr(llm_client, '_get_json', get)
    monkeypatch.setattr(llm_client, '_post_json', post)
    result = probe_connection(provider='ollama', provider_choice='ollama_cloud',
                              model='gpt-oss:120b', base_url=base, api_key='fixture-key')
    assert calls == ['https://ollama.com/api/tags', 'https://ollama.com/v1/chat/completions']
    assert result['test_payload']['ok'] is True


@pytest.mark.parametrize('suffix', ['', '/api', '/v1', '/api/v1/'])
def test_local_proxy_prefix_survives_ollama_normalization(suffix):
    root = 'http://localhost:11434/proxy'
    assert service_base_url('ollama', root + suffix) == root
    assert runtime_base_url('ollama', root + suffix) == root + '/v1'
    assert runtime_base_url('openrouter', 'https://openrouter.ai/api/v1') == 'https://openrouter.ai/api/v1'

from unittest.mock import Mock, patch

import pytest

from scripts.start_practice import wait_ready


def test_launcher_reports_backend_exit_without_opening_browser():
    process = Mock()
    process.poll.return_value = 1
    process.returncode = 1
    with pytest.raises(RuntimeError, match='Server exited'):
        wait_ready('http://127.0.0.1:12345/v1/health', process)


def test_launcher_waits_for_a_successful_health_response():
    process = Mock()
    process.poll.return_value = None
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.status = 200
    with patch('scripts.start_practice.urlopen', side_effect=[OSError('starting'), response]), patch('scripts.start_practice.time.sleep'):
        wait_ready('http://127.0.0.1:12345/v1/health', process)


def test_launcher_and_backend_use_the_same_practice_timeout_constant():
    from assess_core.settings import PRACTICE_LLM_TIMEOUT_SEC, practice_llm_timeout_sec
    from scripts.start_practice import backend_environment

    with patch.dict('scripts.start_practice.os.environ', {}, clear=True):
        env = backend_environment(8802)
        backend_timeout = practice_llm_timeout_sec()

    assert PRACTICE_LLM_TIMEOUT_SEC == 180.0
    assert float(env['LLM_TIMEOUT_SEC']) == PRACTICE_LLM_TIMEOUT_SEC
    assert backend_timeout == PRACTICE_LLM_TIMEOUT_SEC


@pytest.mark.parametrize('configured', ['0', '-1', 'nan', 'inf', '-inf'])
def test_practice_timeout_rejects_non_positive_or_non_finite_override(configured):
    from assess_core.settings import PRACTICE_LLM_TIMEOUT_SEC, practice_llm_timeout_sec

    with patch.dict('assess_core.settings.os.environ', {'LLM_TIMEOUT_SEC': configured}, clear=True):
        assert practice_llm_timeout_sec() == PRACTICE_LLM_TIMEOUT_SEC

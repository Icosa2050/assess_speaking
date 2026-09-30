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

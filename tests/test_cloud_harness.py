"""Terminal polling must tolerate a worker publishing its final job snapshot."""
import json
from unittest.mock import Mock

import pytest

from tests.helpers import cloud_harness


@pytest.mark.parametrize('partial', ['', '{"worker_pid":', None])
def test_wait_retries_incomplete_job_publication(tmp_path, monkeypatch, partial):
    app = cloud_harness.Backend.__new__(cloud_harness.Backend)
    app.root = tmp_path
    app.deadline = 1
    app.request = Mock(return_value={'status': 'completed'})
    path = tmp_path / 'app/jobs/asmt_fixture.json'
    path.parent.mkdir(parents=True)
    if partial is not None:
        path.write_text(partial)
    monkeypatch.setattr(cloud_harness.time, 'monotonic', lambda: 0)
    pause = Mock(side_effect=lambda seconds: path.write_text(json.dumps({'worker_pid': 123})))
    monkeypatch.setattr(cloud_harness.time, 'sleep', pause)
    exited = Mock(side_effect=ProcessLookupError)
    monkeypatch.setattr(cloud_harness.os, 'kill', exited)
    assert app.wait('asmt_fixture') == {'status': 'completed'}
    assert app.request.call_count == 2
    pause.assert_called_once_with(.05)
    exited.assert_called_once_with(123, 0)


def test_wait_keeps_deadline_for_permanently_damaged_job(tmp_path, monkeypatch):
    app = cloud_harness.Backend.__new__(cloud_harness.Backend)
    app.root = tmp_path
    app.deadline = 1
    app.request = Mock(return_value={'status': 'completed'})
    app.dispatches = Mock(return_value=[])
    path = tmp_path / 'app/jobs/asmt_fixture.json'
    path.parent.mkdir(parents=True)
    path.write_text('')
    monkeypatch.setattr(cloud_harness.time, 'monotonic', Mock(side_effect=[0, 2]))
    monkeypatch.setattr(cloud_harness.time, 'sleep', Mock())
    with pytest.raises(AssertionError, match='60-second fixture deadline.*JSONDecodeError'):
        app.wait('asmt_fixture')

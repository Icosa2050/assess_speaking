import json
import os
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app_backend.config import build_backend_runtime_config
from app_backend.contracts import AssessmentCreateRequest, AssessmentCreateResponse, ErrorCode, JobStatus
from app_backend.jobs import AssessmentBusyError, JobManager, QUIET_WARNING_MARGIN_SEC, _job_worker, _read_json, _update_job_file, _write_json, prunable_job_metadata_files
from app_core.cloud_policy import locked_file
from assessment_runtime.runner import AssessmentRunResult


class BackendJobsTests(unittest.TestCase):
    def test_cloud_submit_persists_deadline_inputs(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            upload = manager.register_upload(data=b"audio", filename="clip.wav")
            request = AssessmentCreateRequest(audio_id=upload.audio_id, whisper="tiny", provider="chatgpt",
                llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                task_family="generic", theme="test", target_duration_sec=90)
            cloud = SimpleNamespace(primary=None, settings=SimpleNamespace(asr_provider="groq", asr_model="whisper"))
            process = mock.Mock()
            process.is_alive.return_value = False
            with mock.patch.object(manager._ctx, "Process", return_value=process):
                created = manager.submit(request, cloud_runtime=cloud)
            job_file = config.jobs_dir / f"{created.assessment_id}.json"
            payload = json.loads(job_file.read_text(encoding="utf-8"))
            self.assertTrue(payload["request"]["cloud_broker"])
            self.assertEqual(payload["request"]["asr_provider"], "groq")
            self.assertNotIn("llm_api_key", payload["request"])
            manager._processes.pop(created.assessment_id)
            payload["status"] = "running"
            payload["phase"] = "transcribing"
            job_file.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(manager.get_status(created.assessment_id).quiet_warning_after_sec, 660 + QUIET_WARNING_MARGIN_SEC)

    def test_cancel_provider_attempts_every_matching_job_before_reporting_busy(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            for assessment_id in ("asmt_one", "asmt_two"):
                (config.jobs_dir / f"{assessment_id}.json").write_text(json.dumps({
                    "request": {"provider": "chatgpt"}, "status": "running"}), encoding="utf-8")
                manager._processes[assessment_id] = mock.Mock()
            with mock.patch.object(manager, "cancel", side_effect=[AssessmentBusyError("still stopping"), None]) as cancel:
                with self.assertRaises(AssessmentBusyError):
                    manager.cancel_provider("chatgpt")
            self.assertEqual([call.args[0] for call in cancel.call_args_list], ["asmt_one", "asmt_two"])

    def test_active_cloud_call_keeps_provider_busy_after_worker_exit(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            assessment_id = "asmt_cloud_pending"
            (config.jobs_dir / f"{assessment_id}.json").write_text(json.dumps({
                "request": {"provider": "chatgpt"}, "status": "cancelled"}), encoding="utf-8")
            active = threading.Event()
            active.set()
            manager._active_cloud_requests[assessment_id] = active
            self.assertTrue(manager.has_active_provider("chatgpt"))
            with self.assertRaises(AssessmentBusyError):
                manager.cancel_provider("chatgpt")
            active.clear()
            self.assertFalse(manager.has_active_provider("chatgpt"))

    def test_finishing_cloud_services_prune_request_events(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            for index in range(20):
                assessment_id = f"asmt_cloud_{index}"
                active = threading.Event()
                manager._active_cloud_requests[assessment_id] = active
                state = {"alive": True}
                process = mock.Mock()
                process.is_alive.side_effect = lambda: state["alive"]
                channel = mock.Mock()
                channel.recv.return_value = {"rpc_id": index}

                def finish(_message):
                    self.assertTrue(active.is_set())
                    state["alive"] = False
                    return "done"

                runtime = SimpleNamespace(handle=finish)
                manager._serve_cloud(process, channel, runtime, active)
                channel.send.assert_called_once_with({"value": "done", "rpc_id": index})
                channel.close.assert_called_once()
                self.assertNotIn(assessment_id, manager._active_cloud_requests)
            self.assertEqual(manager._active_cloud_requests, {})

    def test_cloud_service_error_prunes_only_its_event_even_if_channel_close_fails(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            active = threading.Event()
            other = threading.Event()
            other.set()
            manager._active_cloud_requests.update({"asmt_error": active, "asmt_other": other})
            state = {"alive": True}
            process = mock.Mock()
            process.is_alive.side_effect = lambda: state["alive"]
            channel = mock.Mock()
            channel.recv.return_value = {"rpc_id": "error"}
            channel.close.side_effect = OSError("pipe close failed")

            def fail(_message):
                self.assertTrue(active.is_set())
                state["alive"] = False
                raise RuntimeError("private provider failure")

            with self.assertRaisesRegex(OSError, "pipe close failed"):
                manager._serve_cloud(process, channel, SimpleNamespace(handle=fail), active)
            channel.send.assert_called_once_with({"error": "Cloud request failed. Your recording and completed stages are retained.", "rpc_id": "error"})
            self.assertFalse(active.is_set())
            self.assertEqual(manager._active_cloud_requests, {"asmt_other": other})

    def test_cloud_wait_releases_submit_and_retention_locks_then_rechecks(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            active = threading.Event()
            active.set()
            manager._active_cloud_requests["asmt_finishing"] = active
            waiting = threading.Event()
            resume = threading.Event()
            retention_acquired = threading.Event()
            outcomes = []
            request = AssessmentCreateRequest(audio_id="unused", whisper="tiny", provider="ollama",
                llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                task_family="generic", theme="test", target_duration_sec=90)

            def pause_wait(_seconds):
                waiting.set()
                self.assertTrue(resume.wait(2))

            def take_retention_lock():
                with locked_file(config.jobs_dir / "retention.lock"):
                    retention_acquired.set()

            with mock.patch("app_backend.jobs.time.sleep", side_effect=pause_wait), mock.patch.object(
                manager, "_submit", return_value="admitted"
            ) as submit:
                worker = threading.Thread(target=lambda: outcomes.append(manager.submit(request)))
                worker.start()
                try:
                    self.assertTrue(waiting.wait(2))
                    self.assertTrue(manager._submit_lock.acquire(blocking=False))
                    manager._submit_lock.release()
                    retention = threading.Thread(target=take_retention_lock)
                    retention.start()
                    self.assertTrue(retention_acquired.wait(1))
                finally:
                    active.clear()
                    resume.set()
                    worker.join(2)
                    if "retention" in locals():
                        retention.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(outcomes, ["admitted"])
            submit.assert_called_once()

    def test_cloud_wait_rechecks_request_id_before_starting_another_worker(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            active = threading.Event()
            active.set()
            manager._active_cloud_requests["asmt_finishing"] = active
            waiting = threading.Event()
            resume = threading.Event()
            outcomes = []
            request = AssessmentCreateRequest(audio_id="same-audio", request_id="same-submit", whisper="tiny",
                provider="ollama", llm_model="test", expected_language="en", feedback_language="en",
                speaker_id="test", task_family="generic", theme="test", target_duration_sec=90)

            def pause_wait(_seconds):
                waiting.set()
                self.assertTrue(resume.wait(2))

            with mock.patch("app_backend.jobs.time.sleep", side_effect=pause_wait), mock.patch.object(manager, "_submit") as submit:
                worker = threading.Thread(target=lambda: outcomes.append(manager.submit(request)))
                worker.start()
                try:
                    self.assertTrue(waiting.wait(2))
                    _write_json(config.jobs_dir / "asmt_accepted.json", {
                        "assessment_id": "asmt_accepted", "status": "queued", "request": request.model_dump(),
                    })
                finally:
                    active.clear()
                    resume.set()
                    worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(outcomes[0].assessment_id, "asmt_accepted")
            submit.assert_not_called()

    def test_cloud_wait_scans_metadata_only_on_entry_and_before_admission(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            old_job = config.jobs_dir / "asmt_old.json"
            _write_json(old_job, {"assessment_id": "asmt_old", "status": "failed", "request": {"request_id": "other"}})
            request = AssessmentCreateRequest(audio_id="unused", request_id="new-submit", whisper="tiny",
                provider="ollama", llm_model="test", expected_language="en", feedback_language="en",
                speaker_id="test", task_family="generic", theme="test", target_duration_sec=90)
            active = threading.Event()
            active.set()
            manager._active_cloud_requests["asmt_finishing"] = active
            reads = []
            pauses = []
            read_json = _read_json

            def counted_read(path):
                if path == old_job:
                    reads.append(path)
                return read_json(path)

            def finish_after_three_ticks(_seconds):
                pauses.append(True)
                if len(pauses) == 3:
                    active.clear()

            with mock.patch("app_backend.jobs._read_json", side_effect=counted_read), mock.patch(
                "app_backend.jobs.time.sleep", side_effect=finish_after_three_ticks
            ), mock.patch.object(manager, "_submit", return_value="admitted"):
                self.assertEqual(manager.submit(request), "admitted")
            self.assertEqual(len(pauses), 3)
            self.assertEqual(len(reads), 2)

    def test_two_waiting_submits_share_one_accepted_request(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            active = threading.Event()
            active.set()
            manager._active_cloud_requests["asmt_finishing"] = active
            request = AssessmentCreateRequest(audio_id="same-audio", request_id="same-submit", whisper="tiny",
                provider="ollama", llm_model="test", expected_language="en", feedback_language="en",
                speaker_id="test", task_family="generic", theme="test", target_duration_sec=90)
            both_waiting = threading.Event()
            resume = threading.Event()
            count_lock = threading.Lock()
            wait_count = 0
            outcomes = []

            def pause_wait(_seconds):
                nonlocal wait_count
                with count_lock:
                    wait_count += 1
                    if wait_count == 2:
                        both_waiting.set()
                self.assertTrue(resume.wait(2))

            def accept(_request, **_kwargs):
                _write_json(config.jobs_dir / "asmt_accepted.json", {
                    "assessment_id": "asmt_accepted", "status": "queued", "request": request.model_dump(),
                })
                return AssessmentCreateResponse(assessment_id="asmt_accepted", status=JobStatus.QUEUED)

            with mock.patch("app_backend.jobs.time.sleep", side_effect=pause_wait), mock.patch.object(
                manager, "_submit", side_effect=accept
            ) as submit:
                workers = [threading.Thread(target=lambda: outcomes.append(manager.submit(request))) for _ in range(2)]
                for worker in workers:
                    worker.start()
                try:
                    self.assertTrue(both_waiting.wait(2))
                finally:
                    active.clear()
                    resume.set()
                    for worker in workers:
                        worker.join(2)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertEqual([outcome.assessment_id for outcome in outcomes], ["asmt_accepted", "asmt_accepted"])
            submit.assert_called_once()

    def test_real_cloud_service_blocks_retry_for_two_seconds_until_paid_call_finishes(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            assessment_id = "asmt_paid"
            _write_json(config.jobs_dir / f"{assessment_id}.json", {
                "assessment_id": assessment_id, "status": "running", "phase": "scoring_rubric",
                "request": {"provider": "chatgpt", "audio_id": "same-audio"},
            })
            state = {"alive": True}
            process = mock.Mock()
            process.is_alive.side_effect = lambda: state["alive"]
            process.terminate.side_effect = lambda: state.update(alive=False)
            manager._processes[assessment_id] = process
            active = threading.Event()
            manager._active_cloud_requests[assessment_id] = active
            channel = mock.Mock()
            channel.recv.return_value = {"rpc_id": "paid"}
            handling = threading.Event()
            release = threading.Event()

            def handle(_message):
                handling.set()
                self.assertTrue(release.wait(5))
                return "finished"

            service = threading.Thread(target=manager._serve_cloud,
                args=(process, channel, SimpleNamespace(handle=handle), active))
            service.start()
            try:
                self.assertTrue(handling.wait(2))
                self.assertTrue(manager.has_active_provider("chatgpt"))
                self.assertEqual(manager.cancel(assessment_id).status, JobStatus.CANCELLED)
                request = AssessmentCreateRequest(audio_id="same-audio", whisper="tiny", provider="ollama",
                    llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                    task_family="generic", theme="test", target_duration_sec=90)
                with mock.patch.object(manager, "_submit", return_value="admitted") as submit:
                    started = time.monotonic()
                    with self.assertRaisesRegex(AssessmentBusyError, "provider request is still finishing"):
                        manager.submit(request)
                    elapsed = time.monotonic() - started
                    self.assertGreaterEqual(elapsed, 1.8)
                    self.assertLess(elapsed, 5.0)
                    submit.assert_not_called()
                    release.set()
                    service.join(2)
                    self.assertFalse(service.is_alive())
                    self.assertEqual(manager._active_cloud_requests, {})
                    self.assertEqual(manager.submit(request), "admitted")
                    submit.assert_called_once()
            finally:
                release.set()
                service.join(2)

    def test_unstarted_cloud_service_is_removed_and_worker_stopped(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            upload = manager.register_upload(data=b"audio", filename="clip.wav")
            request = AssessmentCreateRequest(audio_id=upload.audio_id, whisper="tiny", provider="chatgpt",
                llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                task_family="generic", theme="test", target_duration_sec=90)
            cloud = SimpleNamespace(primary=None, settings=SimpleNamespace(asr_provider="groq", asr_model="whisper"))
            state = {"alive": True}
            process = mock.Mock()
            process.is_alive.side_effect = lambda: state["alive"]
            process.terminate.side_effect = lambda: state.update(alive=False)
            with mock.patch.object(manager._ctx, "Process", return_value=process), mock.patch(
                "app_backend.jobs.threading.Thread", side_effect=RuntimeError("thread unavailable")
            ):
                with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                    manager.submit(request, cloud_runtime=cloud)
            self.assertEqual(manager._active_cloud_requests, {})
            self.assertEqual(manager._processes, {})
            process.terminate.assert_called_once()
            job_file = next(config.jobs_dir.glob("asmt_*.json"))
            self.assertEqual(_read_json(job_file)["status"], "cancelled")

    def test_cloud_wait_budget_starts_when_active_call_is_observed(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            active = threading.Event()
            active.set()
            manager._active_cloud_requests["asmt_finishing"] = active
            request = AssessmentCreateRequest(audio_id="unused", whisper="tiny", provider="ollama",
                llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                task_family="generic", theme="test", target_duration_sec=90)
            clock = {"now": 0.0}
            def finish_slow_scan():
                clock["now"] = 10.0

            with mock.patch("app_backend.jobs.time.monotonic", side_effect=lambda: clock["now"]), mock.patch.object(
                manager, "wait_for_finished_workers", side_effect=finish_slow_scan
            ), mock.patch(
                "app_backend.jobs.time.sleep", side_effect=lambda _seconds: active.clear()
            ), mock.patch.object(manager, "_submit", return_value="admitted") as submit:
                self.assertEqual(manager.submit(request), "admitted")
            submit.assert_called_once()

    def test_status_warning_deadline_uses_configured_model_timeout_and_cloud_phase(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            job_file = config.jobs_dir / "asmt_deadline.json"
            job_file.write_text(json.dumps({"assessment_id": "asmt_deadline", "status": "running",
                "phase": "scoring_rubric", "request": {"cloud_broker": False}}), encoding="utf-8")
            with mock.patch.dict(os.environ, {"LLM_TIMEOUT_SEC": "240.5"}):
                self.assertEqual(manager.get_status("asmt_deadline").quiet_warning_after_sec, 241 + QUIET_WARNING_MARGIN_SEC)
            job_file.write_text(json.dumps({"assessment_id": "asmt_deadline", "status": "running",
                "phase": "transcribing", "request": {"cloud_broker": False}}), encoding="utf-8")
            self.assertIsNone(manager.get_status("asmt_deadline").quiet_warning_after_sec)
            job_file.write_text(json.dumps({"assessment_id": "asmt_deadline", "status": "running",
                "phase": "transcribing", "request": {"cloud_broker": True, "asr_provider": "groq"}}), encoding="utf-8")
            self.assertGreaterEqual(manager.get_status("asmt_deadline").quiet_warning_after_sec, 660 + QUIET_WARNING_MARGIN_SEC)

    def test_status_timestamps_change_only_for_meaningful_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            job_file = Path(directory) / "asmt_test.json"
            job_file.write_text(json.dumps({"status": "queued", "phase": "queued", "created_at": "2026-01-01T00:00:00+00:00"}))
            with mock.patch("app_backend.jobs._now_iso", side_effect=["2026-01-01T00:00:01+00:00", "2026-01-01T00:00:02+00:00"]):
                started = _update_job_file(job_file, status="running", phase="transcribing")
                repeated = _update_job_file(job_file, status="running", phase="transcribing")
                scoring = _update_job_file(job_file, status="running", phase="scoring_rubric")
            self.assertEqual(started["updated_at"], "2026-01-01T00:00:01+00:00")
            self.assertEqual(repeated["updated_at"], started["updated_at"])
            self.assertEqual(scoring["phase_started_at"], "2026-01-01T00:00:02+00:00")

    def test_job_status_stays_readable_while_a_new_status_is_staged(self):
        with tempfile.TemporaryDirectory() as directory:
            job_file = Path(directory) / "asmt_staged.json"
            old = {"assessment_id": "asmt_staged", "status": "running", "phase": "transcribing"}
            new = {"assessment_id": "asmt_staged", "status": "completed", "phase": "done", "payload": {"report": "x" * 100_000}}
            _write_json(job_file, old)
            replace = os.replace

            def inspect_before_replace(source, destination):
                self.assertEqual(destination, job_file)
                self.assertEqual(_read_json(job_file), old)
                self.assertEqual(json.loads(Path(source).read_text(encoding="utf-8")), new)
                replace(source, destination)

            with mock.patch("app_backend.jobs.os.replace", side_effect=inspect_before_replace):
                _write_json(job_file, new)
            self.assertEqual(_read_json(job_file), new)
            self.assertEqual(list(Path(directory).iterdir()), [job_file])

            with mock.patch("app_backend.jobs.os.replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    _write_json(job_file, old)
            self.assertEqual(_read_json(job_file), new)
            self.assertEqual(list(Path(directory).iterdir()), [job_file])

    def test_prunable_job_metadata_files_handles_mixed_case_terminal_statuses(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8766)
            now = datetime(2026, 4, 22, 10, 0, tzinfo=UTC)
            old_completed = config.jobs_dir / "old-completed.json"
            old_completed.write_text(
                json.dumps({"status": "Completed", "completed_at": (now - timedelta(days=31)).isoformat()}),
                encoding="utf-8",
            )

            candidates = prunable_job_metadata_files(config.jobs_dir, now=now)

        self.assertEqual(candidates, [old_completed])

    def test_cancel_does_not_overwrite_completed_job_after_process_join(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            assessment_id = "asmt_race"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps(
                    {
                        "assessment_id": assessment_id,
                        "status": "running",
                        "phase": "running",
                        "progress": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            process = mock.Mock()
            process.is_alive.return_value = True

            def complete_on_join(timeout=None):
                job_file.write_text(
                    json.dumps(
                        {
                            "assessment_id": assessment_id,
                            "status": "completed",
                            "phase": "done",
                            "progress": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )

            process.join.side_effect = complete_on_join
            manager._processes[assessment_id] = process

            status = manager.cancel(assessment_id)

        self.assertEqual(status.status.value, "completed")

    def test_cancel_keeps_stubborn_worker_tracked_and_blocks_submission(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            assessment_id = "asmt_stubborn"
            (config.jobs_dir / f"{assessment_id}.json").write_text(json.dumps({
                "assessment_id": assessment_id, "status": "running", "phase": "scoring_rubric", "progress": 0.5,
            }), encoding="utf-8")
            process = mock.Mock()
            process.is_alive.return_value = True
            manager._processes[assessment_id] = process
            with self.assertRaises(AssessmentBusyError):
                manager.cancel(assessment_id)
            process.terminate.assert_called_once()
            process.kill.assert_called_once()
            self.assertIs(manager._processes[assessment_id], process)
            running_status = manager.get_status(assessment_id)
            self.assertEqual(running_status.status, JobStatus.RUNNING)
            self.assertIsNotNone(running_status.worker_seen_at)
            with self.assertRaises(AssessmentBusyError):
                manager.submit(AssessmentCreateRequest(audio_id="unused", whisper="tiny", provider="ollama",
                    llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                    task_family="generic", theme="test", target_duration_sec=90))

    def test_cancel_accepts_process_exit_during_terminate_or_kill(self):
        for signal_name in ("terminate", "kill"):
            with self.subTest(signal_name=signal_name), tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
                config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
                manager = JobManager(config)
                assessment_id = f"asmt_exit_{signal_name}"
                _write_json(config.jobs_dir / f"{assessment_id}.json", {
                    "assessment_id": assessment_id, "status": "running", "phase": "scoring_rubric",
                })
                state = {"alive": True}
                process = mock.Mock()
                process.is_alive.side_effect = lambda: state["alive"]

                def exit_during_signal():
                    state["alive"] = False
                    raise ProcessLookupError("already exited")

                if signal_name == "terminate":
                    process.terminate.side_effect = exit_during_signal
                else:
                    process.kill.side_effect = exit_during_signal
                manager._processes[assessment_id] = process
                status = manager.cancel(assessment_id)
                self.assertEqual(status.status, JobStatus.CANCELLED)
                self.assertEqual(status.last_phase, "scoring_rubric")
                self.assertNotIn(assessment_id, manager._processes)

    def test_provider_request_must_finish_before_a_cancelled_job_can_be_retried(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            assessment_id = "asmt_cloud"
            (config.jobs_dir / f"{assessment_id}.json").write_text(json.dumps({
                "assessment_id": assessment_id, "status": "running", "phase": "scoring_rubric", "progress": 0.5,
            }), encoding="utf-8")
            state = {"alive": True}
            process = mock.Mock()
            process.is_alive.side_effect = lambda: state["alive"]
            process.terminate.side_effect = lambda: state.update(alive=False)
            manager._processes[assessment_id] = process
            active = threading.Event()
            active.set()
            manager._active_cloud_requests[assessment_id] = active
            self.assertEqual(manager.cancel(assessment_id).status, JobStatus.CANCELLED)
            with self.assertRaisesRegex(AssessmentBusyError, "provider request is still finishing"):
                manager.submit(AssessmentCreateRequest(audio_id="unused", whisper="tiny", provider="ollama",
                    llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                    task_family="generic", theme="test", target_duration_sec=90))
            active.clear()
            with mock.patch.object(manager, "_submit", side_effect=ValueError("admitted")):
                with self.assertRaisesRegex(ValueError, "admitted"):
                    manager.submit(AssessmentCreateRequest(audio_id="unused", whisper="tiny", provider="ollama",
                        llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                        task_family="generic", theme="test", target_duration_sec=90))

    def test_cancel_waits_for_submit_to_register_worker(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            upload = manager.register_upload(data=b"audio", filename="clip.wav")
            started = threading.Event()
            release = threading.Event()
            cancel_attempted = threading.Event()
            state = {"alive": True}
            process = mock.Mock()
            process.is_alive.side_effect = lambda: state["alive"]
            process.start.side_effect = lambda: (started.set(), release.wait(2))
            process.terminate.side_effect = lambda: state.update(alive=False)
            request = AssessmentCreateRequest(audio_id=upload.audio_id, whisper="tiny", provider="ollama",
                llm_model="test", expected_language="en", feedback_language="en", speaker_id="test",
                task_family="generic", theme="test", target_duration_sec=90)
            created = []
            cancelled = []
            with mock.patch.object(manager._ctx, "Process", return_value=process):
                submit_thread = threading.Thread(target=lambda: created.append(manager.submit(request)))
                submit_thread.start()
                self.assertTrue(started.wait(2))
                job_file = next(config.jobs_dir.glob("asmt_*.json"))
                cancel_thread = threading.Thread(target=lambda: (cancel_attempted.set(), cancelled.append(manager.cancel(job_file.stem))))
                cancel_thread.start()
                self.assertTrue(cancel_attempted.wait(2))
                self.assertEqual(json.loads(job_file.read_text())["status"], "queued")
                release.set()
                submit_thread.join(2)
                cancel_thread.join(2)
            self.assertFalse(submit_thread.is_alive())
            self.assertFalse(cancel_thread.is_alive())
            self.assertEqual(created[0].assessment_id, job_file.stem)
            self.assertEqual(cancelled[0].status, JobStatus.CANCELLED)
            self.assertEqual(cancelled[0].last_phase, "queued")
            self.assertEqual(json.loads(job_file.read_text())["status"], "cancelled")

    def test_recovered_job_reports_terminal_update_timestamps(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            assessment_id = "asmt_recovered"
            (config.jobs_dir / f"{assessment_id}.json").write_text(json.dumps({
                "assessment_id": assessment_id, "status": "running", "phase": "transcribing",
                "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:01+00:00",
            }), encoding="utf-8")
            manager = JobManager(config)
            status = manager.get_status(assessment_id)
            self.assertEqual(status.status, JobStatus.FAILED)
            self.assertEqual(status.last_phase, "transcribing")
            self.assertEqual(status.updated_at, status.phase_started_at)
            self.assertNotEqual(status.updated_at, status.created_at)
            from fastapi.testclient import TestClient
            from app_backend.app import create_app
            response = TestClient(create_app(config)).get(f"/v1/assessments/{assessment_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["created_at"], status.created_at)
            self.assertEqual(response.json()["updated_at"], status.updated_at)
            self.assertEqual(response.json()["phase_started_at"], status.phase_started_at)

    def test_failed_worker_start_allows_retry_with_same_request_id(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8769)
            manager = JobManager(config)
            upload = manager.register_upload(data=b"fake-audio", filename="sample.wav")
            request = AssessmentCreateRequest(
                audio_id=upload.audio_id, request_id="retry-start", whisper="tiny",
                provider="ollama", llm_model="demo-model", expected_language="it",
                feedback_language="en", speaker_id="speaker", task_family="free_monologue",
                theme="Travel", target_duration_sec=90,
            )
            failed = mock.Mock()
            failed.start.side_effect = OSError("spawn failed")
            retried = mock.Mock()
            with mock.patch.object(manager._ctx, "Process", side_effect=[failed, retried]):
                with self.assertRaisesRegex(OSError, "spawn failed"):
                    manager.submit(request)
                self.assertEqual(list(config.jobs_dir.glob("*.json")), [])
                created = manager.submit(request)
            retried.start.assert_called_once_with()
            self.assertIs(manager._processes[created.assessment_id], retried)
            self.assertTrue(manager._resolve_audio_path(upload.audio_id).exists())

    def test_submit_does_not_persist_llm_api_key_in_job_metadata(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8769)
            manager = JobManager(config)
            upload = manager.register_upload(data=b"fake-audio", filename="sample.wav")
            process = mock.Mock()

            with mock.patch.object(manager._ctx, "Process", return_value=process) as process_factory:
                created = manager.submit(
                    AssessmentCreateRequest(
                        audio_id=upload.audio_id,
                        whisper="tiny",
                        provider="openrouter",
                        llm_model="demo-model",
                        expected_language="it",
                        feedback_language="en",
                        speaker_id="speaker",
                        task_family="free_monologue",
                        theme="Travel",
                        target_duration_sec=90,
                        llm_api_key="sk-secret",
                    )
                )

            job_payload = json.loads((config.jobs_dir / f"{created.assessment_id}.json").read_text(encoding="utf-8"))
            self.assertNotIn("llm_api_key", job_payload["request"])
            worker_request = process_factory.call_args.kwargs["args"][1]
            self.assertEqual(worker_request["llm_api_key"], "sk-secret")
            process.start.assert_called_once_with()

    def test_worker_persists_completed_payload_summary_and_runtime_env(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8770)
            assessment_id = "asmt_worker_success"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps({"assessment_id": assessment_id, "status": "queued", "phase": "queued", "progress": 0.0}),
                encoding="utf-8",
            )
            audio_path = config.app_data.uploads_dir / "sample.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"fake-audio")
            report_path = config.app_data.reports_dir / "report.json"
            saved_payload = {
                "report": {
                    "scores": {"final": 4.2, "band": "B2"},
                    "metrics": {"duration_sec": 31},
                    "checks": {"min_words_pass": True, "language_pass": True, "content_validity_pass": True},
                    "coaching": {"next_focus": "Use clearer connectors."},
                }
            }
            request_payload = {
                "whisper": "small",
                "provider": "openrouter",
                "llm_model": "demo-model",
                "target_cefr": "B2",
                "theme": "travel",
                "task_family": "free_monologue",
                "speaker_id": "speaker",
                "target_duration_sec": 90,
                "expected_language": "it",
                "language_profile_key": "it",
                "feedback_language": "en",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "dry_run": True,
                "log_dir": str(config.app_data.reports_dir),
                "label": "trial",
                "notes": "notes",
                "llm_api_key": "sk-worker",
                "openrouter_http_referer": "https://example.test/app",
                "openrouter_app_title": "Vostavo Desktop",
            }

            def fake_execute(run_request, status_callback):
                self.assertEqual(run_request.audio, audio_path)
                self.assertEqual(run_request.whisper_model, "small")
                self.assertEqual(run_request.provider, "openrouter")
                self.assertEqual(run_request.llm_timeout_sec, 180.0)
                self.assertTrue(run_request.dry_run)
                status_callback("transcribing")
                return AssessmentRunResult(
                    meta={},
                    output={"report": {}},
                    stdout_json="{}",
                    report=saved_payload["report"],
                    report_path=report_path,
                    saved_payload=saved_payload,
                )

            with mock.patch("app_backend.jobs.validate_audio_duration"), mock.patch.dict("app_backend.jobs.os.environ", {}, clear=True), mock.patch(
                "app_backend.jobs.execute_assessment_run",
                side_effect=fake_execute,
            ):
                _job_worker(str(job_file), request_payload, str(audio_path))

                self.assertEqual(os.environ["LLM_API_KEY"], "sk-worker")
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "sk-worker")
                self.assertEqual(os.environ["OPENROUTER_HTTP_REFERER"], "https://example.test/app")
                self.assertEqual(os.environ["OPENROUTER_APP_TITLE"], "Vostavo Desktop")

            payload = json.loads(job_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], JobStatus.COMPLETED.value)
            self.assertEqual(payload["phase"], "done")
            self.assertEqual(payload["progress"], 1.0)
            self.assertEqual(payload["payload"], saved_payload)
            self.assertEqual(payload["summary"]["score_overall"], 4.2)
            self.assertEqual(payload["summary"]["band"], "B2")
            self.assertEqual(payload["summary"]["next_focus"], "Use clearer connectors.")
            self.assertEqual(payload["report_path"], str(report_path.resolve()))

    def test_worker_environment_llm_timeout_overrides_practice_default(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8770)
            job_file = config.jobs_dir / "asmt_timeout_override.json"
            job_file.write_text(json.dumps({"status": "queued"}), encoding="utf-8")
            request_payload = {
                "whisper": "small",
                "provider": "ollama",
                "theme": "travel",
                "task_family": "free_monologue",
                "speaker_id": "speaker",
                "target_duration_sec": 90,
                "log_dir": str(config.app_data.reports_dir),
            }
            with mock.patch.dict("app_backend.jobs.os.environ", {"LLM_TIMEOUT_SEC": "42.5"}, clear=True), mock.patch(
                "app_backend.jobs.validate_audio_duration"
            ), mock.patch("app_backend.jobs.execute_assessment_run", side_effect=RuntimeError("stop after capture")) as execute:
                _job_worker(str(job_file), request_payload, str(config.app_data.uploads_dir / "sample.wav"))

            self.assertEqual(execute.call_args.args[0].llm_timeout_sec, 42.5)

    def test_worker_persists_local_provider_failure_code(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8771)
            assessment_id = "asmt_worker_failure"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps({"assessment_id": assessment_id, "status": "queued", "phase": "queued", "progress": 0.0}),
                encoding="utf-8",
            )
            audio_path = config.app_data.uploads_dir / "sample.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"fake-audio")
            request_payload = {
                "whisper": "small",
                "provider": "ollama",
                "llm_model": "demo-model",
                "theme": "travel",
                "task_family": "free_monologue",
                "speaker_id": "speaker",
                "target_duration_sec": 90,
                "log_dir": str(config.app_data.reports_dir),
            }

            with mock.patch("app_backend.jobs.validate_audio_duration"), mock.patch(
                "app_backend.jobs.execute_assessment_run",
                side_effect=RuntimeError("connection refused by local provider"),
            ):
                _job_worker(str(job_file), request_payload, str(audio_path))

            payload = json.loads(job_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], JobStatus.FAILED.value)
            self.assertEqual(payload["phase"], "failed")
            self.assertEqual(payload["progress"], 1.0)
            self.assertEqual(payload["error"]["code"], ErrorCode.LOCAL_PROVIDER_NOT_RUNNING.value)
            self.assertIn("connection refused", payload["error"]["detail"])

    def test_worker_reports_short_recording_without_creating_a_review(self):
        from assessment_runtime.recording_policy import RecordingTooShortError

        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8771)
            job_file = config.jobs_dir / "asmt_short.json"
            job_file.write_text(json.dumps({"assessment_id": "asmt_short", "status": "queued"}), encoding="utf-8")
            request_payload = {"theme": "Synthetic topic", "speaker_id": "synthetic", "task_family": "free_monologue",
                               "target_duration_sec": 60, "whisper": "small", "provider": "ollama",
                               "log_dir": str(config.app_data.reports_dir)}
            with mock.patch("app_backend.jobs.validate_audio_duration"), mock.patch(
                "app_backend.jobs.execute_assessment_run", side_effect=RecordingTooShortError(),
            ):
                _job_worker(str(job_file), request_payload, str(config.app_data.uploads_dir / "short.wav"))
            payload = json.loads(job_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], JobStatus.FAILED.value)
            self.assertEqual(payload["error"]["code"], ErrorCode.RECORDING_TOO_SHORT.value)
            self.assertIn("at least 30 seconds", payload["error"]["detail"])
            self.assertIsNone(payload.get("payload"))
            self.assertIsNone(payload.get("report_path"))

    def test_get_status_marks_unfinished_job_failed_when_worker_exited(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8772)
            manager = JobManager(config)
            assessment_id = "asmt_dead_worker"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps({"assessment_id": assessment_id, "status": "running", "phase": "transcribing", "progress": 0.0}),
                encoding="utf-8",
            )
            process = mock.Mock()
            process.is_alive.return_value = False
            process.exitcode = 9
            manager._processes[assessment_id] = process

            status = manager.get_status(assessment_id)

        self.assertEqual(status.status, JobStatus.FAILED)
        self.assertEqual(status.phase, "failed")
        self.assertEqual(status.progress, 1.0)
        self.assertEqual(status.error.code, ErrorCode.RUNTIME)
        self.assertIn("exited unexpectedly with code 9", status.error.detail)
        process.join.assert_called_once_with(timeout=0.1)
        self.assertNotIn(assessment_id, manager._processes)

    def test_cancel_marks_queued_job_cancelled_without_worker_process(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8773)
            manager = JobManager(config)
            assessment_id = "asmt_cancel_queued"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps({"assessment_id": assessment_id, "status": "queued", "phase": "queued", "progress": 0.0}),
                encoding="utf-8",
            )

            status = manager.cancel(assessment_id)

        self.assertEqual(status.status, JobStatus.CANCELLED)
        self.assertEqual(status.phase, "cancelled")
        self.assertEqual(status.progress, 1.0)
        self.assertEqual(status.error.code, ErrorCode.CANCELLATION)

    def test_shutdown_logs_worker_termination_failures(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8768)
            manager = JobManager(config)
            process = mock.Mock()
            process.pid = 12345
            process.is_alive.return_value = True
            manager._processes["asmt_stuck"] = process

            with mock.patch("app_backend.jobs.os.kill", side_effect=OSError("denied")), self.assertLogs(
                "app_backend.jobs",
                level="WARNING",
            ) as logs:
                manager.shutdown()

        self.assertIn("Could not terminate assessment worker", "\n".join(logs.output))
        process.join.assert_called_once_with(timeout=0.5)
        self.assertNotIn("asmt_stuck", manager._processes)


if __name__ == "__main__":
    unittest.main()


def test_completed_worker_teardown_is_drained_before_a_following_submission(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8765)
    manager = JobManager(config)
    process = mock.Mock()
    process.is_alive.return_value = True
    manager._processes['asmt_completed'] = process
    (config.jobs_dir / 'asmt_completed.json').write_text(json.dumps({'assessment_id': 'asmt_completed', 'status': 'completed'}))
    process.join.side_effect = lambda timeout: setattr(process.is_alive, 'return_value', False)
    request = AssessmentCreateRequest(audio_id='aud_fixture', whisper='tiny', provider='ollama', llm_model='fixture', expected_language='en', feedback_language='en', speaker_id='fixture', task_family='free_monologue', theme='travel', target_duration_sec=90)
    with mock.patch.object(manager, '_submit', return_value='accepted'):
        assert manager.submit(request) == 'accepted'
    process.join.assert_any_call(timeout=1.0)
    assert 'asmt_completed' not in manager._processes


def test_active_worker_is_not_joined_or_cancelled_to_admit_a_second_submission(tmp_path):
    import pytest
    from app_backend.jobs import AssessmentBusyError
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8765)
    manager = JobManager(config)
    process = mock.Mock()
    process.is_alive.return_value = True
    manager._processes['asmt_active'] = process
    (config.jobs_dir / 'asmt_active.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(AssessmentBusyError):
        manager.submit(AssessmentCreateRequest(audio_id='aud_fixture', whisper='tiny', provider='ollama', llm_model='fixture', expected_language='en', feedback_language='en', speaker_id='fixture', task_family='free_monologue', theme='travel', target_duration_sec=90))
    process.join.assert_not_called()
    process.terminate.assert_not_called()

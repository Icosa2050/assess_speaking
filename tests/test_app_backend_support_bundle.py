from datetime import UTC, datetime
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from app_backend.config import build_backend_runtime_config
from app_backend.contracts import SupportBundleCreateRequest
from app_backend.support_bundle import (
    REDACTED_VALUE,
    build_storage_summary,
    create_support_bundle,
    support_bundle_path,
)


class SupportBundleTests(unittest.TestCase):
    def test_storage_summary_counts_known_app_data_areas(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8771)
            report_file = config.app_data.reports_dir / "session.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text("{}", encoding="utf-8")
            cache_file = config.app_data.cache_root / "models" / "marker.txt"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text("cached", encoding="utf-8")

            summary = build_storage_summary(config)

            self.assertEqual(summary.app_data_root, str(config.app_data.root))
            self.assertEqual(summary.cache_root, str(config.app_data.cache_root))
            self.assertEqual(summary.areas["reports"].file_count, 1)
            self.assertEqual(summary.areas["reports"].size_bytes, 2)
            self.assertEqual(summary.areas["cache"].file_count, 1)
            self.assertEqual(support_bundle_path(config, "bundle_test").parent, config.app_data.temp_dir / "support-bundles")

    def test_create_support_bundle_includes_optional_trees_and_redacts_sensitive_content(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8772)
            config.state_file.write_text("{not json", encoding="utf-8")
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            config.log_file.write_text(
                'backend ready\napi_key="log-secret"\nsecret_ref="connection:primary"\n',
                encoding="utf-8",
            )

            reports_dir = config.app_data.reports_dir
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": "sess_1",
                        "api_token": "report-secret",
                        "connection": {"secret_ref": "connection:primary"},
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "notes.txt").write_text(
                'password="text-secret"\nsecret_ref="connection:primary"\n',
                encoding="utf-8",
            )

            recordings_dir = config.app_data.recordings_dir
            recordings_dir.mkdir(parents=True, exist_ok=True)
            (recordings_dir / "sample.wav").write_bytes(b"fake-recording")

            uploads_dir = config.app_data.uploads_dir
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / "metadata.csv").write_text('authorization="Bearer upload-secret"\n', encoding="utf-8")

            config.jobs_dir.mkdir(parents=True, exist_ok=True)
            old_job = config.jobs_dir / "old.json"
            old_job.write_text('{"status":"completed","token":"old-secret"}', encoding="utf-8")
            recent_job = config.jobs_dir / "recent.json"
            recent_job.write_text("{invalid", encoding="utf-8")
            old_time = datetime(2026, 4, 20, tzinfo=UTC).timestamp()
            recent_time = datetime(2026, 4, 21, tzinfo=UTC).timestamp()
            old_job.touch()
            recent_job.touch()
            old_job.stat()
            recent_job.stat()

            os.utime(old_job, (old_time, old_time))
            os.utime(recent_job, (recent_time, recent_time))

            response = create_support_bundle(
                config,
                SupportBundleCreateRequest(
                    include_reports=True,
                    include_recordings=True,
                    include_uploads=True,
                    client_snapshot={
                        "has_active_connection": True,
                        "llm_api_key": "client-secret",
                        "secret_ref": "connection:primary",
                    },
                    client_diagnostics=[
                        {
                            "key": "runtime",
                            "detail_args": {
                                "nested": [{"password": "diag-secret"}],
                                "secret_ref": "connection:primary",
                            },
                        }
                    ],
                ),
            )

            bundle_file = support_bundle_path(config, response.bundle_id)
            self.assertEqual(response.filename, bundle_file.name)
            self.assertGreater(response.size_bytes, 0)

            with zipfile.ZipFile(bundle_file) as archive:
                names = set(archive.namelist())
                self.assertIn("backend_state.json", names)
                self.assertIn("logs/backend.log", names)
                self.assertIn("reports/session.json", names)
                self.assertIn("reports/notes.txt", names)
                self.assertIn("recordings/sample.wav", names)
                self.assertIn("uploads/metadata.csv", names)
                self.assertIn("jobs/old.json", names)
                self.assertNotIn("jobs/recent.json", names)

                backend_state = json.loads(archive.read("backend_state.json"))
                self.assertEqual(backend_state["base_url"], config.base_url)

                client_snapshot = json.loads(archive.read("shell/client_snapshot.json"))
                self.assertEqual(client_snapshot["llm_api_key"], REDACTED_VALUE)
                self.assertNotIn("secret_ref", client_snapshot)

                client_diagnostics = json.loads(archive.read("shell/client_diagnostics.json"))
                self.assertEqual(
                    client_diagnostics[0]["detail_args"]["nested"][0]["password"],
                    REDACTED_VALUE,
                )
                self.assertNotIn("secret_ref", client_diagnostics[0]["detail_args"])

                report_payload = json.loads(archive.read("reports/session.json"))
                self.assertEqual(report_payload["api_token"], REDACTED_VALUE)
                self.assertNotIn("secret_ref", report_payload["connection"])

                notes = archive.read("reports/notes.txt").decode("utf-8")
                self.assertNotIn("text-secret", notes)
                self.assertNotIn("connection:primary", notes)

                upload_metadata = archive.read("uploads/metadata.csv").decode("utf-8")
                self.assertIn(REDACTED_VALUE, upload_metadata)
                self.assertNotIn("upload-secret", upload_metadata)
                self.assertEqual(archive.read("recordings/sample.wav"), b"fake-recording")

                manifest = json.loads(archive.read("manifest.json"))
                self.assertTrue(manifest["include_reports"])
                self.assertTrue(manifest["include_recordings"])
                self.assertTrue(manifest["include_uploads"])
                self.assertEqual(manifest["redaction"]["secret_ref_policy"], "full_redaction")
                self.assertGreaterEqual(manifest["redaction"]["removed_secret_refs"], 4)
                self.assertGreaterEqual(manifest["redaction"]["redacted_secret_values"], 5)


if __name__ == "__main__":
    unittest.main()

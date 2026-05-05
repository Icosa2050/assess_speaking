import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import zipfile

from fastapi.testclient import TestClient

from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_backend.contracts import (
    AssessmentCreateResponse,
    AssessmentStatusResponse,
    CANONICAL_PRODUCT_API_ROUTES,
    JobStatus,
    LOCAL_RUNTIME_MANAGEMENT_ROUTES,
    LOCAL_SUPPORT_API_ROUTES,
)
import assessment_runtime.theme_library as theme_library_store
from app_shell.bootstrap import bootstrap_app_environment
from app_shell import secret_store
from app_shell.runtime_connections import serialize_connections
from app_shell.services import build_provider_connection, save_provider_connection, set_default_provider_connection
from app_shell.state import AppPreferences, AppShellState


class FakeKeyringModule:
    def __init__(self) -> None:
        self.secrets: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.secrets.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.secrets[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.secrets.pop((service, account), None)


def persist_runtime_settings_seed(config, state: AppShellState) -> None:
    theme_library_store.save_workspace_prefs(
        config.app_data.reports_dir,
        {
            "ui_locale": state.prefs.ui_locale,
            "provider": state.prefs.provider,
            "model": state.prefs.model,
            "llm_base_url": state.prefs.llm_base_url,
            "whisper_model": state.prefs.whisper_model,
            "whisper_cache_dir": state.prefs.whisper_cache_dir,
            "openrouter_http_referer": state.prefs.openrouter_http_referer,
            "openrouter_app_title": state.prefs.openrouter_app_title,
            "active_connection_id": state.prefs.active_connection_id,
            "connections": serialize_connections(list(state.prefs.connections or [])),
            "setup_complete": bool(state.prefs.setup_complete or state.prefs.connections),
            "log_dir": str(config.app_data.reports_dir),
        },
    )


class BackendApiTests(unittest.TestCase):
    def test_runtime_config_honors_app_data_and_cache_overrides(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                port=8764,
            )

        self.assertEqual(config.app_data.root, Path(app_dir).resolve())
        self.assertEqual(config.app_data.cache_root, Path(cache_dir).resolve())
        self.assertEqual(config.state_file, Path(app_dir).resolve() / "backend_state.json")
        self.assertEqual(config.jobs_dir, Path(app_dir).resolve() / "jobs")
        self.assertEqual(config.log_file, Path(app_dir).resolve() / "logs" / "backend.log")

    def test_health_and_samples_endpoints_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8765)
            client = TestClient(create_app(config))
            health = client.get("/v1/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ready")

            samples = client.get("/v1/samples")
            self.assertEqual(samples.status_code, 200)
            self.assertTrue(samples.json()["items"])

            storage = client.get("/v1/maintenance/storage")
            self.assertEqual(storage.status_code, 200)
            self.assertIn("areas", storage.json())
            self.assertIn("tmp", storage.json()["areas"])

    def test_history_endpoint_serializes_real_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            report_path = reports_dir / "report.json"
            report_path.write_text(json.dumps({"report": {"session_id": "sess-1"}}), encoding="utf-8")
            (reports_dir / "history.csv").write_text(
                "timestamp,session_id,schema_version,speaker_id,learning_language,task_family,theme,"
                "audio,whisper,llm,label,target_duration_sec,duration_sec,wpm,word_count,duration_pass,"
                "topic_pass,language_pass,fluency,cohesion,accuracy,range,overall,final_score,band,"
                "requires_human_review,top_priority_1,top_priority_2,top_priority_3,"
                "grammar_error_categories,coherence_issue_categories,report_path\n"
                "2026-05-05T22:22:30,sess-1,2,manual-smoke,it,travel_narrative,Travel Story,"
                "sample.wav,large-v3,llama3.2:3b,,90,16.05,190.6,51,false,,true,,,,,,3.26,3,"
                "true,Keep speaking,Use connectors,Add detail,,,"
                f"{report_path}\n",
                encoding="utf-8",
            )
            client = TestClient(create_app(build_backend_runtime_config(log_dir=reports_dir, port=8775)))

            history = client.get("/v1/history")
            self.assertEqual(history.status_code, 200)
            row = history.json()["items"][0]
            self.assertEqual(row["timestamp"], "2026-05-05T22:22:30")
            self.assertEqual(row["session_id"], "sess-1")
            self.assertEqual(row["learning_language"], "it")
            self.assertEqual(row["top_priorities"], ["Keep speaking", "Use connectors", "Add detail"])

            detail = client.get("/v1/history/sess-1")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json()["payload"]["report"]["session_id"], "sess-1")

    def test_contract_endpoint_freezes_phase2_local_api_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8770)
            client = TestClient(create_app(config))

            contract = client.get("/v1/contract")
            self.assertEqual(contract.status_code, 200)
            payload = contract.json()
            self.assertEqual(payload["contract_version"], "phase2-local-v1")
            self.assertEqual(payload["session_mode"], "local_guest")
            self.assertEqual(payload["auth_mode"], "none")
            self.assertEqual(payload["tenancy_mode"], "none")
            self.assertEqual(payload["storage_mode"], "local_app_data")
            self.assertEqual(payload["product_routes"], list(CANONICAL_PRODUCT_API_ROUTES))
            self.assertEqual(payload["local_support_routes"], list(LOCAL_SUPPORT_API_ROUTES))
            self.assertEqual(payload["local_runtime_routes"], list(LOCAL_RUNTIME_MANAGEMENT_ROUTES))
            self.assertNotIn("/v1/support-bundles", payload["product_routes"])
            self.assertNotIn("/v1/assessments", payload["local_support_routes"])

            openapi = client.get("/openapi.json")
            self.assertEqual(openapi.status_code, 200)
            paths = openapi.json()["paths"]
            self.assertEqual(paths["/v1/health"]["get"]["tags"], ["product-api"])
            self.assertEqual(paths["/v1/history"]["get"]["tags"], ["product-api"])
            self.assertEqual(paths["/v1/runtime/settings"]["get"]["tags"], ["local-runtime"])
            self.assertEqual(paths["/v1/maintenance/storage"]["get"]["tags"], ["local-support"])
            self.assertEqual(paths["/v1/support-bundles"]["post"]["tags"], ["local-support"])

    def test_local_guest_dev_origins_receive_cors_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8771)
            client = TestClient(create_app(config))

            response = client.options(
                "/v1/maintenance/storage",
                headers={
                    "Origin": "http://127.0.0.1:4173",
                    "Access-Control-Request-Method": "GET",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:4173")

            tauri_response = client.options(
                "/v1/health",
                headers={
                    "Origin": "https://tauri.localhost",
                    "Access-Control-Request-Method": "GET",
                },
            )
            self.assertEqual(tauri_response.status_code, 200)
            self.assertEqual(tauri_response.headers.get("access-control-allow-origin"), "https://tauri.localhost")

    def test_upload_endpoint_persists_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8766)
            client = TestClient(create_app(config))
            response = client.post(
                "/v1/uploads",
                files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(Path(payload["stored_path"]).exists())
            self.assertTrue(payload["audio_id"].startswith("aud_"))

    def test_runtime_settings_endpoints_manage_saved_connections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_data_dir = Path(tmpdir) / "app-data"
            cache_dir = Path(tmpdir) / "cache"
            reports_dir = app_data_dir / "reports"
            config = build_backend_runtime_config(
                log_dir=reports_dir,
                app_data_dir=app_data_dir,
                cache_dir=cache_dir,
                port=8772,
            )
            keyring = FakeKeyringModule()
            keyring_status = secret_store.SecretStoreStatus(
                persistent=True,
                backend_name="mock-keyring",
            )

            with mock.patch("app_shell.secret_store._load_keyring_module", return_value=(keyring, keyring_status)):
                bootstrap_app_environment(
                    log_dir=config.app_data.reports_dir,
                    app_data_dir=config.app_data.root,
                    cache_dir=config.app_data.cache_root,
                    whisper_cache_dir=config.app_data.whisper_cache_dir,
                )
                state = AppShellState(prefs=AppPreferences())
                state.prefs.log_dir = str(config.app_data.reports_dir)
                state.prefs.whisper_cache_dir = str(config.app_data.whisper_cache_dir)
                state.prefs.ui_locale = "it"
                state.prefs.whisper_model = "medium"

                primary = build_provider_connection(
                    provider_choice="openrouter",
                    label="Primary cloud",
                    model="google/gemini-3.1-pro-preview",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="saved-key",
                    openrouter_http_referer="https://example.test/app",
                    openrouter_app_title="Vostavo Desktop",
                )
                save_provider_connection(state, primary, api_key="saved-key", persist_draft=False)

                secondary = build_provider_connection(
                    provider_choice="lmstudio_local",
                    label="Desk LM Studio",
                    model="qwen-local",
                    base_url="http://localhost:1234/v1",
                    api_key="",
                )
                save_provider_connection(state, secondary, api_key="", persist_draft=False)
                self.assertTrue(set_default_provider_connection(state, primary.connection_id, persist_draft=False))
                persist_runtime_settings_seed(config, state)

                with mock.patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": "0"}), TestClient(create_app(config)) as client:
                    fetched = client.get("/v1/runtime/settings")
                    self.assertEqual(fetched.status_code, 200)
                    payload = fetched.json()
                    self.assertEqual(payload["ui_locale"], "it")
                    self.assertEqual(payload["whisper_model"], "medium")
                    self.assertEqual(payload["active_connection_id"], primary.connection_id)
                    self.assertEqual(len(payload["connections"]), 2)
                    primary_payload = next(item for item in payload["connections"] if item["connection_id"] == primary.connection_id)
                    secondary_payload = next(item for item in payload["connections"] if item["connection_id"] == secondary.connection_id)
                    self.assertEqual(primary_payload["secret_state"], "present")
                    self.assertEqual(primary_payload["openrouter_http_referer"], "https://example.test/app")
                    self.assertEqual(primary_payload["openrouter_app_title"], "Vostavo Desktop")
                    self.assertEqual(secondary_payload["secret_state"], "absent")

                    updated = client.put(
                        "/v1/runtime/settings",
                        json={
                            "ui_locale": "de",
                            "whisper_model": "large-v3",
                            "clear_saved_secret": False,
                            "connection": {
                                "connection_id": secondary.connection_id,
                                "provider_choice": "lmstudio_local",
                                "label": "Desk LM Studio Updated",
                                "model": "qwen-local-2",
                                "base_url": "http://localhost:1234/v1",
                                "api_key": "",
                                "openrouter_http_referer": "",
                                "openrouter_app_title": "",
                            },
                        },
                    )
                    self.assertEqual(updated.status_code, 200)
                    updated_payload = updated.json()
                    self.assertEqual(updated_payload["ui_locale"], "de")
                    self.assertEqual(updated_payload["whisper_model"], "large-v3")
                    self.assertEqual(updated_payload["active_connection_id"], secondary.connection_id)
                    updated_secondary = next(
                        item for item in updated_payload["connections"] if item["connection_id"] == secondary.connection_id
                    )
                    self.assertTrue(updated_secondary["is_default"])
                    self.assertEqual(updated_secondary["label"], "Desk LM Studio Updated")
                    self.assertEqual(updated_secondary["model"], "qwen-local-2")

                    defaulted = client.post(f"/v1/runtime/settings/connections/{primary.connection_id}/default")
                    self.assertEqual(defaulted.status_code, 200)
                    defaulted_payload = defaulted.json()
                    self.assertEqual(defaulted_payload["active_connection_id"], primary.connection_id)
                    defaulted_primary = next(
                        item for item in defaulted_payload["connections"] if item["connection_id"] == primary.connection_id
                    )
                    self.assertTrue(defaulted_primary["is_default"])

                    cleared = client.put(
                        "/v1/runtime/settings",
                        json={
                            "ui_locale": "de",
                            "whisper_model": "large-v3",
                            "clear_saved_secret": True,
                            "connection": {
                                "connection_id": primary.connection_id,
                                "provider_choice": "openrouter",
                                "label": "Primary cloud",
                                "model": "google/gemini-3.1-pro-preview",
                                "base_url": "https://openrouter.ai/api/v1",
                                "api_key": "",
                                "openrouter_http_referer": "https://example.test/app",
                                "openrouter_app_title": "Vostavo Desktop",
                            },
                        },
                    )
                    self.assertEqual(cleared.status_code, 200)
                    cleared_primary = next(
                        item for item in cleared.json()["connections"] if item["connection_id"] == primary.connection_id
                    )
                    self.assertEqual(cleared_primary["secret_state"], "missing")
                    self.assertFalse(cleared_primary["has_api_key"])

                    deleted = client.delete(f"/v1/runtime/settings/connections/{secondary.connection_id}")
                    self.assertEqual(deleted.status_code, 200)
                    self.assertEqual(len(deleted.json()["connections"]), 1)

    def test_runtime_settings_drops_saved_secret_when_provider_identity_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_data_dir = Path(tmpdir) / "app-data"
            cache_dir = Path(tmpdir) / "cache"
            reports_dir = app_data_dir / "reports"
            config = build_backend_runtime_config(
                log_dir=reports_dir,
                app_data_dir=app_data_dir,
                cache_dir=cache_dir,
                port=8774,
            )
            keyring = FakeKeyringModule()
            keyring_status = secret_store.SecretStoreStatus(
                persistent=True,
                backend_name="mock-keyring",
            )

            with mock.patch("app_shell.secret_store._load_keyring_module", return_value=(keyring, keyring_status)):
                bootstrap_app_environment(
                    log_dir=config.app_data.reports_dir,
                    app_data_dir=config.app_data.root,
                    cache_dir=config.app_data.cache_root,
                    whisper_cache_dir=config.app_data.whisper_cache_dir,
                )
                state = AppShellState(prefs=AppPreferences())
                state.prefs.log_dir = str(config.app_data.reports_dir)
                state.prefs.whisper_cache_dir = str(config.app_data.whisper_cache_dir)

                provider_changed = build_provider_connection(
                    provider_choice="openrouter",
                    label="Cloud source",
                    model="google/gemini-3.1-pro-preview",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="provider-key",
                    openrouter_http_referer="https://example.test/app",
                    openrouter_app_title="Vostavo Desktop",
                )
                save_provider_connection(state, provider_changed, api_key="provider-key", persist_draft=False)

                base_url_changed = build_provider_connection(
                    provider_choice="openrouter",
                    label="Gateway source",
                    model="google/gemini-3.1-pro-preview",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="gateway-key",
                    openrouter_http_referer="https://example.test/app",
                    openrouter_app_title="Vostavo Desktop",
                )
                save_provider_connection(state, base_url_changed, api_key="gateway-key", persist_draft=False)
                persist_runtime_settings_seed(config, state)

                with mock.patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": "0"}), TestClient(create_app(config)) as client:
                    changed_provider = client.put(
                        "/v1/runtime/settings",
                        json={
                            "ui_locale": "en",
                            "whisper_model": "large-v3",
                            "clear_saved_secret": False,
                            "connection": {
                                "connection_id": provider_changed.connection_id,
                                "provider_choice": "lmstudio_local",
                                "label": "Local source",
                                "model": "qwen-local",
                                "base_url": "http://localhost:1234/v1",
                                "api_key": "",
                                "openrouter_http_referer": "",
                                "openrouter_app_title": "",
                            },
                        },
                    )
                    self.assertEqual(changed_provider.status_code, 200)
                    changed_provider_payload = next(
                        item
                        for item in changed_provider.json()["connections"]
                        if item["connection_id"] == provider_changed.connection_id
                    )
                    self.assertEqual(changed_provider_payload["provider_choice"], "lmstudio_local")
                    self.assertEqual(changed_provider_payload["secret_state"], "absent")
                    self.assertFalse(changed_provider_payload["has_api_key"])
                    self.assertNotIn((secret_store.SERVICE_NAME, provider_changed.secret_ref), keyring.secrets)

                    changed_url = client.put(
                        "/v1/runtime/settings",
                        json={
                            "ui_locale": "en",
                            "whisper_model": "large-v3",
                            "clear_saved_secret": False,
                            "connection": {
                                "connection_id": base_url_changed.connection_id,
                                "provider_choice": "openrouter",
                                "label": "Gateway source",
                                "model": "google/gemini-3.1-pro-preview",
                                "base_url": "https://gateway.example.test/v1",
                                "api_key": "",
                                "openrouter_http_referer": "https://example.test/app",
                                "openrouter_app_title": "Vostavo Desktop",
                            },
                        },
                    )
                    self.assertEqual(changed_url.status_code, 200)
                    changed_url_payload = next(
                        item
                        for item in changed_url.json()["connections"]
                        if item["connection_id"] == base_url_changed.connection_id
                    )
                    self.assertEqual(changed_url_payload["base_url"], "https://gateway.example.test/v1")
                    self.assertEqual(changed_url_payload["secret_state"], "missing")
                    self.assertFalse(changed_url_payload["has_api_key"])
                    self.assertNotIn((secret_store.SERVICE_NAME, base_url_changed.secret_ref), keyring.secrets)

    def test_runtime_test_connection_and_whisper_endpoints_return_local_runtime_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8773)
            client = TestClient(create_app(config))
            with mock.patch(
                "app_backend.app.test_runtime_connection",
                return_value={
                    "provider": "ollama",
                    "base_url": "http://localhost:11434/v1",
                    "service_base_url": "http://localhost:11434",
                    "health_endpoint": "http://localhost:11434/api/tags",
                    "models": ["llama3", "qwen2.5"],
                    "test_payload": {
                        "tested_at": "2026-04-30T12:00:00+00:00",
                        "content_preview": "ok",
                    },
                },
            ) as mock_test, mock.patch(
                "app_backend.app.whisper_model_status",
                return_value={
                    "model": "medium",
                    "repo_id": "systran/faster-whisper-medium",
                    "cached": False,
                    "cached_path": "",
                    "recommended": True,
                    "recommendation_reason": "Good practice tier",
                },
            ) as mock_status, mock.patch(
                "app_backend.app.download_whisper_model",
                return_value={
                    "model": "medium",
                    "repo_id": "systran/faster-whisper-medium",
                    "cached": True,
                    "cached_path": "/tmp/medium",
                    "recommended": True,
                    "recommendation_reason": "Good practice tier",
                },
            ) as mock_download:
                tested = client.post(
                    "/v1/runtime/settings/test-connection",
                    json={
                        "connection": {
                            "provider_choice": "ollama_local",
                            "label": "Ollama",
                            "model": "",
                            "base_url": "http://localhost:11434",
                            "api_key": "",
                            "openrouter_http_referer": "",
                            "openrouter_app_title": "",
                        }
                    },
                )
                self.assertEqual(tested.status_code, 200)
                tested_payload = tested.json()
                self.assertEqual(tested_payload["provider"], "ollama")
                self.assertEqual(tested_payload["discovered_models"], ["llama3", "qwen2.5"])
                self.assertEqual(tested_payload["tested_at"], "2026-04-30T12:00:00+00:00")
                mock_test.assert_called_once()

                status = client.get("/v1/runtime/whisper-models/medium")
                self.assertEqual(status.status_code, 200)
                self.assertFalse(status.json()["cached"])
                mock_status.assert_called_once_with("medium")

                downloaded = client.post("/v1/runtime/whisper-models/medium/download")
                self.assertEqual(downloaded.status_code, 200)
                self.assertTrue(downloaded.json()["cached"])
                self.assertEqual(downloaded.json()["cached_path"], "/tmp/medium")
                mock_download.assert_called_once_with("medium")

    def test_assessment_routes_delegate_to_job_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8767)
            app = create_app(config)
            client = TestClient(app)
            fake_submit = AssessmentCreateResponse(assessment_id="asmt_1", status=JobStatus.QUEUED)
            fake_status = AssessmentStatusResponse(
                assessment_id="asmt_1",
                status=JobStatus.COMPLETED,
                phase="done",
                progress=1.0,
                summary=None,
                error=None,
                report_path=None,
                payload={"report": {"session_id": "sess-1"}},
            )
            with mock.patch.object(app.state.job_manager, "submit", return_value=fake_submit) as mock_submit, \
                    mock.patch.object(app.state.job_manager, "get_status", return_value=fake_status) as mock_status, \
                    mock.patch.object(app.state.job_manager, "cancel", return_value=fake_status) as mock_cancel:
                created = client.post(
                    "/v1/assessments",
                    json={
                        "audio_id": "aud_1",
                        "whisper": "small",
                        "provider": "openrouter",
                        "llm_model": "google/gemini-3.1-pro-preview",
                        "expected_language": "en",
                        "feedback_language": "en",
                        "speaker_id": "bern",
                        "task_family": "free_monologue",
                        "theme": "travel",
                        "target_duration_sec": 90,
                        "dry_run": True,
                    },
                )
                self.assertEqual(created.status_code, 200)
                self.assertEqual(created.json()["assessment_id"], "asmt_1")
                mock_submit.assert_called_once()
                self.assertTrue(mock_submit.call_args.args[0].dry_run)

                status = client.get("/v1/assessments/asmt_1")
                self.assertEqual(status.status_code, 200)
                self.assertEqual(status.json()["status"], "completed")
                mock_status.assert_called_once_with("asmt_1")

                cancelled = client.post("/v1/assessments/asmt_1/cancel")
                self.assertEqual(cancelled.status_code, 200)
                self.assertEqual(cancelled.json()["phase"], "done")
                mock_cancel.assert_called_once_with("asmt_1")

    def test_maintenance_cleanup_endpoint_supports_dry_run_and_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8768)
            temp_file = config.app_data.temp_dir / "stale.tmp"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text("cleanup-me", encoding="utf-8")
            bundle_dir = config.app_data.temp_dir / "support-bundles"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            fresh_bundle = bundle_dir / "fresh.zip"
            fresh_bundle.write_text("bundle", encoding="utf-8")
            old_job = config.jobs_dir / "old-job.json"
            old_job.parent.mkdir(parents=True, exist_ok=True)
            old_job.write_text('{"status":"completed","completed_at":"2026-03-01T00:00:00+00:00"}', encoding="utf-8")
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            config.log_file.write_text("active", encoding="utf-8")
            rotated_log = config.app_data.logs_dir / "backend.log.1"
            rotated_log.write_text("rotated", encoding="utf-8")
            report_file = config.app_data.reports_dir / "session.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text("report", encoding="utf-8")
            client = TestClient(create_app(config))

            dry_run = client.post("/v1/maintenance/cleanup", json={"target": "tmp", "dry_run": True})
            self.assertEqual(dry_run.status_code, 200)
            self.assertEqual(dry_run.json()["target"], "tmp")
            self.assertTrue(temp_file.exists())
            self.assertEqual(dry_run.json()["deleted_file_count"], 1)
            self.assertTrue(fresh_bundle.exists())

            cleaned = client.post("/v1/maintenance/cleanup", json={"target": "all_safe", "dry_run": False})
            self.assertEqual(cleaned.status_code, 200)
            self.assertFalse(temp_file.exists())
            self.assertTrue(fresh_bundle.exists())
            self.assertFalse(old_job.exists())
            self.assertFalse(rotated_log.exists())
            self.assertTrue(config.log_file.exists())
            self.assertTrue(report_file.exists())

            invalid = client.post("/v1/maintenance/cleanup", json={"target": "bad-target", "dry_run": True})
            self.assertEqual(invalid.status_code, 422)

    def test_support_bundle_create_and_download_endpoints_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8769)
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            config.log_file.write_text(
                'runtime boot ok\nllm_api_key="top-secret"\nsecret_ref="connection:primary"\n',
                encoding="utf-8",
            )
            config.jobs_dir.mkdir(parents=True, exist_ok=True)
            (config.jobs_dir / "asmt_recent.json").write_text(
                json.dumps(
                    {
                        "assessment_id": "asmt_recent",
                        "status": "queued",
                        "request": {
                            "provider": "openrouter",
                            "llm_api_key": "job-secret",
                            "connection": {
                                "secret_ref": "connection:primary",
                            },
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config.app_data.reports_dir.mkdir(parents=True, exist_ok=True)
            (config.app_data.reports_dir / "session.json").write_text('{"session_id":"sess_1"}', encoding="utf-8")
            config.app_data.recordings_dir.mkdir(parents=True, exist_ok=True)
            (config.app_data.recordings_dir / "sample.wav").write_bytes(b"fake-recording")
            config.app_data.uploads_dir.mkdir(parents=True, exist_ok=True)
            (config.app_data.uploads_dir / "sample.wav").write_bytes(b"fake-upload")

            with TestClient(create_app(config)) as client:
                with mock.patch(
                    "app_backend.app._serialize_diagnostics",
                    return_value=mock.Mock(
                        items=[
                            mock.Mock(
                                model_dump=mock.Mock(
                                    return_value={
                                        "key": "runtime_health",
                                        "status": "ok",
                                        "title_key": "diagnostics.runtime_title",
                                        "detail_key": "diagnostics.runtime_detail",
                                        "detail_args": {"detail": "healthy"},
                                    }
                                )
                            )
                        ]
                    ),
                ):
                    created = client.post(
                        "/v1/support-bundles",
                        json={
                            "include_runtime_health": True,
                            "client_snapshot": {
                                "has_active_connection": True,
                                "provider_requires_auth": True,
                                "has_saved_secret": False,
                                "credentials_missing": True,
                                "secure_storage_persistent": True,
                                "credential_state": "missing",
                                "llm_api_key": "leaked-from-client",
                                "secret_ref": "connection:primary",
                            },
                            "client_diagnostics": [
                                {
                                    "key": "runtime_api_key",
                                    "status": "error",
                                    "detail_args": {
                                        "api_key": "diag-secret",
                                        "secret_ref": "connection:primary",
                                    },
                                }
                            ],
                        },
                    )
                self.assertEqual(created.status_code, 200)
                payload = created.json()
                self.assertTrue(payload["bundle_id"].startswith("bundle_"))
                bundle_id = payload["bundle_id"]

                downloaded = client.get(f"/v1/support-bundles/{bundle_id}")
                self.assertEqual(downloaded.status_code, 200)
                self.assertEqual(downloaded.headers["content-type"], "application/zip")

                missing = client.get("/v1/support-bundles/bundle_missing")
                self.assertEqual(missing.status_code, 404)

            bundle_path = config.app_data.temp_dir / "support-bundles" / f"{bundle_id}.zip"
            self.assertTrue(bundle_path.exists())
            with zipfile.ZipFile(bundle_path) as archive:
                names = set(archive.namelist())
                self.assertIn("manifest.json", names)
                self.assertIn("backend_state.json", names)
                self.assertIn("backend_diagnostics.json", names)
                self.assertIn("runtime_metadata.json", names)
                self.assertIn("storage_summary.json", names)
                self.assertIn("shell/client_snapshot.json", names)
                self.assertIn("shell/client_diagnostics.json", names)
                self.assertIn("jobs/asmt_recent.json", names)
                self.assertIn("logs/backend.log", names)
                self.assertFalse(any(name.startswith("reports/") for name in names))
                self.assertFalse(any(name.startswith("recordings/") for name in names))
                self.assertFalse(any(name.startswith("uploads/") for name in names))

                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["redaction"]["secret_ref_policy"], "full_redaction")
                self.assertGreaterEqual(manifest["redaction"]["removed_secret_refs"], 3)
                self.assertGreaterEqual(manifest["redaction"]["redacted_secret_values"], 3)

                runtime_metadata = json.loads(archive.read("runtime_metadata.json"))
                self.assertIn("auth_mode", runtime_metadata["runtime_metadata"])

                client_snapshot = json.loads(archive.read("shell/client_snapshot.json"))
                self.assertNotIn("secret_ref", client_snapshot)
                self.assertEqual(client_snapshot["llm_api_key"], "[redacted]")

                client_diagnostics = json.loads(archive.read("shell/client_diagnostics.json"))
                self.assertEqual(client_diagnostics[0]["detail_args"]["api_key"], "[redacted]")
                self.assertNotIn("secret_ref", client_diagnostics[0]["detail_args"])
                self.assertTrue(any(item["key"] == "runtime_health" for item in client_diagnostics))

                job_payload = json.loads(archive.read("jobs/asmt_recent.json"))
                self.assertEqual(job_payload["request"]["llm_api_key"], "[redacted]")
                self.assertNotIn("secret_ref", job_payload["request"]["connection"])

                log_content = archive.read("logs/backend.log").decode("utf-8")
                self.assertNotIn("top-secret", log_content)
                self.assertNotIn("connection:primary", log_content)


if __name__ == "__main__":
    unittest.main()

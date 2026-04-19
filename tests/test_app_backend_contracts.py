import unittest

from app_backend.contracts import AssessmentCreateRequest, ErrorCode, JobStatus


class BackendContractTests(unittest.TestCase):
    def test_assessment_create_request_requires_audio_id(self):
        payload = AssessmentCreateRequest(
            audio_id="aud_1",
            whisper="small",
            provider="openrouter",
            llm_model="google/gemini-3.1-pro-preview",
            expected_language="en",
            feedback_language="en",
            speaker_id="bern",
            task_family="free_monologue",
            theme="travel",
            target_duration_sec=90,
        )
        self.assertEqual(payload.audio_id, "aud_1")
        self.assertEqual(payload.provider, "openrouter")

    def test_error_codes_include_backend_and_local_provider_failures(self):
        self.assertEqual(ErrorCode.BACKEND_UNAVAILABLE.value, "backend_unavailable")
        self.assertEqual(ErrorCode.LOCAL_PROVIDER_NOT_RUNNING.value, "local_provider_not_running")

    def test_job_statuses_cover_full_lifecycle(self):
        statuses = {item.value for item in JobStatus}
        self.assertEqual(
            statuses,
            {"queued", "running", "completed", "failed", "cancelled"},
        )


if __name__ == "__main__":
    unittest.main()

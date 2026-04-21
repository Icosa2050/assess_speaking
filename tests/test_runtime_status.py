import unittest

from app_shell.i18n import t
from app_shell.runtime_resolver import RuntimeConfig
from app_shell.runtime_status import job_status_message


def _translate(locale: str):
    return lambda key, **kwargs: t(key, locale=locale, **kwargs)


class RuntimeStatusTests(unittest.TestCase):
    def test_job_status_message_mentions_openrouter_and_model_when_remote(self):
        message = job_status_message(
            "running",
            RuntimeConfig(
                provider="openrouter",
                model="google/gemini-3.1-pro-preview",
                base_url="https://openrouter.ai/api/v1",
                is_local=False,
            ),
            translate=_translate("en"),
        )
        self.assertEqual(
            message,
            "Your assessment is running via OpenRouter with model `google/gemini-3.1-pro-preview`.",
        )

    def test_job_status_message_mentions_local_provider_and_model_when_running_locally(self):
        message = job_status_message(
            "running",
            RuntimeConfig(
                provider="ollama",
                model="llama3",
                base_url="http://localhost:11434/v1",
                is_local=True,
            ),
            translate=_translate("en"),
        )
        self.assertEqual(
            message,
            "Your assessment is running locally with Ollama model `llama3`.",
        )

    def test_job_status_message_localizes_provider_aware_copy(self):
        message = job_status_message(
            "queued",
            RuntimeConfig(
                provider="openrouter",
                model="google/gemini-3.1-pro-preview",
                base_url="https://openrouter.ai/api/v1",
                is_local=False,
            ),
            translate=_translate("de"),
        )
        self.assertEqual(
            message,
            "Deine Auswertung wartet. Fuer die Review wird OpenRouter mit Modell `google/gemini-3.1-pro-preview` verwendet.",
        )

    def test_job_status_message_falls_back_to_generic_unknown_copy(self):
        self.assertEqual(
            job_status_message("mystery", None, translate=_translate("en")),
            "The assessment is still processing.",
        )


if __name__ == "__main__":
    unittest.main()

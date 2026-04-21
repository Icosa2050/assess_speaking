from __future__ import annotations

from collections.abc import Callable

from app_shell.i18n import t
from app_shell.runtime_providers import default_connection_label, normalize_provider
from app_shell.runtime_resolver import RuntimeConfig

TranslateFn = Callable[..., str]


def provider_display_name(provider: str, *, translate: TranslateFn = t) -> str:
    normalized_provider = normalize_provider(provider)
    translated = translate(f"settings.provider_option_{normalized_provider}")
    if not translated.startswith("["):
        return translated
    return default_connection_label(normalized_provider)


def job_status_message(status: str, runtime: RuntimeConfig | None = None, *, translate: TranslateFn = t) -> str:
    normalized_status = str(status or "").strip().lower()
    if runtime is not None:
        provider = provider_display_name(runtime.provider, translate=translate)
        model = str(runtime.model or "").strip()
        if normalized_status == "queued" and provider and model:
            translated = translate("speak.job_status_queued_provider", provider=provider, model=model)
            if not translated.startswith("["):
                return translated
        if normalized_status == "running" and provider and model:
            key = "speak.job_status_running_local_provider" if runtime.is_local else "speak.job_status_running_remote_provider"
            translated = translate(key, provider=provider, model=model)
            if not translated.startswith("["):
                return translated
    translated = translate(f"speak.job_status_{normalized_status or 'unknown'}")
    if translated.startswith("["):
        return translate("speak.job_status_unknown")
    return translated

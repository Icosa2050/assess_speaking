from __future__ import annotations

from collections.abc import Callable

from app_core.i18n import t
from app_core.runtime_providers import default_connection_label, normalize_provider
from app_core.runtime_resolver import RuntimeConfig

TranslateFn = Callable[..., str]


def _is_missing_translation(key: str, translated: str) -> bool:
    return translated == f"[{key}]"


def provider_display_name(provider: str, *, translate: TranslateFn = t) -> str:
    normalized_provider = normalize_provider(provider)
    key = f"settings.provider_option_{normalized_provider}"
    translated = translate(key)
    if _is_missing_translation(key, translated):
        return default_connection_label(normalized_provider)
    return translated


def job_status_message(status: str, runtime: RuntimeConfig | None = None, *, translate: TranslateFn = t) -> str:
    normalized_status = str(status or "").strip().lower()
    if runtime is not None:
        provider = provider_display_name(runtime.provider, translate=translate)
        model = str(runtime.model or "").strip()
        if normalized_status == "queued" and provider and model:
            key = "speak.job_status_queued_provider"
            translated = translate(key, provider=provider, model=model)
            if not _is_missing_translation(key, translated):
                return translated
        if normalized_status == "running" and provider and model:
            key = "speak.job_status_running_local_provider" if runtime.is_local else "speak.job_status_running_remote_provider"
            translated = translate(key, provider=provider, model=model)
            if not _is_missing_translation(key, translated):
                return translated
    key = f"speak.job_status_{normalized_status or 'unknown'}"
    translated = translate(key)
    if _is_missing_translation(key, translated):
        return translate("speak.job_status_unknown")
    return translated

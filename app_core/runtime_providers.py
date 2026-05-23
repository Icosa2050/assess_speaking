from __future__ import annotations

import hashlib

DEFAULT_PROVIDER = "openrouter"
SUPPORTED_PROVIDERS = ("openrouter", "ollama", "lmstudio", "openai_compatible")
SETUP_PROVIDER_CHOICES = ("ollama_local", "ollama_cloud", "lmstudio_local", "openrouter", "openai_compatible")
LEGACY_SETUP_PROVIDER_ALIASES = {
    "ollama": "ollama_local",
    "lmstudio": "lmstudio_local",
}

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai_compatible": "",
}

DEFAULT_SETUP_BASE_URLS = {
    "ollama_local": "http://localhost:11434",
    "ollama_cloud": "https://ollama.com/api",
    "lmstudio_local": "http://localhost:1234/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai_compatible": "",
}


def normalize_provider(provider: str | None) -> str:
    candidate = str(provider or "").strip().lower()
    return candidate if candidate in SUPPORTED_PROVIDERS else DEFAULT_PROVIDER


def normalize_setup_provider_choice(choice: str | None) -> str:
    candidate = str(choice or "").strip().lower()
    candidate = LEGACY_SETUP_PROVIDER_ALIASES.get(candidate, candidate)
    return candidate if candidate in SETUP_PROVIDER_CHOICES else "openrouter"


def provider_kind_from_choice(choice: str | None) -> str:
    normalized_choice = normalize_setup_provider_choice(choice)
    if normalized_choice.startswith("ollama"):
        return "ollama"
    if normalized_choice.startswith("lmstudio"):
        return "lmstudio"
    return normalize_provider(normalized_choice)


def is_local_setup_choice(choice: str | None) -> bool:
    return normalize_setup_provider_choice(choice) in {"ollama_local", "lmstudio_local"}


def default_connection_label(choice: str | None) -> str:
    candidate = str(choice or "").strip().lower()
    labels = {
        "ollama_local": "Ollama Local",
        "ollama_cloud": "Ollama Cloud",
        "lmstudio_local": "LM Studio Local",
        "openrouter": "OpenRouter",
        "openai_compatible": "OpenAI-compatible",
        "ollama": "Ollama",
        "lmstudio": "LM Studio",
    }
    if candidate in labels:
        return labels[candidate]
    normalized_choice = normalize_setup_provider_choice(choice)
    return labels.get(normalized_choice, "Runtime Connection")


def default_base_url(provider: str | None) -> str:
    return DEFAULT_BASE_URLS.get(normalize_provider(provider), "")


def default_setup_base_url(choice: str | None) -> str:
    return DEFAULT_SETUP_BASE_URLS.get(normalize_setup_provider_choice(choice), "")


def resolved_base_url(provider: str | None, base_url: str | None) -> str:
    candidate = str(base_url or "").strip()
    if candidate:
        return candidate.rstrip("/")
    default = default_base_url(provider)
    return default.rstrip("/")


def runtime_base_url(provider: str | None, base_url: str | None) -> str:
    normalized_provider = normalize_provider(provider)
    resolved = resolved_base_url(normalized_provider, base_url)
    if normalized_provider == "ollama" and resolved and not resolved.endswith("/v1"):
        return f"{resolved.rstrip('/')}/v1"
    return resolved.rstrip("/")


def service_base_url(provider: str | None, base_url: str | None) -> str:
    normalized_provider = normalize_provider(provider)
    candidate = str(base_url or "").strip()
    if not candidate:
        if normalized_provider == "ollama":
            return DEFAULT_SETUP_BASE_URLS["ollama_local"].rstrip("/")
        return default_base_url(normalized_provider).rstrip("/")
    resolved = candidate.rstrip("/")
    if normalized_provider == "ollama" and resolved.endswith("/v1"):
        return resolved[: -len("/v1")]
    return resolved


def supports_optional_bearer_token(provider: str | None) -> bool:
    return normalize_provider(provider) in {"ollama", "lmstudio", "openai_compatible"}


def requires_api_key(provider: str | None) -> bool:
    return normalize_provider(provider) == "openrouter"


def secret_account_name(provider: str | None, base_url: str | None) -> str:
    normalized_provider = normalize_provider(provider)
    normalized_url = runtime_base_url(normalized_provider, base_url)
    digest = hashlib.sha256(f"{normalized_provider}|{normalized_url}".encode("utf-8")).hexdigest()[:16]
    return f"llm:{normalized_provider}:{digest}"


def connection_secret_ref(connection_id: str) -> str:
    return f"connection:{str(connection_id or '').strip()}"

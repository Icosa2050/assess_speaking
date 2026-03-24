from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from app_shell.page_helpers import go_to, render_shell_summary
from app_shell.runtime_resolver import active_connection
from app_shell.secret_store import secret_store_status
from app_shell.services import (
    DEFAULT_WHISPER_OPTIONS,
    build_provider_connection,
    download_whisper_model,
    hydrate_state_from_storage,
    needs_runtime_setup,
    provider_choice_for_connection,
    save_provider_connection,
    test_runtime_connection,
    whisper_model_status,
)
from app_shell.state import (
    APP_NAME,
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_APP_TITLE,
    DEFAULT_OPENROUTER_HTTP_REFERER,
    get_app_state,
    set_current_page,
)

PROVIDER_CHOICES = {
    "ollama_local": "Ollama local",
    "ollama_cloud": "Ollama cloud",
    "lmstudio_local": "LM Studio local",
    "openrouter": "OpenRouter",
    "openai_compatible": "Generic OpenAI-compatible",
}


def _safe_index(options: list[str], value: str, default: int = 0) -> int:
    if value in options:
        return options.index(value)
    return default


st.set_page_config(page_title=f"{APP_NAME} · Runtime Setup", page_icon="🧭", layout="wide")
state = hydrate_state_from_storage(set_current_page("runtime_setup"))

st.title("Runtime setup")
st.caption("Prepare Whisper and the first inference connection before using the rest of the app shell.")
render_shell_summary(state)

current_connection = active_connection(state.prefs)
if not needs_runtime_setup(state):
    st.success("Runtime setup is already complete. You can still use this page to replace or update the active connection.")

setup_message = st.session_state.pop("runtime_setup_message", "")
setup_error = st.session_state.pop("runtime_setup_error", "")
test_message = st.session_state.pop("runtime_setup_test_message", "")
test_error = st.session_state.pop("runtime_setup_test_error", "")
if setup_message:
    st.success(setup_message)
if setup_error:
    st.error(setup_error)
if test_message:
    st.success(test_message)
if test_error:
    st.warning(test_error)

provider_options = list(PROVIDER_CHOICES)
default_provider_choice = provider_choice_for_connection(current_connection, state.prefs.provider)
default_label = current_connection.label if current_connection else PROVIDER_CHOICES[default_provider_choice]
default_base_url = current_connection.base_url if current_connection else state.prefs.llm_base_url
default_model = current_connection.default_model if current_connection else state.prefs.model
default_api_key = state.prefs.llm_api_key or state.prefs.openrouter_api_key
default_referer = str(
    (current_connection.provider_metadata if current_connection else {}).get("http_referer")
    or state.prefs.openrouter_http_referer
    or DEFAULT_OPENROUTER_HTTP_REFERER
)
default_app_title = str(
    (current_connection.provider_metadata if current_connection else {}).get("app_title")
    or state.prefs.openrouter_app_title
    or DEFAULT_OPENROUTER_APP_TITLE
)

with st.container(border=True):
    st.subheader("Section A · Whisper")
    whisper_model = st.selectbox(
        "Whisper model",
        options=list(DEFAULT_WHISPER_OPTIONS),
        index=_safe_index(list(DEFAULT_WHISPER_OPTIONS), state.prefs.whisper_model, default=_safe_index(list(DEFAULT_WHISPER_OPTIONS), "small")),
        key="runtime_setup_whisper_model",
    )
    st.caption("Guidance: tiny is fastest, small is a balanced default, and large-v3 is the best quality option.")
    availability = whisper_model_status(whisper_model)
    if availability["cached"]:
        st.caption(f"Cache status: ready at {availability['cached_path']}")
    else:
        st.caption("Cache status: not downloaded yet.")
    if st.button("Download model", key="runtime_setup_download_model", width="stretch"):
        try:
            with st.spinner("Downloading the selected Whisper model..."):
                result = download_whisper_model(whisper_model)
            st.session_state["runtime_setup_message"] = f"Whisper model is ready at {result['cached_path']}."
        except Exception as exc:
            st.session_state["runtime_setup_error"] = f"Whisper download failed: {exc}"
        st.rerun()

with st.container(border=True):
    st.subheader("Section B · Inference provider")
    provider_choice = st.selectbox(
        "Provider",
        options=provider_options,
        index=_safe_index(provider_options, default_provider_choice),
        format_func=lambda value: PROVIDER_CHOICES[value],
        key="runtime_setup_provider_choice",
    )
    if provider_choice == "ollama_cloud":
        st.caption("Cloud endpoints can expose different models and auth behavior than local Ollama.")
    elif provider_choice == "openai_compatible":
        st.caption("Use this for custom OpenAI-compatible endpoints when none of the named presets fit.")

with st.container(border=True):
    st.subheader("Section C · Connection details")
    label = st.text_input("Connection label", value=default_label, key="runtime_setup_label")
    base_url = st.text_input("Base URL", value=default_base_url, key="runtime_setup_base_url")
    model = st.text_input("Model", value=default_model or DEFAULT_MODEL, key="runtime_setup_model")
    api_key = st.text_input("Bearer token / API key", value=default_api_key, type="password", key="runtime_setup_api_key")
    openrouter_http_referer = st.text_input(
        "OpenRouter HTTP-Referer",
        value=default_referer,
        disabled=provider_choice != "openrouter",
        key="runtime_setup_openrouter_http_referer",
    )
    openrouter_app_title = st.text_input(
        "OpenRouter app title",
        value=default_app_title,
        disabled=provider_choice != "openrouter",
        key="runtime_setup_openrouter_app_title",
    )
    storage_status = secret_store_status()
    st.caption(f"Secret storage backend: {storage_status.backend_name}. {storage_status.detail or 'Persistent secure storage is available.'}")

with st.container(border=True):
    st.subheader("Section D · Test connection")
    st.caption("Run a quick health check first, then a short smoke test. Test results are advisory and never block save.")
    test_col, save_col, skip_col = st.columns(3)
    with test_col:
        test_clicked = st.button("Test connection", key="runtime_setup_test_connection", width="stretch")
    with save_col:
        save_clicked = st.button("Save connection", key="runtime_setup_save_connection", width="stretch")
    with skip_col:
        home_clicked = st.button("Back to home", key="runtime_setup_back_home", width="stretch")

if home_clicked:
    go_to("streamlit_app.py", return_to="runtime_setup")

if test_clicked:
    try:
        with st.spinner("Testing connection..."):
            result = test_runtime_connection(
                provider=provider_choice,
                provider_choice=provider_choice,
                model=model.strip() or DEFAULT_MODEL,
                base_url=base_url.strip(),
                api_key=api_key.strip(),
                openrouter_http_referer=openrouter_http_referer.strip() or DEFAULT_OPENROUTER_HTTP_REFERER,
                openrouter_app_title=openrouter_app_title.strip() or DEFAULT_OPENROUTER_APP_TITLE,
            )
        test_payload = result.get("test_payload") or {}
        st.session_state["runtime_setup_last_test_status"] = "passed"
        st.session_state["runtime_setup_last_tested_at"] = str(
            test_payload.get("tested_at") or datetime.now(UTC).isoformat(timespec="seconds")
        )
        preview = str(test_payload.get("content_preview") or "-").strip() or "-"
        st.session_state["runtime_setup_test_message"] = (
            f"Health check passed at {result['health_endpoint']}. "
            f"Smoke test reached {result['base_url']}. Preview: {preview}"
        )
    except Exception as exc:
        st.session_state["runtime_setup_last_test_status"] = f"failed: {exc}"
        st.session_state["runtime_setup_last_tested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        st.session_state["runtime_setup_test_error"] = f"Connection test failed: {exc}"
    st.rerun()

if save_clicked:
    state = get_app_state()
    state.prefs.whisper_model = whisper_model
    connection = build_provider_connection(
        provider_choice=provider_choice,
        label=label.strip(),
        model=model.strip() or DEFAULT_MODEL,
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        openrouter_http_referer=openrouter_http_referer.strip() or DEFAULT_OPENROUTER_HTTP_REFERER,
        openrouter_app_title=openrouter_app_title.strip() or DEFAULT_OPENROUTER_APP_TITLE,
        existing_connection=current_connection if current_connection and current_connection.connection_id == getattr(state.prefs, "active_connection_id", "") else None,
    )
    connection.last_test_status = str(st.session_state.get("runtime_setup_last_test_status") or "")
    connection.last_tested_at = str(st.session_state.get("runtime_setup_last_tested_at") or "")
    secret_status = save_provider_connection(state, connection, api_key=api_key.strip(), persist_draft=False)
    if secret_status.persistent:
        st.session_state["runtime_setup_message"] = "Runtime setup saved."
    else:
        st.session_state["runtime_setup_message"] = f"Runtime setup saved. Secret storage is running in {secret_status.backend_name} mode."
    st.rerun()

from __future__ import annotations

import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import configure_page, go_to, render_page_intro, render_shell_summary
from app_shell.runtime_resolver import active_connection, resolve_connection_runtime
from app_shell.secret_store import secret_store_status
from app_shell.services import (
    DEFAULT_WHISPER_OPTIONS,
    build_provider_connection,
    delete_provider_connection,
    download_whisper_model,
    needs_runtime_setup,
    provider_choice_for_connection,
    save_provider_connection,
    set_default_provider_connection,
    test_runtime_connection,
    whisper_model_status,
)
from app_shell.state import (
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_APP_TITLE,
    DEFAULT_OPENROUTER_HTTP_REFERER,
    SUPPORTED_UI_LOCALES,
    get_app_state,
)

PROVIDER_CHOICES = {
    "ollama_local": "Ollama local",
    "ollama": "Ollama local",
    "ollama_cloud": "Ollama cloud",
    "lmstudio_local": "LM Studio local",
    "lmstudio": "LM Studio local",
    "openrouter": "OpenRouter",
    "openai_compatible": "Generic OpenAI-compatible",
}


def _safe_index(options: list[str], value: str, default: int = 0) -> int:
    if value in options:
        return options.index(value)
    return default


def _connection_option_label(connection) -> str:
    return f"{connection.label} · {connection.default_model}"


def _populate_connection_form(state, connection) -> None:
    provider_choice = provider_choice_for_connection(connection, state.prefs.provider)
    runtime = resolve_connection_runtime(connection) if connection is not None else None
    metadata = dict(connection.provider_metadata or {}) if connection is not None else {}
    st.session_state["settings_provider"] = provider_choice
    st.session_state["settings_connection_label"] = (
        connection.label if connection is not None else PROVIDER_CHOICES.get(provider_choice, "Runtime connection")
    )
    st.session_state["settings_model"] = connection.default_model if connection is not None else (state.prefs.model or DEFAULT_MODEL)
    st.session_state["settings_base_url"] = connection.base_url if connection is not None else (state.prefs.llm_base_url or "")
    st.session_state["settings_openrouter_api_key"] = (
        runtime.api_key if runtime is not None else (state.prefs.llm_api_key or state.prefs.openrouter_api_key or "")
    )
    st.session_state["settings_openrouter_http_referer"] = str(
        metadata.get("http_referer") or state.prefs.openrouter_http_referer or DEFAULT_OPENROUTER_HTTP_REFERER
    )
    st.session_state["settings_openrouter_app_title"] = str(
        metadata.get("app_title") or state.prefs.openrouter_app_title or DEFAULT_OPENROUTER_APP_TITLE
    )


def _connection_detail_line(connection) -> str:
    provider_label = PROVIDER_CHOICES.get(provider_choice_for_connection(connection), connection.provider_kind)
    details = [provider_label, connection.default_model]
    if connection.is_default:
        details.append("default")
    if connection.last_test_status:
        details.append(connection.last_test_status)
    return " · ".join(part for part in details if part)


state = configure_page("settings", "nav.settings", icon="⚙️")

render_page_intro("settings.title", "settings.body")
render_shell_summary(state)

success_message = st.session_state.pop("settings_success", "")
if success_message:
    st.success(success_message if isinstance(success_message, str) else t("settings.saved"))
secret_message = st.session_state.pop("settings_secret_message", "")
if secret_message:
    st.info(secret_message)
provider_test_message = st.session_state.pop("settings_test_message", "")
provider_test_error = st.session_state.pop("settings_test_error", "")
if provider_test_message:
    st.success(provider_test_message)
if provider_test_error:
    st.warning(provider_test_error)
whisper_message = st.session_state.pop("settings_whisper_message", "")
whisper_error = st.session_state.pop("settings_whisper_error", "")
if whisper_message:
    st.success(whisper_message)
if whisper_error:
    st.error(whisper_error)

current_connection = active_connection(state.prefs)
if needs_runtime_setup(state):
    st.info("Guided runtime setup is recommended before ordinary Settings.")
    if st.button("Open guided setup", key="settings_open_setup", width="stretch"):
        go_to("pages/00_Setup.py", return_to="settings")

connection_options = ["__new__"] + [connection.connection_id for connection in state.prefs.connections]
default_connection_id = current_connection.connection_id if current_connection else "__new__"
pending_connection_id = str(st.session_state.pop("settings_pending_connection_id", "") or "").strip()
if pending_connection_id in connection_options:
    st.session_state["settings_connection_id"] = pending_connection_id
if st.session_state.get("settings_connection_id") not in connection_options:
    st.session_state["settings_connection_id"] = default_connection_id
selected_connection_id = st.selectbox(
    "Saved connection",
    options=connection_options,
    index=_safe_index(connection_options, default_connection_id),
    format_func=lambda value: "Create new connection" if value == "__new__" else _connection_option_label(
        next(item for item in state.prefs.connections if item.connection_id == value)
    ),
    key="settings_connection_id",
)
editing_connection = next((item for item in state.prefs.connections if item.connection_id == selected_connection_id), None)
if st.session_state.get("settings_form_connection_id") != selected_connection_id:
    _populate_connection_form(state, editing_connection)
    st.session_state["settings_form_connection_id"] = selected_connection_id
legacy_provider_aliases = {"ollama": "ollama_local", "lmstudio": "lmstudio_local"}
if "settings_provider" in st.session_state and st.session_state["settings_provider"] not in PROVIDER_CHOICES:
    st.session_state["settings_provider"] = legacy_provider_aliases.get(
        str(st.session_state["settings_provider"]),
        provider_choice_for_connection(editing_connection, state.prefs.provider),
    )

with st.container(border=True):
    st.subheader("Saved connections")
    if not state.prefs.connections:
        st.caption("No saved connections yet. Use the form below or the guided setup page to create the first one.")
    for connection in state.prefs.connections:
        st.markdown(f"**{connection.label}**")
        st.caption(_connection_detail_line(connection))
        if connection.last_tested_at:
            st.caption(f"Last tested: {connection.last_tested_at}")
        edit_col, default_col, delete_col = st.columns(3)
        if edit_col.button("Edit", key=f"settings_edit_{connection.connection_id}", width="stretch"):
            st.session_state["settings_pending_connection_id"] = connection.connection_id
            _populate_connection_form(state, connection)
            st.session_state["settings_form_connection_id"] = connection.connection_id
            st.rerun()
        if default_col.button(
            "Make default",
            key=f"settings_default_{connection.connection_id}",
            disabled=connection.is_default,
            width="stretch",
        ):
            if set_default_provider_connection(state, connection.connection_id, persist_draft=False):
                updated_connection = next((item for item in state.prefs.connections if item.connection_id == connection.connection_id), None)
                st.session_state["settings_pending_connection_id"] = connection.connection_id
                _populate_connection_form(state, updated_connection)
                st.session_state["settings_form_connection_id"] = connection.connection_id
                st.session_state["settings_success"] = f"Default connection changed to {connection.label}."
            st.rerun()
        if delete_col.button("Delete", key=f"settings_delete_{connection.connection_id}", width="stretch"):
            deleted_label = connection.label
            if delete_provider_connection(state, connection.connection_id, persist_draft=False):
                next_connection = active_connection(state.prefs)
                next_connection_id = next_connection.connection_id if next_connection is not None else "__new__"
                st.session_state["settings_pending_connection_id"] = next_connection_id
                _populate_connection_form(state, next_connection)
                st.session_state["settings_form_connection_id"] = next_connection_id
                st.session_state["settings_success"] = f"Deleted connection {deleted_label}."
            st.rerun()

ui_locale = st.selectbox(
    t("settings.ui_locale"),
    options=list(SUPPORTED_UI_LOCALES),
    index=_safe_index(list(SUPPORTED_UI_LOCALES), state.prefs.ui_locale),
    format_func=lambda value: t(f"locale.{value}"),
    key="settings_ui_locale",
)

provider_choice = st.selectbox(
    t("settings.provider"),
    options=list(PROVIDER_CHOICES),
    index=_safe_index(list(PROVIDER_CHOICES), provider_choice_for_connection(editing_connection, state.prefs.provider)),
    format_func=lambda value: PROVIDER_CHOICES.get(value, value),
    key="settings_provider",
)

label_default = editing_connection.label if editing_connection else PROVIDER_CHOICES[provider_choice]
label = st.text_input("Connection label", value=label_default, key="settings_connection_label")

model = st.text_input(
    t("settings.model"),
    value=(editing_connection.default_model if editing_connection else (state.prefs.model or DEFAULT_MODEL)),
    key="settings_model",
)

base_url_default = editing_connection.base_url if editing_connection else (state.prefs.llm_base_url or "")
base_url = st.text_input(
    t("settings.base_url"),
    value=base_url_default,
    key="settings_base_url",
)

api_key = st.text_input(
    t("settings.api_key"),
    value=state.prefs.llm_api_key or state.prefs.openrouter_api_key or "",
    type="password",
    key="settings_openrouter_api_key",
)

openrouter_http_referer = st.text_input(
    t("settings.openrouter_http_referer"),
    value=str(
        (editing_connection.provider_metadata if editing_connection else {}).get("http_referer")
        or state.prefs.openrouter_http_referer
        or DEFAULT_OPENROUTER_HTTP_REFERER
    ),
    disabled=provider_choice != "openrouter",
    key="settings_openrouter_http_referer",
)

openrouter_app_title = st.text_input(
    t("settings.openrouter_app_title"),
    value=str(
        (editing_connection.provider_metadata if editing_connection else {}).get("app_title")
        or state.prefs.openrouter_app_title
        or DEFAULT_OPENROUTER_APP_TITLE
    ),
    disabled=provider_choice != "openrouter",
    key="settings_openrouter_app_title",
)

storage_status = secret_store_status()
st.caption(
    t(
        "settings.secret_storage_note",
        backend=storage_status.backend_name,
        detail=storage_status.detail or t("settings.secret_storage_ok"),
    )
)

whisper_model = st.selectbox(
    t("settings.whisper_model"),
    options=list(DEFAULT_WHISPER_OPTIONS),
    index=_safe_index(list(DEFAULT_WHISPER_OPTIONS), state.prefs.whisper_model, default=_safe_index(list(DEFAULT_WHISPER_OPTIONS), "small")),
    key="settings_whisper_model",
)
availability = whisper_model_status(whisper_model)
if availability["cached"]:
    st.caption(t("settings.whisper_cached", path=availability["cached_path"]))
else:
    st.caption(t("settings.whisper_not_cached"))
if availability["recommendation_reason"]:
    st.caption(availability["recommendation_reason"])

download_model = st.button(t("settings.whisper_download"), key="settings_whisper_download", width="stretch")
test_connection = st.button(t("settings.test_connection"), key="settings_test_connection", width="stretch")
saved = st.button(t("settings.save"), key="settings_save", width="stretch")

if download_model:
    try:
        with st.spinner(t("settings.whisper_downloading")):
            result = download_whisper_model(whisper_model)
        st.session_state["settings_whisper_message"] = t("settings.whisper_downloaded", path=result["cached_path"])
    except Exception as exc:
        st.session_state["settings_whisper_error"] = t("settings.whisper_download_failed", detail=str(exc))
    st.rerun()

if test_connection:
    try:
        with st.spinner(t("settings.testing_connection")):
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
        st.session_state["settings_last_test_status"] = "passed"
        st.session_state["settings_last_tested_at"] = str(test_payload.get("tested_at") or "")
        preview = str(test_payload.get("content_preview") or "").strip() or "-"
        st.session_state["settings_test_message"] = (
            f"Health check passed at {result['health_endpoint']}. "
            f"Smoke test reached {result['base_url']}. Preview: {preview}"
        )
    except Exception as exc:
        st.session_state["settings_last_test_status"] = f"failed: {exc}"
        st.session_state["settings_last_tested_at"] = ""
        st.session_state["settings_test_error"] = t("settings.test_failed", detail=str(exc))
    st.rerun()

if saved:
    state = get_app_state()
    state.prefs.ui_locale = ui_locale
    state.prefs.whisper_model = whisper_model
    connection = build_provider_connection(
        provider_choice=provider_choice,
        label=label.strip(),
        model=model.strip() or DEFAULT_MODEL,
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        openrouter_http_referer=openrouter_http_referer.strip() or DEFAULT_OPENROUTER_HTTP_REFERER,
        openrouter_app_title=openrouter_app_title.strip() or DEFAULT_OPENROUTER_APP_TITLE,
        existing_connection=editing_connection,
    )
    connection.last_test_status = str(st.session_state.get("settings_last_test_status") or connection.last_test_status or "")
    connection.last_tested_at = str(st.session_state.get("settings_last_tested_at") or connection.last_tested_at or "")
    secret_status = save_provider_connection(state, connection, api_key=api_key.strip(), persist_draft=False)
    st.session_state["settings_pending_connection_id"] = connection.connection_id
    st.session_state["settings_form_connection_id"] = connection.connection_id
    if secret_status.persistent:
        st.session_state["settings_secret_message"] = t("settings.secret_saved", backend=secret_status.backend_name)
    elif api_key.strip():
        st.session_state["settings_secret_message"] = t("settings.secret_session_only", detail=secret_status.detail or secret_status.backend_name)
    st.session_state["settings_success"] = t("settings.saved")
    st.rerun()

with st.container(border=True):
    st.subheader(t("settings.return_title"))
    if st.button(t("settings.back"), key="settings_back", width="stretch"):
        state = get_app_state()
        if state.nav.return_to == "review":
            go_to("pages/03_Review.py", return_to="settings")
        elif state.nav.return_to == "speak":
            go_to("pages/02_Speak.py", return_to="settings")
        elif state.nav.return_to == "setup":
            go_to("pages/01_Session_Setup.py", return_to="settings")
        elif state.nav.return_to == "library":
            go_to("pages/05_Library.py", return_to="settings")
        elif state.nav.return_to == "history":
            go_to("pages/04_History.py", return_to="settings")
        else:
            go_to("streamlit_app.py", return_to="settings")

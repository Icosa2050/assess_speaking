from __future__ import annotations

import streamlit as st

import app_shell.backend_client as backend_client
import app_shell.services as shell_services
from app_shell.i18n import t
from app_shell.page_helpers import (
    configure_page,
    describe_whisper_download_event,
    format_byte_count,
    go_to,
    render_page_intro,
    render_shell_summary,
    storage_area_rows,
)
from app_shell.runtime_providers import default_setup_base_url
from app_shell.runtime_resolver import active_connection, resolve_connection_runtime
from app_shell.secret_store import delete_secret, secret_store_status
from app_shell.services import (
    DEFAULT_WHISPER_OPTIONS,
    build_provider_connection,
    delete_provider_connection,
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

PROVIDER_CHOICES = (
    "ollama_local",
    "ollama_cloud",
    "lmstudio_local",
    "openrouter",
    "openai_compatible",
)

PROVIDER_SUGGESTED_MODELS = {
    "ollama_local": "llama3",
    "ollama_cloud": "llama3",
    "lmstudio_local": "qwen2.5",
    "openrouter": DEFAULT_MODEL,
    "openai_compatible": "",
}


def _safe_index(options: list[str], value: str, default: int = 0) -> int:
    if value in options:
        return options.index(value)
    return default


def _provider_label(provider_choice: str) -> str:
    return t(f"settings.provider_option_{provider_choice}")


def _connection_option_label(connection) -> str:
    return f"{connection.label} · {connection.default_model}"


def _provider_choice_hint(connection, fallback_provider: str = "") -> str:
    return provider_choice_for_connection(connection, fallback_provider)


def _default_connection_form_values(provider_choice: str) -> dict[str, str]:
    selected_choice = str(provider_choice or "").strip()
    return {
        "label": _provider_label(selected_choice) if selected_choice in PROVIDER_CHOICES else t("settings.connection_label_default"),
        "model": str(PROVIDER_SUGGESTED_MODELS.get(selected_choice, "") or "").strip(),
        "base_url": shell_services.sanitize_setup_base_url(selected_choice, default_setup_base_url(selected_choice)),
        "api_key": "",
        "referer": DEFAULT_OPENROUTER_HTTP_REFERER,
        "app_title": DEFAULT_OPENROUTER_APP_TITLE,
    }


def _populate_connection_form(connection, fallback_provider: str = "") -> None:
    provider_choice = _provider_choice_hint(connection, fallback_provider)
    metadata = dict(connection.provider_metadata or {}) if connection is not None else {}
    defaults = _default_connection_form_values(provider_choice)
    st.session_state["settings_provider"] = provider_choice
    st.session_state["settings_last_provider_choice"] = provider_choice
    st.session_state["settings_connection_label"] = (
        connection.label if connection is not None else defaults["label"]
    )
    st.session_state["settings_model"] = connection.default_model if connection is not None else defaults["model"]
    st.session_state["settings_base_url"] = connection.base_url if connection is not None else defaults["base_url"]
    st.session_state["settings_api_key"] = defaults["api_key"]
    st.session_state["settings_openrouter_http_referer"] = str(
        metadata.get("http_referer") or defaults["referer"]
    )
    st.session_state["settings_openrouter_app_title"] = str(
        metadata.get("app_title") or defaults["app_title"]
    )
    st.session_state.pop("settings_clear_secret_confirmation", None)
    st.session_state.pop("settings_clear_secret_requested", None)


def _connection_detail_line(connection) -> str:
    provider_choice = provider_choice_for_connection(connection)
    provider_label = _provider_label(provider_choice) if provider_choice in PROVIDER_CHOICES else connection.provider_kind
    details = [provider_label, connection.default_model]
    if connection.is_default:
        details.append(t("settings.default_badge"))
    if connection.last_test_status:
        details.append(connection.last_test_status)
    return " · ".join(part for part in details if part)


def _settings_log_dir() -> str | None:
    current_state = get_app_state()
    return str(current_state.prefs.log_dir or "").strip() or None


def _render_support_storage(storage_payload: object) -> None:
    rows = storage_area_rows(storage_payload)
    if not rows:
        st.caption(t("settings.support_storage_empty"))
        return
    st.markdown(f"**{t('settings.support_storage_title')}**")
    for row in rows:
        st.markdown(
            f"**{t(str(row['label_key']))}**  \n"
            f"{t('settings.support_storage_row_detail', size=row['size_label'], file_count=row['file_count'])}"
        )
        if row["path"]:
            st.caption(str(row["path"]))


def _cleanup_message_key(dry_run: bool) -> str:
    return "settings.support_cleanup_preview_success" if dry_run else "settings.support_cleanup_run_success"


def _run_support_cleanup(*, dry_run: bool) -> None:
    try:
        result = backend_client.post_maintenance_cleanup(
            {"target": "all_safe", "dry_run": dry_run},
            log_dir=_settings_log_dir(),
        )
        st.session_state["settings_support_message"] = t(
            _cleanup_message_key(dry_run),
            file_count=result.deleted_file_count,
            size=format_byte_count(result.freed_bytes),
        )
        st.session_state["settings_support_warnings"] = list(result.warnings or [])
        if not dry_run:
            st.session_state.pop("settings_support_storage", None)
    except Exception as exc:  # quality: allow[broad-except] backend support errors become localized Settings feedback
        st.session_state["settings_support_error"] = t("settings.support_cleanup_error", detail=str(exc))
    st.rerun()


def _create_support_bundle(
    *,
    include_reports: bool,
    include_recordings: bool,
    include_uploads: bool,
    include_runtime_health: bool,
) -> None:
    try:
        with st.spinner(t("settings.support_bundle_creating")):
            created = shell_services.create_support_bundle_archive(
                get_app_state(),
                include_reports=include_reports,
                include_recordings=include_recordings,
                include_uploads=include_uploads,
                include_runtime_health=include_runtime_health,
            )
        st.session_state["settings_support_message"] = t(
            "settings.support_bundle_success",
            filename=created.filename,
            size=format_byte_count(created.size_bytes),
            expires_at=created.expires_at,
        )
    except Exception as exc:  # quality: allow[broad-except] backend support errors become localized Settings feedback
        st.session_state["settings_support_error"] = t("settings.support_bundle_error", detail=str(exc))
    st.rerun()


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
support_message = st.session_state.pop("settings_support_message", "")
support_error = st.session_state.pop("settings_support_error", "")
support_warnings = list(st.session_state.pop("settings_support_warnings", []) or [])

current_connection = active_connection(state.prefs)
if needs_runtime_setup(state):
    st.info(t("settings.needs_setup_info"))
    if st.button(t("settings.open_setup"), key="settings_open_setup", width="stretch"):
        go_to("pages/00_Setup.py", return_to="settings")

connection_options = ["__new__"] + [connection.connection_id for connection in state.prefs.connections]
default_connection_id = current_connection.connection_id if current_connection else "__new__"
pending_connection_id = str(st.session_state.pop("settings_pending_connection_id", "") or "").strip()
if pending_connection_id in connection_options:
    st.session_state["settings_connection_id"] = pending_connection_id
if st.session_state.get("settings_connection_id") not in connection_options:
    st.session_state["settings_connection_id"] = default_connection_id
selected_connection_id = st.selectbox(
    t("settings.saved_connection"),
    options=connection_options,
    index=_safe_index(connection_options, default_connection_id),
    format_func=lambda value: t("settings.create_new_connection") if value == "__new__" else _connection_option_label(
        next(item for item in state.prefs.connections if item.connection_id == value)
    ),
    key="settings_connection_id",
)
editing_connection = next((item for item in state.prefs.connections if item.connection_id == selected_connection_id), None)
if st.session_state.get("settings_form_connection_id") != selected_connection_id:
    _populate_connection_form(editing_connection, state.prefs.provider)
    st.session_state["settings_form_connection_id"] = selected_connection_id
if "settings_provider" in st.session_state and st.session_state["settings_provider"] not in PROVIDER_CHOICES:
    st.session_state["settings_provider"] = _provider_choice_hint(editing_connection, state.prefs.provider)

with st.container(border=True):
    st.subheader(t("settings.saved_connections"))
    if not state.prefs.connections:
        st.caption(t("settings.no_saved_connections"))
    for connection in state.prefs.connections:
        st.markdown(f"**{connection.label}**")
        st.caption(_connection_detail_line(connection))
        if connection.last_tested_at:
            st.caption(t("settings.last_tested", value=connection.last_tested_at))
        edit_col, default_col, delete_col = st.columns(3)
        if edit_col.button(t("settings.edit"), key=f"settings_edit_{connection.connection_id}", width="stretch"):
            st.session_state["settings_pending_connection_id"] = connection.connection_id
            _populate_connection_form(connection, state.prefs.provider)
            st.session_state["settings_form_connection_id"] = connection.connection_id
            st.rerun()
        if default_col.button(
            t("settings.make_default"),
            key=f"settings_default_{connection.connection_id}",
            disabled=connection.is_default,
            width="stretch",
        ):
            if set_default_provider_connection(state, connection.connection_id, persist_draft=False):
                updated_connection = next((item for item in state.prefs.connections if item.connection_id == connection.connection_id), None)
                st.session_state["settings_pending_connection_id"] = connection.connection_id
                _populate_connection_form(updated_connection, state.prefs.provider)
                st.session_state["settings_form_connection_id"] = connection.connection_id
                st.session_state["settings_success"] = t("settings.default_changed", label=connection.label)
            st.rerun()
        if delete_col.button(t("settings.delete"), key=f"settings_delete_{connection.connection_id}", width="stretch"):
            deleted_label = connection.label
            if delete_provider_connection(state, connection.connection_id, persist_draft=False):
                next_connection = active_connection(state.prefs)
                next_connection_id = next_connection.connection_id if next_connection is not None else "__new__"
                st.session_state["settings_pending_connection_id"] = next_connection_id
                _populate_connection_form(next_connection, state.prefs.provider)
                st.session_state["settings_form_connection_id"] = next_connection_id
                st.session_state["settings_success"] = t("settings.deleted_connection", label=deleted_label)
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
    index=_safe_index(list(PROVIDER_CHOICES), _provider_choice_hint(editing_connection, state.prefs.provider)),
    format_func=_provider_label,
    key="settings_provider",
)

provider_defaults = _default_connection_form_values(provider_choice)
editing_provider_choice = _provider_choice_hint(editing_connection, state.prefs.provider) if editing_connection is not None else ""
same_provider_as_existing = editing_connection is not None and provider_choice == editing_provider_choice
existing_runtime = resolve_connection_runtime(editing_connection) if same_provider_as_existing and editing_connection is not None else None
existing_secret_available = bool(existing_runtime and existing_runtime.api_key)
existing_secret_missing = bool(editing_connection is not None and same_provider_as_existing and editing_connection.secret_ref and not existing_secret_available)
clear_secret_requested = bool(
    editing_connection is not None
    and same_provider_as_existing
    and st.session_state.get("settings_clear_secret_requested") == editing_connection.connection_id
)
clear_secret_confirmation = bool(
    editing_connection is not None
    and same_provider_as_existing
    and st.session_state.get("settings_clear_secret_confirmation") == editing_connection.connection_id
)

label_default = editing_connection.label if editing_connection else _provider_label(provider_choice)
label = st.text_input(t("settings.connection_label"), value=label_default, key="settings_connection_label")

model = st.text_input(
    t("settings.model"),
    value=(editing_connection.default_model if editing_connection else provider_defaults["model"]),
    key="settings_model",
)

base_url_default = editing_connection.base_url if editing_connection else provider_defaults["base_url"]
base_url = st.text_input(
    t("settings.base_url"),
    value=base_url_default,
    key="settings_base_url",
)

api_key = st.text_input(
    t("settings.api_key"),
    value=provider_defaults["api_key"],
    type="password",
    key="settings_api_key",
)

openrouter_http_referer = st.text_input(
    t("settings.openrouter_http_referer"),
    value=str(
        (editing_connection.provider_metadata if editing_connection else {}).get("http_referer")
        or provider_defaults["referer"]
    ),
    disabled=provider_choice != "openrouter",
    key="settings_openrouter_http_referer",
)

openrouter_app_title = st.text_input(
    t("settings.openrouter_app_title"),
    value=str(
        (editing_connection.provider_metadata if editing_connection else {}).get("app_title")
        or provider_defaults["app_title"]
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

api_key_value = str(api_key or "").strip()
effective_api_key = "" if clear_secret_requested else api_key_value
if not effective_api_key and same_provider_as_existing and not clear_secret_requested and existing_runtime is not None:
    effective_api_key = str(existing_runtime.api_key or "").strip()

if clear_secret_requested:
    st.warning(t("settings.secret_clear_pending"))
elif existing_secret_available:
    st.caption(t("settings.secret_saved_state"))
    st.caption(t("settings.secret_keep_blank_hint"))
elif existing_secret_missing:
    st.warning(t("settings.secret_missing_state"))

if same_provider_as_existing and editing_connection is not None and editing_connection.secret_ref:
    if clear_secret_confirmation:
        st.warning(t("settings.clear_saved_key_confirm"))
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button(t("settings.clear_saved_key_confirm_button"), key="settings_clear_saved_key_confirm", width="stretch"):
            st.session_state["settings_clear_secret_requested"] = editing_connection.connection_id
            st.session_state.pop("settings_clear_secret_confirmation", None)
            st.rerun()
        if cancel_col.button(t("settings.clear_saved_key_cancel"), key="settings_clear_saved_key_cancel", width="stretch"):
            st.session_state.pop("settings_clear_secret_confirmation", None)
            st.rerun()
    elif clear_secret_requested:
        if st.button(t("settings.clear_saved_key_undo"), key="settings_clear_saved_key_undo", width="stretch"):
            st.session_state.pop("settings_clear_secret_requested", None)
            st.rerun()
    elif st.button(t("settings.clear_saved_key"), key="settings_clear_saved_key", width="stretch"):
        st.session_state["settings_clear_secret_confirmation"] = editing_connection.connection_id
        st.rerun()

download_status = st.empty()
download_detail = st.empty()
download_progress = st.empty()
download_model = st.button(t("settings.whisper_download"), key="settings_whisper_download", width="stretch")
test_connection = st.button(t("settings.test_connection"), key="settings_test_connection", width="stretch")
saved = st.button(t("settings.save"), key="settings_save", width="stretch")

if download_model:
    progress_bar = download_progress.progress(0)

    def _update_download_progress(event: dict[str, object]) -> None:
        status = describe_whisper_download_event(event)
        download_status.info(str(status["headline"]))
        download_detail.caption(str(status["detail"]))
        progress_bar.progress(int(status["progress_percent"]))

    try:
        result = shell_services.download_whisper_model(whisper_model, progress_callback=_update_download_progress)
        st.session_state["settings_whisper_message"] = t("settings.whisper_downloaded", path=result["cached_path"])
    except Exception as exc:  # quality: allow[broad-except] download boundary should become localized settings feedback
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
                api_key=effective_api_key,
                openrouter_http_referer=openrouter_http_referer.strip() or DEFAULT_OPENROUTER_HTTP_REFERER,
                openrouter_app_title=openrouter_app_title.strip() or DEFAULT_OPENROUTER_APP_TITLE,
            )
        test_payload = result.get("test_payload") or {}
        st.session_state["settings_last_test_status"] = "passed"
        st.session_state["settings_last_tested_at"] = str(test_payload.get("tested_at") or "")
        preview = str(test_payload.get("content_preview") or "").strip() or "-"
        st.session_state["settings_test_message"] = t(
            "settings.test_success",
            provider=_provider_label(provider_choice) if provider_choice in PROVIDER_CHOICES else provider_choice,
            base_url=result["base_url"],
            preview=preview,
        )
    except Exception as exc:  # quality: allow[broad-except] provider test errors should become localized settings feedback
        st.session_state["settings_last_test_status"] = f"failed: {exc}"
        st.session_state["settings_last_tested_at"] = ""
        st.session_state["settings_test_error"] = t("settings.test_failed", detail=str(exc))
    st.rerun()

if saved:
    state = get_app_state()
    state.prefs.ui_locale = ui_locale
    state.prefs.whisper_model = whisper_model
    if editing_connection is not None and editing_connection.secret_ref and (clear_secret_requested or (not api_key_value and not same_provider_as_existing)):
        delete_secret(editing_connection.secret_ref)
    connection = build_provider_connection(
        provider_choice=provider_choice,
        label=label.strip(),
        model=model.strip() or DEFAULT_MODEL,
        base_url=base_url.strip(),
        api_key=effective_api_key,
        openrouter_http_referer=openrouter_http_referer.strip() or DEFAULT_OPENROUTER_HTTP_REFERER,
        openrouter_app_title=openrouter_app_title.strip() or DEFAULT_OPENROUTER_APP_TITLE,
        existing_connection=editing_connection,
    )
    connection.last_test_status = str(st.session_state.get("settings_last_test_status") or connection.last_test_status or "")
    connection.last_tested_at = str(st.session_state.get("settings_last_tested_at") or connection.last_tested_at or "")
    secret_status = save_provider_connection(state, connection, api_key=effective_api_key, persist_draft=False)
    st.session_state["settings_pending_connection_id"] = connection.connection_id
    st.session_state["settings_form_connection_id"] = connection.connection_id
    st.session_state.pop("settings_clear_secret_confirmation", None)
    st.session_state.pop("settings_clear_secret_requested", None)
    if clear_secret_requested:
        st.session_state["settings_secret_message"] = t("settings.secret_cleared")
    elif api_key_value and secret_status.persistent:
        st.session_state["settings_secret_message"] = t("settings.secret_saved", backend=secret_status.backend_name)
    elif api_key_value:
        st.session_state["settings_secret_message"] = t("settings.secret_session_only", detail=secret_status.detail or secret_status.backend_name)
    st.session_state["settings_success"] = t("settings.saved")
    st.rerun()

with st.container(border=True):
    st.subheader(t("settings.support_title"))
    st.caption(t("settings.support_body"))
    st.info(t("settings.support_privacy_note"))
    if support_message:
        st.success(support_message)
    if support_error:
        st.error(support_error)
    for warning in support_warnings:
        st.warning(t("settings.support_backend_warning", detail=warning))

    if st.button(t("settings.support_refresh_storage"), key="settings_support_refresh_storage", width="stretch"):
        try:
            storage = backend_client.get_maintenance_storage(log_dir=_settings_log_dir())
            st.session_state["settings_support_storage"] = storage.model_dump(mode="json")
            st.session_state["settings_support_message"] = t("settings.support_storage_refreshed")
        except Exception as exc:  # quality: allow[broad-except] backend support errors become localized Settings feedback
            st.session_state["settings_support_error"] = t("settings.support_storage_error", detail=str(exc))
        st.rerun()

    _render_support_storage(st.session_state.get("settings_support_storage", {}))

    st.markdown(f"**{t('settings.support_cleanup_title')}**")
    st.caption(t("settings.support_cleanup_body"))
    preview_col, cleanup_col = st.columns(2)
    if preview_col.button(t("settings.support_cleanup_preview"), key="settings_support_cleanup_preview", width="stretch"):
        _run_support_cleanup(dry_run=True)
    if cleanup_col.button(t("settings.support_cleanup_run"), key="settings_support_cleanup_run", width="stretch"):
        _run_support_cleanup(dry_run=False)

    st.markdown(f"**{t('settings.support_bundle_title')}**")
    st.caption(t("settings.support_bundle_body"))
    include_reports = st.checkbox(
        t("settings.support_bundle_include_reports"),
        value=False,
        key="settings_support_include_reports",
    )
    include_recordings = st.checkbox(
        t("settings.support_bundle_include_recordings"),
        value=False,
        key="settings_support_include_recordings",
    )
    include_uploads = st.checkbox(
        t("settings.support_bundle_include_uploads"),
        value=False,
        key="settings_support_include_uploads",
    )
    include_runtime_health = st.checkbox(
        t("settings.support_bundle_include_runtime_health"),
        value=False,
        key="settings_support_include_runtime_health",
    )
    if st.button(t("settings.support_bundle_create"), key="settings_support_create_bundle", width="stretch"):
        _create_support_bundle(
            include_reports=include_reports,
            include_recordings=include_recordings,
            include_uploads=include_uploads,
            include_runtime_health=include_runtime_health,
        )

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

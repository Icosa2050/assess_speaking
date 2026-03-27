from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

import app_shell.services as shell_services
from app_shell.i18n import t
from app_shell.page_helpers import describe_whisper_download_event, go_to, render_shell_summary, resolve_page_title_locale
from app_shell.runtime_providers import default_connection_label, default_setup_base_url
from app_shell.runtime_resolver import active_connection, resolve_connection_runtime
from app_shell.secret_store import secret_store_status
from app_shell.services import (
    DEFAULT_WHISPER_OPTIONS,
    build_provider_connection,
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

LOCAL_PROVIDER_CHOICES = {"ollama_local", "lmstudio_local"}


def _safe_index(options: list[str], value: str, default: int = 0) -> int:
    if value in options:
        return options.index(value)
    return default


def _default_provider_choice(current_connection, fallback_provider: str = "") -> str:
    return provider_choice_for_connection(current_connection, fallback_provider)


def _suggested_model(provider_choice: str, *, current_connection) -> str:
    selected_choice = str(provider_choice or "").strip()
    current_choice = _default_provider_choice(current_connection)
    if current_connection is not None and selected_choice == current_choice:
        return str(current_connection.default_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return str(PROVIDER_SUGGESTED_MODELS.get(selected_choice, "") or "").strip()


def _provider_label(provider_choice: str) -> str:
    return t(f"runtime_setup.provider_options.{provider_choice}.label")


def _provider_hint(provider_choice: str) -> str:
    return t(f"runtime_setup.provider_options.{provider_choice}.hint")


def _connection_defaults(provider_choice: str, *, current_connection, fallback_provider: str = "") -> dict[str, str]:
    selected_choice = str(provider_choice or "").strip()
    current_choice = _default_provider_choice(current_connection, fallback_provider)
    using_current_connection = current_connection is not None and selected_choice == current_choice
    runtime = resolve_connection_runtime(current_connection) if using_current_connection else None
    metadata = dict(current_connection.provider_metadata or {}) if using_current_connection else {}
    return {
        "label": str(
            current_connection.label if using_current_connection else default_connection_label(selected_choice)
        ).strip(),
        "base_url": str(
            shell_services.sanitize_setup_base_url(selected_choice, current_connection.base_url)
            if using_current_connection
            else shell_services.sanitize_setup_base_url(selected_choice, default_setup_base_url(selected_choice))
        ).strip(),
        "model": _suggested_model(selected_choice, current_connection=current_connection),
        "api_key": str(runtime.api_key or "").strip() if runtime is not None else "",
        "referer": str(
            metadata.get("http_referer")
            or DEFAULT_OPENROUTER_HTTP_REFERER
        ).strip(),
        "app_title": str(
            metadata.get("app_title")
            or DEFAULT_OPENROUTER_APP_TITLE
        ).strip(),
    }


def _seed_form_state(*, current_connection, fallback_provider: str = "") -> None:
    default_choice = _default_provider_choice(current_connection, fallback_provider)
    selected_choice = str(st.session_state.get("runtime_setup_provider_choice") or default_choice).strip()
    if selected_choice not in PROVIDER_CHOICES:
        selected_choice = default_choice
    connection_signature = str(current_connection.connection_id if current_connection is not None else "__new__")
    seed_signature = str(st.session_state.get("runtime_setup_seed_signature") or "").strip()
    defaults = _connection_defaults(default_choice, current_connection=current_connection, fallback_provider=fallback_provider)
    if seed_signature != connection_signature:
        st.session_state["runtime_setup_provider_choice"] = default_choice
        st.session_state["runtime_setup_label"] = defaults["label"]
        st.session_state["runtime_setup_base_url"] = defaults["base_url"]
        st.session_state["runtime_setup_model"] = defaults["model"]
        st.session_state["runtime_setup_api_key"] = defaults["api_key"]
        st.session_state["runtime_setup_openrouter_http_referer"] = defaults["referer"]
        st.session_state["runtime_setup_openrouter_app_title"] = defaults["app_title"]
        st.session_state["runtime_setup_last_provider_choice"] = default_choice
        st.session_state["runtime_setup_seed_signature"] = connection_signature
        return
    defaults = _connection_defaults(selected_choice, current_connection=current_connection, fallback_provider=fallback_provider)
    st.session_state.setdefault("runtime_setup_provider_choice", selected_choice)
    st.session_state.setdefault("runtime_setup_label", defaults["label"])
    st.session_state.setdefault("runtime_setup_base_url", defaults["base_url"])
    st.session_state.setdefault("runtime_setup_model", defaults["model"])
    st.session_state.setdefault("runtime_setup_api_key", defaults["api_key"])
    st.session_state.setdefault("runtime_setup_openrouter_http_referer", defaults["referer"])
    st.session_state.setdefault("runtime_setup_openrouter_app_title", defaults["app_title"])
    st.session_state.setdefault("runtime_setup_last_provider_choice", selected_choice)


def _queue_form_updates(**updates: str) -> None:
    pending = dict(st.session_state.get("runtime_setup_pending_form_updates") or {})
    for key, value in updates.items():
        pending[key] = value
    st.session_state["runtime_setup_pending_form_updates"] = pending


def _apply_pending_form_updates() -> None:
    pending = dict(st.session_state.pop("runtime_setup_pending_form_updates", {}) or {})
    for key, value in pending.items():
        st.session_state[key] = value


def _clear_model_discovery_state() -> None:
    for key in (
        "runtime_setup_detected_models",
        "runtime_setup_detected_models_signature",
        "runtime_setup_detected_model_choice",
        "runtime_setup_model_detection_message",
        "runtime_setup_model_detection_error",
    ):
        st.session_state.pop(key, None)


def _model_discovery_signature(provider_choice: str, base_url: str) -> str:
    return f"{str(provider_choice or '').strip()}|{shell_services.sanitize_setup_base_url(provider_choice, base_url)}"


def _normalize_base_url_input() -> None:
    provider_choice = str(st.session_state.get("runtime_setup_provider_choice") or "").strip()
    current_value = str(st.session_state.get("runtime_setup_base_url") or "").strip()
    sanitized = shell_services.sanitize_setup_base_url(provider_choice, current_value)
    if sanitized != current_value:
        st.session_state["runtime_setup_base_url"] = sanitized
    current_signature = _model_discovery_signature(provider_choice, sanitized)
    if st.session_state.get("runtime_setup_detected_models_signature") != current_signature:
        _clear_model_discovery_state()


def _apply_detected_model_choice() -> None:
    selected_model = str(st.session_state.get("runtime_setup_detected_model_choice") or "").strip()
    if selected_model:
        st.session_state["runtime_setup_model"] = selected_model
        st.session_state["runtime_setup_last_autofilled_model"] = selected_model


def _sync_provider_dependent_fields(*, current_connection, fallback_provider: str = "") -> None:
    selected_choice = str(st.session_state.get("runtime_setup_provider_choice") or "").strip()
    previous_choice = str(
        st.session_state.get("runtime_setup_last_provider_choice")
        or _default_provider_choice(current_connection, fallback_provider)
    ).strip()
    previous_defaults = _connection_defaults(previous_choice, current_connection=current_connection, fallback_provider=fallback_provider)
    next_defaults = _connection_defaults(selected_choice, current_connection=current_connection, fallback_provider=fallback_provider)
    managed_fields = {
        "runtime_setup_label": "label",
        "runtime_setup_base_url": "base_url",
        "runtime_setup_model": "model",
        "runtime_setup_openrouter_http_referer": "referer",
        "runtime_setup_openrouter_app_title": "app_title",
    }
    for field_key, default_key in managed_fields.items():
        current_value = str(st.session_state.get(field_key) or "").strip()
        previous_value = str(previous_defaults[default_key] or "").strip()
        if not current_value or current_value == previous_value:
            st.session_state[field_key] = next_defaults[default_key]
    st.session_state["runtime_setup_last_provider_choice"] = selected_choice
    _clear_model_discovery_state()


page_title_locale = resolve_page_title_locale()
st.set_page_config(
    page_title=f"{APP_NAME} · {t('runtime_setup.title', locale=page_title_locale)}",
    page_icon="🧭",
    layout="wide",
)
state = hydrate_state_from_storage(set_current_page("runtime_setup"))

st.title(t("runtime_setup.title"))
st.caption(t("runtime_setup.body"))
render_shell_summary(state)

current_connection = active_connection(state.prefs)
_seed_form_state(current_connection=current_connection, fallback_provider=state.prefs.provider)
_apply_pending_form_updates()
if not needs_runtime_setup(state):
    st.caption(t("runtime_setup.current_connection_ready"))

setup_message = st.session_state.pop("runtime_setup_message", "")
setup_error = st.session_state.pop("runtime_setup_error", "")
test_message = st.session_state.pop("runtime_setup_test_message", "")
test_error = st.session_state.pop("runtime_setup_test_error", "")
if setup_message:
    st.success(setup_message)
if setup_error:
    st.error(setup_error)
if test_message:
    st.success(t("runtime_setup.last_test_passed"))
if test_error:
    st.warning(t("runtime_setup.last_test_failed"))

provider_options = list(PROVIDER_CHOICES)
default_provider_choice = _default_provider_choice(current_connection, state.prefs.provider)

with st.container(border=True):
    st.subheader(t("runtime_setup.section_whisper"))
    whisper_model = st.selectbox(
        t("runtime_setup.whisper_model"),
        options=list(DEFAULT_WHISPER_OPTIONS),
        index=_safe_index(list(DEFAULT_WHISPER_OPTIONS), state.prefs.whisper_model, default=_safe_index(list(DEFAULT_WHISPER_OPTIONS), "small")),
        key="runtime_setup_whisper_model",
    )
    st.caption(t("runtime_setup.whisper_guidance"))
    availability = whisper_model_status(whisper_model)
    if availability["cached"]:
        st.caption(t("runtime_setup.cache_ready", path=availability["cached_path"]))
    else:
        st.caption(t("runtime_setup.cache_missing"))
    download_status = st.empty()
    download_detail = st.empty()
    download_progress = st.empty()
    if st.button(t("runtime_setup.download_model"), key="runtime_setup_download_model", width="stretch"):
        progress_bar = download_progress.progress(0)

        def _update_download_progress(event: dict[str, object]) -> None:
            status = describe_whisper_download_event(event)
            download_status.info(str(status["headline"]))
            download_detail.caption(str(status["detail"]))
            progress_bar.progress(int(status["progress_percent"]))

        try:
            result = shell_services.download_whisper_model(whisper_model, progress_callback=_update_download_progress)
            st.session_state["runtime_setup_message"] = t("runtime_setup.whisper_ready", path=result["cached_path"])
        except Exception as exc:
            st.session_state["runtime_setup_error"] = t("runtime_setup.whisper_download_failed", detail=str(exc))
        st.rerun()

with st.container(border=True):
    st.subheader(t("runtime_setup.section_provider"))
    provider_choice = st.selectbox(
        t("runtime_setup.provider_label"),
        options=provider_options,
        index=_safe_index(provider_options, str(st.session_state.get("runtime_setup_provider_choice") or default_provider_choice)),
        format_func=_provider_label,
        key="runtime_setup_provider_choice",
        on_change=_sync_provider_dependent_fields,
        kwargs={"current_connection": current_connection, "fallback_provider": state.prefs.provider},
    )
    st.caption(_provider_hint(provider_choice))
    if provider_choice == "ollama_cloud":
        st.caption(t("runtime_setup.ollama_cloud_note"))
    elif provider_choice == "openai_compatible":
        st.caption(t("runtime_setup.openai_compatible_note"))

with st.container(border=True):
    st.subheader(t("runtime_setup.section_connection"))
    provider_defaults = _connection_defaults(
        provider_choice,
        current_connection=current_connection,
        fallback_provider=state.prefs.provider,
    )
    label = st.text_input(
        t("runtime_setup.connection_label"),
        key="runtime_setup_label",
        placeholder=provider_defaults["label"],
    )
    base_url = st.text_input(
        t("runtime_setup.base_url"),
        key="runtime_setup_base_url",
        placeholder=provider_defaults["base_url"] or t("runtime_setup.base_url_placeholder"),
        help=t("runtime_setup.base_url_help"),
        on_change=_normalize_base_url_input,
    )
    model = st.text_input(
        t("runtime_setup.model"),
        key="runtime_setup_model",
        placeholder=provider_defaults["model"] or DEFAULT_MODEL,
        help=_provider_hint(provider_choice),
    )
    if provider_choice in LOCAL_PROVIDER_CHOICES:
        detect_clicked = st.button(t("runtime_setup.detect_local_models"), key="runtime_setup_detect_local_models", width="stretch")
        if detect_clicked:
            try:
                with st.spinner(t("runtime_setup.detecting_local_models")):
                    detection = shell_services.discover_runtime_models(
                        provider=provider_choice,
                        provider_choice=provider_choice,
                        base_url=base_url.strip(),
                        api_key=str(st.session_state.get("runtime_setup_api_key") or "").strip(),
                        timeout_sec=5.0,
                    )
                detected_models = list(detection.get("models") or [])
                sanitized_base_url = shell_services.sanitize_setup_base_url(provider_choice, base_url.strip())
                st.session_state["runtime_setup_detected_models"] = detected_models
                st.session_state["runtime_setup_detected_models_signature"] = _model_discovery_signature(
                    provider_choice,
                    sanitized_base_url or base_url.strip(),
                )
                if detected_models:
                    current_model = str(st.session_state.get("runtime_setup_model") or "").strip()
                    last_autofilled = str(st.session_state.get("runtime_setup_last_autofilled_model") or "").strip()
                    queued_updates: dict[str, str] = {}
                    if sanitized_base_url and sanitized_base_url != base_url.strip():
                        queued_updates["runtime_setup_base_url"] = sanitized_base_url
                    if not current_model or current_model == provider_defaults["model"] or current_model == last_autofilled:
                        queued_updates["runtime_setup_model"] = detected_models[0]
                        st.session_state["runtime_setup_last_autofilled_model"] = detected_models[0]
                    selected_model = current_model if current_model in detected_models else queued_updates.get("runtime_setup_model", detected_models[0])
                    st.session_state["runtime_setup_detected_model_choice"] = selected_model
                    st.session_state["runtime_setup_model_detection_message"] = t(
                        "runtime_setup.detected_local_models_message",
                        count=len(detected_models),
                        endpoint=detection["health_endpoint"],
                    )
                    st.session_state.pop("runtime_setup_model_detection_error", None)
                    if queued_updates:
                        _queue_form_updates(**queued_updates)
                        st.rerun()
                else:
                    st.session_state["runtime_setup_model_detection_message"] = t(
                        "runtime_setup.detected_local_models_empty",
                        endpoint=detection["health_endpoint"],
                    )
                    st.session_state.pop("runtime_setup_model_detection_error", None)
            except Exception as exc:
                _clear_model_discovery_state()
                st.session_state["runtime_setup_model_detection_error"] = t(
                    "runtime_setup.detected_local_models_failed",
                    detail=str(exc),
                )
        current_signature = _model_discovery_signature(provider_choice, st.session_state.get("runtime_setup_base_url", ""))
        detected_models = (
            list(st.session_state.get("runtime_setup_detected_models") or [])
            if st.session_state.get("runtime_setup_detected_models_signature") == current_signature
            else []
        )
        model_detection_message = str(st.session_state.get("runtime_setup_model_detection_message") or "").strip()
        model_detection_error = str(st.session_state.get("runtime_setup_model_detection_error") or "").strip()
        if model_detection_message:
            st.info(model_detection_message)
        if model_detection_error:
            st.warning(model_detection_error)
        if detected_models:
            model_choice = st.selectbox(
                t("runtime_setup.detected_local_models_label"),
                options=detected_models,
                index=_safe_index(
                    detected_models,
                    str(st.session_state.get("runtime_setup_detected_model_choice") or st.session_state.get("runtime_setup_model") or detected_models[0]),
                ),
                key="runtime_setup_detected_model_choice",
                on_change=_apply_detected_model_choice,
            )
            st.caption(t("runtime_setup.detected_local_model_selected", model=model_choice))
        else:
            st.caption(t("runtime_setup.detected_local_models_help"))
    api_key = st.text_input(t("runtime_setup.api_key"), type="password", key="runtime_setup_api_key")
    openrouter_http_referer = st.text_input(
        t("runtime_setup.openrouter_http_referer"),
        disabled=provider_choice != "openrouter",
        key="runtime_setup_openrouter_http_referer",
    )
    openrouter_app_title = st.text_input(
        t("runtime_setup.openrouter_app_title"),
        disabled=provider_choice != "openrouter",
        key="runtime_setup_openrouter_app_title",
    )
    if provider_defaults["base_url"] or provider_defaults["model"]:
        suggested_parts = [
            t("runtime_setup.suggested_base_url", value=provider_defaults["base_url"])
            if provider_defaults["base_url"]
            else "",
            t("runtime_setup.suggested_model", value=provider_defaults["model"])
            if provider_defaults["model"]
            else "",
        ]
        st.caption(
            t("runtime_setup.suggested_defaults", values=", ".join(part for part in suggested_parts if part))
        )
    storage_status = secret_store_status()
    st.caption(
        t(
            "runtime_setup.secret_storage_note",
            backend=storage_status.backend_name,
            detail=storage_status.detail or t("runtime_setup.secret_storage_ok"),
        )
    )

with st.container(border=True):
    st.subheader(t("runtime_setup.section_actions"))
    st.caption(
        t("runtime_setup.actions_body")
    )
    if test_message:
        st.success(test_message)
    if test_error:
        st.warning(test_error)
    test_col, save_col, skip_col = st.columns(3)
    with test_col:
        test_clicked = st.button(t("runtime_setup.test_connection"), key="runtime_setup_test_connection", width="stretch")
    with save_col:
        save_clicked = st.button(t("runtime_setup.save_connection"), key="runtime_setup_save_connection", width="stretch")
    with skip_col:
        home_clicked = st.button(t("runtime_setup.back_home"), key="runtime_setup_back_home", width="stretch")

if home_clicked:
    go_to("streamlit_app.py", return_to="runtime_setup")

if test_clicked:
    try:
        sanitized_base_url = shell_services.sanitize_setup_base_url(provider_choice, base_url.strip())
        with st.spinner(t("runtime_setup.testing_connection")):
            result = test_runtime_connection(
                provider=provider_choice,
                provider_choice=provider_choice,
                model=model.strip(),
                base_url=sanitized_base_url,
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
        tested_model = str(test_payload.get("model") or model.strip() or provider_defaults["model"] or DEFAULT_MODEL).strip()
        st.session_state["runtime_setup_test_message"] = t(
            "runtime_setup.test_message",
            endpoint=result["health_endpoint"],
            base_url=result["base_url"],
            model=tested_model,
            preview=preview,
        )
    except Exception as exc:
        st.session_state["runtime_setup_last_test_status"] = f"failed: {exc}"
        st.session_state["runtime_setup_last_tested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        st.session_state["runtime_setup_test_error"] = t("runtime_setup.test_error", detail=str(exc))
    st.rerun()

if save_clicked:
    state = get_app_state()
    state.prefs.whisper_model = whisper_model
    sanitized_base_url = shell_services.sanitize_setup_base_url(provider_choice, base_url.strip())
    resolved_model = (
        model.strip()
        or str(st.session_state.get("runtime_setup_detected_model_choice") or "").strip()
        or provider_defaults["model"]
        or DEFAULT_MODEL
    )
    connection = build_provider_connection(
        provider_choice=provider_choice,
        label=label.strip(),
        model=resolved_model,
        base_url=sanitized_base_url,
        api_key=api_key.strip(),
        openrouter_http_referer=openrouter_http_referer.strip() or DEFAULT_OPENROUTER_HTTP_REFERER,
        openrouter_app_title=openrouter_app_title.strip() or DEFAULT_OPENROUTER_APP_TITLE,
        existing_connection=current_connection if current_connection and current_connection.connection_id == getattr(state.prefs, "active_connection_id", "") else None,
    )
    connection.last_test_status = str(st.session_state.get("runtime_setup_last_test_status") or "")
    connection.last_tested_at = str(st.session_state.get("runtime_setup_last_tested_at") or "")
    secret_status = save_provider_connection(state, connection, api_key=api_key.strip(), persist_draft=False)
    if secret_status.persistent:
        st.session_state["runtime_setup_message"] = t("runtime_setup.save_success")
    else:
        st.session_state["runtime_setup_message"] = t(
            "runtime_setup.save_success_session_only",
            backend=secret_status.backend_name,
        )
    st.rerun()

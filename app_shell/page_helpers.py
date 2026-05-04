from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitAPIException

from app_shell.bootstrap import PROJECT_ROOT, bootstrap_app_environment
from app_shell.i18n import t
from app_shell.services import hydrate_state_from_storage, load_workspace_prefs, resolve_log_dir
from app_shell.state import (
    APP_NAME,
    AppShellState,
    DEFAULT_UI_LOCALE,
    SUPPORTED_UI_LOCALES,
    get_app_state,
    set_current_page,
    set_return_to,
)
from app_shell.visual_system import inject_visual_system

BRANDING_DIR = PROJECT_ROOT / "branding"
BRAND_ICON_PATH = BRANDING_DIR / "vostavo-icon-playful.svg"
BRAND_LOGO_PATH = BRANDING_DIR / "vostavo-logo-playful.svg"


def resolve_page_title_locale(log_dir: str | os.PathLike[str] | None = None) -> str:
    bootstrap_app_environment(log_dir=log_dir)
    resolved_log_dir = resolve_log_dir(log_dir)
    prefs = load_workspace_prefs(resolved_log_dir)
    nested_log_dir = str(prefs.get("log_dir") or "").strip()
    if nested_log_dir:
        nested_resolved = resolve_log_dir(nested_log_dir)
        if nested_resolved != resolved_log_dir:
            nested_prefs = load_workspace_prefs(nested_resolved)
            nested_locale = str(nested_prefs.get("ui_locale") or "").strip().lower()
            if nested_locale in SUPPORTED_UI_LOCALES:
                return nested_locale
    locale = str(prefs.get("ui_locale") or "").strip().lower()
    if locale in SUPPORTED_UI_LOCALES:
        return locale
    return DEFAULT_UI_LOCALE


def configure_page(page_id: str, title_key: str, *, icon: str) -> AppShellState:
    state = get_app_state()
    bootstrap_app_environment(
        log_dir=getattr(getattr(state, "prefs", None), "log_dir", None),
        whisper_cache_dir=getattr(getattr(state, "prefs", None), "whisper_cache_dir", None),
    )
    page_title_locale = resolve_page_title_locale()
    st.set_page_config(
        page_title=f"{APP_NAME} · {t(title_key, locale=page_title_locale)}",
        page_icon=icon,
        layout="wide",
    )
    inject_visual_system()
    render_brand_mark()
    state = set_current_page(page_id)
    return hydrate_state_from_storage(state)


def render_brand_mark(*, width: int = 88) -> None:
    if BRAND_ICON_PATH.exists():
        st.sidebar.image(str(BRAND_ICON_PATH), width=width)


def render_brand_logo(*, width: int = 340) -> None:
    if BRAND_LOGO_PATH.exists():
        st.image(str(BRAND_LOGO_PATH), width=width)


def render_page_intro(title_key: str, body_key: str | None = None) -> None:
    st.title(t(title_key))
    if body_key:
        st.caption(t(body_key))


def render_shell_summary(state: AppShellState) -> None:
    st.caption(
        t(
            "common.session_summary",
            session_id=state.draft.session_id,
            ui_locale=state.prefs.ui_locale,
            learning_language=state.draft.learning_language_label,
            cefr=state.draft.cefr_level,
            speaker_id=state.draft.speaker_id or "—",
        )
    )


def render_startup_diagnostics(
    diagnostics: list[object],
    *,
    title_key: str = "diagnostics.title",
    body_key: str = "diagnostics.body",
    render_mode: str = "alerts",
) -> None:
    with st.container(border=True):
        st.subheader(t(title_key))
        st.caption(t(body_key))
        status_renderers = {
            "ok": st.success,
            "warning": st.warning,
            "error": st.error,
            "info": st.info,
        }
        checklist_icons = {
            "ok": "OK",
            "warning": "Warn",
            "error": "Fix",
            "info": "Info",
        }
        for item in diagnostics:
            status = str(getattr(item, "status", "info"))
            title = t(getattr(item, "title_key"))
            detail = t(getattr(item, "detail_key"), **dict(getattr(item, "detail_args", {}) or {}))
            target_page = diagnostic_target_page(item)
            action_label_key = diagnostic_action_label_key(item)
            if render_mode == "checklist":
                icon = checklist_icons.get(status, checklist_icons["info"])
                st.markdown(f"**[{icon}] {title}**  \n{detail}")
                continue
            render = status_renderers.get(status, st.info)
            render(f"{title}: {detail}")
            if target_page and action_label_key and st.button(
                t(action_label_key),
                key=f"diagnostic::{getattr(item, 'key', 'item')}",
            ):
                go_to(target_page)


def diagnostic_target_page(item: object) -> str | None:
    detail_args = dict(getattr(item, "detail_args", {}) or {})
    target_page = str(detail_args.get("target_page") or "").strip()
    return target_page or None


def diagnostic_action_label_key(item: object) -> str | None:
    target_page = diagnostic_target_page(item)
    if not target_page:
        return None
    detail_args = dict(getattr(item, "detail_args", {}) or {})
    action_label_key = str(detail_args.get("action_label_key") or "diagnostics.maintenance_open_settings").strip()
    return action_label_key or None


def format_byte_count(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "unknown size"
    value = float(num_bytes)
    if value < 0:
        return "unknown size"
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


STORAGE_AREA_ORDER = ("tmp", "jobs", "logs", "reports", "recordings", "uploads", "cache")


def _summary_value(summary: object, key: str, default: object = "") -> object:
    if isinstance(summary, dict):
        return summary.get(key, default)
    return getattr(summary, key, default)


def storage_area_rows(storage_summary: object) -> list[dict[str, object]]:
    if isinstance(storage_summary, dict):
        raw_areas = storage_summary.get("areas", {})
    else:
        raw_areas = getattr(storage_summary, "areas", {})
    areas = dict(raw_areas or {})
    ordered_keys = [key for key in STORAGE_AREA_ORDER if key in areas]
    ordered_keys.extend(sorted(key for key in areas if key not in set(ordered_keys)))
    rows: list[dict[str, object]] = []
    for area_key in ordered_keys:
        summary = areas[area_key]
        file_count = int(_summary_value(summary, "file_count", 0) or 0)
        size_bytes = int(_summary_value(summary, "size_bytes", 0) or 0)
        rows.append(
            {
                "area": area_key,
                "label_key": f"settings.storage_area_{area_key}",
                "path": str(_summary_value(summary, "path", "") or ""),
                "file_count": file_count,
                "size_bytes": size_bytes,
                "size_label": format_byte_count(size_bytes),
            }
        )
    return rows


def describe_whisper_download_event(event: dict[str, object]) -> dict[str, object]:
    stage = str(event.get("stage") or "")
    current_file = Path(str(event.get("current_file") or "")).name
    downloaded_bytes = int(event.get("downloaded_bytes") or 0)
    total_bytes = int(event.get("total_bytes") or 0)
    completed_files = int(event.get("completed_files") or 0)
    total_files = int(event.get("total_files") or 0)
    progress_percent = 0
    if total_bytes > 0:
        progress_percent = max(0, min(100, int(downloaded_bytes * 100 / total_bytes)))
    elif stage in {"finalizing", "ready"}:
        progress_percent = 100

    if stage == "checking_cache":
        return {
            "headline": t("runtime_setup.download_status.checking_cache_headline"),
            "detail": t("runtime_setup.download_status.checking_cache_detail"),
            "progress_percent": 0,
        }
    if stage == "starting_download":
        pending_files = int(event.get("pending_files") or 0)
        pending_bytes = int(event.get("pending_bytes") or 0)
        detail = (
            t(
                "runtime_setup.download_status.starting_download_detail",
                pending_files=pending_files,
                pending_bytes=format_byte_count(pending_bytes),
            )
            if pending_bytes > 0
            else t(
                "runtime_setup.download_status.starting_download_detail_files_only",
                pending_files=pending_files,
            )
        )
        return {
            "headline": t("runtime_setup.download_status.starting_download_headline"),
            "detail": detail,
            "progress_percent": progress_percent,
        }
    if stage == "downloading":
        detail = (
            t(
                "runtime_setup.download_status.downloading_detail",
                downloaded_bytes=format_byte_count(downloaded_bytes),
                total_bytes=format_byte_count(total_bytes),
                completed_files=completed_files,
                total_files=total_files,
            )
            if total_files > 0
            else t(
                "runtime_setup.download_status.downloading_detail_bytes_only",
                downloaded_bytes=format_byte_count(downloaded_bytes),
                total_bytes=format_byte_count(total_bytes),
            )
        )
        return {
            "headline": t(
                "runtime_setup.download_status.downloading_headline",
                current_file=current_file or t("runtime_setup.download_status.model_files_fallback"),
            ),
            "detail": detail,
            "progress_percent": progress_percent,
        }
    if stage == "finalizing":
        return {
            "headline": t("runtime_setup.download_status.finalizing_headline"),
            "detail": t("runtime_setup.download_status.finalizing_detail"),
            "progress_percent": 100,
        }
    if stage == "ready":
        return {
            "headline": t("runtime_setup.download_status.ready_headline"),
            "detail": str(event.get("cached_path") or "").strip() or t("runtime_setup.download_status.ready_detail"),
            "progress_percent": 100,
        }
    return {
        "headline": t("runtime_setup.download_status.default_headline"),
        "detail": t("runtime_setup.download_status.default_detail"),
        "progress_percent": progress_percent,
    }


def go_to(page_path: str, *, return_to: str | None = None) -> None:
    state = get_app_state()
    set_return_to(return_to or state.nav.current_page)
    if os.environ.get("APP_SHELL_SKIP_BOOTSTRAP") == "1":
        st.session_state["_next_page"] = page_path
        st.stop()
    candidates = [page_path]
    if page_path.startswith("pages/"):
        candidates.append(page_path.split("/", 1)[1])
    elif page_path.endswith(".py") and page_path != "streamlit_app.py":
        candidates.append(f"pages/{page_path}")
    last_error: StreamlitAPIException | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            st.switch_page(candidate)
            return
        except StreamlitAPIException as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def render_guard(message_key: str, action_label_key: str, target_page: str) -> None:
    st.warning(t(message_key))
    if st.button(t(action_label_key), key=f"guard::{target_page}"):
        go_to(target_page)
    st.stop()

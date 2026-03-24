from __future__ import annotations

from statistics import mean

import pandas as pd
import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import configure_page, render_page_intro, render_shell_summary
from app_shell.review_components import render_report_panels
from app_shell.services import load_history_records, load_report_payload, review_summary
from assessment_runtime.progress_analysis import format_top_counts, latest_priorities, task_family_progress

RECENT_HISTORY_JUMP_COUNT = 4
ALL_HISTORY_LANGUAGES = "__all__"


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_family_label(value: object) -> object:
    if pd.isna(value):
        return value
    raw = str(value or "").strip()
    if not raw:
        return value
    localized = t(f"task_family.{raw}")
    if isinstance(localized, str) and localized.startswith("[") and localized.endswith("]"):
        return raw.replace("_", " ")
    return localized


def _speaker_scoped_records(records: list[object], speaker_id: str) -> list[object]:
    if not speaker_id:
        return list(records)
    return [record for record in records if getattr(record, "speaker_id", "") == speaker_id]


def _normalized_language(value: object) -> str:
    return str(value or "").strip().lower()


def _history_language_options(records: list[object]) -> list[str]:
    return sorted({_normalized_language(getattr(record, "learning_language", "")) for record in records if _normalized_language(getattr(record, "learning_language", ""))})


def _history_language_label(value: str) -> str:
    if value == ALL_HISTORY_LANGUAGES:
        return t("history.language_filter_all")
    if not value:
        return t("history.none")
    return value.upper()


def _language_scoped_records(records: list[object], language_code: str) -> list[object]:
    if not language_code or language_code == ALL_HISTORY_LANGUAGES:
        return list(records)
    return [record for record in records if _normalized_language(getattr(record, "learning_language", "")) == language_code]


def _history_summary(records: list[object]) -> dict[str, object]:
    final_values = [
        value
        for record in records
        if (value := _safe_float(getattr(record, "final_score", None))) is not None
    ]
    wpm_values = [
        value
        for record in records
        if (value := _safe_float(getattr(record, "wpm", None))) is not None
    ]
    band_values = [
        value
        for record in records
        if (value := _safe_int(getattr(record, "band", None))) is not None
    ]
    latest = records[-1] if records else None
    return {
        "count": len(records),
        "avg_final": round(mean(final_values), 2) if final_values else None,
        "best_final": round(max(final_values), 2) if final_values else None,
        "avg_wpm": round(mean(wpm_values), 1) if wpm_values else None,
        "best_band": max(band_values) if band_values else None,
        "latest": latest,
    }


def _trend_frame(records: list[object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "timestamp": getattr(record, "timestamp", None),
                "final_score": getattr(record, "final_score", None),
                "overall": getattr(record, "overall", None),
                "wpm": getattr(record, "wpm", None),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame()
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return pd.DataFrame()
    for column in ("final_score", "overall", "wpm"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index("timestamp")


def _task_family_frame(records: list[object]) -> pd.DataFrame:
    rows = []
    for row in task_family_progress(records):
        rows.append(
            {
                t("history.task_family_name"): _task_family_label(row["task_family"]),
                t("history.task_family_runs"): row["count"],
                t("history.task_family_avg_final"): row["avg_final"],
                t("history.task_family_latest_final"): row["latest_final"],
                t("history.task_family_grammar"): format_top_counts(row["grammar_counts"]),
                t("history.task_family_coherence"): format_top_counts(row["coherence_counts"]),
            }
        )
    return pd.DataFrame(rows)


def _attempts_frame(records: list[object]) -> pd.DataFrame:
    rows = []
    for record in reversed(records):
        rows.append(
            {
                t("history.table_timestamp"): getattr(record, "timestamp", None),
                t("history.table_session"): getattr(record, "session_id", ""),
                t("history.table_speaker"): getattr(record, "speaker_id", ""),
                t("history.table_language"): _history_language_label(_normalized_language(getattr(record, "learning_language", ""))),
                t("history.table_theme"): getattr(record, "theme", ""),
                t("history.table_task_family"): _task_family_label(getattr(record, "task_family", "")),
                t("history.table_score"): getattr(record, "final_score", None),
                t("history.table_band"): getattr(record, "band", None),
            }
        )
    return pd.DataFrame(rows)


def _list_or_placeholder(values: list[str]) -> str:
    cleaned = [str(value) for value in values if str(value).strip()]
    return ", ".join(cleaned) if cleaned else t("history.none")


def _detail_record_options(records: list[object]) -> list[object]:
    return [record for record in reversed(records) if getattr(record, "report_path", "")]


def _detail_option_label(record: object) -> str:
    timestamp = getattr(record, "timestamp", None)
    if hasattr(timestamp, "strftime"):
        timestamp_label = timestamp.strftime("%Y-%m-%d %H:%M")
    else:
        timestamp_label = str(timestamp or "-")
    language = _history_language_label(_normalized_language(getattr(record, "learning_language", "")))
    theme = str(getattr(record, "theme", "") or "-")
    family = str(_task_family_label(getattr(record, "task_family", "")) or "-")
    return f"{timestamp_label} · {language} · {theme} · {family}"


def _recent_jump_label(record: object) -> str:
    timestamp = getattr(record, "timestamp", None)
    if hasattr(timestamp, "strftime"):
        timestamp_label = timestamp.strftime("%m-%d %H:%M")
    else:
        timestamp_label = str(timestamp or "-")
    language = _history_language_label(_normalized_language(getattr(record, "learning_language", "")))
    family = str(_task_family_label(getattr(record, "task_family", "")) or "-")
    return f"{language} · {timestamp_label} · {family}"


def _set_history_detail_report(report_path: str) -> None:
    st.session_state["history_detail_report"] = report_path


state = configure_page("history", "nav.history", icon="📊")

render_page_intro("history.title", "history.body")
render_shell_summary(state)

speaker_scope = str(state.draft.speaker_id or "").strip()

try:
    raw_records = _speaker_scoped_records(load_history_records(state.prefs.log_dir), speaker_scope)
    history_error = ""
except Exception:  # pragma: no cover - I/O boundary
    raw_records = []
    history_error = t("history.error_loading")

if history_error:
    with st.container(border=True):
        st.error(history_error)
elif not raw_records:
    with st.container(border=True):
        st.info(t("history.empty"))
else:
    available_languages = _history_language_options(raw_records)
    selected_language = ALL_HISTORY_LANGUAGES
    if available_languages:
        preferred_language = _normalized_language(getattr(state.draft, "learning_language", ""))
        default_index = 0
        if preferred_language and preferred_language in available_languages:
            default_index = available_languages.index(preferred_language) + 1
        with st.container(border=True):
            selected_language = str(
                st.selectbox(
                    t("history.language_filter"),
                    options=[ALL_HISTORY_LANGUAGES, *available_languages],
                    format_func=_history_language_label,
                    index=default_index,
                    key="history_learning_language",
                )
            )

    records = _language_scoped_records(raw_records, selected_language)
    if not records:
        with st.container(border=True):
            st.info(t("history.empty_filtered"))
        st.stop()

    summary = _history_summary(records)
    trend_frame = _trend_frame(records)
    family_frame = _task_family_frame(records)
    priorities = latest_priorities(records)

    with st.container(border=True):
        if speaker_scope and selected_language != ALL_HISTORY_LANGUAGES:
            st.caption(
                t(
                    "history.scope_current_speaker_language",
                    speaker=speaker_scope,
                    language=_history_language_label(selected_language),
                    count=summary["count"],
                )
            )
        elif speaker_scope:
            st.caption(t("history.scope_current_speaker", speaker=speaker_scope, count=summary["count"]))
        elif selected_language != ALL_HISTORY_LANGUAGES:
            st.caption(
                t(
                    "history.scope_all_language",
                    language=_history_language_label(selected_language),
                    count=summary["count"],
                )
            )
        else:
            st.caption(t("history.scope_all", count=summary["count"]))
        metric_cols = st.columns(5)
        metric_cols[0].metric(t("history.metric_runs"), str(summary["count"]))
        metric_cols[1].metric(
            t("history.metric_avg_final"),
            f"{summary['avg_final']:.2f}" if isinstance(summary.get("avg_final"), (int, float)) else t("history.none"),
        )
        metric_cols[2].metric(
            t("history.metric_best_final"),
            f"{summary['best_final']:.2f}" if isinstance(summary.get("best_final"), (int, float)) else t("history.none"),
        )
        metric_cols[3].metric(
            t("history.metric_avg_wpm"),
            f"{summary['avg_wpm']:.1f}" if isinstance(summary.get("avg_wpm"), (int, float)) else t("history.none"),
        )
        metric_cols[4].metric(
            t("history.metric_best_band"),
            str(summary["best_band"]) if summary.get("best_band") is not None else t("history.none"),
        )

    with st.container(border=True):
        st.subheader(t("history.trends_title"))
        if len(trend_frame.index) >= 2:
            score_columns = [column for column in ("final_score", "overall") if column in trend_frame.columns and trend_frame[column].notna().any()]
            if score_columns:
                st.caption(t("history.score_chart_title"))
                st.line_chart(trend_frame[score_columns], width="stretch")
            if "wpm" in trend_frame.columns and trend_frame["wpm"].notna().any():
                st.caption(t("history.pace_chart_title"))
                st.line_chart(trend_frame[["wpm"]], width="stretch")
        else:
            st.info(t("history.trends_not_enough"))

        priority_cols = st.columns(3)
        with priority_cols[0]:
            st.caption(t("history.priority_latest"))
            st.write(_list_or_placeholder(priorities.get("latest", [])))
        with priority_cols[1]:
            st.caption(t("history.priority_new"))
            st.write(_list_or_placeholder(priorities.get("new", [])))
        with priority_cols[2]:
            st.caption(t("history.priority_resolved"))
            st.write(_list_or_placeholder(priorities.get("resolved", [])))

    with st.container(border=True):
        st.subheader(t("history.task_family_title"))
        if family_frame.empty:
            st.info(t("history.task_family_empty"))
        else:
            st.dataframe(family_frame, width="stretch", hide_index=True)

    selected_report_path = ""
    with st.container(border=True):
        st.subheader(t("history.attempts_title"))
        detail_records = _detail_record_options(records)
        if detail_records:
            st.caption(t("history.attempts_jump_hint"))
            jump_records = detail_records[:RECENT_HISTORY_JUMP_COUNT]
            jump_cols = st.columns(len(jump_records))
            for idx, record in enumerate(jump_records):
                report_path = str(getattr(record, "report_path", ""))
                jump_cols[idx].button(
                    _recent_jump_label(record),
                    key=f"history_jump_{idx}",
                    width="stretch",
                    on_click=_set_history_detail_report,
                    args=(report_path,),
                )
            selected_report_path = str(
                st.selectbox(
                t("history.details_select"),
                options=[str(getattr(record, "report_path", "")) for record in detail_records],
                format_func=lambda report_path: _detail_option_label(
                    next(
                        record
                        for record in detail_records
                        if str(getattr(record, "report_path", "")) == report_path
                    )
                ),
                key="history_detail_report",
            )
            )
        st.dataframe(_attempts_frame(records), width="stretch", hide_index=True)

    with st.container(border=True):
        st.subheader(t("history.details_title"))
        if not detail_records:
            st.info(t("history.details_unavailable"))
        else:
            if selected_report_path not in {
                str(getattr(record, "report_path", "")) for record in detail_records
            }:
                selected_report_path = str(getattr(detail_records[0], "report_path", ""))
                st.session_state["history_detail_report"] = selected_report_path
            selected_record = next(
                record for record in detail_records if str(getattr(record, "report_path", "")) == selected_report_path
            )
            payload = load_report_payload(selected_report_path)
            if payload is None:
                st.error(t("history.details_error"))
            else:
                st.caption(
                    t(
                        "history.details_caption",
                        session=getattr(selected_record, "session_id", "") or "-",
                        language=_history_language_label(_normalized_language(getattr(selected_record, "learning_language", ""))),
                        theme=getattr(selected_record, "theme", "") or "-",
                    )
                )
                summary = review_summary(payload)
                render_report_panels(
                    summary,
                    transcript=str(summary.get("transcript") or ""),
                    notes=str(summary.get("notes") or ""),
                    key_prefix="history",
                )

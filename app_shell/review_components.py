from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import go_to
from app_shell.visual_system import render_kicker


@dataclass(frozen=True)
class ReportStatusSummary:
    level: str
    message: str
    short_label: str


def _gate_label(gate_key: str) -> str:
    mapping = {
        "language_pass": "review.gate_language",
        "topic_pass": "review.gate_theme",
        "duration_pass": "review.gate_duration",
        "min_words_pass": "review.gate_words",
    }
    return t(mapping[gate_key]) if gate_key in mapping else gate_key.replace("_", " ")


def _mode_label(mode: str) -> str:
    if mode == "hybrid":
        return t("review.mode_hybrid")
    if mode == "deterministic_only":
        return t("review.mode_deterministic_only")
    return t("review.mode_unknown")


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _warning_message(value: str) -> str:
    warning_key = str(value or "").strip()
    if not warning_key:
        return ""
    mapping = {
        "coaching_unavailable": "review.warning_codes.coaching_unavailable",
        "llm_unavailable": "review.warning_codes.llm_unavailable",
        "llm_skipped_low_word_count": "review.warning_codes.llm_skipped_low_word_count",
        "llm_invalid_schema": "review.warning_codes.llm_invalid_schema",
        "asr_pause_mismatch": "review.warning_codes.asr_pause_mismatch",
    }
    locale_key = mapping.get(warning_key)
    if locale_key:
        return t(locale_key)
    return warning_key.replace("_", " ")


def _gate_status(value: bool | None) -> str:
    if value is True:
        return t("review.gate_pass")
    if value is False:
        return t("review.gate_fail")
    return t("review.gate_unknown")


def _localized_warning_messages(summary: dict) -> list[str]:
    messages: list[str] = []
    for item in _as_text_list(summary.get("warnings")):
        message = _warning_message(item)
        if message:
            messages.append(message)
    return messages


def report_status_summary(summary: dict) -> ReportStatusSummary:
    failed_gates = summary.get("failed_gates") or []
    warning_messages = _localized_warning_messages(summary)
    is_done = not summary.get("requires_human_review") and not failed_gates
    if summary.get("requires_human_review"):
        level = "warning"
        messages = [t("review.status_review")]
        short_label = t("review.status_short_review")
    elif failed_gates:
        labels = ", ".join(_gate_label(item) for item in failed_gates)
        level = "info"
        messages = [t("review.status_unstable", gates=labels)]
        short_label = t("review.status_short_unstable")
    else:
        level = "success"
        messages = [t("review.status_done")]
        short_label = t("review.status_short_done")
    if warning_messages:
        level = "warning"
        messages.append(t("review.warnings", value=" ".join(warning_messages)))
        if is_done:
            short_label = t("review.status_short_unstable")
    return ReportStatusSummary(level=level, message=" ".join(messages), short_label=short_label)


def _report_status_banner(summary: dict) -> tuple[str, str]:
    status = report_status_summary(summary)
    return status.level, status.message


def render_report_status(summary: dict) -> None:
    level, message = _report_status_banner(summary)
    renderers = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
    }
    renderers.get(level, st.info)(message)


def _answer_result_text(summary: dict) -> str:
    score = summary.get("score_overall")
    if summary.get("requires_human_review"):
        return t("review.answer_result_review")
    if isinstance(score, (int, float)):
        return t(
            "review.answer_result_value",
            score=f"{score:.1f}",
            band=str(summary.get("band") or "-"),
        )
    return t("review.answer_result_pending")


def _answer_why_text(summary: dict) -> str:
    failed_gates = summary.get("failed_gates") or []
    strengths = _as_text_list(summary.get("strengths"))
    if failed_gates:
        return t("review.answer_why_gates", value=", ".join(_gate_label(item) for item in failed_gates))
    if strengths:
        return t("review.answer_why_strength", value=strengths[0])
    return str(summary.get("coach_summary") or t("review.answer_why_placeholder"))


def _answer_next_text(summary: dict) -> str:
    next_focus = str(summary.get("next_focus") or "").strip()
    next_exercise = str(summary.get("next_exercise") or "").strip()
    priorities = _as_text_list(summary.get("priorities"))
    if next_focus:
        return t("review.answer_next_focus", value=next_focus)
    if next_exercise:
        return t("review.answer_next_exercise", value=next_exercise)
    if priorities:
        return t("review.answer_next_priority", value=priorities[0])
    return t("review.answer_next_placeholder")


def _render_progress_items(summary: dict) -> None:
    items = summary.get("progress_items") or []
    rendered_items: list[str] = []
    for item in items:
        kind = item.get("kind")
        value = item.get("value")
        if value is None:
            continue
        if kind == "previous_session":
            rendered_items.append(t("review.progress_previous_session", value=value))
        elif kind == "delta_final":
            if isinstance(value, (int, float)):
                rendered_items.append(t("review.progress_delta_final", value=f"{value:+.2f}"))
        elif kind == "delta_overall":
            if isinstance(value, (int, float)):
                rendered_items.append(t("review.progress_delta_overall", value=f"{value:+.2f}"))
        elif kind == "delta_wpm":
            if isinstance(value, (int, float)):
                rendered_items.append(t("review.progress_delta_wpm", value=f"{value:+.2f}"))
        elif kind == "new_priorities":
            if isinstance(value, list):
                rendered_items.append(t("review.progress_new_priorities", value=", ".join(str(item) for item in value)))
        elif kind == "resolved_priorities":
            if isinstance(value, list):
                rendered_items.append(t("review.progress_resolved_priorities", value=", ".join(str(item) for item in value)))
        elif kind == "repeating_grammar":
            if isinstance(value, list):
                rendered_items.append(t("review.progress_repeating_grammar", value=", ".join(str(item) for item in value)))
        elif kind == "repeating_coherence":
            if isinstance(value, list):
                rendered_items.append(t("review.progress_repeating_coherence", value=", ".join(str(item) for item in value)))
    if not rendered_items:
        st.caption(t("review.progress_unavailable"))
        return
    st.subheader(t("review.progress_title"))
    for item in rendered_items:
        st.markdown(f"- {item}")


def _localized_cefr_descriptor(level: str) -> str:
    normalized = str(level or "").strip().lower()
    if not normalized:
        return ""
    key = f"review.baseline_descriptors.{normalized}"
    translated = t(key)
    return "" if translated == f"[{key}]" else translated


def _localized_language_label(language_code: str) -> str:
    normalized = str(language_code or "").strip().lower()
    if not normalized:
        return ""
    key = f"locale.{normalized}"
    translated = t(key)
    return normalized.upper() if translated == f"[{key}]" else translated


def _baseline_caption(summary: dict, baseline: dict) -> str:
    level = str(baseline.get("level") or "").strip().upper()
    descriptor = _localized_cefr_descriptor(level)
    language_label = _localized_language_label(str(summary.get("learning_language") or ""))
    if language_label and descriptor:
        return t("review.baseline_target_caption", level=level, language=language_label, comment=descriptor)
    if language_label:
        return t("review.baseline_target_caption_no_comment", level=level, language=language_label)
    if descriptor:
        return t("review.baseline_level_caption", level=level, comment=descriptor)
    return t("review.baseline_level_caption_no_comment", level=level)


def _render_baseline(summary: dict) -> None:
    baseline = summary.get("baseline")
    if not isinstance(baseline, dict):
        return
    st.subheader(t("review.baseline_title"))
    st.caption(_baseline_caption(summary, baseline))
    rows = []
    for metric, entry in (baseline.get("targets") or {}).items():
        rows.append(
            {
                t("review.baseline_metric"): metric,
                t("review.baseline_expected"): entry.get("expected"),
                t("review.baseline_actual"): entry.get("actual"),
                t("review.baseline_ok"): t("review.gate_pass" if entry.get("ok") else "review.gate_fail"),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_report_panels(summary: dict, *, transcript: str = "", notes: str = "", key_prefix: str = "review") -> None:
    with st.container(border=True):
        render_kicker(t("review.answers_title"))
        st.subheader(t("review.answers_title"))
        render_report_status(summary)
        answer_cols = st.columns(3)
        with answer_cols[0]:
            st.subheader(t("review.answer_result_title"))
            st.write(_answer_result_text(summary))
        with answer_cols[1]:
            st.subheader(t("review.answer_why_title"))
            st.write(_answer_why_text(summary))
        with answer_cols[2]:
            st.subheader(t("review.answer_next_title"))
            st.write(_answer_next_text(summary))

        score = summary.get("score_overall")
        metric_cols = st.columns(4)
        metric_cols[0].metric(t("review.score"), f"{score:.1f}" if isinstance(score, (int, float)) else "-")
        metric_cols[1].metric(t("review.band"), str(summary.get("band") or "-"))
        metric_cols[2].metric(
            t("review.metric_deterministic"),
            f"{summary.get('deterministic_score'):.1f}" if isinstance(summary.get("deterministic_score"), (int, float)) else "-",
        )
        metric_cols[3].metric(
            t("review.metric_llm"),
            f"{summary.get('llm_score'):.1f}" if isinstance(summary.get("llm_score"), (int, float)) else "-",
        )
        st.caption(t("review.mode_caption", value=_mode_label(str(summary.get("mode") or ""))))
        if st.button(t("common.scoring_guide"), key=f"{key_prefix}_open_scoring_guide"):
            go_to("pages/07_Scoring_Guide.py")

    with st.container(border=True):
        render_kicker(t("review.coaching_tab"))
        st.subheader(t("review.summary_title"))
        st.write(summary.get("coach_summary") or t("review.summary_placeholder"))
        columns = st.columns(2)
        with columns[0]:
            st.caption(t("review.strengths_title"))
            strengths = summary.get("strengths") or []
            if strengths:
                for item in strengths:
                    st.markdown(f"- {item}")
            else:
                st.write(t("review.no_strengths"))
        with columns[1]:
            st.caption(t("review.priorities_title"))
            priorities = summary.get("priorities") or []
            if priorities:
                for item in priorities:
                    st.markdown(f"- {item}")
            else:
                st.write(t("review.no_priorities"))
        if isinstance(summary.get("next_focus"), str) and summary["next_focus"].strip():
            st.info(t("review.next_focus", value=summary["next_focus"]))
        if isinstance(summary.get("next_exercise"), str) and summary["next_exercise"].strip():
            st.info(t("review.next_exercise", value=summary["next_exercise"]))
        issue_cols = st.columns(2)
        with issue_cols[0]:
            st.caption(t("review.recurring_grammar_title"))
            recurring_grammar = _as_text_list(summary.get("recurring_grammar"))
            st.write(", ".join(recurring_grammar) if recurring_grammar else t("review.none"))
        with issue_cols[1]:
            st.caption(t("review.recurring_coherence_title"))
            recurring_coherence = _as_text_list(summary.get("recurring_coherence"))
            st.write(", ".join(recurring_coherence) if recurring_coherence else t("review.none"))
        _render_progress_items(summary)
        _render_baseline(summary)

    with st.container(border=True):
        render_kicker(t("review.gates_title"))
        st.subheader(t("review.gates_title"))
        gates = summary.get("gates") or {}
        gate_cols = st.columns(4)
        gate_cols[0].metric(t("review.gate_language"), _gate_status(gates.get("language_pass")))
        gate_cols[1].metric(t("review.gate_theme"), _gate_status(gates.get("topic_pass")))
        gate_cols[2].metric(t("review.gate_duration"), _gate_status(gates.get("duration_pass")))
        gate_cols[3].metric(t("review.gate_words"), _gate_status(gates.get("min_words_pass")))

    with st.container(border=True):
        render_kicker(t("review.details_tab"))
        evidence_cols = st.columns([0.42, 0.58], gap="large")
        with evidence_cols[0]:
            st.subheader(t("review.label_title"))
            st.text_input(
                t("review.label_title"),
                value=str(summary.get("label") or "").strip() or t("review.label_placeholder"),
                key=f"{key_prefix}_saved_label_view",
                disabled=True,
                label_visibility="collapsed",
            )
            st.subheader(t("review.notes_title"))
            st.text_area(
                t("review.notes_title"),
                value=notes or t("review.notes_placeholder"),
                key=f"{key_prefix}_saved_notes_view",
                height=110,
                disabled=True,
                label_visibility="collapsed",
            )
            st.subheader(t("review.details_title"))
            detail_cols = st.columns(2)
            detail_cols[0].metric(t("review.metric_report_id"), str(summary.get("report_id") or "-"))
            detail_cols[1].metric(t("review.metric_mode"), _mode_label(str(summary.get("mode") or "")))
            with st.expander(t("review.raw_payload"), expanded=False):
                st.json(summary.get("payload") or {})
        with evidence_cols[1]:
            st.subheader(t("review.transcript_title"))
            st.text_area(
                t("review.transcript_title"),
                value=transcript or t("review.transcript_placeholder"),
                key=f"{key_prefix}_transcript_view",
                height=360,
                disabled=True,
                label_visibility="collapsed",
            )

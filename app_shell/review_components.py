from __future__ import annotations

import pandas as pd
import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import go_to
from app_shell.visual_system import render_kicker


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


def _gate_status(value: bool | None) -> str:
    if value is True:
        return t("review.gate_pass")
    if value is False:
        return t("review.gate_fail")
    return t("review.gate_unknown")


def render_report_status(summary: dict) -> None:
    if summary.get("requires_human_review"):
        st.warning(t("review.status_review"))
        return
    failed_gates = summary.get("failed_gates") or []
    if failed_gates:
        labels = ", ".join(_gate_label(item) for item in failed_gates)
        st.info(t("review.status_unstable", gates=labels))
        return
    st.success(t("review.status_done"))


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


def _render_baseline(summary: dict) -> None:
    baseline = summary.get("baseline")
    if not isinstance(baseline, dict):
        return
    st.subheader(t("review.baseline_title"))
    st.caption(t("review.baseline_caption", level=baseline.get("level", "-"), comment=baseline.get("comment", "")))
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
    render_report_status(summary)

    with st.container(border=True):
        render_kicker(t("review.summary_title"))
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
        render_kicker(t("review.gates_title"))
        st.subheader(t("review.gates_title"))
        gates = summary.get("gates") or {}
        gate_cols = st.columns(4)
        gate_cols[0].metric(t("review.gate_language"), _gate_status(gates.get("language_pass")))
        gate_cols[1].metric(t("review.gate_theme"), _gate_status(gates.get("topic_pass")))
        gate_cols[2].metric(t("review.gate_duration"), _gate_status(gates.get("duration_pass")))
        gate_cols[3].metric(t("review.gate_words"), _gate_status(gates.get("min_words_pass")))

    coaching_col, transcript_col = st.columns([1.06, 0.94], gap="large")

    with coaching_col:
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
            warnings = _as_text_list(summary.get("warnings"))
            if warnings:
                st.warning(t("review.warnings", value=", ".join(warnings)))
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

    with transcript_col:
        with st.container(border=True):
            render_kicker(t("review.transcript_tab"))
            st.subheader(t("review.notes_title"))
            st.text_area(
                t("review.notes_title"),
                value=notes or t("review.notes_placeholder"),
                key=f"{key_prefix}_saved_notes_view",
                height=110,
                disabled=True,
                label_visibility="collapsed",
            )
            st.subheader(t("review.transcript_title"))
            st.text_area(
                t("review.transcript_title"),
                value=transcript or t("review.transcript_placeholder"),
                key=f"{key_prefix}_transcript_view",
                height=320,
                disabled=True,
                label_visibility="collapsed",
            )

        with st.container(border=True):
            render_kicker(t("review.details_tab"))
            st.subheader(t("review.details_title"))
            detail_cols = st.columns(2)
            detail_cols[0].metric(t("review.metric_report_id"), str(summary.get("report_id") or "-"))
            detail_cols[1].metric(t("review.metric_mode"), _mode_label(str(summary.get("mode") or "")))
            with st.expander(t("review.raw_payload"), expanded=False):
                st.json(summary.get("payload") or {})

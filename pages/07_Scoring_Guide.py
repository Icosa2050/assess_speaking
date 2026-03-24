from __future__ import annotations

import pandas as pd
import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import configure_page, render_page_intro, render_shell_summary
from app_shell.scoring_guide import build_scoring_guide_data

state = configure_page("guide", "nav.guide", icon="ℹ️")

render_page_intro("guide.title", "guide.body")
render_shell_summary(state)

guide = build_scoring_guide_data()
formula = guide["formula"]

with st.container(border=True):
    st.subheader(t("guide.summary_title"))
    st.write(t("guide.summary_body"))
    summary_cols = st.columns(5)
    summary_cards = [
        ("guide.card_overall_title", "guide.card_overall_body"),
        ("guide.card_band_title", "guide.card_band_body"),
        ("guide.card_deterministic_title", "guide.card_deterministic_body"),
        ("guide.card_rubric_title", "guide.card_rubric_body"),
        ("guide.card_gates_title", "guide.card_gates_body"),
    ]
    for column, (title_key, body_key) in zip(summary_cols, summary_cards):
        with column:
            st.caption(t(title_key))
            st.write(t(body_key))

with st.container(border=True):
    st.subheader(t("guide.formula_title"))
    st.markdown(
        f"- {t('guide.formula_hybrid', deterministic_weight=formula['deterministic_weight_pct'], rubric_weight=formula['rubric_weight_pct'])}"
    )
    st.markdown(f"- {t('guide.formula_deterministic_only')}")
    st.markdown(f"- {t('guide.formula_topic_cap', cap=formula['topic_fail_cap_score'])}")
    st.markdown(
        f"- {t('guide.formula_decimal', min_score=guide['score_scale']['min'], max_score=guide['score_scale']['max'])}"
    )

with st.container(border=True):
    st.subheader(t("guide.deterministic_title"))
    st.write(t("guide.deterministic_body"))
    deterministic_rows = []
    for signal in guide["deterministic_signals"]:
        signal_id = signal["id"]
        if signal_id == "wpm":
            target = t(
                "guide.det_target_wpm",
                target=signal["target"],
                tolerance=signal["tolerance"],
            )
        elif signal_id == "pause_ratio":
            target = t("guide.det_target_pause", ceiling=signal["ceiling"])
        elif signal_id == "filler_ratio":
            target = t("guide.det_target_filler", ceiling=signal["ceiling"])
        elif signal_id == "cohesion_markers":
            target = t("guide.det_target_cohesion", target=signal["target"])
        else:
            target = t("guide.det_target_complexity", target=signal["target"])
        deterministic_rows.append(
            {
                t("guide.det_column_signal"): t(f"guide.det_signal_{signal_id}"),
                t("guide.det_column_target"): target,
                t("guide.det_column_weight"): f"{signal['weight_pct']}%",
            }
        )
    st.dataframe(pd.DataFrame(deterministic_rows), width="stretch", hide_index=True)

with st.container(border=True):
    st.subheader(t("guide.rubric_title"))
    st.write(t("guide.rubric_body"))
    rubric_rows = [
        {
            t("guide.rubric_column_dimension"): t(f"guide.dimension_{dimension_id}"),
            t("guide.rubric_column_description"): t(f"guide.rubric_desc_{dimension_id}"),
        }
        for dimension_id in guide["rubric_dimensions"]
    ]
    st.dataframe(pd.DataFrame(rubric_rows), width="stretch", hide_index=True)

with st.container(border=True):
    st.subheader(t("guide.gates_title"))
    st.write(t("guide.gates_body"))
    gate_rows = []
    for gate in guide["gates"]:
        gate_id = gate["id"]
        if gate_id == "duration_pass":
            rule = t("guide.gate_rule_duration_pass", ratio=gate["duration_pass_ratio_pct"])
        elif gate_id == "min_words_pass":
            rule = t("guide.gate_rule_min_words_pass", count=gate["min_word_count"])
        elif gate_id == "topic_pass":
            rule = t("guide.gate_rule_topic_pass", cap=gate["topic_fail_cap_score"])
        else:
            rule = t("guide.gate_rule_language_pass")
        gate_rows.append(
            {
                t("guide.gates_column_gate"): t(f"review.gate_{'theme' if gate_id == 'topic_pass' else 'words' if gate_id == 'min_words_pass' else 'duration' if gate_id == 'duration_pass' else 'language'}"),
                t("guide.gates_column_rule"): rule,
            }
        )
    st.dataframe(pd.DataFrame(gate_rows), width="stretch", hide_index=True)

with st.container(border=True):
    st.subheader(t("guide.cefr_title"))
    st.write(t("guide.cefr_body"))
    translated_dimensions = ", ".join(t(f"guide.dimension_{dimension_id}") for dimension_id in guide["cefr_dimensions"])
    st.caption(t("guide.cefr_dimensions_caption", dimensions=translated_dimensions))
    cefr_rows = [
        {
            t("guide.cefr_column_language"): row["label"],
            t("guide.cefr_column_b2"): row["B2"],
            t("guide.cefr_column_c1"): row["C1"],
            t("guide.cefr_column_c2"): row["C2"],
        }
        for row in guide["cefr_thresholds"]
    ]
    st.dataframe(pd.DataFrame(cefr_rows), width="stretch", hide_index=True)
    st.caption(t("guide.cefr_footer"))

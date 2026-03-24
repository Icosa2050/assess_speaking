import unittest
from unittest.mock import patch

import app_shell.review_components as review_components
from app_shell.i18n import t


class ReviewComponentsTests(unittest.TestCase):
    def test_helper_mappings_cover_known_and_unknown_values(self):
        self.assertEqual(review_components._gate_label("language_pass"), t("review.gate_language"))
        self.assertEqual(review_components._gate_label("custom_gate"), "custom gate")
        self.assertEqual(review_components._mode_label("hybrid"), t("review.mode_hybrid"))
        self.assertEqual(review_components._mode_label("deterministic_only"), t("review.mode_deterministic_only"))
        self.assertEqual(review_components._mode_label("mystery"), t("review.mode_unknown"))
        self.assertEqual(review_components._as_text_list("one"), ["one"])
        self.assertEqual(review_components._as_text_list(["one", "", 2]), ["one", "2"])
        self.assertEqual(review_components._as_text_list(None), [])
        self.assertEqual(review_components._gate_status(True), t("review.gate_pass"))
        self.assertEqual(review_components._gate_status(False), t("review.gate_fail"))
        self.assertEqual(review_components._gate_status(None), t("review.gate_unknown"))

    def test_render_report_status_routes_to_warning_info_and_success(self):
        with patch.object(review_components.st, "warning") as warning, \
                patch.object(review_components.st, "info") as info, \
                patch.object(review_components.st, "success") as success:
            review_components.render_report_status({"requires_human_review": True})
            review_components.render_report_status({"failed_gates": ["topic_pass", "duration_pass"]})
            review_components.render_report_status({})

        warning.assert_called_once_with(t("review.status_review"))
        info.assert_called_once_with(t("review.status_unstable", gates="Theme, Duration"))
        success.assert_called_once_with(t("review.status_done"))

    def test_render_progress_items_renders_all_supported_kinds(self):
        summary = {
            "progress_items": [
                {"kind": "previous_session", "value": "sess-1"},
                {"kind": "delta_final", "value": 0.25},
                {"kind": "delta_overall", "value": -0.2},
                {"kind": "delta_wpm", "value": 12.0},
                {"kind": "new_priorities", "value": ["clearer conclusions"]},
                {"kind": "resolved_priorities", "value": ["agreement"]},
                {"kind": "repeating_grammar", "value": ["Verb tense"]},
                {"kind": "repeating_coherence", "value": ["Sequencing"]},
            ]
        }
        with patch.object(review_components.st, "subheader") as subheader, \
                patch.object(review_components.st, "markdown") as markdown, \
                patch.object(review_components.st, "caption") as caption:
            review_components._render_progress_items(summary)

        subheader.assert_called_once_with(t("review.progress_title"))
        self.assertEqual(len(markdown.call_args_list), 8)
        caption.assert_not_called()

    def test_render_progress_items_shows_unavailable_when_items_do_not_render(self):
        summary = {"progress_items": [{"kind": "delta_final", "value": "n/a"}]}
        with patch.object(review_components.st, "caption") as caption, \
                patch.object(review_components.st, "subheader") as subheader, \
                patch.object(review_components.st, "markdown") as markdown:
            review_components._render_progress_items(summary)

        caption.assert_called_once_with(t("review.progress_unavailable"))
        subheader.assert_not_called()
        markdown.assert_not_called()

    def test_render_baseline_renders_dataframe_when_targets_exist(self):
        summary = {
            "baseline": {
                "level": "B2",
                "comment": "Stable baseline.",
                "targets": {
                    "grammar": {"expected": "few errors", "actual": "few errors", "ok": True},
                    "coherence": {"expected": "clear structure", "actual": "clear structure", "ok": False},
                },
            }
        }
        with patch.object(review_components.st, "subheader") as subheader, \
                patch.object(review_components.st, "caption") as caption, \
                patch.object(review_components.st, "dataframe") as dataframe:
            review_components._render_baseline(summary)

        subheader.assert_called_once_with(t("review.baseline_title"))
        caption.assert_called_once_with(t("review.baseline_caption", level="B2", comment="Stable baseline."))
        dataframe.assert_called_once()
        rendered_frame = dataframe.call_args.args[0]
        self.assertEqual(list(rendered_frame.columns), [
            t("review.baseline_metric"),
            t("review.baseline_expected"),
            t("review.baseline_actual"),
            t("review.baseline_ok"),
        ])
        self.assertEqual(len(rendered_frame), 2)

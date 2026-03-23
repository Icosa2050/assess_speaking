import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.repo_quality_audit import (
    scan_duplicate_functions,
    scan_missing_tests,
    scan_python_quality,
    scan_root_layout,
    scan_translation_issues,
)


class RepoQualityAuditTests(unittest.TestCase):
    def test_root_layout_scan_reports_artifacts_and_unexpected_modules(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "assess_speaking.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "rogue_tool.py").write_text("print('rogue')\n", encoding="utf-8")
            (root / "-o").write_text("artifact\n", encoding="utf-8")

            findings = scan_root_layout(
                root,
                allowed_root_python_files={"assess_speaking.py"},
                root_artifact_names={"-o"},
                root_python_sprawl_threshold=10,
            )
            checks = {finding.check for finding in findings}
            self.assertIn("root-generated-artifact", checks)
            self.assertIn("unexpected-root-python-module", checks)

    def test_root_layout_scan_reports_python_sprawl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            allowed: set[str] = set()
            for name in ("one.py", "two.py", "three.py"):
                (root / name).write_text("value = 1\n", encoding="utf-8")
                allowed.add(name)

            findings = scan_root_layout(
                root,
                allowed_root_python_files=allowed,
                root_artifact_names=set(),
                root_python_sprawl_threshold=2,
            )
            self.assertTrue(any(finding.check == "root-python-module-sprawl" for finding in findings))

    def test_translation_scan_reports_missing_keys_and_placeholder_drift(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "locales").mkdir()
            (root / "locales" / "en.json").write_text(
                textwrap.dedent(
                    """
                {
                  "home": {
                    "title": "Title",
                    "body": "Hello {name}"
                  },
                  "unused": {
                    "key": "Unused"
                  }
                }
                """
                ).strip(),
                encoding="utf-8",
            )
            (root / "locales" / "de.json").write_text(
                textwrap.dedent(
                    """
                {
                  "home": {
                    "body": "Hallo {vorname}"
                  }
                }
                """
                ).strip(),
                encoding="utf-8",
            )
            (root / "app.py").write_text(
                textwrap.dedent(
                    """
                from app_shell.i18n import t

                def render() -> None:
                    print(t("home.title"))
                    print(t("missing.key"))
                """
                ).strip(),
                encoding="utf-8",
            )

            findings = scan_translation_issues(
                root,
                excluded_dirs=set(),
                locale_markers=("locales",),
                translation_functions=("t",),
            )
            checks = {finding.check for finding in findings}
            self.assertIn("locale-missing-keys", checks)
            self.assertIn("locale-placeholder-mismatch", checks)
            self.assertIn("translation-key-missing", checks)
            self.assertIn("locale-unused-keys", checks)

    def test_python_quality_scan_reports_silent_except_and_mutable_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "service.py").write_text(
                textwrap.dedent(
                    """
                def load_items(items=[]):
                    return items

                def run():
                    try:
                        return load_items()
                    except Exception:
                        pass
                """
                ).strip(),
                encoding="utf-8",
            )

            findings = scan_python_quality(root, excluded_dirs=set(), include_tests=False)
            checks = {finding.check for finding in findings}
            self.assertIn("mutable-default-argument", checks)
            self.assertIn("silent-except", checks)

    def test_duplicate_function_scan_reports_structural_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "duplicates.py").write_text(
                textwrap.dedent(
                    """
                def build_alpha(values):
                    cleaned = []
                    for value in values:
                        if value:
                            cleaned.append(value.strip())
                    cleaned.sort()
                    result = ",".join(cleaned)
                    return result

                def build_beta(items):
                    trimmed = []
                    for item in items:
                        if item:
                            trimmed.append(item.strip())
                    trimmed.sort()
                    result = ",".join(trimmed)
                    return result
                """
                ).strip(),
                encoding="utf-8",
            )

            findings = scan_duplicate_functions(root, excluded_dirs=set(), include_tests=False)
            self.assertTrue(any(finding.check == "duplicate-function-logic" for finding in findings))

    def test_missing_tests_scan_flags_untested_modules(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "tests").mkdir()
            (root / "app_shell").mkdir()
            (root / "app_shell" / "__init__.py").write_text("", encoding="utf-8")
            (root / "app_shell" / "i18n.py").write_text(
                "\n".join(f"value_{index} = {index}" for index in range(40)),
                encoding="utf-8",
            )
            (root / "analytics.py").write_text(
                "\n".join(f"metric_{index} = {index}" for index in range(40)),
                encoding="utf-8",
            )
            (root / "tests" / "test_app_shell_i18n.py").write_text(
                textwrap.dedent(
                    """
                from app_shell import i18n

                def test_import():
                    assert i18n is not None
                """
                ).strip(),
                encoding="utf-8",
            )

            findings = scan_missing_tests(root, excluded_dirs=set())
            flagged_paths = {finding.path for finding in findings}
            self.assertIn("analytics.py", flagged_paths)
            self.assertNotIn("app_shell/i18n.py", flagged_paths)


if __name__ == "__main__":
    unittest.main()

import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.root_import_audit import (
    build_root_import_audit,
    classify_module,
    discover_root_python_files,
    extract_local_imports,
    render_text_report,
)


class RootImportAuditTests(unittest.TestCase):
    def test_discover_root_python_files_ignores_nested_modules(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "alpha.py").write_text("value = 1\n", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "beta.py").write_text("value = 2\n", encoding="utf-8")

            discovered = discover_root_python_files(root)
            self.assertEqual([path.name for path in discovered], ["alpha.py"])

    def test_extract_local_imports_collects_root_module_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = root / "alpha.py"
            path.write_text(
                textwrap.dedent(
                    """
                    import beta
                    from gamma import build
                    import json
                    """
                ).strip(),
                encoding="utf-8",
            )

            imports = extract_local_imports(path, {"alpha", "beta", "gamma"})
            self.assertEqual(imports, {"beta", "gamma"})

    def test_build_root_import_audit_tracks_inbound_and_outbound_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "alpha.py").write_text("import beta\nfrom gamma import build\n", encoding="utf-8")
            (root / "beta.py").write_text("value = 1\n", encoding="utf-8")
            (root / "gamma.py").write_text("import beta\n", encoding="utf-8")

            audit = build_root_import_audit(root)
            modules = {item["name"]: item for item in audit["modules"]}

            self.assertEqual(modules["alpha"]["direct_imports"], ["beta", "gamma"])
            self.assertEqual(modules["beta"]["inbound_imports"], ["alpha", "gamma"])
            self.assertEqual(audit["inbound_hubs"][0]["name"], "beta")

    def test_classify_module_marks_unknown_names_as_unclassified(self):
        self.assertEqual(classify_module("unknown_tool"), "unclassified")

    def test_render_text_report_marks_empty_groups_explicitly(self):
        audit = {
            "root": "/tmp/repo",
            "root_python_file_count": 2,
            "entrypoints": ["assess_speaking.py", "streamlit_app.py"],
            "groups": {},
            "inbound_hubs": [],
            "high_outbound_modules": [],
        }

        report = render_text_report(audit)

        self.assertIn("Groups:\n- none", report)


if __name__ == "__main__":
    unittest.main()

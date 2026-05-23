import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.check_quality import DEFAULT_TARGETS, scan_quality


class CheckQualityTests(unittest.TestCase):
    def test_broad_except_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "app_core").mkdir()
            (root / "app_core" / "services.py").write_text(
                textwrap.dedent(
                    """
                    def run():
                        try:
                            return 1
                        except Exception:
                            return 0
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            findings = scan_quality(root, targets=["app_core/services.py"])

        self.assertEqual([(finding.check, finding.line) for finding in findings], [("broad-except", 4)])

    def test_allow_comment_suppresses_broad_except(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "app_core").mkdir()
            (root / "app_core" / "services.py").write_text(
                textwrap.dedent(
                    """
                    def run():
                        try:
                            return 1
                        # quality: allow[broad-except] provider health boundary
                        except Exception:
                            return 0
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            findings = scan_quality(root, targets=["app_core/services.py"])

        self.assertEqual(findings, [])

    def test_bare_except_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "app_backend").mkdir()
            (root / "app_backend" / "jobs.py").write_text(
                textwrap.dedent(
                    """
                    def run():
                        try:
                            return 1
                        except:
                            return 0
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            findings = scan_quality(root, targets=["app_backend"])

        self.assertEqual([(finding.check, finding.line) for finding in findings], [("bare-except", 4)])

    def test_subprocess_is_banned_in_app_core_and_allowed_in_scripts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "app_core").mkdir()
            (root / "scripts").mkdir()
            (root / "app_core" / "services.py").write_text(
                textwrap.dedent(
                    """
                    import subprocess

                    def run():
                        return subprocess.run(["echo", "hi"])
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (root / "scripts" / "run_app.py").write_text(
                textwrap.dedent(
                    """
                    import subprocess

                    def run():
                        return subprocess.run(["echo", "hi"])
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            findings = scan_quality(root, targets=["app_core/services.py", "scripts/run_app.py"])

        self.assertEqual([(finding.check, finding.path) for finding in findings], [("subprocess-shellout", "app_core/services.py")])

    def test_allow_comment_suppresses_subprocess_shellout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "app_core").mkdir()
            (root / "app_core" / "services.py").write_text(
                textwrap.dedent(
                    """
                    import subprocess

                    def run():
                        # quality: allow[subprocess-shellout] backend bootstrap boundary
                        return subprocess.run(["echo", "hi"])
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            findings = scan_quality(root, targets=["app_core/services.py"])

        self.assertEqual(findings, [])

    def test_default_targets_include_core_runtime(self):
        self.assertIn("app_core", DEFAULT_TARGETS)


if __name__ == "__main__":
    unittest.main()

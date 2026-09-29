from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PlaywrightWrapperIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Test shell argument forwarding without depending on an npm install.
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.repo_root = Path(workspace.name)
        scripts = self.repo_root / "scripts"
        scripts.mkdir()
        for name in ("playwright_research.sh", "playwright_celi.sh"):
            shutil.copy2(REPO_ROOT / "scripts" / name, scripts / name)
        cli = self.repo_root / "frontend/node_modules/playwright-core/lib/tools/cli-client/cli.js"
        cli.parent.mkdir(parents=True)
        cli.touch()

    def _make_fake_node(self, tmp_path: Path) -> Path:
        # Capture the actual project CLI invocation without starting a browser daemon.
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        node = bin_dir / "node"
        node.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['FAKE_PWCLI_OUT']).write_text(json.dumps({'argv': sys.argv[1:]}))\n",
            encoding="utf-8",
        )
        node.chmod(0o755)
        return bin_dir

    def _run_wrapper(
        self,
        script_name: str,
        args: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_path = tmp_path / "pwcli-args.json"
            bin_dir = self._make_fake_node(tmp_path)
            env = os.environ.copy()
            env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
            env["FAKE_PWCLI_OUT"] = str(out_path)
            if env_overrides:
                env.update(env_overrides)
            completed = subprocess.run(
                [str(self.repo_root / "scripts" / script_name), *args],
                cwd=self.repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{script_name} failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            payload["tmp_dir"] = tmp_dir
            return payload

    def test_research_wrapper_adds_default_session_and_persistent_open(self) -> None:
        payload = self._run_wrapper(
            "playwright_research.sh",
            ["open", "https://example.com/?q=1"],
        )
        self.assertEqual(
            payload["argv"],
            [
                str(self.repo_root / "frontend/node_modules/playwright-core/lib/tools/cli-client/cli.js"),
                "--session",
                "research",
                "--config",
                str(self.repo_root / ".playwright" / "research-cli.config.json"),
                "open",
                "--persistent",
                "https://example.com/?q=1",
                f"--profile={self.repo_root / '.playwright/profiles/research-chromium'}",
            ],
        )

    def test_research_wrapper_respects_explicit_persistent_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            profile_dir = tmp_path / "profiles" / "custom"
            output_dir = tmp_path / "output" / "custom"
            payload = self._run_wrapper(
                "playwright_research.sh",
                ["open", "--persistent", "https://example.com/"],
                env_overrides={
                    "PLAYWRIGHT_RESEARCH_SESSION": "custom-session",
                    "PLAYWRIGHT_RESEARCH_PROFILE_DIR": str(profile_dir),
                    "PLAYWRIGHT_RESEARCH_OUTPUT_DIR": str(output_dir),
                },
            )
            self.assertEqual(
                payload["argv"],
                [
                    str(self.repo_root / "frontend/node_modules/playwright-core/lib/tools/cli-client/cli.js"),
                    "--session",
                    "custom-session",
                    "--config",
                    str(self.repo_root / ".playwright" / "research-cli.config.json"),
                    "open",
                    "--persistent",
                    "https://example.com/",
                    f"--profile={profile_dir}",
                ],
            )
            self.assertTrue(profile_dir.exists())
            self.assertTrue(output_dir.exists())

    def test_celi_wrapper_delegates_to_research_wrapper_with_celi_defaults(self) -> None:
        payload = self._run_wrapper(
            "playwright_celi.sh",
            ["snapshot"],
        )
        self.assertEqual(
            payload["argv"],
            [
                str(self.repo_root / "frontend/node_modules/playwright-core/lib/tools/cli-client/cli.js"),
                "--session",
                "celi",
                "--config",
                str(self.repo_root / ".playwright" / "celi-cli.config.json"),
                "snapshot",
            ],
        )


if __name__ == "__main__":
    unittest.main()

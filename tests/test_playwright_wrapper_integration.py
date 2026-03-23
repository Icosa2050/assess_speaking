from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PlaywrightWrapperIntegrationTests(unittest.TestCase):
    def _make_fake_codex_home(self, tmp_path: Path) -> tuple[Path, Path]:
        codex_home = tmp_path / "codex-home"
        wrapper_path = codex_home / "skills" / "playwright" / "scripts" / "playwright_cli.sh"
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper_path.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
python3 - "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "argv": sys.argv[1:],
    "playwright_cli_session": os.environ.get("PLAYWRIGHT_CLI_SESSION"),
}
Path(os.environ["FAKE_PWCLI_OUT"]).write_text(json.dumps(payload), encoding="utf-8")
PY
""",
            encoding="utf-8",
        )
        wrapper_path.chmod(0o755)
        return codex_home, wrapper_path

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
            codex_home, _ = self._make_fake_codex_home(tmp_path)
            env = os.environ.copy()
            env["CODEX_HOME"] = str(codex_home)
            env["FAKE_PWCLI_OUT"] = str(out_path)
            if env_overrides:
                env.update(env_overrides)
            completed = subprocess.run(
                [str(REPO_ROOT / "scripts" / script_name), *args],
                cwd=REPO_ROOT,
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
                "--session",
                "research",
                "--config",
                str(REPO_ROOT / ".playwright" / "research-cli.config.json"),
                "open",
                "--persistent",
                "https://example.com/?q=1",
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
                    "--session",
                    "custom-session",
                    "--config",
                    str(REPO_ROOT / ".playwright" / "research-cli.config.json"),
                    "open",
                    "--persistent",
                    "https://example.com/",
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
                "--session",
                "celi",
                "--config",
                str(REPO_ROOT / ".playwright" / "celi-cli.config.json"),
                "snapshot",
            ],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PlaywrightTaskSetupTests(unittest.TestCase):
    def test_research_config_is_valid_and_uses_dedicated_profile(self) -> None:
        config_path = REPO_ROOT / ".playwright" / "research-cli.config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["browser"]["launchOptions"]["channel"], "chrome")
        self.assertFalse(payload["browser"]["launchOptions"]["headless"])
        self.assertEqual(payload["browser"]["userDataDir"], ".playwright/profiles/research")
        self.assertEqual(payload["outputDir"], "output/playwright/research")

    def test_celi_config_is_valid_and_uses_dedicated_profile(self) -> None:
        config_path = REPO_ROOT / ".playwright" / "celi-cli.config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["browser"]["launchOptions"]["channel"], "chrome")
        self.assertFalse(payload["browser"]["launchOptions"]["headless"])
        self.assertEqual(payload["browser"]["userDataDir"], ".playwright/profiles/celi")
        self.assertEqual(payload["outputDir"], "output/playwright/celi")

    def test_research_helper_script_points_to_repo_local_config_and_profile(self) -> None:
        script_path = REPO_ROOT / "scripts" / "playwright_research.sh"
        content = script_path.read_text(encoding="utf-8")
        self.assertIn(".playwright/research-cli.config.json", content)
        self.assertIn(".playwright/profiles/research", content)
        self.assertIn("--session", content)
        self.assertIn("--persistent", content)

    def test_celi_helper_script_delegates_to_research_helper(self) -> None:
        script_path = REPO_ROOT / "scripts" / "playwright_celi.sh"
        content = script_path.read_text(encoding="utf-8")
        self.assertIn(".playwright/celi-cli.config.json", content)
        self.assertIn(".playwright/profiles/celi", content)
        self.assertIn("playwright_research.sh", content)


if __name__ == "__main__":
    unittest.main()

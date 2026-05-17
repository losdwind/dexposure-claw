"""Smoke tests for the DeXposure Claw portable agent extension."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CLAW_ROOT = REPO_ROOT / "claw"
SRC_ROOT = CLAW_ROOT / "src"


class DeXposureClawTests(unittest.TestCase):
    def setUp(self) -> None:
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))

    def test_canonical_pack_layout_exists(self):
        required = [
            "pyproject.toml",
            "package.json",
            "pack/skills/run-full-suite/SKILL.md",
            "pack/commands/dexposure-benchmark.md",
            "adapters/claude-code/plugin.json.j2",
            "adapters/claude-code/mcp.json.j2",
            "src/dexposure_claw/cli.py",
            "src/dexposure_claw/mcp_server.py",
        ]

        for relative_path in required:
            self.assertTrue(
                (CLAW_ROOT / relative_path).exists(),
                f"missing {relative_path}",
            )

    def test_build_claude_code_plugin_renders_dist_from_canonical_pack(self):
        from dexposure_claw.installer import build_claude_code_plugin

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp) / "dexposure-claw"
            result = build_claude_code_plugin(output_dir=output_dir)

            self.assertEqual(result["target"], "claude-code")
            manifest_path = output_dir / ".claude-plugin" / "plugin.json"
            mcp_path = output_dir / ".mcp.json"
            skill_path = output_dir / "skills" / "run-full-suite" / "SKILL.md"
            command_path = output_dir / "commands" / "dexposure-benchmark.md"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(mcp_path.exists())
            self.assertTrue(skill_path.exists())
            self.assertTrue(command_path.exists())

            manifest = json.loads(manifest_path.read_text())
            mcp = json.loads(mcp_path.read_text())
            self.assertEqual(manifest["name"], "dexposure-claw")
            self.assertIn("dexposure", mcp["mcpServers"])
            self.assertEqual(mcp["mcpServers"]["dexposure"]["command"], "node")
            self.assertTrue((output_dir / "bin" / "dexposure-claw.js").exists())
            self.assertTrue((output_dir / "src" / "dexposure_claw" / "cli.py").exists())
            self.assertIn("dexposure-claw", skill_path.read_text().lower())

    def test_platform_config_snippets_are_generated_without_mutating_home(self):
        from dexposure_claw.installer import render_install_snippet

        hermes = render_install_snippet("hermes")
        codex = render_install_snippet("codex")
        mcp_only = render_install_snippet("mcp")

        self.assertIn("mcp_servers:", hermes)
        self.assertIn("dexposure-claw", hermes)
        self.assertIn("[mcp_servers.dexposure]", codex)
        self.assertIn("dexposure-claw", codex)
        self.assertIn('"mcpServers"', mcp_only)
        self.assertIn('"dexposure"', mcp_only)


if __name__ == "__main__":
    unittest.main()

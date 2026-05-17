"""Build and install runtime adapters for DeXposure Claw."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from string import Template
from typing import Any


PACKAGE_NAME = "dexposure-claw"
MCP_COMMAND = "dexposure-claw"


def claw_root() -> Path:
    """Return the source-tree root for the DeXposure Claw package."""
    env_root = os.environ.get("DEXPOSURE_CLAW_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def pack_root() -> Path:
    return claw_root() / "pack"


def adapters_root() -> Path:
    return claw_root() / "adapters"


def dist_root() -> Path:
    return claw_root() / "dist"


def _copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    items = list(source.iterdir())
    if not items:
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in items:
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _render_template(path: Path, values: dict[str, Any]) -> str:
    # Keep templates dependency-free. We accept both $name and {{ name }} forms.
    text = path.read_text()
    for key, value in values.items():
        text = text.replace("{{ " + key + " }}", str(value))
        text = text.replace("{{" + key + "}}", str(value))
    return Template(text).safe_substitute(values)


def build_claude_code_plugin(output_dir: Path | None = None) -> dict[str, Any]:
    """Build a Claude Code plugin from the canonical pack materials."""
    root = output_dir or dist_root() / "claude-code" / PACKAGE_NAME
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    values = {
        "package_name": PACKAGE_NAME,
        "mcp_command": MCP_COMMAND,
        "description": (
            "Run and audit DeXposure-Bench financial-risk workflows from "
            "Claude Code."
        ),
    }
    claude_adapter = adapters_root() / "claude-code"

    plugin_dir = root / ".claude-plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.json").write_text(
        _render_template(claude_adapter / "plugin.json.j2", values)
    )
    (root / ".mcp.json").write_text(
        _render_template(claude_adapter / "mcp.json.j2", values)
    )

    for component in ("skills", "commands", "agents", "hooks"):
        _copy_tree_contents(pack_root() / component, root / component)
    _copy_tree_contents(claw_root() / "bin", root / "bin")
    _copy_tree_contents(claw_root() / "src", root / "src")
    readme = claw_root() / "README.md"
    if readme.exists():
        shutil.copy2(readme, root / "README.md")

    return {
        "target": "claude-code",
        "path": str(root),
        "manifest": str(plugin_dir / "plugin.json"),
    }


def render_install_snippet(target: str) -> str:
    """Render a manual install snippet for a runtime target."""
    values = {
        "package_name": PACKAGE_NAME,
        "mcp_command": MCP_COMMAND,
    }
    if target == "mcp":
        return json.dumps(
            {
                "mcpServers": {
                    "dexposure": {
                        "command": MCP_COMMAND,
                        "args": ["mcp"],
                    }
                }
            },
            indent=2,
        )

    template_map = {
        "hermes": adapters_root() / "hermes" / "config.yaml.j2",
        "codex": adapters_root() / "codex" / "config.toml.j2",
    }
    try:
        return _render_template(template_map[target], values)
    except KeyError as exc:
        raise ValueError(f"Unknown install target: {target}") from exc


def install(target: str | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Install or print instructions for a target runtime."""
    chosen = target or "manual"
    if chosen == "claude-code":
        result = build_claude_code_plugin(output_dir=output_dir)
        result["message"] = (
            "Claude Code plugin built. Load it with: "
            f"claude --plugin-dir {result['path']}"
        )
        return result
    if chosen in {"hermes", "codex", "mcp"}:
        return {
            "target": chosen,
            "snippet": render_install_snippet(chosen),
        }
    return {
        "target": "manual",
        "options": ["claude-code", "hermes", "codex", "mcp"],
        "message": "Choose a target: claude-code, hermes, codex, or mcp.",
    }

"""Command-line interface for DeXposure Claw."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .installer import build_claude_code_plugin, install, render_install_snippet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dexposure-claw")
    sub = parser.add_subparsers(dest="command")

    install_parser = sub.add_parser("install", help="Install into an agent runtime")
    install_parser.add_argument(
        "target",
        nargs="?",
        choices=["claude-code", "hermes", "codex", "mcp"],
        help="Runtime target. Omit to print available options.",
    )
    install_parser.add_argument("--output-dir", type=Path)

    build = sub.add_parser("build", help="Build runtime adapter artifacts")
    build.add_argument("target", choices=["claude-code"])
    build.add_argument("--output-dir", type=Path)

    sub.add_parser("mcp", help="Start the MCP server")
    sub.add_parser("health", help="Print package health information")

    snippet = sub.add_parser("snippet", help="Print manual install config")
    snippet.add_argument("target", choices=["hermes", "codex", "mcp"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        print(json.dumps(build_claude_code_plugin(args.output_dir), indent=2))
        return 0
    if args.command == "install":
        print(json.dumps(install(args.target, args.output_dir), indent=2))
        return 0
    if args.command == "snippet":
        print(render_install_snippet(args.target))
        return 0
    if args.command == "health":
        print(json.dumps({"status": "ok", "package": "dexposure-claw"}, indent=2))
        return 0
    if args.command == "mcp":
        from .mcp_server import main as mcp_main

        return mcp_main()

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

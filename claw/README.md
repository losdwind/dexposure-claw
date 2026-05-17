# DeXposure Claw

DeXposure Claw is a pluggable financial-risk agent extension for
Claude Code, Hermes, OpenAI Codex, and other MCP-compatible runtimes.

It exposes DeXposure-Bench and paper-evaluation workflows as:

- MCP tools for benchmark execution, reporting, and audit metadata.
- Runtime skills for repeatable financial-risk workflows.
- Adapter builds for Claude Code plugins and other agent runtimes.

## Quick Start

From the repository root:

```bash
pipx install ./claw
dexposure-claw install
```

For editable local development:

```bash
python -m pip install -e ./claw
dexposure-claw health
```

## Claude Code

For local development:

```bash
dexposure-claw build claude-code
claude --plugin-dir claw/dist/claude-code/dexposure-claw
```

## MCP

Any MCP client can start the tool server with:

```bash
dexposure-claw mcp
```

## OpenAI Codex

Codex consumes DeXposure Claw through MCP:

```bash
codex mcp add dexposure -- dexposure-claw mcp
codex mcp get dexposure
```

After the npm wrapper is published, clients can use:

```bash
codex mcp add dexposure -- npx -y @dexposure/claw mcp
```

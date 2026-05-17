# DeXposure Claw

DeXposure Claw is a pluggable financial-risk agent extension for
Claude Code, Hermes, OpenAI Codex, and other MCP-compatible runtimes.

It exposes DeXposure-Bench and DeXposure-Agent workflows as:

- MCP tools for benchmark execution, reporting, and audit metadata.
- Runtime skills for repeatable financial-risk workflows.
- Adapter builds for Claude Code plugins and other agent runtimes.

## Quick Start

```bash
pipx install dexposure-claw
dexposure-claw install
```

or, through the optional npm wrapper:

```bash
npx @dexposure/claw install
```

## Claude Code

For local development:

```bash
dexposure-claw build claude-code
claude --plugin-dir DeXposure_Agent/dexposure_claw/dist/claude-code/dexposure-claw
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

For npm-based installs, use the wrapper command instead:

```bash
codex mcp add dexposure -- npx -y @dexposure/claw mcp
```

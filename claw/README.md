# DeXposure Claw

DeXposure Claw is a pluggable financial-risk agent extension for
Claude Code, Hermes, OpenAI Codex, and other MCP-compatible runtimes.

It exposes DeXposure-Bench and paper-evaluation workflows as:

- MCP tools for benchmark execution, reporting, and audit metadata.
- Runtime skills for repeatable financial-risk workflows.
- Adapter builds for Claude Code plugins and other agent runtimes.

## Quick Start

### Node.js / npm

For agent-runtime users, npm is the most convenient entrypoint. The npm package
ships a Node.js binary that starts the Claw Python runtime under the hood, so it
requires Node.js 18+ and Python 3.10+.

From the repository root:

```bash
npm exec --package ./claw -- dexposure-claw health
npm exec --package ./claw -- dexposure-claw mcp
```

After publishing:

```bash
npx -y @dexposure/claw health
npx -y @dexposure/claw mcp
```

To install globally from a checkout:

```bash
npm install -g ./claw
dexposure-claw health
```

### Python

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

With the npm package, clients can use:

```bash
codex mcp add dexposure -- npx -y @dexposure/claw mcp
```

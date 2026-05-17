# DeXposure Claw

DeXposure Claw is a lightweight agent-runtime extension for the DeXposure-Claw
DeFi risk-monitoring project. It provides a `dexposure-claw` CLI and a
dependency-free stdio MCP server for Claude Code, OpenAI Codex, Hermes, and
other MCP-compatible runtimes.

Claw is a research and monitoring extension. It does not execute trades, sign
transactions, custody assets, or provide investment advice.

## Current Capabilities

The current package focuses on runtime setup and benchmark discovery:

| Capability | Status |
| --- | --- |
| CLI health check | Available |
| Claude Code adapter build | Available |
| MCP stdio server | Available |
| Install snippets for Hermes, Codex, and generic MCP clients | Available |
| DeXposure-Bench catalog listing | Available |
| Direct benchmark execution through MCP | Planned |
| Direct report generation through MCP | Planned |

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

## MCP Tools

Start the MCP server with:

```bash
dexposure-claw mcp
```

The server currently exposes:

| Tool | Purpose |
| --- | --- |
| `dexposure_health` | Check that the DeXposure Claw MCP server is reachable. |
| `dexposure_install_snippet` | Return install config for Hermes, Codex, or generic MCP clients. |
| `dexposure_list_benchmarks` | List the six DeXposure-Bench benchmark IDs and readable names. |

## Claude Code

For local development:

```bash
dexposure-claw build claude-code
claude --plugin-dir claw/dist/claude-code/dexposure-claw
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

## Package Checks

Before publishing or opening a package-related pull request, run:

```bash
python -m compileall claw/src/dexposure_claw
npm --prefix ./claw run pack:check
```

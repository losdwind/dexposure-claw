# DeXposure-Claw

| [**Researcher README**](#researcher-path) | [**Agent User README**](#agent-user-path) |
| --- | --- |
| Reproduce the paper, inspect DeXposure-Bench, and rerun tables and figures. | Configure Claude Code, Hermes, Codex, or another MCP-compatible client. |

DeXposure-Claw is an open-source DeFi risk-monitoring agent project. It is the
deployable agent instantiation of a forecast-grounded regulatory decision
framework for DeFi exposure graphs.

## What This Repository Contains

This repository contains three connected artifacts:

- **Framework:** a forecast--measure--decide--gate decision architecture that
  turns graph forecasts into auditable supervisory risk tickets.
- **Benchmark:** **DeXposure-Bench**, a six-axis evaluation suite for temporal
  forecasting, early warning, calibration, stress testing, decision quality, and
  robustness.
- **Agent:** **DeXposure-Claw**, a conservative MCP-compatible package that
  exposes health checks, install snippets, and the benchmark catalog to agent
  runtimes.

The Claw package is intentionally conservative today. The MCP server currently
lists benchmark metadata and runtime setup information; it does not yet run the
full benchmark suite directly through MCP.

## Financial Safety

DeXposure-Claw is a research and monitoring tool. It is not financial advice,
investment advice, legal advice, or an automated trading system.

The agent does not custody assets, execute trades, sign transactions, or make
portfolio decisions. Treat its outputs as decision-support signals that require
human review, independent validation, and domain-specific risk controls.

Known limitations:

- DeFi data can be delayed, incomplete, manipulated, or missing protocol context.
- Forecasts are probabilistic and can fail during regime shifts or adversarial
  market conditions.
- Benchmark performance does not guarantee live-market performance.
- Regulatory, accounting, and compliance obligations are outside the model scope.

## Researcher Path

Use this path if you are reviewing the paper, reproducing tables and figures, or
auditing the DeXposure-Bench evaluation harness.

Useful entry points:

- Paper source: [`paper/DeXposure-Agent.tex`](paper/DeXposure-Agent.tex)
- Compiled paper: [`paper/DeXposure-Agent.pdf`](paper/DeXposure-Agent.pdf)
- Experiment SOP: [`paper/CLAUDE.md`](paper/CLAUDE.md)
- Benchmark runners: [`paper/experiments/`](paper/experiments/)

### Key Paper Results

The main claim is not that the foundation model wins every point-forecasting
metric. The paper evaluates whether forecast-grounded decision-making improves a
deployable supervisory agent under a benchmark that exposes costs as well as
benefits.

- End-to-end decision quality: DeXposure-Claw improves ticket F1 from `0.0076`
  to `0.0233` against the persistence+rules baseline (`+208%`).
- Explanation quality: LLM-as-judge score improves from `2.03/5` for pure LLM
  to `2.69/5` for gated FM+LLM (`+33%`).
- Calibration: prediction-interval coverage is `0.913` against a `0.90` target,
  with ECE `0.013`.
- Deployment caveat: FM+LLM variants improve recall and explanations, but expose
  a false-intervention-rate trade-off that is measured explicitly by the
  benchmark and ablations.

### Reproducing The Paper

The paper workflow is more demanding than the Claw health check.

Prerequisites:

- Python 3.12.9 for the root project environment
- `uv` for dependency management
- Git LFS data and checkpoints
- CUDA-capable GPU for full benchmark reproduction

Basic flow:

```bash
git lfs pull
uv sync
uv run pytest paper/tests
uv run python paper/experiments/run_all.py
```

The shared `data/`, `checkpoints/`, and `lib/` directories are referenced via
relative paths from the repository root, so run commands from the repository
root. See [`paper/CLAUDE.md`](paper/CLAUDE.md) for GPU server access, sync flow,
benchmark runners, logging, checkpoints, and LLM evaluation notes.

### DeXposure-Bench Tasks

The benchmark catalog uses readable labels with explicit IDs:

| ID | Label |
| --- | --- |
| `b1_forecast` | B1 Forecast / Temporal risk forecasting |
| `b2_warning` | B2 Warning / Streaming early warning |
| `b3_calibration` | B3 Calibration / Predictive uncertainty |
| `b4_stress` | B4 Stress / What-if scenario fidelity |
| `b5_decision` | B5 Decision / Supervisory ticket quality |
| `b6_robustness` | B6 Robustness / Data-quality sensitivity |

### Data And Weights

The repository includes metadata, sample graph snapshots, and released model
checkpoint files. Some large files are managed with Git LFS.

The released DeXposure-FM checkpoint README is
[`checkpoints/dexposure-fm-release/README.md`](checkpoints/dexposure-fm-release/README.md).
It documents model horizons, input features, training periods, and reported
forecast metrics.

## Agent User Path

Use this path if you only want to verify that the agent extension starts.

Prerequisites:

- Node.js 18+
- Python 3.10+
- A checkout of this repository

From the repository root:

```bash
npm exec --package ./claw -- dexposure-claw health
npm exec --package ./claw -- dexposure-claw mcp
```

Expected health output:

```json
{
  "status": "ok",
  "package": "dexposure-claw"
}
```

After the package is published to npm:

```bash
npx -y @dexposure/claw health
npx -y @dexposure/claw mcp
```

Python users can install the same package from a checkout:

```bash
pipx install ./claw
dexposure-claw health
dexposure-claw mcp
```

## MCP Usage

Register the MCP server with OpenAI Codex:

```bash
codex mcp add dexposure -- dexposure-claw mcp
codex mcp get dexposure
```

The current MCP tools are:

| Tool | Purpose |
| --- | --- |
| `dexposure_health` | Check that the MCP server is reachable. |
| `dexposure_install_snippet` | Print install snippets for supported runtimes. |
| `dexposure_list_benchmarks` | List DeXposure-Bench IDs and readable names. |

## Repository Layout

- [`paper/`](paper/) -- paper source, experiment code, benchmark scripts, tables,
  and generated figures.
- [`claw/`](claw/) -- installable DeXposure Claw agent extension for Claude Code,
  Hermes, OpenAI Codex, and MCP-compatible runtimes.
- [`data/`](data/) -- weekly graph snapshots and metadata stored with Git LFS.
- [`checkpoints/dexposure-fm-release/`](checkpoints/dexposure-fm-release/) --
  trained FM weights for h1, h4, and h8-h12 forecasts.
- [`lib/`](lib/) -- GraphPFN and LiMiX source used by the active FM predictor.
- [`docs/`](docs/) -- additional project notes and risk-monitoring references.

## Development

For editable Claw development:

```bash
python -m pip install -e ./claw
dexposure-claw build claude-code
claude --plugin-dir claw/dist/claude-code/dexposure-claw
```

For root paper development:

```bash
uv sync
uv run pytest paper/tests
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution expectations.

## Security

Never commit API keys, wallet keys, private RPC credentials, or exchange
credentials. See [`SECURITY.md`](SECURITY.md) for vulnerability reporting and
secret-handling guidance.

## Archive

Previous-generation code and reference papers are kept under
[`archive/`](archive/) as read-only references. Do not run from `archive/`.

The FM weights at `checkpoints/dexposure-fm-release/` are the productized output
of the archived training pipeline, and the active paper consumes them directly.
To recover the pre-restructure layout, check out the `pre-archive-refactor` tag.

## License

Apache-2.0. Third-party components vendored under `archive/code/lib/` and
`lib/limix/` retain their own notices; see [`NOTICE`](NOTICE) and
`archive/LICENSES/` where applicable.

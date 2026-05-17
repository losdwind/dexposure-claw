# DeXposure-Claw

This repository contains both the **DeXposure-Claw** paper draft and a
root-level **Claw** software package for running the agent workflow.

DeXposure-Claw is a forecast-driven DeFi risk-monitoring and decision
recommendation system built on top of the **DeXposure-FM** graph time-series
foundation model.

## Repository layout

- [`DeXposure_Agent/`](DeXposure_Agent/) -- paper source, experiment code,
  benchmark scripts, tables, and generated figures.
- [`claw/`](claw/) -- installable DeXposure Claw agent extension for Claude
  Code, Hermes, OpenAI Codex, and MCP-compatible runtimes.
- [`data/`](data/) -- weekly graph snapshots and metadata stored with Git LFS.
- [`checkpoints/dexposure-fm-release/`](checkpoints/dexposure-fm-release/) --
  trained FM weights for h1, h4, and h8-h12 forecasts.
- [`lib/`](lib/) -- GraphPFN + LiMiX source used by the active FM predictor.

See [`DeXposure_Agent/CLAUDE.md`](DeXposure_Agent/CLAUDE.md) for the experiment
SOP, including GPU server access, sync flow, benchmark runners, and LLM eval.

## Try the Claw software

From a checkout of this repository:

```bash
pipx install ./claw
dexposure-claw health
dexposure-claw mcp
```

For editable local development:

```bash
python -m pip install -e ./claw
dexposure-claw build claude-code
claude --plugin-dir claw/dist/claude-code/dexposure-claw
```

To register the MCP server with OpenAI Codex:

```bash
codex mcp add dexposure -- dexposure-claw mcp
codex mcp get dexposure
```

The package source is under [`claw/src/dexposure_claw/`](claw/src/dexposure_claw/),
while runtime commands, skills, and adapter templates live under
[`claw/pack/`](claw/pack/) and [`claw/adapters/`](claw/adapters/).

## Reproducing the paper

1. Activate the project environment with `uv sync` from the repo root.
2. Follow [`DeXposure_Agent/CLAUDE.md`](DeXposure_Agent/CLAUDE.md) for the
   end-to-end experiment SOP. GPU work runs on a separate vast.ai instance.
3. The benchmark runner is
   [`DeXposure_Agent/experiments/run_all.py`](DeXposure_Agent/experiments/run_all.py).

The shared `data/`, `checkpoints/`, and `lib/` directories are referenced via
relative paths from the repo root, so commands should run with the repo root as
the current working directory.

## Archive

Previous-generation code and reference papers are kept under
[`archive/`](archive/) (read-only references):

- `archive/code/` -- DeXposure-FM training pipeline (`run_full_experiment.py`,
  `run_task2_model_based.py`, ...) plus the GraphPFN training entries
  (`bin/`, `exp/`) and FM-side `analysis/` and `autoresearch/`. The
  GraphPFN/LiMiX library source (`lib/`) lives at the repo root because
  the active paper imports it.
- `archive/papers/` -- DeXposure-FM PDF + review notes, GraphPFN paper,
  external references (GoT, arXiv preprint).
- `archive/docs/` -- FM-era `EXPERIMENT_PLAN.md`, `ACTION_CONDITIONED_ROADMAP.txt`,
  writing guide.
- `archive/LICENSES/` -- LimiX and TabICL licenses (third-party libs used by
  `archive/code/lib/`).
- `archive/.tools/` -- legacy AI-tool project state (Sisyphus, Serena).

Do not run from `archive/`. The FM weights at
`checkpoints/dexposure-fm-release/` are the productized output of the archived
training pipeline, and the active paper consumes them directly.

To recover the pre-restructure layout, check out the `pre-archive-refactor`
tag.

## License

Apache-2.0. Third-party components vendored under `archive/code/lib/`
(LimiX, TabICL) -- see `archive/LICENSES/`.

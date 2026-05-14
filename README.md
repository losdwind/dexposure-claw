# graph-dexposure

This repo hosts the **DeXposure-Agent** paper (forecast-driven risk monitoring
on DeFi exposure graphs), built on top of the **DeXposure-FM** foundation model.

## Active work

The current paper lives entirely under [`DeXposure_Agent/`](DeXposure_Agent/).
See [`DeXposure_Agent/CLAUDE.md`](DeXposure_Agent/CLAUDE.md) for the experiment
SOP (GPU server access, sync flow, benchmark runners, LLM eval).

Shared resources at repo root:

- `data/` -- weekly graph snapshots (Git LFS) + `meta_df.csv`
- `checkpoints/dexposure-fm-release/` -- trained FM weights (h1, h4, h8-h12)

Both paths are referenced by `DeXposure_Agent/dexposure_agent/fm_predictor.py`
via relative paths from the repo root, so commands are expected to run with
the repo root as cwd.

## Archive

Previous-generation code and reference papers are kept under
[`archive/`](archive/) (read-only references):

- `archive/code/` -- DeXposure-FM training pipeline (`run_full_experiment.py`,
  `run_task2_model_based.py`, ...) plus the GraphPFN/LiMiX backbone
  libraries (`lib/`, `bin/`, `exp/`) and FM-side `analysis/` and
  `autoresearch/`.
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

To recover the pre-restructure layout, check out the
`pre-archive-refactor` tag.

## Reproducing the current paper

1. Activate the project environment (`uv sync` from repo root; GPU work
   happens on a separate vast.ai instance with `torch 2.11.0+cu126`).
2. Follow [`DeXposure_Agent/CLAUDE.md`](DeXposure_Agent/CLAUDE.md) for the
   end-to-end experiment SOP.
3. The benchmark runner is
   [`DeXposure_Agent/experiments/run_all.py`](DeXposure_Agent/experiments/run_all.py).

## License

Apache-2.0. Third-party components vendored under `archive/code/lib/`
(LimiX, TabICL) -- see `archive/LICENSES/`.

# Archive

Read-only historical material for this repo.

## Contents

| Path | What lives here |
|------|-----------------|
| `code/lib/` | GraphPFN + LiMiX library source (upstream fork) |
| `code/bin/` | GraphPFN original training and evaluation scripts |
| `code/exp/` | GraphPFN experiment artifacts (~13 MB) |
| `code/dexposure_fm/` | FM helper modules (macroprudential tools, network statistics) |
| `code/analysis/` | FM-paper analysis scripts (AUPRC baselines, neg/pos ratio, training-time analysis) |
| `code/autoresearch/` | FM-side hyperparameter search (distinct from `DeXposure_Agent/autoresearch/`) |
| `code/run_full_experiment.py` | DeXposure-FM main training entry (ROLAND, Frozen, DeXposure-FM, Persistence baselines) |
| `code/run_task2_model_based.py` | Task II forward-looking risk experiments (~115 KB) |
| `code/run_macroprudential_tools.py` | SIS / spillover / contagion CLI |
| `code/run_optuna_search.py` | Optuna hyperparameter search |
| `code/Makefile` | Convenience targets for the FM-era pipeline |
| `papers/` | DeXposure-FM PDF, FM-paper review, GraphPFN paper, GoT, arXiv preprint |
| `docs/` | `EXPERIMENT_PLAN.md`, `ACTION_CONDITIONED_ROADMAP.txt`, writing guide |
| `LICENSES/` | LimiX and TabICL third-party licenses (tracks `code/lib/`) |
| `.tools/` | Legacy AI-tool project state (Sisyphus boulder.json, Serena project.yml) |

## Provenance

- These directories were active during the DeXposure-FM paper (ICML 2026
  GFM workshop submission).
- They were moved here on 2026-05-15 as part of the archive restructure
  described in
  [`docs/superpowers/specs/2026-05-15-archive-restructure-design.md`](../docs/superpowers/specs/2026-05-15-archive-restructure-design.md).
- The last commit before the restructure is tagged
  `pre-archive-refactor` (see `git show pre-archive-refactor`).

## Reproducing the FM training pipeline

The FM weights are already published at
`checkpoints/dexposure-fm-release/` (h1, h4, h8-h12), so the active paper
(DeXposure_Agent) does not need to retrain. If retraining is needed:

1. Restore the original layout: `git checkout pre-archive-refactor` (or
   manually `cp -r archive/code/* .` into a worktree).
2. Recreate the training environment: Python 3.12.9, CUDA 12.1,
   PyTorch 2.2.1+cu121, DGL 2.1.0+cu121 (see the original `pyproject.toml`
   on the tag).
3. Follow the commands in `archive/docs/EXPERIMENT_PLAN.md`.

The `archive/code/run_full_experiment.py` script saves model state via
`torch.save` only for the DeXposure-FM/GraphPFN families. ROLAND
baselines are trained on the fly each run and weights are not persisted.

## Do not run from here

Running scripts in place would resolve relative paths against `archive/code/`
and pollute it with intermediate outputs. Always materialize a working copy
elsewhere first.

# Archive Restructure Design

Date: 2026-05-15
Branch: feat/dexposure-agent-plugin
Status: approved, pending implementation plan

## Problem

The repo currently mixes three layers in one tree:

- **Layer A — GraphPFN upstream**: `lib/`, `bin/`, `exp/`, `LICENSES/`,
  `graphpfn-paper.pdf`, plus `pyproject.toml` still naming itself "GraphPFN".
- **Layer B — DeXposure-FM (previous paper)**: `dexposure_fm/`, four
  `run_*.py` training entries (130 KB + 115 KB among them), `analysis/`,
  a root `autoresearch/`, FM-era docs (`EXPERIMENT_PLAN.md`,
  `ACTION_CONDITIONED_ROADMAP.txt`), and the FM PDF + review notes.
- **Layer C — DeXposure-Agent (current paper)**: `DeXposure_Agent/`.

`DeXposure_Agent/` is already self-contained — no Python in it imports
`lib/`, `bin/`, `dexposure_fm/`, `analysis/`, or any root `run_*.py`. The
FM dependency is already crystallized into `checkpoints/dexposure-fm-release/`.
But the surrounding clutter is currently:

1. Polluting context windows during paper work.
2. Causing real stale-path bugs (the previous session found 11 sites still
   pointing to a long-deleted `DeXposure/data/` layout).
3. Hiding which files are live vs dormant.

## Goal

Reorganize so the live paper (`DeXposure_Agent/`) sits next to its shared
resources (`data/`, `checkpoints/`) at the repo root, while Layer A + B
move under a single `archive/` subtree that is read-only reference material.
No git history is destroyed: every move uses `git mv`, and a tag
`pre-archive-refactor` marks the pre-restructure state.

## Out of Scope

- Migrating Layer A + B to a separate GitHub repo (rejected option).
- Flattening `DeXposure_Agent/` to repo root (rejected option).
- Reconciling `pyproject.toml` deps with the actual GPU-server environment
  (`torch 2.11.0+cu126` vs declared `torch==2.2.1+cu121`). That predates
  this work and stays a separate ticket.

## Final Layout

```
graph-dexposure/
|
+-- DeXposure_Agent/             current paper (unchanged in this refactor)
|   +-- dexposure_agent/
|   +-- experiments/
|   +-- scripts/
|   +-- plugin/, sections/, figures/
|   +-- autoresearch/
|   +-- docs/, tests/, results/
|   +-- CLAUDE.md
|
+-- data/                        shared: weekly snapshots + meta_df (unchanged)
+-- checkpoints/                 shared: FM weights (unchanged)
|
+-- archive/                     historical code + reference docs
|   +-- README.md                inventory + last-known-good commit
|   +-- code/                    Layer A + B source
|   |   +-- lib/                 GraphPFN + LiMiX library source
|   |   +-- bin/                 GraphPFN training entries
|   |   +-- exp/                 GraphPFN experiment artifacts (~13 MB)
|   |   +-- dexposure_fm/        FM helper modules
|   |   +-- analysis/            FM analysis scripts
|   |   +-- autoresearch/        FM hyperparameter search (root one, not the Agent one)
|   |   +-- run_full_experiment.py
|   |   +-- run_macroprudential_tools.py
|   |   +-- run_optuna_search.py
|   |   +-- run_task2_model_based.py
|   |   +-- Makefile             FM-era Makefile
|   +-- papers/                  all reference PDFs
|   |   +-- DeXposure_FM.pdf
|   |   +-- DeXposure-FM_Paper_Review.pdf
|   |   +-- graphpfn-paper.pdf
|   |   +-- GoT_2-3.pdf
|   |   +-- 2605.05145v1.pdf
|   +-- docs/                    FM-era docs
|   |   +-- EXPERIMENT_PLAN.md
|   |   +-- ACTION_CONDITIONED_ROADMAP.txt
|   |   +-- academic-writing-cs-financial.md
|   +-- LICENSES/                LimiX/TabICL licenses (track with lib/)
|   +-- .tools/                  legacy AI tool state
|       +-- sisyphus/
|       +-- serena/
|
+-- pyproject.toml               renamed "GraphPFN" -> "dexposure-agent"
+-- uv.lock
+-- README.md                    rewritten: short pointer to DeXposure_Agent/ + archive/
+-- CLAUDE.md                    unchanged (6-line workstyle directives)
+-- LICENSE, NOTICE, CODE_OF_CONDUCT.md, CONTRIBUTING.md
+-- .gitignore                   updated (see below)
+-- .gitattributes               unchanged
+-- .github/, .claude/           unchanged
```

## Move + Delete Inventory

### A. `git mv` to archive/ (17 items, history preserved)

To `archive/code/`:
- `lib/`, `bin/`, `exp/`, `dexposure_fm/`, `analysis/`, `autoresearch/`
- `run_full_experiment.py`, `run_macroprudential_tools.py`,
  `run_optuna_search.py`, `run_task2_model_based.py`
- `Makefile`

To `archive/papers/` (rename to remove spaces/colons):
- `DeXposure_FM.pdf` -> `archive/papers/DeXposure_FM.pdf`
- `DeXposure-FM Paper Review.pdf` -> `archive/papers/DeXposure-FM_Paper_Review.pdf`
- `graphpfn-paper.pdf` -> `archive/papers/graphpfn-paper.pdf`
- `GoT.2:3.pdf` -> `archive/papers/GoT_2-3.pdf`
- `2605.05145v1.pdf` -> `archive/papers/2605.05145v1.pdf` (untracked, `mv` not `git mv`)

To `archive/docs/`:
- `EXPERIMENT_PLAN.md`, `ACTION_CONDITIONED_ROADMAP.txt`,
  `academic-writing-cs-financial.md`
- `LICENSES/` -> `archive/LICENSES/`

To `archive/.tools/`:
- `.sisyphus/` -> `archive/.tools/sisyphus/`
- `.serena/` -> `archive/.tools/serena/`

### B. Direct delete (4 items)

- `icml2026/` (LaTeX build artifacts: `.aux/.bbl/.blg/.log/.fls/.fdb_latexmk/.out`)
- `.run_full_experiment.lock`
- `tests/` (top-level, only contains `__init__.py`; real tests live in `DeXposure_Agent/tests/`)
- `results/` (top-level, empty; real results live in `DeXposure_Agent/results/`)

### C. Unchanged (13 items)

- `DeXposure_Agent/`, `data/`, `checkpoints/`
- `README.md` (content changes, file stays), `CLAUDE.md`, `pyproject.toml` (content changes)
- `uv.lock`, `LICENSE`, `NOTICE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`
- `.gitignore` (content changes), `.gitattributes`
- `.github/`, `.claude/`

## Config Changes

### `pyproject.toml`

| Field | Current | New |
|---|---|---|
| `name` | `"GraphPFN"` | `"dexposure-agent"` |
| `description` | `"Official implementation of GraphPFN"` | `"DeXposure-Agent: forecast-driven risk monitoring on DeFi exposure graphs"` |
| `[tool.setuptools.packages.find].where` | `["."]` | `["DeXposure_Agent"]` |
| `[tool.setuptools.packages.find].include` | `["lib*", "dexposure_fm*"]` | `["dexposure_agent*", "experiments*"]` |
| `[tool.ruff].extend-exclude` | `["lib/limix", "lib/tabpfn"]` | `["archive/**"]` |
| `[tool.pyright].exclude` | `["cache", "data", "exp", "local"]` | `["cache", "data", "archive", "checkpoints", "local"]` |

Dependencies untouched.

### `.gitignore`

Add:
```
# Process locks
.run_full_experiment.lock
*.lock

# Legacy AI tool state (now archived)
.sisyphus/
.serena/

# Build outputs
build/
```

Remove the stale `DeXposure/` entry (L148). LaTeX patterns already present.

### Root `README.md` (full rewrite)

Replace the entire FM-era README with a short pointer:

```markdown
# graph-dexposure

This repo hosts the DeXposure-Agent paper (forecast-driven risk monitoring
on DeFi exposure graphs), built on top of the DeXposure-FM foundation model.

## Active work

The current paper lives entirely under [`DeXposure_Agent/`](DeXposure_Agent/).
See [`DeXposure_Agent/CLAUDE.md`](DeXposure_Agent/CLAUDE.md) for the experiment SOP.

Shared resources at repo root:
- `data/` -- weekly graph snapshots (Git LFS) + meta_df.csv
- `checkpoints/dexposure-fm-release/` -- trained FM weights (h1/h4/h8-h12)

## Archive

Previous-generation code and papers are kept under [`archive/`](archive/):
- `archive/code/` -- DeXposure-FM training pipeline (run_full_experiment.py, ...)
  and the GraphPFN/LiMiX backbone libraries (`lib/`, `bin/`, `exp/`)
- `archive/papers/` -- DeXposure-FM PDF, GraphPFN paper, review notes, references
- `archive/docs/` -- FM-era EXPERIMENT_PLAN, ROADMAP, writing guide

These are read-only references; do not run from `archive/`.

## License

Apache-2.0. Third-party components: see `archive/LICENSES/` (LimiX, TabICL).
```

### `archive/README.md` (new)

Short inventory (~30 lines):
- What lives under each subdirectory
- Last commit known to run the FM training pipeline end-to-end
- The Python environment that produced it (torch 2.2.1+cu121,
  dgl 2.1.0+cu121, CUDA 12.1)
- Pointer to `archive/docs/EXPERIMENT_PLAN.md` for detailed reproduction args
- Note: FM weights already at `checkpoints/dexposure-fm-release/`; archive
  code is mainly needed if retraining

### Root `Makefile`

Delete from repo root. A copy stays at `archive/code/Makefile` for history.
`DeXposure_Agent/` uses `scripts/` runners directly; no new root Makefile.

### Root `CLAUDE.md`

Unchanged (current content is 6 lines of workstyle directives, structurally
independent).

## Migration Order (7 phases, 3 commits)

### Phase 0 — Tag pre-refactor state (no commit, no working-tree change)
Capture the current HEAD as a recovery point before anything moves:
```
git tag -a pre-archive-refactor -m "Last commit before archive/ restructure"
```
The tag is created locally; it is pushed in Phase 6.

### Phase 1 — Direct deletes
```
git rm -r icml2026/ tests/ results/
git rm .run_full_experiment.lock
git commit -m "chore: remove stale top-level dirs and process lock"
```

### Phase 2 — Build archive skeleton
```
mkdir -p archive/{code,papers,docs,.tools}
```
(no commit; folds into Phase 3)

### Phase 3 — `git mv` to archive
Move in batches by destination (code, papers, docs, tools). Renames that
strip spaces/colons happen as a second `git mv` step on the target name,
not combined with the move, because git/setuptools handle that more
predictably:
```
git mv lib bin exp dexposure_fm analysis autoresearch archive/code/
git mv run_full_experiment.py run_macroprudential_tools.py \
       run_optuna_search.py run_task2_model_based.py Makefile \
       archive/code/

git mv DeXposure_FM.pdf archive/papers/DeXposure_FM.pdf
git mv "DeXposure-FM Paper Review.pdf" archive/papers/DeXposure-FM_Paper_Review.pdf
git mv graphpfn-paper.pdf archive/papers/graphpfn-paper.pdf
git mv "GoT.2:3.pdf" archive/papers/GoT_2-3.pdf
mv 2605.05145v1.pdf archive/papers/                     # untracked, plain mv

git mv EXPERIMENT_PLAN.md ACTION_CONDITIONED_ROADMAP.txt \
       academic-writing-cs-financial.md archive/docs/
git mv LICENSES archive/LICENSES

git mv .sisyphus archive/.tools/sisyphus
git mv .serena archive/.tools/serena

git commit -m "refactor: archive DeXposure-FM training code, papers, and FM-era docs"
```

### Phase 4 — Config updates
- Edit `pyproject.toml`
- Edit `.gitignore`
- Rewrite root `README.md`
- Write `archive/README.md`

```
git add pyproject.toml .gitignore README.md archive/README.md
git commit -m "refactor: update root config and docs for archived layout"
```

### Phase 5 — Verification (no commit)
```
# Package metadata still resolves
uv sync --dry-run

# Current paper imports still work
cd DeXposure_Agent
python -c "
import sys; sys.path.insert(0, '.')
from dexposure_agent.fm_predictor import FMPredictor
from experiments.run_all import build_arg_parser
print('imports OK')
"

# FM checkpoint relative path still resolves
ls ../checkpoints/dexposure-fm-release/*.pt

# Data still in place
ls ../data/historical-network_week_*.json ../data/meta_df.csv

# run_all CLI parses
cd .. && python DeXposure_Agent/experiments/run_all.py --help | head -10

# Existing tests still green
cd DeXposure_Agent && python -m pytest tests/ -x --tb=short
```

Each check that fails rolls back its phase (Phase 3 or 4) and gets fixed in
isolation. The verification phase intentionally produces no commit so a
clean rollback is `git reset --hard HEAD~1` (or `~2`).

### Phase 6 — Push
After Phase 5 verification passes:
```
git push origin feat/dexposure-agent-plugin
git push origin pre-archive-refactor
```

## Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| `pyproject.toml [tool.setuptools.packages.find]` misconfigured -> `uv sync` package discovery breaks | Medium | Phase 5a `uv sync --dry-run` catches this before push; revert to current behavior if needed |
| `fm_predictor.py` hardcoded `"checkpoints/dexposure-fm-release/"` fails because of cwd shift | Low | `checkpoints/` stays at repo root; DeXposure_Agent entries already assume cwd=repo root |
| GPU server copy gets out of sync | Low | rsync flow only syncs `DeXposure_Agent/`, `data/`, `checkpoints/`; archive never goes to server |
| LFS pointer files accidentally inflate to real blobs during `git mv` | Low | `.gitattributes` is untouched; git preserves pointer encoding through renames |
| GitHub Actions break | Low | Confirmed `.github/` references zero archive-bound paths |
| Future need to retrain FM | Medium | `pre-archive-refactor` tag pinpoints the last working layout; `archive/code/` is complete and runnable in place |

## Verification Checklist

After Phase 5 completes, all of the following must hold:

- [ ] `uv sync --dry-run` reports no package-discovery errors
- [ ] `python -c "from dexposure_agent.fm_predictor import FMPredictor; from experiments.run_all import build_arg_parser"` succeeds (run from `DeXposure_Agent/`)
- [ ] `checkpoints/dexposure-fm-release/dexposure-fm-h{1,4,8-h12}.pt` all present
- [ ] `data/historical-network_week_*.json` and `data/meta_df.csv` all present
- [ ] `python DeXposure_Agent/experiments/run_all.py --help` returns usage
- [ ] `python -m pytest DeXposure_Agent/tests/` exits 0
- [ ] No `grep -r "from lib\|from dexposure_fm\|run_full_experiment\|import lib" DeXposure_Agent/` results
- [ ] `pre-archive-refactor` tag points at the commit before Phase 1

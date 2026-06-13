Write Markdown in a style that remains as compatible with plain text as possible.
Launch all cloud GPU training exclusively via `cloud/train.sh <training command>` (SkyPilot + Vast.ai; the instance auto-destroys after the job finishes, so it can never be left running by mistake). Never create Vast instances manually via the web console or CLI, and never use bare `sky launch` without autodown. Code sync is handled automatically by SkyPilot (`workdir: .`, exclusions in `.skyignore`) -- no manual rsync needed. After launching, confirm with `sky status`; collect logs with `sky logs <cluster>`.
Training artifacts (checkpoints, logs) must be uploaded at the end of the run step in `cloud/train.yaml` -- the instance disk is destroyed afterwards.
Make full use of available compute resources and avoid leaving them idle. If cloud training is needed, plan the sequence of tasks all at once instead of running only one small task and then waiting for me to say what to do next.
Always set up comprehensive logging.
Make sure there is a mechanism for saving model weights and checkpoints.

When replying to me, if you use model or method identifiers such as M1, M2, B1, B2, C1, or C2, always include their abbreviated names or another understandable label to reduce cognitive load.

## Overleaf sync (EMNLP paper)

The EMNLP paper `paper-emnlp-industry/` mirrors an Overleaf project the advisor also edits.
- Overleaf project: https://www.overleaf.com/project/6a21893fcad07ee92afa84d5
- Overleaf git remote: https://git.overleaf.com/6a21893fcad07ee92afa84d5 (default branch `main`).
- A dedicated local clone lives at `../overleaf-dexposure-emnlp` (sibling of this repo, NOT tracked here). The git token is stored only in `~/.netrc` (chmod 600) and that clone's local `.git/config` -- it is NEVER written into any tracked file. This repo is public on GitHub, so do not put the Overleaf token (or any secret) in CLAUDE.md or any committed file.
- Overleaf has its own structure with the `.tex`/`figures` at the repo root (no `paper-emnlp-industry/` prefix), and the advisor may have edits there. Before pushing, `git pull` the clone and `diff` against `paper-emnlp-industry/` to confirm you are only changing your own edits -- never blind-overwrite. Sync = copy the changed source files + figures into the clone, commit, `git push origin main`.

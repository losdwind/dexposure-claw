Write Markdown in a style that remains as compatible with plain text as possible.
Launch all cloud GPU training exclusively via `cloud/train.sh <training command>` (SkyPilot + Vast.ai; the instance auto-destroys after the job finishes, so it can never be left running by mistake). Never create Vast instances manually via the web console or CLI, and never use bare `sky launch` without autodown. Code sync is handled automatically by SkyPilot (`workdir: .`, exclusions in `.skyignore`) -- no manual rsync needed. After launching, confirm with `sky status`; collect logs with `sky logs <cluster>`.
Training artifacts (checkpoints, logs) must be uploaded at the end of the run step in `cloud/train.yaml` -- the instance disk is destroyed afterwards.
Make full use of available compute resources and avoid leaving them idle. If cloud training is needed, plan the sequence of tasks all at once instead of running only one small task and then waiting for me to say what to do next.
Always set up comprehensive logging.
Make sure there is a mechanism for saving model weights and checkpoints.

When replying to me, if you use model or method identifiers such as M1, M2, B1, B2, C1, or C2, always include their abbreviated names or another understandable label to reduce cognitive load.

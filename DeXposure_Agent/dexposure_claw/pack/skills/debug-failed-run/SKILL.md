---
description: Debug a failed DeXposure Claw benchmark run from logs and manifests.
---

# Debug Failed Run

Use this when a benchmark run has unavailable predictors, runtime errors, missing
artifacts, or incomplete reports.

## Workflow

1. Inspect `results.json` summary counts.
2. Separate expected skips from unavailable predictors and runtime errors.
3. Check checkpoint hashes, git commit, data directory, and GPU API status.
4. Re-run the smallest failing benchmark/method pair before re-running the suite.
5. Record the fix in the run audit log or a self-improvement proposal.

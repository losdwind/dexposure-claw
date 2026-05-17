# Contributing

Thanks for contributing to DeXposure-Claw.

This repository contains both a research paper workflow and an agent-runtime
package. Please keep changes focused and make the intended layer clear in your
pull request.

## Development Setup

Root paper and benchmark workflow:

```bash
uv sync
python -m pytest paper/tests
```

Claw package workflow:

```bash
python -m pip install -e ./claw
dexposure-claw health
npm --prefix ./claw run pack:check
```

Some data and checkpoint files are managed with Git LFS. If a reproduction
command fails because a model or graph file is missing, first confirm that LFS
objects have been fetched.

## Code Quality

Use the narrowest relevant checks for your change:

```bash
python -m compileall claw/src/dexposure_claw paper/dexposure_agent
python -m pytest paper/tests
npm --prefix ./claw run pack:check
```

If you add or modify benchmark code, include the exact command needed to
reproduce the result and the expected output location.

## Documentation Quality

For user-facing changes, update documentation with:

- what changed;
- who should use it;
- prerequisites;
- exact commands;
- expected success output;
- known limitations or safety boundaries.

For model or method IDs such as M1, M2, B1, or B2, include a readable label
next to the identifier.

## Financial-Risk Contributions

Changes that affect risk signals, forecasts, benchmark scoring, or decision
recommendations should explain:

- the data window used;
- the benchmark task affected;
- the metric or qualitative criterion affected;
- whether the change can alter user-facing risk conclusions;
- any new failure mode or assumption.

Do not add trading, transaction-signing, wallet-control, or exchange-execution
behavior without prior design discussion and explicit safety review.

## Pull Requests

- Keep PRs focused and small when possible.
- Do not mix paper-result changes with unrelated package or documentation work.
- Include tests or reproduction commands when behavior changes.
- Mention any skipped checks and why they were skipped.
- Never include secrets, API keys, private keys, wallet material, or private RPC
  credentials in commits, logs, screenshots, or issue comments.

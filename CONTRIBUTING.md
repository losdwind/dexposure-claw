# Contributing

Thanks for contributing!

## Development setup

This project uses `uv` for Python dependency management.

```
uv sync
```

If you need the full DeXposure dataset and did not fetch it via Git LFS:

```
uv run python bin/download_dataset.py
```

## Code quality

- Lint: `make lint`
- Format: `make format`

## Pull requests

- Keep PRs focused and small when possible.
- If you add a new experiment or metric, include a short description and the command needed to reproduce it.


# Security Policy

DeXposure-Claw is a research and monitoring project for DeFi risk workflows. It
must be treated as decision-support software, not as an execution or custody
system.

## Supported Versions

The `main` branch is the active development target. Security fixes are applied
there first.

## Reporting A Vulnerability

Please report suspected vulnerabilities privately before opening a public issue.
If there is no published security contact for the project, contact the
maintainers through the repository owner and include:

- affected component or file path;
- steps to reproduce;
- expected impact;
- whether secrets, model artifacts, data files, or external services are
  involved;
- any temporary mitigation you have already tested.

Do not include active API keys, wallet keys, private RPC credentials, or private
exchange credentials in reports.

## Secret Handling

Never commit secrets. This includes:

- LLM provider API keys;
- wallet private keys or seed phrases;
- exchange API keys;
- private RPC URLs;
- database credentials;
- cloud access tokens;
- signing keys.

The repository ignores `.env` files, but that is only a last line of defense.
If a secret is exposed in a terminal, screenshot, commit, issue, pull request,
or chat message, rotate it immediately.

## Financial Safety Boundaries

The project should not:

- execute trades;
- sign blockchain transactions;
- transfer funds;
- custody assets;
- bypass human review for financial decisions;
- present model output as guaranteed live-market truth.

Risk recommendations should be traceable to data, model version, benchmark task,
and timestamp whenever practical.

## Dependency And Artifact Safety

Large data files, model checkpoints, and vendored third-party components should
keep clear provenance. When adding or updating them, document:

- source;
- license or terms;
- checksum when practical;
- expected path;
- command used to obtain or regenerate the artifact.

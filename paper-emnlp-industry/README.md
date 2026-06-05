# paper-emnlp-industry/

EMNLP 2026 Industry Track submission workspace.
Parallel to [`../paper/`](../paper/) (TKDE version, untouched).

## Status (2026-06-02 — DELIVERABLE)

[`main.pdf`](main.pdf) compiles cleanly. 15 total pages:

| Page | Content |
|------|---------|
| 1    | Title + Abstract + §1 Introduction (start) |
| 2    | §1 Introduction (end) + §2 Related Work |
| 3    | §3 Pipeline |
| 4    | Figure 1 (system overview) + §4 Evaluation Harness |
| 5    | §5 Empirical Results + Figure 2 (layer-wise) |
| 6    | Headline Table 1 + §6 Lessons + §7 Conclusion |
| 7    | Limitations + Ethics + References (start) |
| 8    | References (end) |
| 9–15 | Appendices A–E |

**Main text: pages 1–6 → within the 6-page budget.**
**Limitations / Ethics / References / Appendices: pages 6–15 → outside budget per CFP.**

### Quality checks passed

- Compile clean (no overfull/underfull warnings, no unresolved refs)
- All citations resolved, all figures present
- All section labels resolve correctly
- `Forecast-FM.Forecast(Gt,Xt,h)` math expressions render cleanly via `\text{}`
- Anonymisation: only literal `DeXposure` mention in submission is in the citation to the published dataset paper (`Wu et al. 2025`) — see "Known risk" below

### Known risk before submission

1. **Public GitHub repo is the deanonymisation vector.** The published dataset paper Wu et al. 2025 is titled "*Dexposure: A Dataset and Benchmarks ...*" and will render in the references list with that title. A reviewer who searches for the term will find the public `DeXposure-Claw` repo (Aijie's). **Fix: privatise the repo for the EMNLP review window (2026-06-16 to 2026-08-20).** Do NOT post the arXiv preprint until after the EMNLP commit decision.

## Compile

```bash
cd paper-emnlp-industry
latexmk -pdf main.tex
```

Style files `acl.sty` and `acl_natbib.bst` are already in this folder.

## Thesis lock (Plan Mode INSIGHT, revised 2026-06-02)

```
[INSIGHT: thesis_statement]
(G3 lead, G1 consequence)
"We define a regulator-aligned absolute-loss ground truth and use it
 to score, for the first time, an integrated FM+monitor+scenario+LLM+gate
 pipeline against what a supervisor would actually prioritise. The
 characterisation produces the first joint measurement of forecast
 quality, monitor lead time, and ticket quality on the same target,
 and the first quantification of the false-intervention cost for an
 LLM agent operating in this setting."

Honest constraints ((b) deployment status):
- "deployment-ready" / "deployable", NEVER "deployed"
- "operational characterisation on historical data", NEVER "field study"
- "audit logs from benchmark runs", NEVER "production audit logs"
```

## Anonymisation macros

Declared in `main.tex`. Flip in camera-ready by editing those four lines:

| Submission name | Real name (camera-ready) |
|-----------------|--------------------------|
| `\sysname`      | DeXposure-Claw           |
| `\fmname`       | DeXposure-FM             |
| `\benchname`    | DeXposure-Bench          |
| `\corpusname`   | DeXposure dataset        |

## File map

```
paper-emnlp-industry/
├── README.md                       <- this file
├── OUTLINE.md                      <- original 6-page plan + appendix map
├── ABSTRACT.md                     <- Draft A (final, embedded in main.tex)
├── main.tex                        <- compiles to main.pdf
├── main.pdf                        <- deliverable
├── references.bib                  <- copy of paper/reference.bib
├── acl.sty                         <- EMNLP/ACL 2026 style
├── acl_natbib.bst                  <- ACL bibliography style
├── figures/
│   ├── fig1_system_overview.tex    <- anonymised TikZ (Forecast-FM not DeXposure-FM)
│   ├── fig4_method_comparison.pdf  <- copied as-is
│   └── fig4_method_comparison.png  <- copied as-is
└── sections/
    ├── 1-intro.tex                 <- G3-lead version
    ├── 2-related.tex               <- bench-first order
    ├── 3-pipeline.tex              <- with Figure 1 inline
    ├── 4-bench.tex
    ├── 5-results.tex               <- with Table 1 + Figure 2 inline
    ├── 6-lessons.tex               <- L1–L4
    ├── 7-conclusion.tex
    ├── 8-limitations.tex
    ├── 9-ethics.tex
    ├── A-fm-notation.tex
    ├── B-pipeline-math.tex         <- with Algorithm 1
    ├── C-bench-details.tex
    ├── D-reference-methods.tex
    ├── E-full-tables.tex           <- wires A1–A5 + T4
    ├── A1-b1_forecast_AllHorizons.tex  <- copied from TKDE
    ├── A2-b4_stress_Detail.tex         <- copied
    ├── A3-b6_robustness_Detail.tex     <- copied
    ├── A4-b2_warning_Budget.tex        <- copied, sec:exp → sec:results
    ├── A5-A1_isolated_Detail.tex       <- copied
    └── T4-Ablation.tex                 <- copied, anonymised
```

## Pre-submission checklist (when you're ready to submit)

- [ ] Privatise `DeXposure-Claw` GitHub repo (CRITICAL — see Known risk above)
- [ ] Hold off on arXiv preprint posting until EMNLP commit decision (2026-08-20)
- [ ] Final read-through of [main.pdf](main.pdf) (1 hour)
- [ ] Verify EMNLP 2026 Industry Track submission portal accepts the PDF
- [ ] Submit before deadline 2026-06-16, 23:59 UTC-12 (AoE)

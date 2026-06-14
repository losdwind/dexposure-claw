# paper-emnlp-industry/

EMNLP 2026 Industry Track submission workspace.
Parallel to [`../paper/`](../paper/) (TKDE version, untouched).

## Status (2026-06-10 — DELIVERABLE, post-review revision)

[`main.pdf`](main.pdf) compiles cleanly. 18 total pages:

| Page  | Content |
|-------|---------|
| 1     | Title + Abstract + §1 Introduction (start) |
| 2     | §1 Introduction (end) + §2 Related Work + §3 Preliminaries (start) |
| 3     | §3 (end) + §4 Pipeline + Figure 1 |
| 4     | §4 (end) + §5 Experiments (benchmark setup) |
| 5     | §5.1 Results (start) |
| 6     | Tables 1–3 + §5.1 (end) + §6 Discussion and Conclusion + Limitations (start) |
| 7     | Limitations (end) + Ethics + References (start) |
| 8–9   | References (end) |
| 10–18 | Appendices A–E (Figure 2 layer-wise now lives in E) |

**Main text: §1–§6 end on page 6 → within the 6-page budget.**
**Limitations / Ethics / References / Appendices: pages 7–18 → outside budget per CFP.**

### Quality checks passed (re-run 2026-06-10)

- Compile clean (no overfull boxes, no unresolved refs; remaining
  Underfull warnings are cosmetic, inside the TikZ figure)
- All citations resolved, all figures present
- All section labels resolve correctly
- Anonymisation: only literal `DeXposure` mention in the compiled
  PDF is the citation title of the published dataset paper
  (`Wu et al. 2025`) — verified by full-text scan; see "Known risk"

### 2026-06-10 review fixes (summary)

1. CRITICAL: `\title{}` hard-coded "DeXposure-Claw" → now `\sysname`;
   `\sysname` placeholder renamed `Claw-System` → `Sentinel`
   (old one contained the searchable word "Claw").
2. CRITICAL: abstract/intro claimed the *raw-data* LLM over-triggers,
   but m2 (snapshot LLM) has FIR 0.000; over-triggering appears when
   the LLM is fed FM evidence (m6/m7, FIR 0.448/0.437). Narrative
   rewritten as a two-failure-mode story consistent with Table 1.
3. CRITICAL: "all three judges preserve m7≥m6>m2" was false for
   Gemini (m2 2.69 > m6 2.55) → claims weakened to "all rank m7 top".
4. Appendix A FM training spec corrected to match
   `checkpoints/dexposure-fm-release/config.json` (March 2020 –
   January 2025; 104 train / 12 val / 8 internal-test weeks);
   previous text said "(March 2020–January 2024)" + wrong val window.
5. A2 stress-table scenario labels were rotated (S2 "cross-protocol"
   etc.) → relabelled to match `dexposure_agent/scenario.py` and
   Appendix B (S2 bridge cluster, S3 stablecoin, S4 sector lending,
   S5 correlated top-10).
6. §E Panel B gate symbol fixed (τ_data=0 disables A1; τ_conf=0 is
   held fixed), standalone lessons folded into §7 Discussion and
   Conclusion, m5→m6 F1 text 0.0189 →
   0.0190, n=29 vs ~33 weeks explained (h=4 lookahead truncation),
   "$10 per run" cost claim scoped to full leaderboard sweep,
   judge scale (1–5) declared in Table 1 caption, persistence
   "matches" → "beats" FM on PageRank MAE, plus spelling/nbsp/
   terminology unification (British throughout, `\benchname{}` for
   bare "bench", Claude~Opus~4.7 nbsp, etc.).
7. Figure 2 (layer-wise) moved to Appendix E to keep §1–§6 within
   the 6-page budget after the honest-narrative rewrite.

### 2026-06-10 figure/table audit (second pass)

8. **L=26 → L=42.** §3 and Appendix B claimed a 26-week rolling
   baseline, but `config.py` flipped `rolling_window` 26→42 on
   2026-03-26 (commit 06b2ec9), BEFORE all 2026-05 runs — so b5
   (m1/m5) and the T4/A5 ablations actually ran with L=42. Only
   `b2_early_warning.py` uses a hard-coded 26-week baseline; B now
   states both. Also added K=5 for attribution top-K (config
   `top_k`). All other §3/B hyperparameters verified against
   `AgentConfig` (τ_data 0.7, τ_conf 0.6, z=2, π_min 0.2, M=50,
   λ=0.5, damping 0.85 / 10 iters, ε=1e-8).
9. Table 1: added Prec. column (m1 0.720 / m2 0.575 / m5 0.600 /
   m6 0.570 / m7 0.580, from b5_tkde_v4 + llm_eval summaries) —
   fixes the dangling "0.720, in §E" pointer (§E never contained
   per-method b5 precision).
10. §5 judge-robustness: added per-week judge std (Gemini 1.3–1.5
    vs Opus/GPT 0.3–0.6, from rejudge_aggregated.json) to put the
    Gemini m2>m6 inversion in noise context.
11. Bolding errors in appendix tables ("best" markers are factual
    claims): A2 S2 loss-MAE bold was on m1 0.0150 but the best is
    m3 0.0116; A3 low_data_10pct HHI bold was on m5 0.0587 but
    best is m4 0.0586, and Δrel bold was on m5 +0.607 but smallest
    is m3 +0.276; A1 h8 HHI best (0.0398) was unbolded. Tie policy
    unified ("ties share the bold") across A1/A2/A3 and stated in
    captions. A3 caption now also notes EvolveGCN's ≈0 Δrel under
    the two feature regimes.
12. Figure 1 notation: MC-sample superscript `(1..S)` → `(1..M)`
    (M=50; S is the scenario symbol), Layer-2 output `losses L_t`
    → `scenarios R_t` to match Algorithm 1. Fixed in both
    `figures/fig1_system_overview.tex` and the generator
    `paper/scripts/redraw_emnlp_industry_fig1.py`.

### 2026-06-11 statistical-rigor pass (third pass)

13. **New Appendix F** (`sections/F-statistics.tex`): 95% bootstrap
    CIs (10k week resamples) + paired sign-flip permutation tests
    (20k) for all headline comparisons, plus a matched-budget
    recall@k / precision@k analysis. Backing artifacts:
    `paper/results/bootstrap_stats.json`,
    `paper/results/matched_budget.json`; scripts:
    `paper/scripts/bootstrap_stats.py`,
    `paper/scripts/matched_budget.py` (pure local compute, no API).
14. Significance landscape: FM-signal F1 lift p=0.0002; full-stack
    vs rules F1 p<1e-4; Sonnet F1 lift p=0.0004; FIR emergence
    p<1e-4 — all confirmed. NOT significant: gate's F1/judge
    effects (p=0.41/1.0), judge +9.4% under Opus 4.8 (p=0.23; only
    GPT-5.5 significant), Sonnet judge +0.20 (p=0.21). §5 wording
    calibrated accordingly ("strict Pareto" → "Pareto"; judge axis
    labelled directional).
15. Matched-budget result (new §5 paragraph + Table F.3): at
    matched top-k budgets the LLM variants still ~double the rules
    baseline's precision@k (p≤0.0055), but FM-fed and raw-snapshot
    LLMs become indistinguishable — the FM's F1 gain comes from
    supporting a larger target set at undiminished precision, not
    a sharper ranking top. Honest mechanism now stated explicitly.
16. m1 per-week data from a LOCAL re-run (GPU instance destroyed):
    `paper/results/local_rerun_m1/` via a new weekly-dump hook in
    `experiments/b5_decision_quality.py`. Reproduction status:
    recall@k, pool size (283.5), FIR reproduce exactly; F1 0.0081
    (published 0.0076 inside the CI [0.0041, 0.0123]); precision
    0.633 vs published 0.720 (borderline tickets flip across
    environments at <1 ticket/week) — disclosed in Table F.1 note.
17. Per-method targets/week measured: m1 2.1, m2 4.9, m7 6.3,
    m6 6.5, m7×Sonnet 7.6.

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

One switch in `main.tex` drives the whole paper. Flip the single line:

- `\anonfalse` — real names (working copy, e.g. the shared Overleaf with the advisor)
- `\anontrue`  — placeholders (EMNLP double-blind submission)

| Macro         | `\anontrue` (submission) | `\anonfalse` (real)   |
|---------------|--------------------------|-----------------------|
| `\sysname`    | Sentinel                 | DeXposure-Claw        |
| `\fmname`     | Forecast-FM              | DeXposure-FM          |
| `\benchname`  | Decision-Bench           | DeXposure-Bench       |
| `\corpusname` | Exposure-Corpus          | DeXposure             |

Before submitting, set `\anontrue` and confirm content still ends on
page 6 (real names are longer and spill onto page 7; the placeholders
fit). Every name in the body goes through these macros — no literal
real name remains except the third-party dataset citation
(Wu et al. 2025), which is a legitimate reference and stays.

2026-06-10 revision: `\sysname` placeholder renamed from `Claw-System`
to `Sentinel` — the old placeholder contained the distinctive word
"Claw", and a search for "Claw DeFi" surfaces the public
`DeXposure-Claw` repo. The `% was: DeXposure-*` comments were also
removed from `main.tex`; this table is now the only mapping record.
Additionally fixed: `\title{}` previously hard-coded the literal
real name `DeXposure-Claw` (the worst leak); it now uses `\sysname`.

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
    ├── 3-pre.tex                   <- preliminaries / notation
    ├── 4-pipeline.tex              <- with Figure 1 inline
    ├── 5-experiments.tex           <- Experiments setup / evaluation harness
    ├── 5-results.tex               <- Results subsection + Tables 1–3
    ├── 6-discussion-conclusion.tex <- Discussion and Conclusion
    ├── limitations.tex             <- required unnumbered section
    ├── ethics.tex                  <- required unnumbered section
    ├── AppA-algorithm-details.tex
    ├── AppB-implementation-details.tex
    ├── AppC-experimental-results.tex
    └── T4-Ablation.tex
```

## Pre-submission checklist (when you're ready to submit)

- [ ] Privatise `DeXposure-Claw` GitHub repo (CRITICAL — see Known risk above)
- [ ] Hold off on arXiv preprint posting until EMNLP commit decision (2026-08-20)
- [ ] Final read-through of [main.pdf](main.pdf) (1 hour)
- [ ] Verify EMNLP 2026 Industry Track submission portal accepts the PDF
- [ ] Submit before deadline 2026-06-16, 23:59 UTC-12 (AoE)

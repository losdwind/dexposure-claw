# DeXposure-Agent Evaluation Recovery Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the experimental design so the paper can make defensible claims about DeXposure-Agent rather than relying on benchmarks that currently collapse to persistence or measure the wrong thing.

**Architecture:** Keep the current pipeline intact, but fix the evaluation surface around it. The recovery work is split into three layers: first expose the FM signal that already exists, then remove benchmark design errors that currently force m5_fm_rules and m1_persistence_rules to look identical, then align the paper text and reproducibility claims with what is actually implemented and run.

**Tech Stack:** Python 3.12, NumPy, SciPy, NetworkX, pytest, JSON benchmark outputs, LaTeX.

---

## File Map

**Modify**
- `DeXposure_Agent/experiments/b1_risk_forecasting.py`: add node-ranking and edge-level forecast metrics; add horizon/volatility slices.
- `DeXposure_Agent/experiments/b2_early_warning.py`: make warning generation depend on predicted graphs and predicted future metric changes.
- `DeXposure_Agent/experiments/b3_uncertainty_calibration.py`: replace pseudo-calibration with conformalized intervals or clearly scoped empirical uncertainty.
- `DeXposure_Agent/experiments/b5_decision_quality.py`: replace broken audit metric with precision/recall-style target metrics and stricter stressed-protocol definition.
- `DeXposure_Agent/experiments/b6_robustness.py`: remove or relabel fake low-data regimes; keep only inference-time corruption unless retraining is added.
- `DeXposure_Agent/experiments/predict_helper.py`: route implemented baselines explicitly and fail closed for unimplemented methods.
- `DeXposure_Agent/experiments/run_all.py`: expose the real applicability matrix and include new baseline/statistics runners.
- `DeXposure_Agent/scripts/run_benchmarks_sequential.py`: run the repaired benchmark set, not just `m5_fm_rules` and `m1_persistence_rules`.
- `DeXposure_Agent/sections/Exp.tex`: update evaluation protocol, benchmark definitions, and reproducibility claims.
- `DeXposure_Agent/DeXposure-Agent.tex`: remove unsupported “significantly outperforming SOTA” language until results justify it.
- `DeXposure_Agent/EXPERIMENT_PLAN.txt`: sync the paper-facing plan to the repaired benchmark definitions.
- `DeXposure_Agent/CLAUDE.md`: sync operational notes and defaults (`pi_min`, `rolling_window`, `top_k`, active baselines).

**Create**
- `DeXposure_Agent/experiments/metrics_ranking.py`: `ndcg_at_k`, `map_at_k`, `precision_at_k`, `recall_at_k`.
- `DeXposure_Agent/experiments/metrics_edge.py`: edge-level AUROC/AUPRC helpers for graph prediction.
- `DeXposure_Agent/experiments/stats.py`: paired bootstrap CI and Wilcoxon helpers.
- `DeXposure_Agent/tests/test_b1_metrics.py`: ranking and edge-metric unit tests.
- `DeXposure_Agent/tests/test_b2_early_warning.py`: tests that b2_warning changes when predicted graph changes.
- `DeXposure_Agent/tests/test_b3_uncertainty.py`: tests for conformal or interval-calibration logic.
- `DeXposure_Agent/tests/test_b5_decision_quality.py`: tests for revised completeness/precision-at-K logic.
- `DeXposure_Agent/tests/test_b6_robustness.py`: tests that low-data regimes are either removed or explicitly marked as evaluation subsampling.

**Optional Create If Needed**
- `DeXposure_Agent/experiments/baseline_runner.py`: thin integration layer if C1/m4_fm_only/C5 wiring becomes too messy inside existing benchmark files.

## Submission Strategy

**Track A: Minimum submittable version**
- Make FM advantages visible in b1_forecast.
- Add at least `m4_fm_only` and `C5` comparisons.
- Fix b2_warning so it uses predictions.
- Fix b5_decision so its target metric is not degenerate.
- Remove unsupported claims in the paper.

**Track B: Stronger version**
- Replace b3_calibration with conformal prediction or another defensible uncertainty method.
- Add 3-seed reporting and paired significance tests.
- Add true walk-forward evaluation.
- Reintroduce a low-data claim only after retraining.

## Chunk 1: Recover the Core Narrative

### Task 1: Make b1_forecast expose FM signal instead of only aggregate graph summaries

**Files:**
- Create: `DeXposure_Agent/experiments/metrics_ranking.py`
- Create: `DeXposure_Agent/experiments/metrics_edge.py`
- Modify: `DeXposure_Agent/experiments/b1_risk_forecasting.py`
- Test: `DeXposure_Agent/tests/test_b1_metrics.py`

- [ ] **Step 1: Write failing tests for ranking metrics**
Run: `pytest DeXposure_Agent/tests/test_b1_metrics.py -q`
Expected: FAIL because `metrics_ranking.py` and `metrics_edge.py` do not exist yet.

- [ ] **Step 2: Add ranking helpers**
Implement `ndcg_at_k`, `map_at_k`, `precision_at_k`, and `recall_at_k` over node risk-score dictionaries with deterministic tie handling.

- [ ] **Step 3: Add edge-level helpers**
Implement AUROC/AUPRC over `(source, target)` edge labels and predicted existence scores. Fail fast when labels are single-class.

- [ ] **Step 4: Extend b1_forecast outputs**
Add these fields to `B1Result` and serialized JSON:
`ndcg_at_10`, `map_at_10`, `edge_auroc`, `edge_auprc`, `high_change_slice`, `stable_slice`.

- [ ] **Step 5: Slice by change regime**
Define a per-snapshot “high-change week” from realized graph turnover or realized metric delta, then report b1_forecast metrics separately for high-change and stable weeks.

- [ ] **Step 6: Verify horizon behavior**
Run: `python DeXposure_Agent/experiments/b1_risk_forecasting.py --method m5_fm_rules --data-dir ...`
Expected: JSON now contains node-ranking and edge-level metrics, and horizon-specific outputs can show whether FM gains increase with `h`.

- [ ] **Step 7: Commit**
```bash
git add DeXposure_Agent/experiments/metrics_ranking.py DeXposure_Agent/experiments/metrics_edge.py DeXposure_Agent/experiments/b1_risk_forecasting.py DeXposure_Agent/tests/test_b1_metrics.py
git commit -m "feat: expose ranking and edge metrics in b1_forecast"
```

### Task 2: Add the baseline floor needed to support any “better than baseline” claim

**Files:**
- Modify: `DeXposure_Agent/experiments/predict_helper.py`
- Modify: `DeXposure_Agent/experiments/run_all.py`
- Modify: `DeXposure_Agent/scripts/run_benchmarks_sequential.py`
- Modify: `DeXposure_Agent/experiments/competitors/baselines.py`
- Modify: `DeXposure_Agent/experiments/competitors/roland_agent.py`

- [ ] **Step 1: Wire `m4_fm_only` immediately**
Route `m4_fm_only` through the same FM predictor used by `m5_fm_rules`, but skip agent-only benchmarks in the runner.

- [ ] **Step 2: Integrate `C5` using the trained ROLAND checkpoints**
Do not leave `C5` as `NotImplementedError`; at minimum it must run on b1_forecast/b4_stress/b6_robustness and emit serialized outputs.

- [ ] **Step 3: Decide whether `C1` fits the deadline**
If `C1` can be wired in under one day, add it. If not, remove it from paper claims and applicability tables now.

- [ ] **Step 4: Make unimplemented methods fail closed**
Do not silently score unimplemented methods with persistence proxies when the paper presents them as real baselines.

- [ ] **Step 5: Expand the sequential runner**
Run at least `m5_fm_rules`, `m1_persistence_rules`, `m4_fm_only`, and `C5` for the repaired benchmarks.

- [ ] **Step 6: Commit**
```bash
git add DeXposure_Agent/experiments/predict_helper.py DeXposure_Agent/experiments/run_all.py DeXposure_Agent/scripts/run_benchmarks_sequential.py DeXposure_Agent/experiments/competitors/baselines.py DeXposure_Agent/experiments/competitors/roland_agent.py
git commit -m "feat: add minimum publishable baseline set"
```

## Chunk 2: Remove Benchmark Design Errors

### Task 3: Redesign b2_warning so early warning depends on the forecast, not the current graph

**Files:**
- Modify: `DeXposure_Agent/experiments/b2_early_warning.py`
- Test: `DeXposure_Agent/tests/test_b2_early_warning.py`

- [ ] **Step 1: Write a regression test**
Construct two synthetic predictions with different future risk orderings and assert that b2_warning produces different alerts.

- [ ] **Step 2: Replace `_rank_nodes_by_risk` input**
Rank nodes from predicted `G_{t+h}` or predicted future metric deltas, not from the observed `G_t`.

- [ ] **Step 3: Keep historical events but label them correctly**
Explicitly mark Terra/Luna, FTX, and SVB as retrospective event-window evaluation, not out-of-sample post-2024 forecasting.

- [ ] **Step 4: Add at least one genuine test-period event if available**
If no credible 2025 event exists in the dataset, state this limitation plainly instead of implying b2_warning is test-split evaluation.

- [ ] **Step 5: Commit**
```bash
git add DeXposure_Agent/experiments/b2_early_warning.py DeXposure_Agent/tests/test_b2_early_warning.py
git commit -m "fix: make b2_warning depend on predicted future risk"
```

### Task 4: Repair b5_decision so target coverage is meaningful

**Files:**
- Modify: `DeXposure_Agent/experiments/b5_decision_quality.py`
- Modify: `DeXposure_Agent/dexposure_agent/config.py`
- Test: `DeXposure_Agent/tests/test_b5_decision_quality.py`

- [ ] **Step 1: Tighten or redefine “truly stressed”**
Raise the threshold substantially or define stressed sets using percentile-based future deterioration.

- [ ] **Step 2: Replace `audit_completeness` with retrieval metrics**
Add `target_precision_at_5`, `target_recall_at_5`, and optionally `target_recall_at_10`.

- [ ] **Step 3: Keep the old field only if needed for backwards compatibility**
If retained, mark it deprecated in JSON output and do not feature it in the paper.

- [ ] **Step 4: Recompute `risk_reduction` honestly**
If no intervention simulator exists, rename the field to reflect what is actually measured rather than calling it “reduction”.

- [ ] **Step 5: Commit**
```bash
git add DeXposure_Agent/experiments/b5_decision_quality.py DeXposure_Agent/dexposure_agent/config.py DeXposure_Agent/tests/test_b5_decision_quality.py
git commit -m "fix: replace degenerate b5_decision target coverage metrics"
```

### Task 5: Stop b6_robustness from making unsupported low-data claims

**Files:**
- Modify: `DeXposure_Agent/experiments/b6_robustness.py`
- Test: `DeXposure_Agent/tests/test_b6_robustness.py`

- [ ] **Step 1: Decide policy**
Choose one:
1. Remove `low_data_10pct` and `low_data_25pct` from the paper and runner.
2. Rename them to `subsampled_eval_10pct` and `subsampled_eval_25pct`.
3. Add actual retraining code and only then keep the low-data claim.

- [ ] **Step 2: Implement the chosen policy**
The default recommendation is option 1 or 2, not retraining, unless the paper deadline allows retraining and checkpoint management.

- [ ] **Step 3: Add tests for naming and behavior**
Make sure no benchmark output or paper section still calls snapshot subsampling a training-data ablation.

- [ ] **Step 4: Commit**
```bash
git add DeXposure_Agent/experiments/b6_robustness.py DeXposure_Agent/tests/test_b6_robustness.py
git commit -m "fix: remove unsupported low-data robustness claims"
```

### Task 6: Re-scope b4_stress so it is not just measuring shock-function determinism

**Files:**
- Modify: `DeXposure_Agent/experiments/b4_stress_test.py`
- Modify: `DeXposure_Agent/sections/Exp.tex`

- [ ] **Step 1: Downscope the claim immediately**
Until historical contagion-event evaluation exists, describe b4_stress as “scenario consistency under predicted vs realized networks”, not as direct evidence of real-world contagion forecasting.

- [ ] **Step 2: Keep b4_stress in the paper only if positioned correctly**
If b4_stress remains nearly identical for `m5_fm_rules` and `m1_persistence_rules`, move it to secondary evidence rather than a headline result.

- [ ] **Step 3: Commit**
```bash
git add DeXposure_Agent/experiments/b4_stress_test.py DeXposure_Agent/sections/Exp.tex
git commit -m "docs: correctly scope b4_stress scenario evaluation claims"
```

## Chunk 3: Restore Scientific Defensibility

### Task 7: Replace b3_calibration with defensible uncertainty reporting

**Files:**
- Modify: `DeXposure_Agent/experiments/b3_uncertainty_calibration.py`
- Create: `DeXposure_Agent/tests/test_b3_uncertainty.py`

- [ ] **Step 1: Choose the minimal defensible method**
Preferred: split-conformal prediction on validation residuals for each monitored metric.
Fallback: rename the current procedure to empirical perturbation intervals and remove calibration claims.

- [ ] **Step 2: Implement conformal intervals**
Use validation residual quantiles to build target-coverage intervals on test predictions. Keep the current Gaussian perturbation path only as an auxiliary dispersion source if needed.

- [ ] **Step 3: Remove fake ECE**
ECE should only be reported if the method actually produces heterogeneous confidence probabilities. Otherwise drop ECE and report coverage error plus interval width.

- [ ] **Step 4: Add interval tests**
Verify that the conformal routine returns monotone interval widths and exact target coverage on toy data where it should.

- [ ] **Step 5: Commit**
```bash
git add DeXposure_Agent/experiments/b3_uncertainty_calibration.py DeXposure_Agent/tests/test_b3_uncertainty.py
git commit -m "fix: replace pseudo-calibration with conformal intervals"
```

### Task 8: Add multi-run reporting and paired significance tests

**Files:**
- Create: `DeXposure_Agent/experiments/stats.py`
- Modify: `DeXposure_Agent/scripts/run_benchmarks_sequential.py`
- Modify: `DeXposure_Agent/sections/Exp.tex`

- [ ] **Step 1: Add a seed loop**
Expose `--seeds 42,43,44` or equivalent in the runner and serialize per-seed outputs.

- [ ] **Step 2: Add paired statistics**
Implement paired bootstrap confidence intervals and Wilcoxon signed-rank tests over per-snapshot or per-week differences between `m5_fm_rules` and each baseline.

- [ ] **Step 3: Only claim significance where the test is actually run**
If statistics are not ready before submission, remove “significantly” from the paper.

- [ ] **Step 4: Commit**
```bash
git add DeXposure_Agent/experiments/stats.py DeXposure_Agent/scripts/run_benchmarks_sequential.py DeXposure_Agent/sections/Exp.tex
git commit -m "feat: add multi-seed reporting and paired significance tests"
```

### Task 9: Sync the paper and ops docs to the repaired implementation

**Files:**
- Modify: `DeXposure_Agent/sections/Exp.tex`
- Modify: `DeXposure_Agent/DeXposure-Agent.tex`
- Modify: `DeXposure_Agent/EXPERIMENT_PLAN.txt`
- Modify: `DeXposure_Agent/CLAUDE.md`

- [ ] **Step 1: Remove mismatched defaults**
Make `pi_min`, `rolling_window`, `top_k`, and run-count claims match the code that is actually executed.

- [ ] **Step 2: Remove unsupported wording**
Delete “significantly outperforming state-of-the-art methods” until the repaired experiments actually support it.

- [ ] **Step 3: Rewrite benchmark descriptions honestly**
Document that b2_warning is retrospective event-window evaluation, b4_stress is scenario-based, and b6_robustness low-data claims are removed or renamed.

- [ ] **Step 4: Commit**
```bash
git add DeXposure_Agent/sections/Exp.tex DeXposure_Agent/DeXposure-Agent.tex DeXposure_Agent/EXPERIMENT_PLAN.txt DeXposure_Agent/CLAUDE.md
git commit -m "docs: align paper claims with repaired evaluation pipeline"
```

## Recommended Execution Order

1. Task 1 and Task 2.
Reason: they create the minimum evidence that FM adds something beyond persistence.

2. Task 3 and Task 4.
Reason: b2_warning and b5_decision currently contain the most visibly broken benchmark logic.

3. Task 9.
Reason: remove overclaiming early so the paper does not drift farther from reality while code changes are underway.

4. Task 5 and Task 6.
Reason: these are mostly claim hygiene and benchmark scoping; they matter, but they do not rescue the main story alone.

5. Task 7 and Task 8.
Reason: high value, but more work. These should land before submission if time allows.

## Exit Criteria

- `m5_fm_rules` beats `m1_persistence_rules` on at least one forecast-centric metric family that matters operationally: node ranking, edge prediction, or high-change-week forecasting.
- The benchmark runner includes at least one model-only baseline and one non-FM graph baseline.
- No benchmark in the paper is known to be method-invariant by construction.
- No paper sentence claims significance, SOTA superiority, calibrated uncertainty, or low-data robustness without direct supporting evidence.
- All benchmark JSON outputs and paper tables use the same definitions and defaults.

Plan complete and saved to `DeXposure_Agent/docs/superpowers/plans/2026-03-26-evaluation-recovery.md`. Ready to execute?

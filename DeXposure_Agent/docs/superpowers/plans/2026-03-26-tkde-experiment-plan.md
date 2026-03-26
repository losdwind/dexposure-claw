# DeXposure-Agent: TKDE-Oriented Experiment Plan

> **Target venue:** IEEE Transactions on Knowledge and Data Engineering (TKDE)
> **Paper framing:** Temporal graph risk monitoring & query framework (NOT "AI Agent")
> **Date:** 2026-03-26
> **Supersedes:** 2026-03-26-evaluation-recovery.md (original recovery plan)

---

## Framing Shift: Why This Changes Everything

TKDE explicitly rejects "core machine learning theory, neural networks, and fuzzy systems."
The FM backbone (GraphPFN) cannot be the contribution. Instead, the contributions are:

1. DeFi exposure network as a new temporal graph data type + 43.7M-entry dataset
2. Forecast-to-risk-signal transformation framework (predicted graph -> metrics -> alerts -> recommendations)
3. Standardized evaluation benchmark for temporal graph risk monitoring (B1-B8)
4. Safety gating mechanism with automatic service degradation

This means the experiments must prove the FRAMEWORK works, not that the MODEL is good.

---

## Experiment Overview (B1-B8)

| ID | TKDE Name | Old Name | Change Level | Purpose |
|----|-----------|----------|-------------|---------|
| B1 | Temporal Graph Prediction Quality | Risk Forecasting | Medium | Does FM predict graph structure accurately? |
| B2 | Streaming Anomaly Detection | Early Warning | **Large** | Does prediction-based monitoring beat observation-based? |
| B3 | Prediction Interval Quality | Uncertainty Calibration | Medium | Are MC intervals useful for decision support? |
| B4 | What-if Query Fidelity | Stress Testing | Small (narrative) | Are scenario queries on predicted graphs reliable? |
| B5 | Recommendation Precision | Decision Quality | Medium | Do generated tickets target actually-stressed protocols? |
| B6 | Data Quality Sensitivity | Robustness | Medium | How does the system degrade under imperfect data? |
| B7 | Scalability | **NEW** | New | Runtime vs graph size, horizons, MC samples |
| B8 | Crisis Case Study | **NEW** | New | Qualitative walkthrough of 3 real DeFi crises |

---

## Method / Baseline Revision

### Current state (broken)

Only C0 and C2 are implemented. C0 ~ C2 on most metrics because hybrid FM
preserves too much structure. No model-only or statistical baselines.

### TKDE baseline set

| ID | Name | Type | What it tests | Priority |
|----|------|------|---------------|----------|
| C0 | DeXposure-Framework | FM backbone + full pipeline | The complete system | Must have |
| C2 | Persistence-Framework | G_{t+h} = G_t + full pipeline | Is forecasting needed at all? | Must have |
| C4 | DeXposure-FM-only | FM backbone, no pipeline | Does the pipeline add value beyond raw FM? | **Must have** |
| C_stat | Statistical Monitor | No prediction; EWMA/CUSUM on observed G_t metrics | Do we even need a forecast model? | **Must have** |
| C_curr | Current-Graph Query | Run pipeline on G_t instead of G_hat_{t+h} | Does forecasting improve queries? | **Must have** |
| C5 | ROLAND | Trained ROLAND backbone + pipeline | Alternative temporal GNN | Nice to have |
| C1 | ROLAND-Framework | ROLAND backbone + full pipeline | Alternative backbone in same framework | Nice to have |

Key insight: C_stat and C_curr are the baselines TKDE reviewers will demand.
- C_stat: "Why not just monitor current metrics with EWMA?" -> Because our framework detects future anomalies before they show in current data.
- C_curr: "Why not just run scenarios on the current graph?" -> Because stress testing on predicted graphs is more accurate than on stale graphs.

### Ablation set (unchanged, but reframed)

| ID | Override | TKDE framing |
|----|---------|-------------|
| A1 | tau_data=0.0 | Value of data quality gating |
| A2 | tau_conf=0.0 | Value of confidence-weighted alerts |
| A3 | skip_scenario=True | Value of what-if query engine |
| A5 | rolling_window=999999 | Value of adaptive baselines |
| A6 | horizons=[1] | Value of multi-horizon monitoring |
| A7 | mc_samples=1 | Value of MC uncertainty estimation |
| A8 | unconstrained_actions=True | Value of conservative decision design |

Drop A4 (top_k attribution) -- it's an output formatting detail, not a system design choice.

---

## Detailed Experiment Specifications

### B1: Temporal Graph Prediction Quality

**TKDE question:** How well does the FM backbone predict the evolution of the exposure graph?

**What stays from current B1:**
- Multi-horizon evaluation h in {1, 4, 8, 12}
- Aggregate metric MAE (HHI, Density, Gini)
- Spearman rank correlation on PageRank

**What gets added:**

1. **Edge-level prediction metrics** (new file: metrics_edge.py)
   - Edge existence AUROC: treat edges in G_{t+h} as binary labels, FM existence probabilities as scores
   - Edge existence AUPRC: same, but precision-recall (handles class imbalance)
   - Edge weight MAE: for edges present in both predicted and actual graphs
   - New-edge recall: fraction of genuinely new edges (not in G_t) that FM predicts

2. **Node ranking metrics** (new file: metrics_ranking.py)
   - NDCG@10, NDCG@20: rank protocols by predicted PageRank/degree, compare to actual
   - MAP@10: mean average precision of risk ranking
   - This is the KEY metric for TKDE. If FM predicts *which* protocols become important, that's a knowledge engineering contribution.

3. **Regime-sliced reporting**
   - Split test weeks into "high-change" (top 25% by graph edit distance to previous week) vs "stable"
   - Report all metrics separately. Hypothesis: FM advantage shows on high-change weeks.

4. **Graph structural similarity**
   - Edge set Jaccard: |E_pred intersect E_actual| / |E_pred union E_actual|
   - Weight distribution KL divergence (discretized)

**Baselines for B1:** C0, C2, C4, C5 (if available)
C_stat and C_curr do not apply to B1 (they don't produce predicted graphs).

**Code changes:**
- Create: experiments/metrics_ranking.py
- Create: experiments/metrics_edge.py
- Modify: experiments/b1_risk_forecasting.py (add new metric computation + regime slicing)
- Create: tests/test_b1_metrics.py

---

### B2: Streaming Anomaly Detection (MAJOR REDESIGN)

**TKDE question:** Does prediction-based monitoring detect anomalies earlier and more precisely than observation-based monitoring?

**Current B2 problem:** Alerts are generated from current-graph node ranking, not from predicted graphs. C0 and C2 produce identical results. Three hardcoded historical events are in the training period.

**New B2 design:**

1. **Ground truth redefinition**
   - Primary: "anomalous week" = test week where any monitored metric changes by more than the 90th percentile of historical week-to-week changes
   - Secondary: retain Terra/Luna, FTX, SVB as supplementary analysis (move to B8 case study)
   - This gives ~3-4 anomalous weeks in the 33-week test period (top 10%)

2. **Streaming evaluation protocol**
   For each test week t:
   a. Using all data up to t, compute rolling baselines
   b. Generate alerts using the method's approach
   c. Evaluate: did the alert correctly precede an anomalous week within the next h weeks?

3. **Method-specific alert generation:**
   - C0 (our framework): alert from predicted G_{t+h} metrics vs rolling baseline (current design, but actually using predictions)
   - C2 (persistence): alert from G_t metrics vs rolling baseline (equivalent to assuming nothing changes)
   - C_stat (EWMA): alert when EWMA(metric_t) exceeds control limits (classic statistical process control)
   - C_curr (current-graph query): alert when current metrics exceed z-score threshold (no forecasting)

4. **Metrics:**
   - Alert precision: fraction of alerts that precede a real anomaly within h weeks
   - Alert recall: fraction of anomalous weeks that were preceded by an alert
   - Lead time: weeks between alert and anomaly (higher is better, capped at max horizon)
   - False alarm rate: alerts per non-anomalous week
   - Alert stability: 1 - flip_rate (consistency across consecutive weeks)

**Key hypothesis for paper:**
C0 (prediction-based monitoring) achieves higher lead time than C_stat and C_curr
because it can detect emerging anomalies in the PREDICTED future graph before they
manifest in the OBSERVED current graph. This is the core value proposition.

**Code changes:**
- Rewrite: experiments/b2_early_warning.py (new ground truth + streaming protocol)
- Add: C_stat baseline (EWMA on metric time series, ~50 lines)
- Add: C_curr baseline (current-graph z-score, ~30 lines)
- Create: tests/test_b2_streaming.py

---

### B3: Prediction Interval Quality (MEDIUM REDESIGN)

**TKDE question:** Are the MC-based prediction intervals useful for risk decision support?

**Current B3 problem:** MC noise sigma=0.1 is too small, giving ~1-2% PI coverage vs target 90%. ECE is computed but meaningless without heterogeneous confidence.

**New B3 design:**

1. **Replace MC noise with empirical residual calibration**
   - On validation set: compute residuals (predicted_metric - actual_metric) for each metric and horizon
   - Use residual distribution to set MC noise sigma per metric per horizon
   - OR use split-conformal prediction: sort |residuals|, take the (1-alpha) quantile as interval half-width

2. **Metrics (renamed for honesty):**
   - PI Coverage (target 90%): fraction of actuals inside 90% PI
   - PI Width: mean interval width (narrower is better, given adequate coverage)
   - Coverage Error: |coverage - 0.90| (lower is better)
   - Conditional coverage: coverage computed separately for high-change vs stable weeks
   - DROP ECE unless conformal method actually produces meaningful probability bins

3. **Practical value demonstration:**
   - Show that when PI is wide (high uncertainty), the confidence score C_t(a) in Eq.4 correctly downweights alerts
   - Correlation between PI width and actual prediction error (should be positive)
   - This connects B3 to the safety gating story (contribution #4)

**Baselines for B3:** C0 only (C2 has no meaningful uncertainty; C_stat can use EWMA prediction intervals as comparison)

**Code changes:**
- Modify: experiments/b3_uncertainty_calibration.py
- Create: tests/test_b3_intervals.py

---

### B4: What-if Query Fidelity (SMALL CHANGE, NARRATIVE REFRAME)

**TKDE question:** When we run stress scenarios on the predicted graph, how close are the results to running them on the actual future graph?

**Current B4 is mostly fine.** The experimental design (apply shock to predicted vs actual, compare losses) directly answers this question.

**Changes:**

1. **Add C_curr baseline:**
   Apply scenarios to G_t (current graph) instead of G_hat_{t+h} (predicted graph).
   Hypothesis: scenarios on predicted graphs are more accurate because the graph structure
   at t+h has evolved (new edges, changed weights).

2. **Reframe metrics names:**
   - "Loss MAE" -> "Query Loss Error" (it's a query accuracy metric)
   - "Target Overlap@10" -> "Query Target Precision@10"

3. **Add per-scenario breakdown in results**
   Show which scenarios benefit most from forecasting. Hypothesis: S1 (single protocol failure)
   benefits most because the top protocol identity changes over time.

**Code changes:**
- Modify: experiments/b4_stress_test.py (add C_curr comparison, rename outputs)
- Verify: meta_df.csv is loaded so S2-S4 actually work

---

### B5: Recommendation Precision (MEDIUM REDESIGN)

**TKDE question:** Do the framework's recommendation tickets correctly identify protocols that will actually experience stress?

**Current B5 problems (from recovery plan):**
- STRESS_THRESHOLD=0.20 makes ~3000-8000 protocols "stressed" -> audit_completeness near zero
- risk_reduction has no intervention simulator
- C2 outperforms C0 on ticket precision (persistence is more stable)

**New B5 design:**

1. **Tighten ground truth:**
   - Raise STRESS_THRESHOLD from 0.20 to 0.50 (protocols losing >50% edge weight)
   - Alternative: use percentile-based definition (top 5% most-deteriorating protocols)
   - This reduces the "stressed" set from thousands to tens, making precision/recall meaningful

2. **Replace degenerate metrics:**
   - Keep: ticket_precision, target_stability, suppression_rate, false_intervention_rate
   - Replace audit_completeness with: target_recall@10 (what fraction of truly stressed protocols appear in top-10 tickets)
   - Replace risk_reduction with: score_discrimination (mean ticket score for stressed targets vs non-stressed targets; higher gap is better)

3. **Add ablation connection:**
   - Show B5 results with vs without safety gating (A1, A2)
   - This connects to contribution #4 (safety gating mechanism)

**Code changes:**
- Modify: experiments/b5_decision_quality.py
- Modify: dexposure_agent/config.py (STRESS_THRESHOLD)
- Create: tests/test_b5_precision.py

---

### B6: Data Quality Sensitivity (MEDIUM REDESIGN)

**TKDE question:** How does the framework's output quality degrade as input data quality decreases?

**Current B6 problems:**
- low_data_10pct and low_data_25pct pretend to be training data ablations but only subsample eval snapshots
- Doesn't connect to the data-health gating mechanism

**New B6 design:**

1. **Remove fake low-data regimes:**
   Drop low_data_10pct and low_data_25pct entirely.

2. **Keep inference-time degradation regimes:**
   - partial_graph_30: randomly mask 30% of edges (simulates incomplete on-chain data)
   - partial_graph_50: randomly mask 50% of edges (severe data loss)
   - noisy_features_01: Gaussian noise sigma=0.1 on node features
   - noisy_features_03: Gaussian noise sigma=0.3 on node features (severe)
   - missing_features_20: zero out 20% of node features
   - stale_graph: use G_{t-2} instead of G_t (2-week-old data)

3. **Connect to data-health gate:**
   For each regime:
   a. Compute DH_t (data health score) on the degraded input
   b. Report whether safe_mode activates
   c. Report B1 metrics on degraded input
   d. Report B5 metrics (does safe_mode correctly suppress bad recommendations?)

   This is the KEY contribution for TKDE: the system KNOWS when its data is bad and
   automatically degrades service. Show the correlation between DH_t and actual prediction error.

4. **Graph sparsity analysis (new):**
   Split test period by graph size (number of nodes/edges) and report B1 metrics per quartile.
   Hypothesis: system performs better on denser, more-established networks.

**Code changes:**
- Modify: experiments/b6_robustness.py (remove fake regimes, add new ones, add DH connection)
- Create: tests/test_b6_degradation.py

---

### B7: Scalability (NEW)

**TKDE question:** How does the framework scale with graph size, number of queries, and uncertainty estimation?

**Experimental design:**

1. **Component-level timing:**
   Time each pipeline component separately (data_health, fm_predict, monitor, scenario, decision)
   across all test snapshots. Report mean + std.

2. **Horizon scaling:**
   Run with horizons = {[1], [1,4], [1,4,8], [1,4,8,12]}
   Report total wall-clock time. Expect ~linear scaling.

3. **Scenario scaling:**
   Run with num_scenarios = {1, 3, 5, 10} (add 5 more synthetic scenarios)
   Report scenario engine time.

4. **MC sample scaling:**
   Run with mc_samples = {1, 10, 25, 50, 100}
   Report B3 PI coverage AND wall-clock time.
   Show the quality-cost tradeoff curve.

5. **Graph size effect:**
   Group test snapshots by number of nodes (quartiles).
   Report per-component timing for each quartile.
   This tests whether the framework can handle growing DeFi networks.

**Output:** One table + one figure (quality vs cost tradeoff for MC samples)

**Code changes:**
- Create: experiments/b7_scalability.py
- Modify: dexposure_agent/agent_loop.py (add per-component timing instrumentation if not present)

---

### B8: Crisis Case Study (NEW)

**TKDE question:** How does the framework behave during real-world DeFi crises?

**Not a quantitative benchmark.** This is a qualitative case study section for the paper.

**Design:**
For each of {Terra/Luna, FTX, SVB/USDC}:

1. Run the full pipeline for each week from 8 weeks before to 4 weeks after the event
2. Record for each week:
   - DH_t score
   - Number of alerts and their confidence scores
   - Top-3 alerts with evidence (which metrics, which protocols)
   - Worst-case scenario and its loss estimate
   - Ticket recommendations (action type, targets)
3. Produce a timeline visualization:
   - X-axis: weeks, vertical line at event date
   - Y-axis: system outputs (alert count, confidence, scenario loss)
   - Show how the system's alarm level rises before the event

**Note:** These events are in the TRAINING period, so this is NOT out-of-sample evaluation.
State this explicitly in the paper. The case study demonstrates the framework's interpretability
and output quality, not its predictive accuracy on unseen events.

**Code changes:**
- Create: experiments/b8_case_study.py (runs pipeline on event windows, saves structured JSON)
- Create: scripts/plot_case_study.py (generates timeline figures)

---

## Applicability Matrix (TKDE version)

| Method | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|--------|----|----|----|----|----|----|----|----|
| C0 DeXposure-Framework | Y | Y | Y | Y | Y | Y | Y | Y |
| C2 Persistence-Framework | Y | Y | - | Y | Y | Y | - | - |
| C4 FM-only | Y | - | - | - | - | Y | - | - |
| C_stat EWMA | - | Y | - | - | - | - | - | - |
| C_curr Current-Graph | - | Y | - | Y | - | - | - | - |
| C5 ROLAND (if available) | Y | - | - | - | - | Y | - | - |

---

## Execution Priority

### Phase 1: Minimum submittable (must do)

- [ ] 1a. B1 expansion: add edge-level + node ranking metrics (metrics_ranking.py, metrics_edge.py)
- [ ] 1b. C4 baseline: route FM-only through B1 (trivial: predict_helper already supports it)
- [ ] 1c. B2 redesign: streaming protocol + new ground truth + C_stat/C_curr baselines
- [ ] 1d. B5 fix: tighten stress threshold + replace degenerate metrics
- [ ] 1e. B8 case study: run pipeline on 3 crisis windows (mostly scripting)
- [ ] 1f. Paper text: rewrite Intro contributions + Exp sections for TKDE framing

### Phase 2: Significantly stronger

- [ ] 2a. B6 redesign: data quality sensitivity + DH gate connection
- [ ] 2b. B7 scalability: component timing + scaling curves
- [ ] 2c. B3 fix: empirical residual calibration or conformal prediction
- [ ] 2d. B4 enhancement: add C_curr baseline
- [ ] 2e. Ablations A1-A8: connect to B5 results

### Phase 3: Polish

- [ ] 3a. C5 ROLAND integration
- [ ] 3b. Multi-seed reporting + significance tests (stats.py)
- [ ] 3c. Walk-forward cross-validation (15 folds)
- [ ] 3d. Timeline visualization figures for B8

---

## What to DELETE from the paper

1. "Agent" framing in title/abstract/contributions
   -> Replace with "framework" or "system"
2. "Significantly outperforming SOTA" (no evidence)
3. C6-C10 from applicability matrix (not implementing these)
4. Low-data robustness claims (fake regime)
5. ECE from B3 (meaningless without real probabilities)
6. FM architecture details (put in appendix, 1 paragraph max in main text)
7. Algorithm 1 title "DeXposure-Agent: one epoch" -> "DeXposure: Risk Monitoring Pipeline"

## What to ADD to the paper

1. Data engineering section: temporal graph properties, growth statistics, sparsity analysis
2. Related work on graph anomaly detection (TKDE reviewers will know this literature)
3. Related work on graph-based risk monitoring systems
4. Case study section with timeline figures
5. Scalability analysis table
6. Discussion: when does prediction-based monitoring NOT help? (honest limitations)

---

## Exit Criteria

- C0 beats C_stat and C_curr on B2 lead time (core claim: prediction helps)
- C0 beats C4 on B5 (pipeline adds value beyond raw FM)
- C0 beats C2 on B1 edge-level and node-ranking metrics (FM predicts structure)
- B6 shows correlation between DH_t and prediction error (safety gating works)
- B7 shows sub-minute pipeline time for realistic graph sizes (system is practical)
- B8 produces interpretable crisis timelines (framework is useful)
- No paper sentence claims something the experiments don't support

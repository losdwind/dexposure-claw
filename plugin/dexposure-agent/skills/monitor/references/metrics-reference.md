# Metrics Reference: M1-M7

Full definitions, formulas, and interpretation for the seven network risk metrics computed by the DeXposure-Agent monitoring module.

---

## M1: Systemic Importance Score

**Formula:** max(PageRank(G))

**Description:** The maximum PageRank value across all protocol nodes in the predicted graph. A high value indicates one protocol dominates the credit exposure network as a hub.

**Interpretation:**
- Low (< 0.15): Healthy distributed network, no single point of failure
- Medium (0.15 - 0.25): One protocol has notable centrality; monitor closely
- High (> 0.25): Single-protocol dominance; systemic risk if that protocol fails

**Alert context:** Rising M1 often precedes contagion events. It can indicate that capital is funneling through a single gateway protocol.

---

## M2: Cross-sector Spillover

**Formula:** sum of edge weights connecting nodes in different protocol sectors / total edge weight

**Description:** Fraction of total exposure volume that crosses sector boundaries (e.g., lending -> DEX, stablecoin -> bridge). High spillover means failures propagate across protocol categories.

**Interpretation:**
- Low (< 0.3): Mostly intra-sector exposure; limited cross-contamination
- Medium (0.3 - 0.5): Moderate cross-sector linkage; monitor bridge protocols
- High (> 0.5): Heavy cross-sector integration; single failure can cascade broadly

**Alert context:** Rising M2 is especially concerning when combined with high M1, as it means the dominant node connects across many sectors.

---

## M3: HHI Concentration

**Formula:** sum(s_i^2) where s_i = TVL_i / sum(TVL)

**Description:** Herfindahl-Hirschman Index of node TVL weights. Measures whether total value is concentrated in a few protocols or distributed across many.

**Interpretation:**
- Low HHI (< 0.15): Unconcentrated; competitive distribution
- Medium HHI (0.15 - 0.25): Moderate concentration; a few dominant protocols
- High HHI (> 0.25): Highly concentrated; antitrust-level dominance in traditional markets

**Alert context:** HHI rising while total TVL is stable indicates consolidation; falling while TVL drops indicates capital flight from mid-tier protocols.

---

## M4: Network Density

**Formula:** |E| / (|V| * (|V| - 1))

**Description:** Ratio of observed edges to the maximum possible edges in the directed graph. High density means most protocols are mutually exposed.

**Interpretation:**
- Low (< 0.05): Sparse network; limited direct interdependence
- Medium (0.05 - 0.15): Normal DeFi density; some interconnection
- High (> 0.15): Dense network; high interconnection amplifies contagion

**Alert context:** Sudden density spikes may indicate new yield-farming strategies creating many new exposure links. Sudden drops may indicate protocol exits.

---

## M5: Stress-test Loss

**Formula:** sum of contagion losses across all nodes under the active stress scenario

**Description:** Total simulated TVL loss across the network under the most severe active scenario (see scenario skill). Computed using the DebtRank contagion propagation algorithm.

**Interpretation:** Reported as a fraction of total network TVL.
- Low (< 0.05): Network resilient; losses contained
- Medium (0.05 - 0.15): Significant but survivable contagion
- High (> 0.15): Systemic event; material fraction of DeFi TVL at risk

**Alert context:** M5 is the primary actionable metric. When M5 is HIGH, Recommend-Reduce tickets are generated for the top-loss protocols.

---

## M6: PageRank Concentration (Gini)

**Formula:** Gini coefficient of the PageRank distribution across all nodes

**Description:** Measures inequality in network influence. A Gini of 0 means all protocols are equally influential; Gini of 1 means one protocol has all influence.

**Interpretation:**
- Low (< 0.4): Distributed influence; no single gatekeeper
- Medium (0.4 - 0.6): Some inequality; a cluster of influential protocols
- High (> 0.6): Extreme inequality; one or two protocols gate the network

**Alert context:** M6 and M1 are related but distinct — M1 measures the peak, M6 measures the overall shape. Both high together = concentrated and dominated network.

---

## M7: Gini (Degree Distribution)

**Formula:** Gini coefficient of node in-degree + out-degree distribution

**Description:** Measures inequality in protocol connectivity. High values indicate a hub-and-spoke topology rather than a mesh.

**Interpretation:**
- Low (< 0.35): Relatively flat degree distribution; mesh-like
- Medium (0.35 - 0.55): Moderate hub presence
- High (> 0.55): Strong hub-and-spoke topology; hubs are single points of failure

**Alert context:** Rising M7 over time indicates the network is becoming more hub-dependent. This is a leading indicator that M5 (stress-test loss) will worsen under hub failure scenarios.

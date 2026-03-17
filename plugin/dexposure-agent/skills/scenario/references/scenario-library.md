# Scenario Library: S1-S5

Full specifications for the five built-in stress test scenarios in the DeXposure-Agent.

---

## S1: Single Protocol Failure

**Shock:** 100% TVL loss on the top-1 node by TVL (or by PageRank if specified)

**Target selection:** The protocol with the highest TVL in the current snapshot. If the user specifies a protocol name (e.g., "what if Aave fails"), that protocol is used instead.

**Rationale:** Models a catastrophic exploit, governance hack, or regulatory shutdown of the largest protocol. The most common scenario for systemic risk analysis.

**Parameters:**
```json
{
  "scenario_id": "S1",
  "shock_type": "tvl_loss",
  "shock_fraction": 1.0,
  "target": "top_1_by_tvl"
}
```

**Typical loss range:** 3-15% of network TVL depending on hub centrality.

**When to use:** Initial risk assessment, protocol-specific "what if" queries, board-level stress reports.

---

## S2: Bridge Cluster Failure

**Shock:** 100% TVL loss on all bridge-category protocols simultaneously

**Target selection:** All protocols tagged with sector="bridge" in the protocol registry.

**Rationale:** Models a coordinated bridge exploit (e.g., similar to the Ronin, Wormhole, and Nomad hacks in 2022). Bridges are high-value targets and connect otherwise isolated blockchain ecosystems, making their failure especially contagious.

**Parameters:**
```json
{
  "scenario_id": "S2",
  "shock_type": "tvl_loss",
  "shock_fraction": 1.0,
  "target": "sector:bridge"
}
```

**Typical loss range:** 8-25% of network TVL. Often the highest-loss scenario due to cross-chain connectivity.

**When to use:** Bridge protocol exposure review, cross-chain risk analysis, post-hack scenario planning.

---

## S3: Stablecoin De-peg

**Shock:** 50% value drop on all stablecoin-linked protocols

**Target selection:** Protocols tagged sector="stablecoin" plus any protocols whose primary collateral asset is a major stablecoin (USDC, USDT, DAI, FRAX).

**Rationale:** Models a partial de-peg event similar to UST (May 2022) or USDC temporary de-peg (March 2023). A 50% drop is severe but survivable — full de-peg is captured by S1 applied to the stablecoin issuer.

**Parameters:**
```json
{
  "scenario_id": "S3",
  "shock_type": "value_drop",
  "shock_fraction": 0.5,
  "target": "sector:stablecoin + stablecoin_collateral"
}
```

**Typical loss range:** 5-20% of network TVL. Lending protocols with stablecoin collateral are most exposed.

**When to use:** Stablecoin position review, regulatory stress scenarios, collateral haircut calibration.

---

## S4: Sector-wide Shock (Lending)

**Shock:** 30% TVL drop across the entire lending sector

**Target selection:** All protocols tagged sector="lending".

**Rationale:** Models a sector-wide liquidity crisis such as a rapid interest rate spike, mass liquidations, or a regulatory crackdown on lending protocols. A 30% drop reflects severe but not catastrophic conditions.

**Parameters:**
```json
{
  "scenario_id": "S4",
  "shock_type": "tvl_loss",
  "shock_fraction": 0.3,
  "target": "sector:lending"
}
```

**Typical loss range:** 4-12% of network TVL. Cascades primarily through DEX protocols that depend on lending liquidity.

**When to use:** Interest rate sensitivity analysis, lending protocol concentration review, macro stress scenarios.

---

## S5: Correlated Stress (Top-10 by TVL)

**Shock:** 20% TVL drop on the top-10 protocols by current TVL

**Target selection:** Top 10 protocols ranked by TVL at the epoch date.

**Rationale:** Models a broad market downturn where all major protocols de-risk simultaneously (e.g., crypto bear market, regulatory shock, macro risk-off event). This is the regulatory baseline scenario for capital planning.

**Parameters:**
```json
{
  "scenario_id": "S5",
  "shock_type": "tvl_loss",
  "shock_fraction": 0.2,
  "target": "top_10_by_tvl"
}
```

**Typical loss range:** 10-30% of network TVL. Highest aggregate loss due to broad application; lower per-protocol loss but affects the most interconnected nodes.

**When to use:** Quarterly capital adequacy reviews, regulatory reporting, aggregate portfolio stress testing.

---

## Custom Scenarios

To define a custom scenario beyond S1-S5, specify the shock parameters directly in the API:

```json
{
  "date": "2025-01-15",
  "scenario": "custom",
  "shock_type": "tvl_loss",
  "shock_fraction": 0.75,
  "target_protocols": ["maker", "compound"]
}
```

Custom scenarios are not included in the standard CVaR aggregation unless explicitly added.

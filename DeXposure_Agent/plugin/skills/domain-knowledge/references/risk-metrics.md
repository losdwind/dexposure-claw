# Risk Metrics Reference

Definitions and DeFi-specific interpretation of risk metrics used throughout the DeXposure-Agent. Complements the M1-M7 monitoring metrics (see monitor/references/metrics-reference.md) with broader risk concepts.

---

## Network-Level Metrics

### PageRank

PageRank measures the influence of a node in a directed graph by counting both direct links and the quality of those links. In the DeFi exposure graph, high PageRank means a protocol is both directly exposed to many others AND those others are themselves well-connected.

**DeFi interpretation:** A protocol with high PageRank is a systemic node. Its failure would trigger contagion through many paths simultaneously. PageRank is more meaningful than simple degree because it accounts for the quality of connections.

**Formula:** Iterative solution of r(v) = sum_{u->v} r(u) / out_degree(u), with damping factor d=0.85.

---

### Gini Coefficient

Measures statistical inequality across a distribution. Ranges from 0 (perfect equality) to 1 (perfect inequality).

**DeFi interpretation:** High Gini in PageRank distribution (M6) or degree distribution (M7) indicates a hub-and-spoke network. A few dominant protocols gate the majority of capital flows. This topology concentrates systemic risk but can also be more efficient in normal conditions.

**Formula:** G = (2 * sum_i(i * y_i)) / (n * sum_i(y_i)) - (n+1)/n, where y_i is sorted in ascending order.

---

### Herfindahl-Hirschman Index (HHI)

Measures market concentration. Used in antitrust analysis; repurposed here for TVL concentration.

**DeFi interpretation:** High HHI means a few protocols control most of the DeFi TVL. In traditional finance, HHI > 0.25 triggers regulatory scrutiny. In DeFi, this concentration means systemic events are more likely to affect most value simultaneously.

**Formula:** HHI = sum(s_i^2) where s_i is the TVL share of protocol i.

---

### DebtRank

An iterative contagion algorithm adapted from financial network analysis to compute how losses propagate through the exposure graph.

**DeFi interpretation:** Starting from a shocked node (e.g., a bridge that has been hacked), DebtRank propagates fractional losses to counterparties based on their exposure. Each counterparty then propagates further to its counterparties. The algorithm continues until losses stabilize.

**Key property:** Unlike simple cascade models, DebtRank accounts for partial losses (not just binary failure/survival) and prevents double-counting. A node that receives losses from multiple sources is not counted multiple times.

---

## Protocol-Level Metrics

### TVL (Total Value Locked)

The sum of all assets deposited into a protocol's smart contracts, denominated in USD at current prices.

**Interpretation:** The primary size metric. Large TVL = large systemic importance, but also large target for exploits. TVL velocity (rate of change) is often more informative than TVL level for detecting emerging risks.

**Caveat:** TVL double-counting is common — if ETH is deposited in Lido (stETH), then stETH is deposited in Aave, both protocols show TVL for the same underlying ETH. The DeXposure model normalizes for this.

---

### Loan-to-Value (LTV) Ratio

The ratio of borrowed value to collateral value for a lending position.

**DeFi interpretation:** High system-wide average LTV means lending protocols are operating near their liquidation thresholds. A price drop of (1 - max_LTV) would trigger mass liquidations. The monitoring system tracks protocol-level LTV as an input feature.

---

### Collateralization Ratio

The inverse of LTV — ratio of collateral to borrowed value. Must stay above the liquidation threshold.

**DeFi interpretation:** For stablecoins (e.g., DAI), the collateralization ratio is the primary solvency indicator. Falling below 100% means the stablecoin is undercollateralized and the peg is at risk.

---

### Liquidation Threshold

The LTV ratio at which a lending position is automatically liquidated by the protocol.

**DeFi interpretation:** When collateral prices fall, positions that breach the liquidation threshold are partially closed by liquidators. If many positions reach the threshold simultaneously (liquidation cascade), selling pressure can further depress prices, triggering more liquidations. This feedback loop is the primary contagion mechanism in lending protocols.

---

## Systemic Risk Concepts

### Contagion

The transmission of financial distress from one institution (protocol) to others through direct or indirect exposure linkages.

**DeFi channels:**
1. Collateral linkage: Protocol B's token is used as collateral in Protocol A; B's price drop impairs A's collateral
2. Liquidity linkage: Protocol B's liquidity pool is used to liquidate A's positions; B's drain prevents A's liquidations
3. Governance linkage: shared governance token creates coordinated behavior
4. Code dependency: A imports B's oracle, price feed, or liquidity pool contract

### Value at Risk (VaR)

The maximum loss expected over a given time horizon at a given confidence level.

**Example:** VaR(95%, 1 week) = $100M means there is a 95% probability the portfolio will not lose more than $100M over one week.

### Conditional Value at Risk (CVaR)

The expected loss given that the loss exceeds the VaR threshold. Also called Expected Shortfall.

**DeFi interpretation:** CVaR is more appropriate than VaR for DeFi stress testing because DeFi loss distributions have heavy tails. The worst cases (exploits, cascades) are much worse than VaR would suggest. The DeXposure-Agent reports CVaR-95 across stress scenarios as the primary capital planning metric.

---

## Data Quality Metrics

### Freshness

Time elapsed since the last on-chain snapshot was indexed. Measured in hours. Acceptable threshold: <= 24 hours.

### Missingness Rate

Fraction of node or edge attributes with null or placeholder values. Acceptable threshold: < 10% for nodes, < 15% for edges.

### Health Score

Composite data quality score aggregating freshness, missingness, topology consistency, and discontinuity checks. Range [0,1]. Threshold for safe operation: >= 0.7.

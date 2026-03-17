# DeFi Protocol Categories Reference

Protocol taxonomy used by the DeXposure-Agent. Each protocol in the monitored universe is assigned one primary category for sector-based analysis and scenario targeting.

---

## Lending

Protocols where users deposit assets as collateral and borrow other assets against them. Lending protocols are the core of DeFi credit exposure — they hold the largest collateral pools and are most directly exposed to collateral price drops and liquidation cascades.

**Examples:** Aave, Compound, MakerDAO (DAI minting), Euler, Radiant, Spark

**Key risk characteristics:**
- Liquidation risk: if collateral price falls below the liquidation threshold, positions are force-closed
- Oracle dependency: price feed manipulations can trigger mass liquidations
- Governance risk: parameter changes (LTV ratios, liquidation bonuses) affect exposure levels
- Interconnection: typically the highest-centrality sector in the exposure graph

**Typical TVL range:** $500M - $15B

---

## DEX (Decentralized Exchange)

Automated market makers (AMMs) and on-chain order books. DEXes facilitate token swaps and are critical liquidity routing nodes. They do not extend credit directly but their liquidity depth affects the cost and feasibility of liquidations across lending protocols.

**Examples:** Uniswap, Curve, Balancer, Velodrome, Aerodrome, dYdX (spot), Camelot

**Key risk characteristics:**
- Liquidity depth: thin liquidity amplifies price impact during stress
- LP impermanent loss: concentrated liquidity positions can become insolvent
- MEV exposure: sandwich attacks and arbitrage extract value from traders
- Curve specifically: stablecoin AMM; de-pegs can drain pools rapidly

**Typical TVL range:** $200M - $8B

---

## Liquid Staking

Protocols that accept staked assets (ETH, SOL, etc.) and issue liquid tokens representing the staked position. These liquid tokens (stETH, rETH, etc.) are then used as collateral across lending protocols, creating indirect exposure chains.

**Examples:** Lido Finance (stETH), Rocket Pool (rETH), Frax Ether (frxETH), Mantle (mETH), EigenLayer

**Key risk characteristics:**
- Slashing risk: validator penalties reduce the value of staking tokens
- Depeg risk: liquid staking tokens can trade below parity, affecting collateral values downstream
- Concentration: Lido controls ~30% of ETH staking — a failure would have systemic consequences
- EigenLayer: restaking amplifies exposure by reusing staked ETH as security for multiple services

**Typical TVL range:** $1B - $35B

---

## Bridge

Cross-chain asset transfer protocols that lock assets on one chain and mint representations on another. Bridges are the highest-risk category by historical incident rate — they hold large concentrated pools of value and are prime exploit targets.

**Examples:** Stargate, Across, Hop, Connext, Multichain (defunct), Wormhole, Ronin (historical)

**Key risk characteristics:**
- Smart contract risk: bridge contracts are complex and have been exploited repeatedly
- Centralization: many bridges have admin keys or validator sets that are single points of failure
- Cross-chain dependencies: failures affect all chains the bridge connects
- Historical losses: >$2B lost in bridge exploits (Ronin $625M, Wormhole $320M, Nomad $190M)

**Typical TVL range:** $50M - $2B

---

## Stablecoin

Protocols that issue or maintain USD-pegged tokens. Includes both algorithmic stablecoins and collateral-backed stablecoins. De-peg events propagate across the entire DeFi ecosystem because stablecoins are used as collateral, liquidity, and settlement.

**Examples:** MakerDAO/DAI, Frax Finance/FRAX, crvUSD, USDD, Liquity/LUSD, Ethena/USDe

**Key risk characteristics:**
- De-peg risk: if peg breaks, all protocols using this stablecoin as collateral face impairment
- Algorithmic risk: some stablecoins rely on market mechanisms that can fail under stress (see UST)
- Regulatory risk: USDC temporary freeze (March 2023) showed even centralized stablecoins can de-peg
- Collateral quality: over-collateralized stablecoins depend on the quality of their backing assets

**Typical TVL range:** $200M - $10B

---

## Yield

Yield optimization protocols that automatically route user funds across multiple underlying protocols to maximize returns. These protocols create concentrated, correlated exposure because they hold positions in many protocols simultaneously.

**Examples:** Yearn Finance, Convex Finance, Beefy, Pendle, Harvest Finance, Badger DAO

**Key risk characteristics:**
- Aggregated exposure: failures in any underlying protocol affect yield vault holders
- Strategy risk: automated rebalancing can lock capital in failing protocols
- Complexity: multi-hop strategies are harder to audit and monitor
- Convex specifically: controls large Curve voting power; changes affect Curve pool economics

**Typical TVL range:** $100M - $4B

---

## Derivatives

Perpetuals, options, and structured product protocols. Derivatives protocols create leveraged exposure and are a significant source of liquidation cascades during volatile markets.

**Examples:** GMX, Synthetix, dYdX, Gains Network, Opyn, Ribbon Finance, Premia

**Key risk characteristics:**
- Leverage amplification: small price moves trigger large liquidations
- Counterparty pools: losing traders' losses go to winning traders via a pool — pool insolvency risk
- Oracle manipulation: derivatives are particularly vulnerable to price oracle attacks
- Funding rates: perps funding rates can create incentives for coordinated market manipulation

**Typical TVL range:** $100M - $3B

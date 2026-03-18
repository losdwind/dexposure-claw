---
name: DeFi Risk Domain Knowledge
description: >
  This skill should be used when the user asks about "DeFi protocols",
  "what is TVL", "credit exposure", "systemic risk in DeFi",
  "protocol categories", or needs background on DeFi risk concepts.
version: 0.1.0
---

# DeFi Risk Domain Knowledge

This skill provides background on decentralized finance (DeFi) risk concepts as they relate to the DeXposure-Agent. For detailed protocol listings and metric definitions, see the reference files.

## What Is DeFi Credit Exposure

Credit exposure in DeFi refers to the financial risk arising from one protocol depending on the value or solvency of another. Unlike traditional finance, DeFi protocols compose directly — their smart contracts lock, borrow, or route value through each other, creating implicit credit relationships.

An exposure edge from Protocol A to Protocol B exists when:
- A holds B's token as collateral
- A routes liquidity through B
- A's solvency depends on B's price or liquidity
- Users can atomically move capital between A and B

These relationships form the DeFi credit exposure graph that DeXposure-FM models and predicts.

## Total Value Locked (TVL)

TVL is the primary size metric for DeFi protocols. It measures the total value of assets deposited into a protocol's smart contracts, denominated in USD. TVL is used as:
- Node weight in the exposure graph
- Basis for shock magnitudes in stress tests
- HHI concentration computation (M3)

TVL can be volatile — it changes with asset prices and user deposits/withdrawals. The DeXposure-Agent uses log-scaled TVL to reduce the influence of extreme values.

## Protocol Categories

Seven categories are used in the DeXposure-Agent taxonomy. See `references/defi-protocols.md` for full protocol lists.

**Lending** — Protocols where users deposit collateral and borrow against it. These protocols are central nodes in the exposure graph because they hold large collateral pools.

**DEX (Decentralized Exchange)** — Automated market makers and order books. DEXes are critical routing nodes — exposure flows through them when users rebalance between protocols.

**Liquid Staking** — Protocols that issue liquid tokens representing staked assets. These create indirect exposure between staking yields and downstream protocols that accept staking tokens as collateral.

**Bridge** — Cross-chain asset transfer protocols. High-value targets for exploits; bridge failures have historically caused the largest contagion events.

**Stablecoin** — Protocols that issue or maintain USD-pegged tokens. De-peg events propagate widely because stablecoins are used as collateral and liquidity across the entire ecosystem.

**Yield** — Yield optimizers and aggregators that route capital across multiple underlying protocols simultaneously. Concentrated exposure routing.

**Derivatives** — Perpetuals, options, and structured products. Exposure concentration in derivatives protocols can amplify losses during liquidation cascades.

## Key Risk Concepts

**Systemic Risk** — The risk that failure of one protocol triggers a cascade of failures across the network, harming protocols that had no direct exposure to the failed entity. Measured primarily by M1 (Systemic Importance) and M5 (Stress-test Loss).

**Contagion** — The mechanism by which losses propagate. In DeFi, contagion travels through collateral linkages (Protocol B loses value, Protocol A's collateral drops, A faces liquidations), liquidity linkages (B's pool dries up, A cannot route trades), and psychological linkages (users withdraw from similar protocols).

**Concentration Risk** — Over-reliance on a small number of protocols or protocol categories. Measured by M3 (HHI) and M6 (PageRank Gini).

**Liquidity Risk** — Risk that a protocol cannot meet withdrawal demands. Distinct from credit risk but closely related in DeFi because liquidity crises often trigger contagion.

## Reference Files

- `references/defi-protocols.md` — Protocol categories with examples and key characteristics
- `references/risk-metrics.md` — Risk metric definitions in DeFi context

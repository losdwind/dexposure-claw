---
name: Network Risk Monitoring
description: >
  This skill should be used when the user asks to "check alerts",
  "monitor risk", "what are the current warnings", "any risk alerts",
  "analyze network metrics", or wants to understand monitoring alerts and confidence scores.
version: 0.1.0
---

# Network Risk Monitoring

The monitoring module computes a set of seven network risk metrics (M1-M7) from the predicted graph and compares them against a rolling historical baseline to generate alerts. See `references/metrics-reference.md` for full metric definitions and formulas.

## Alert Generation

Alerts are generated using z-score thresholding against a rolling 30-day baseline of the same metric:

```
z_m = (metric_m - mu_m) / sigma_m
```

An alert fires when `|z_m| >= threshold` (default threshold = 2.0 standard deviations). This is equivalent to flagging values in the outer 5% of the historical distribution.

Alerts have three severity levels:

| z-score | Severity |
|---|---|
| 2.0 - 2.5 | WARN |
| 2.5 - 3.5 | HIGH |
| > 3.5 | CRITICAL |

## Confidence Scoring (Eq. 6)

Each alert carries a confidence score combining forecast uncertainty and historical baseline stability:

```
confidence = (1 - weight_std / weight) * baseline_stability
```

Where `baseline_stability` reflects how consistent the rolling baseline has been (low variance in recent history = high stability). Confidence ranges from 0 to 1. Alerts with confidence < 0.5 are flagged as low-confidence and excluded from decision-making unless explicitly requested.

## Alert Attribution

Each alert includes the top contributing nodes (protocols) ranked by their individual contribution to the metric deviation. For example, a spike in M1 (Systemic Importance) will list the protocols whose PageRank values increased most. This helps analysts quickly identify the root cause.

Example alert output:

```json
{
  "metric": "M1",
  "value": 0.41,
  "z_score": 2.8,
  "severity": "HIGH",
  "confidence": 0.79,
  "top_contributors": ["aave", "compound"],
  "description": "Systemic importance score elevated — top node dominance increased"
}
```

## Reviewing Alerts

Alerts are returned as part of the `/run-epoch` response under the `alerts` key, or can be reviewed after any epoch run. When multiple alerts fire simultaneously, prioritize by severity first, then confidence.

A single epoch generating 3+ CRITICAL alerts with high confidence warrants immediate escalation regardless of safe mode status.

## Reference

See `references/metrics-reference.md` for the complete table of M1-M7 definitions, formulas, and interpretation guidance.

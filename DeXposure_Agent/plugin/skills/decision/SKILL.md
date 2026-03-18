---
name: Risk Decision Recommendations
description: >
  This skill should be used when the user asks "what should we do",
  "recommend actions", "generate tickets", "risk recommendations",
  "what actions to take", or wants to understand the decision playbook.
version: 0.1.0
---

# Risk Decision Recommendations

The decision module translates monitoring alerts and stress test results into structured action tickets. Each ticket represents a specific recommended action for a specific protocol, with a severity level and rationale. See `references/playbook-reference.md` for the full action table.

## How Tickets Are Generated

Tickets are generated after the monitoring and scenario steps complete. The generation process:

1. **Filter by confidence** — Only alerts with confidence >= 0.5 are eligible to generate tickets. Low-confidence alerts are logged but do not produce actions.

2. **Count qualifying alerts per protocol** — Each protocol's ticket type depends on how many qualifying alerts reference it (directly or via attribution).

3. **Apply severity scoring** — The ticket severity is determined by:
   - Maximum z-score among contributing alerts
   - Stress-test loss fraction for that protocol
   - Confidence of the highest-severity alert

4. **Check constraints** — Some actions require SAFE_MODE=0 (data is healthy). If SAFE_MODE=1, high-severity interventions are suppressed (see Data Health skill).

5. **Deduplicate** — If multiple alerts generate the same action for the same protocol, a single ticket is issued with the union of contributing evidence.

## Scoring Formula

The priority score for a ticket is:

```
priority = max_z_score * confidence * (1 + stress_loss_fraction)
```

Higher priority tickets should be addressed first. Tickets are returned sorted by priority descending.

## Safe Mode Suppression

When SAFE_MODE=1, the following ticket types are suppressed and not included in the output:
- Recommend-Reduce
- Contingency

Monitor and Investigate tickets are still generated in safe mode because they do not trigger real-world interventions.

## Output Format

Each ticket in the `tickets` array of AgentOutput follows this structure:

```json
{
  "action": "Recommend-Reduce",
  "protocol": "aave",
  "severity": "High",
  "priority_score": 3.41,
  "evidence": ["M1 z=2.8", "S1 loss=0.09"],
  "description": "Systemic importance elevated; recommend reducing exposure to Aave by 20-30%",
  "suppressed": false
}
```

## Reference

See `references/playbook-reference.md` for the complete action table with all constraints, minimum alert counts, and decision thresholds.

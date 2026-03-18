# Decision Playbook Reference

Complete action table for the DeXposure-Agent decision module. Defines all supported ticket types, their trigger conditions, constraints, and expected outcomes.

---

## Action Table

| Action | Severity | Requires SAFE_MODE=0? | Min alerts | Description |
|---|---|---|---|---|
| Monitor | Low | No | 1 | Add protocol to watchlist; increase monitoring frequency for next 2 epochs |
| Investigate | Medium | No | 1 | Trigger deep-dive analysis; pull additional on-chain data for the protocol |
| Recommend-Reduce | High | Yes | 2 | Suggest reducing exposure to this protocol by 20-30%; route to risk committee |
| Contingency | Critical | Yes | 3 | Activate contingency plan; emergency risk committee convening; possible immediate action |

---

## Action Definitions

### Monitor

**When triggered:** Any single qualifying alert (z >= 2.0, confidence >= 0.5) references the protocol.

**Effect:** The protocol is added to an enhanced monitoring watchlist. In the next epoch, its metrics are compared against a tighter z-score threshold (1.5 instead of 2.0) to catch early signals.

**Safe mode:** Generated in both SAFE_MODE=0 and SAFE_MODE=1. Never triggers real-world action automatically.

**Typical use:** Emerging risk signals that need closer attention but not immediate action.

---

### Investigate

**When triggered:** A HIGH severity alert (z >= 2.5) or two or more WARN alerts reference the protocol.

**Effect:** Flags the protocol for manual deep-dive. The assigned analyst should pull additional data, check on-chain activity, and assess whether the signal is a real risk or a data artifact.

**Safe mode:** Generated in both SAFE_MODE=0 and SAFE_MODE=1. Investigation is informational.

**Typical use:** Unexplained metric spikes, new protocol entering top-10 rapidly, unusual exposure concentration.

---

### Recommend-Reduce

**When triggered:**
- At least 2 qualifying alerts reference the protocol, AND
- At least one alert is severity HIGH or CRITICAL, AND
- Protocol appears in stress test at-risk list for at least one scenario, AND
- SAFE_MODE=0 (data is healthy)

**Effect:** A formal recommendation to reduce portfolio exposure to this protocol by the suggested percentage. Routed to the risk committee for approval before any action is taken.

**Safe mode:** SUPPRESSED when SAFE_MODE=1. Written to the log with `suppressed: true` so the signal is not lost.

**Typical use:** Pre-emptive de-risking before a known event, TVL concentration reduction, protocol health deterioration.

---

### Contingency

**When triggered:**
- At least 3 qualifying alerts reference the protocol, AND
- At least one alert is severity CRITICAL (z >= 3.5), AND
- Protocol appears in at-risk list for 2+ stress scenarios, AND
- SAFE_MODE=0 (data is healthy)

**Effect:** Activates the emergency contingency plan. This is the most severe action — it convenes an emergency risk committee session and may authorize immediate position unwinding or hedging.

**Safe mode:** SUPPRESSED when SAFE_MODE=1. The contingency flag is logged with a note that data health prevented automatic escalation.

**Typical use:** Imminent protocol failure, multi-scenario simultaneous stress, post-exploit contagion detection.

---

## Priority Scoring Details

Tickets within the same action tier are sorted by priority score:

```
priority = max_z_score * confidence * (1 + stress_loss_fraction)
```

Example:
- Protocol A: max_z = 3.1, confidence = 0.82, stress_loss = 0.08 -> priority = 3.1 * 0.82 * 1.08 = 2.75
- Protocol B: max_z = 2.7, confidence = 0.91, stress_loss = 0.15 -> priority = 2.7 * 0.91 * 1.15 = 2.83

Protocol B has a slightly higher priority despite a lower z-score, because its stress loss fraction is higher and its confidence is better calibrated.

---

## Escalation Path

```
Monitor -> Investigate -> Recommend-Reduce -> Contingency
```

Protocols do not need to pass through lower tiers to reach higher ones. If the conditions for Contingency are met directly, a Contingency ticket is issued immediately without requiring prior Monitor or Investigate tickets.

---

## Suppressed Ticket Handling

All suppressed tickets (due to SAFE_MODE=1) are still written to the agent log with full evidence. When safe mode clears in a subsequent epoch, the next run will regenerate appropriate tickets if the signals persist.

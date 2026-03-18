---
name: dexposure
description: Run DeXposure-Agent risk monitoring analysis
argument-hint: "[date] [--scenario S1-S5] [--full]"
---

# /dexposure

Run DeXposure-Agent risk monitoring analysis for a given date. This command orchestrates the full analysis pipeline or targeted sub-analyses depending on the arguments provided.

## Usage Patterns

```
/dexposure
/dexposure 2025-01-15
/dexposure --scenario S2
/dexposure 2025-01-15 --scenario S3
/dexposure --full
/dexposure 2025-01-15 --full
```

## Argument Reference

| Argument | Description |
|---|---|
| `[date]` | Target date in YYYY-MM-DD format. Defaults to today if omitted. |
| `--scenario S1-S5` | Run a specific stress test scenario only (skips full epoch). |
| `--full` | Run full epoch with all horizons (h=1,4,8,12) and all five scenarios. Default runs h=4 + S1-S5. |

## Process Steps

### Default Run (no flags)

1. **Parse arguments** — Extract date (default: today) and any flags.

2. **Check server** — Verify the DeXposure-Agent API server is reachable at the configured base URL (default: http://gpu-server:8000). Report connection errors immediately if unreachable.

3. **Run analysis** — Execute `run-epoch` for the specified date:
   ```
   python plugin/dexposure-agent/scripts/call-api.py run-epoch --date [date] --output json
   ```

4. **Parse output** — Extract data health score, alerts, stress test results, and action tickets from the JSON response.

5. **Present results** — Format and display a structured risk report covering:
   - Data health status and safe mode flag
   - Active alerts sorted by severity
   - Stress test CVaR summary
   - Prioritized action tickets

### Scenario-Only Run (`--scenario S1-S5`)

Skips the full epoch and runs only the specified stress test. Useful for quick targeted analysis or re-running a specific scenario with different parameters.

1. Check server health
2. Run targeted stress test: `POST /stress-test {"date": "[date]", "scenario": "[S1-S5]"}`
3. Present contagion results with propagation narrative

### Full Run (`--full`)

Runs the epoch with all four forecast horizons and all five scenarios, then presents the complete multi-horizon risk trajectory. Takes longer but provides the most complete picture.

## Output Format

The standard output is a structured report:

```
DeXposure-Agent Risk Report
Date: 2025-01-15 | Data Health: 0.84 (HEALTHY)

ALERTS (2)
  HIGH   M1 Systemic Importance  z=2.8  conf=0.79  [aave, compound]
  WARN   M3 HHI Concentration    z=2.1  conf=0.62  [aave]

STRESS TESTS
  S1 Single Failure:     loss=6.2%   CVaR=8.1%
  S2 Bridge Failure:     loss=12.4%  CVaR=18.3%  *** SYSTEMIC
  S3 Stablecoin De-peg:  loss=5.8%   CVaR=9.2%
  S4 Lending Shock:      loss=4.1%   CVaR=6.7%
  S5 Correlated Stress:  loss=9.3%   CVaR=14.1%

ACTION TICKETS (2)
  [3.41] Recommend-Reduce  aave    M1 z=2.8 + S2 loss=12%
  [2.17] Investigate       compound  M1 z=2.8

Run `/dexposure --scenario S2` for detailed bridge failure contagion analysis.
```

## Error Handling

- **Server unreachable:** Report the connection error and the configured base URL. Suggest checking that the DeXposure-Agent server is running on gpu-server.
- **Date not found:** If the requested date has no snapshot, report the error and suggest the nearest available date.
- **Safe mode active:** Prominently flag safe mode in the output header and note which tickets were suppressed.
- **Empty alerts:** If no alerts fire, report "All metrics within normal range" — do not fabricate concern.

## Examples

Run analysis for today:
```
/dexposure
```

Run analysis for a specific historical date:
```
/dexposure 2024-11-01
```

Check what would happen if all bridges fail:
```
/dexposure --scenario S2
```

Get full multi-horizon analysis for quarterly report:
```
/dexposure 2025-01-01 --full
```

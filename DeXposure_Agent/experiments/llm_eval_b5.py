#!/usr/bin/env python3
"""LLM Decision Pipeline for B5 Decision Quality.

Runs LOCALLY. Architecture:
  - FM predictions: fetched from GPU server via SSH tunnel (localhost:8000)
  - LLM decisions:  Claude API called locally
  - Ground truth:   raw snapshots fetched from GPU server /snapshot endpoint

Tests C0-LLM (FM + LLM) and C3 (pure LLM) on test weeks.

Prerequisites:
  1. SSH tunnel to GPU server: ssh -f -N -L 8000:localhost:8000 gpu-server
  2. FM API running on GPU server (verify: curl localhost:8000/health)
  3. ANTHROPIC_API_KEY set locally

Usage:
    python DeXposure_Agent/experiments/llm_eval_b5.py --method C0-LLM --method C3
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FM_API_URL = os.environ.get("FM_API_URL", "http://localhost:8000")

# LLM via OpenRouter (OpenAI-compatible API)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096
STRESS_THRESHOLD = 0.20  # >20% weight drop = truly stressed
STRESS_LOOKAHEAD = 4     # weeks
CONSISTENCY_RUNS = 3     # repeat each prompt N times for consistency check
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# HTTP helpers (FM API via SSH tunnel)
# ---------------------------------------------------------------------------

def _http_get(path: str, retries: int = 3) -> dict:
    import urllib.request, urllib.error
    url = f"{FM_API_URL}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"GET {path} failed (attempt {attempt+1}): {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise


def _http_post(path: str, data: dict, retries: int = 3) -> dict:
    import urllib.request, urllib.error
    url = f"{FM_API_URL}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="POST")
            req.data = json.dumps(data).encode()
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"POST {path} failed (attempt {attempt+1}): {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise


def get_test_dates(test_split: str = "2025-01~2025-08") -> list[str]:
    """Get test dates from the FM API /dates endpoint."""
    data = _http_get("/dates")
    start, end = test_split.split("~")
    return [d for d in data["dates"]
            if d >= f"{start}-01" and d <= f"{end}-31"]


def get_snapshot(date: str) -> dict:
    """Get raw snapshot (no FM prediction) from GPU server."""
    return _http_get(f"/snapshot?date={date}")


def get_forecast(date: str, horizon: int) -> dict:
    """Get FM predicted graph from GPU server."""
    return _http_post("/forecast", {"date": date, "horizon": horizon})


# ---------------------------------------------------------------------------
# Metrics computation (self-contained, no GPU deps)
# ---------------------------------------------------------------------------

def compute_metrics(data: dict) -> dict:
    """Compute network risk metrics from snapshot/forecast data."""
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    n = len(nodes)
    if n == 0:
        return {}

    adj: dict[str, dict[str, float]] = {nid: {} for nid in nodes}
    wdeg: dict[str, float] = {nid: 0.0 for nid in nodes}
    for e in edges:
        src, tgt, w = e["source"], e["target"], e["weight"]
        if src in adj:
            adj[src][tgt] = adj[src].get(tgt, 0.0) + w
        if src in wdeg:
            wdeg[src] += w

    deg_vals = list(wdeg.values())
    total_wd = sum(deg_vals)

    # PageRank (20 iterations)
    pr = {nid: 1.0/n for nid in nodes}
    for _ in range(20):
        new_pr = {}
        for node in nodes:
            rank = 0.15 / n
            for src, targets in adj.items():
                if node in targets:
                    out = sum(targets.values())
                    if out > 0:
                        rank += 0.85 * pr[src] * targets[node] / out
            new_pr[node] = rank
        pr = new_pr
    pr_vals = list(pr.values())

    def _gini(vals):
        nv = len(vals)
        if nv == 0 or sum(vals) == 0:
            return 0.0
        s = sorted(vals)
        total = sum(s)
        cumsum = gs = 0.0
        for v in s:
            cumsum += v
            gs += cumsum
        return 1.0 - (2.0 * gs) / (nv * total)

    return {
        "n_nodes": n,
        "n_edges": len(edges),
        "M1_max_pagerank": round(max(pr_vals), 6),
        "M3_hhi": round(sum((d/total_wd)**2 for d in deg_vals) if total_wd > 0 else 0, 6),
        "M4_density": round(len(edges) / (n*(n-1)) if n > 1 else 0, 6),
        "M6_pagerank_gini": round(_gini(pr_vals), 6),
        "M7_degree_gini": round(_gini(deg_vals), 6),
    }


def run_scenarios(data: dict) -> list[dict]:
    """Run S1-S5 stress scenarios on forecast/snapshot data."""
    SCENARIOS = {
        "S1": {"name": "Top protocol failure", "type": "top_node", "shock_pct": 1.0, "count": 1},
        "S2": {"name": "Bridge cluster failure", "type": "category",
               "categories": ["Bridge", "Cross Chain"], "shock_pct": 1.0},
        "S3": {"name": "Stablecoin de-peg", "type": "category",
               "categories": ["Algo-Stables", "Decentralized Stablecoin", "CDP"], "shock_pct": 0.5},
        "S4": {"name": "Lending sector shock", "type": "category",
               "categories": ["Lending", "Uncollateralized Lending", "RWA Lending", "NFT Lending"],
               "shock_pct": 0.3},
        "S5": {"name": "Correlated stress (top-10)", "type": "top_nodes", "shock_pct": 0.2, "count": 10},
    }

    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    nw: dict[str, float] = defaultdict(float)
    for e in edges:
        nw[e["source"]] += e["weight"]
        nw[e["target"]] += e["weight"]

    results = []
    for sid, spec in SCENARIOS.items():
        shock_type, shock_pct = spec["type"], spec["shock_pct"]
        if shock_type in ("top_node", "top_nodes"):
            cnt = spec.get("count", 1)
            shocked = {n for n, _ in sorted(nw.items(), key=lambda x: x[1], reverse=True)[:cnt]}
        elif shock_type == "category":
            cats_lower = {c.lower() for c in spec.get("categories", [])}
            shocked = {nid for nid, nf in nodes.items()
                       if nf.get("category", "").lower() in cats_lower}
        else:
            shocked = set()

        sum_orig = sum(nw.values())
        sw: dict[str, float] = defaultdict(float)
        for e in edges:
            w = e["weight"]
            if e["source"] in shocked or e["target"] in shocked:
                w *= (1 - shock_pct)
            if w > 0:
                sw[e["source"]] += w
                sw[e["target"]] += w
        sum_shock = sum(sw.values())
        loss = (sum_orig - sum_shock) / sum_orig if sum_orig > 0 else 0

        distressed = sum(1 for nid, ow in nw.items()
                         if ow > 0 and (ow - sw.get(nid, 0)) / ow > 0.5)
        affected = sum(1 for nid in nw if nw[nid] - sw.get(nid, 0) > 0)
        top_aff = sorted(((nid, nw[nid] - sw.get(nid, 0)) for nid in nw
                          if nw[nid] - sw.get(nid, 0) > 0),
                         key=lambda x: x[1], reverse=True)[:5]
        results.append({
            "scenario_id": sid, "scenario_name": spec["name"],
            "system_loss_pct": round(loss * 100, 2),
            "distressed_count": distressed, "affected_nodes": affected,
            "top_affected": [{"protocol": nid, "loss": round(d, 2),
                              "category": nodes.get(nid, {}).get("category", "?")}
                             for nid, d in top_aff],
        })
    return results


# ---------------------------------------------------------------------------
# Ground truth detection
# ---------------------------------------------------------------------------

def detect_truly_stressed(snap_current: dict, snap_future: dict,
                          threshold: float = STRESS_THRESHOLD) -> set[str]:
    """Compare current vs future snapshot to find stressed protocols."""
    def _node_weights(data):
        nw: dict[str, float] = defaultdict(float)
        for e in data.get("edges", []):
            nw[e["source"]] += e["weight"]
            nw[e["target"]] += e["weight"]
        return nw

    wt = _node_weights(snap_current)
    wf = _node_weights(snap_future)
    stressed = set()
    for nid, w in wt.items():
        if w <= 0:
            continue
        drop = (w - wf.get(nid, 0.0)) / w
        if drop > threshold:
            stressed.add(nid)
    return stressed


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a DeFi systemic risk analyst for a regulatory supervisory body.
Your job: given network data, identify protocols at elevated risk of distress
over the next {horizon} weeks and recommend supervisory actions.

You MUST respond with valid JSON only (no markdown, no explanations outside JSON).
Schema:
{{
  "risk_level": "low" | "moderate" | "elevated" | "critical",
  "target_protocols": [
    {{
      "protocol": "<name>",
      "risk_score": <float 0-1>,
      "action": "Monitor" | "Investigate" | "Recommend-Reduce" | "Contingency",
      "reason": "<short explanation citing specific data>"
    }}
  ],
  "rationale": "<2-3 sentence overall assessment citing specific metrics>"
}}

Only flag protocols you genuinely believe are at elevated risk.
Cite specific numbers from the data to support your assessment."""

C0_LLM_USER_TEMPLATE = """Current DeFi network analysis ({date}), forecast horizon = {horizon} weeks.

== FM MODEL PREDICTIONS (predicted graph G_{{t+{horizon}}}) ==
Nodes: {n_nodes} protocols, Edges: {n_edges} weighted exposure links

Top-10 protocols by predicted exposure weight:
{top_protocols}

== NETWORK RISK METRICS ==
{metrics}

== STRESS SCENARIO ANALYSIS (applied to predicted graph) ==
{scenarios}

Based on these FM model predictions, metrics, and stress analysis, identify
which protocols are at elevated risk over the next {horizon} weeks."""

C3_USER_TEMPLATE = """Current DeFi network state ({date}), forecast horizon = {horizon} weeks.

NOTE: You do NOT have access to a predictive model. You must reason from the
current network snapshot only.

== CURRENT NETWORK SNAPSHOT ==
Protocols: {n_nodes} total
Edges: {n_edges} weighted exposure links
Total network weight: {total_weight:.2f}

Top-10 protocols by current exposure weight:
{top_protocols}

Category breakdown:
{category_summary}

== CURRENT NETWORK METRICS ==
{metrics}

Based on this current network state (without predictive model forecasts),
identify which protocols are at elevated risk over the next {horizon} weeks."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_top_protocols(data: dict) -> str:
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    nw: dict[str, float] = defaultdict(float)
    for e in edges:
        nw[e["source"]] += e["weight"]
        nw[e["target"]] += e["weight"]
    top = sorted(nw.items(), key=lambda x: x[1], reverse=True)[:10]
    lines = []
    for i, (nid, w) in enumerate(top, 1):
        cat = nodes.get(nid, {}).get("category", "?")
        lines.append(f"  {i}. {nid} ({cat}) -- weight: {w:.2f}")
    return "\n".join(lines)


def _format_metrics(metrics: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in metrics.items() if k.startswith("M"))


def _format_scenarios(scenarios: list[dict]) -> str:
    lines = []
    for s in scenarios:
        lines.append(f"  {s['scenario_id']} ({s['scenario_name']}): "
                     f"loss={s['system_loss_pct']:.2f}%, "
                     f"distressed={s['distressed_count']}, "
                     f"affected={s['affected_nodes']}")
        for t in s.get("top_affected", [])[:3]:
            lines.append(f"    - {t['protocol']} ({t['category']}): loss={t['loss']:.2f}")
    return "\n".join(lines)


def _format_categories(data: dict) -> str:
    cats: dict[str, int] = defaultdict(int)
    for nid, nf in data.get("nodes", {}).items():
        cats[nf.get("category", "Unknown")] += 1
    top = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:10]
    return "\n".join(f"  {cat}: {cnt} protocols" for cat, cnt in top)


def build_prompt(method: str, date: str, horizon: int,
                 forecast: dict | None, snapshot: dict | None,
                 metrics: dict, scenarios: list[dict] | None) -> tuple[str, str]:
    """Build (system, user) prompt for given method."""
    system = SYSTEM_PROMPT.format(horizon=horizon)

    if method == "C0-LLM":
        assert forecast is not None
        user = C0_LLM_USER_TEMPLATE.format(
            date=date, horizon=horizon,
            n_nodes=forecast.get("n_nodes", 0),
            n_edges=forecast.get("n_edges", 0),
            top_protocols=_format_top_protocols(forecast),
            metrics=_format_metrics(metrics),
            scenarios=_format_scenarios(scenarios or []),
        )
    elif method == "C3":
        assert snapshot is not None
        edges = snapshot.get("edges", [])
        user = C3_USER_TEMPLATE.format(
            date=date, horizon=horizon,
            n_nodes=len(snapshot.get("nodes", {})),
            n_edges=len(edges),
            total_weight=sum(e["weight"] for e in edges),
            top_protocols=_format_top_protocols(snapshot),
            category_summary=_format_categories(snapshot),
            metrics=_format_metrics(metrics),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return system, user


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str) -> dict:
    """Call LLM via OpenRouter (OpenAI-compatible API)."""
    import urllib.request
    import urllib.error

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    req = urllib.request.Request(OPENROUTER_URL, method="POST")
    req.data = json.dumps(payload).encode()
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {OPENROUTER_API_KEY}")

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        err_body = ""
        if hasattr(e, "read"):
            err_body = e.read().decode("utf-8", errors="replace")[:500]
        logger.error(f"LLM API call failed: {e} | {err_body}")
        return {"error": str(e), "risk_level": "unknown",
                "target_protocols": [], "rationale": "", "raw_response": ""}

    latency_ms = (time.time() - t0) * 1000

    choice = body.get("choices", [{}])[0]
    raw = choice.get("message", {}).get("content", "").strip()
    usage = body.get("usage", {})

    # Parse JSON (handle possible markdown wrapper)
    text = raw
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM JSON: {text[:200]}")
        parsed = {"risk_level": "unknown", "target_protocols": [],
                  "rationale": raw[:500]}

    parsed["raw_response"] = raw
    parsed["input_tokens"] = usage.get("prompt_tokens", 0)
    parsed["output_tokens"] = usage.get("completion_tokens", 0)
    parsed["latency_ms"] = round(latency_ms, 1)
    parsed["model"] = body.get("model", LLM_MODEL)
    return parsed


# ---------------------------------------------------------------------------
# Per-week assessment
# ---------------------------------------------------------------------------

def assess_week(llm_output: dict, truly_stressed: set[str]) -> dict:
    """Compute precision, completeness, grounding for one week."""
    targets = {p["protocol"] for p in llm_output.get("target_protocols", [])}

    precision = len(targets & truly_stressed) / len(targets) if targets else 0.0
    completeness = (len(targets & truly_stressed) / len(truly_stressed)
                    if truly_stressed else 1.0)

    # Grounding: fraction of reasons citing numeric data
    grounding = 0.0
    protos = llm_output.get("target_protocols", [])
    if protos:
        grounded = 0
        for p in protos:
            reason = p.get("reason", "")
            has_num = any(c.isdigit() for c in reason)
            has_metric = any(m in reason for m in
                            ["M1", "M3", "M4", "M6", "M7", "loss",
                             "weight", "pagerank", "gini", "hhi", "%"])
            if has_num and has_metric:
                grounded += 1
        grounding = grounded / len(protos)

    return {"targets": sorted(targets), "precision": precision,
            "completeness": completeness, "grounding_score": grounding}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class WeekResult:
    date: str
    method: str
    risk_level: str = ""
    targets: list[str] = field(default_factory=list)
    truly_stressed: list[str] = field(default_factory=list)
    precision: float = 0.0
    completeness: float = 0.0
    grounding_score: float = 0.0
    consistency: float = 1.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


def run_pipeline(
    methods: list[str],
    test_split: str = "2025-01~2025-08",
    horizon: int = STRESS_LOOKAHEAD,
    consistency_runs: int = CONSISTENCY_RUNS,
) -> dict[str, list[WeekResult]]:
    """Run LLM pipeline locally: FM API via tunnel, Claude API locally."""

    # Verify prerequisites
    global OPENROUTER_API_KEY
    if not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set. Export it before running.")
        sys.exit(1)

    try:
        health = _http_get("/health")
        logger.info(f"FM API: {health}")
        if not health.get("fm_available"):
            logger.error("FM model not available on server")
            sys.exit(1)
    except Exception as e:
        logger.error(f"FM API unreachable at {FM_API_URL}: {e}")
        logger.error("Set up SSH tunnel: ssh -f -N -L 8000:localhost:8000 gpu-server")
        sys.exit(1)

    # Get test dates from API
    test_dates = get_test_dates(test_split)
    all_dates = _http_get("/dates")["dates"]
    logger.info(f"Test dates: {len(test_dates)} weeks ({test_dates[0]} to {test_dates[-1]})")

    # Results
    results: dict[str, list[WeekResult]] = {m: [] for m in methods}
    raw_log: list[dict] = []
    total_cost = 0.0

    for wi, date_str in enumerate(test_dates):
        # Find future date for ground truth
        t_idx = all_dates.index(date_str)
        future_idx = t_idx + horizon
        if future_idx >= len(all_dates):
            logger.info(f"[{wi+1}/{len(test_dates)}] {date_str}: no future snapshot, skip")
            continue
        future_date = all_dates[future_idx]

        # Fetch snapshots from GPU server
        snap_current = get_snapshot(date_str)
        snap_future = get_snapshot(future_date)
        truly_stressed = detect_truly_stressed(snap_current, snap_future)

        logger.info(f"[{wi+1}/{len(test_dates)}] {date_str} -> {future_date}: "
                     f"{len(truly_stressed)} truly stressed")

        for method in methods:
            # Collect data for prompt
            forecast = None
            scenarios = None
            if method == "C0-LLM":
                forecast = get_forecast(date_str, horizon)
                metrics = compute_metrics(forecast)
                scenarios = run_scenarios(forecast)
            elif method == "C3":
                metrics = compute_metrics(snap_current)
            else:
                continue

            system, user = build_prompt(
                method, date_str, horizon, forecast, snap_current, metrics, scenarios)

            # Run LLM multiple times for consistency
            run_outputs = []
            for ri in range(consistency_runs):
                out = call_llm(system, user)
                run_outputs.append(out)
                total_cost += (out.get("input_tokens", 0) * 3 +
                               out.get("output_tokens", 0) * 15) / 1_000_000

            primary = run_outputs[0]
            ev = assess_week(primary, truly_stressed)

            # Consistency across runs
            if len(run_outputs) > 1:
                tsets = [{p["protocol"] for p in o.get("target_protocols", [])}
                         for o in run_outputs]
                jacs = [_jaccard(tsets[i], tsets[j])
                        for i in range(len(tsets)) for j in range(i+1, len(tsets))]
                consistency = float(np.mean(jacs)) if jacs else 1.0
            else:
                consistency = 1.0

            wr = WeekResult(
                date=date_str, method=method,
                risk_level=primary.get("risk_level", "unknown"),
                targets=ev["targets"],
                truly_stressed=sorted(truly_stressed),
                precision=ev["precision"],
                completeness=ev["completeness"],
                grounding_score=ev["grounding_score"],
                consistency=consistency,
                input_tokens=primary.get("input_tokens", 0),
                output_tokens=primary.get("output_tokens", 0),
                latency_ms=primary.get("latency_ms", 0),
            )
            results[method].append(wr)

            raw_log.append({
                "date": date_str, "method": method,
                "system_prompt": system, "user_prompt": user,
                "llm_outputs": run_outputs,
                "truly_stressed": sorted(truly_stressed),
                "assessment": ev, "consistency": consistency,
            })

            logger.info(f"  {method}: prec={ev['precision']:.3f} "
                         f"compl={ev['completeness']:.3f} "
                         f"ground={ev['grounding_score']:.3f} "
                         f"consist={consistency:.3f} "
                         f"targets={len(ev['targets'])}")

    # ---------------------------------------------------------------------------
    # Aggregate and save
    # ---------------------------------------------------------------------------
    ts = int(time.time())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for method, wrs in results.items():
        if not wrs:
            continue
        precs = [w.precision for w in wrs]
        compls = [w.completeness for w in wrs]
        grounds = [w.grounding_score for w in wrs]
        consists = [w.consistency for w in wrs]
        stabilities = [_jaccard(set(wrs[i-1].targets), set(wrs[i].targets))
                       for i in range(1, len(wrs))]

        summary = {
            "method": method,
            "n_weeks": len(wrs),
            "ticket_precision": round(float(np.mean(precs)), 4),
            "audit_completeness": round(float(np.mean(compls)), 4),
            "target_stability": round(float(np.mean(stabilities)), 4) if stabilities else 0.0,
            "grounding_score": round(float(np.mean(grounds)), 4),
            "consistency": round(float(np.mean(consists)), 4),
            "total_input_tokens": sum(w.input_tokens for w in wrs),
            "total_output_tokens": sum(w.output_tokens for w in wrs),
        }

        logger.info(f"\n{'='*60}\n{method} SUMMARY:")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")

        with open(RESULTS_DIR / f"LLM-B5_{method}_{ts}.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Save raw audit log
    with open(RESULTS_DIR / f"llm_b5_raw_{ts}.json", "w") as f:
        json.dump(raw_log, f, indent=2, default=str)

    logger.info(f"\nTotal estimated cost: ${total_cost:.2f}")
    logger.info(f"Results saved to {RESULTS_DIR}/")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Decision Pipeline (runs locally)")
    parser.add_argument("--method", action="append", default=[],
                        help="C0-LLM and/or C3 (can repeat)")
    parser.add_argument("--test-split", default="2025-01~2025-08")
    parser.add_argument("--horizon", type=int, default=STRESS_LOOKAHEAD)
    parser.add_argument("--consistency-runs", type=int, default=CONSISTENCY_RUNS)
    args = parser.parse_args()

    if not args.method:
        args.method = ["C0-LLM", "C3"]

    run_pipeline(
        methods=args.method,
        test_split=args.test_split,
        horizon=args.horizon,
        consistency_runs=args.consistency_runs,
    )

#!/usr/bin/env python3
"""LLM Decision Evaluation Pipeline -- Layer 2 + Layer 3 for b5_decision.

Runs LOCALLY (not on GPU server). Architecture:
  - FM predictions: fetched from GPU server via SSH tunnel (localhost:8000)
  - LLM decisions:  Anthropic Claude API called locally
  - Ground truth:   raw snapshots fetched from GPU /snapshot endpoint

Tests four methods:
  m1_persistence_rules  (loaded from existing b5_decision results)
  m5_fm_rules           (loaded from existing b5_decision results)
  m2_snapshot_llm       (text-only snapshot -> Claude)
  m6_fm_llm             (FM predictions + metrics + scenarios -> Claude)
  m7_fm_llm_gated
          FM + LLM + RulesGate  (same prompt, conservative post-hoc action gate)

Layer 2 metrics (LLM reasoning quality):
  - Groundedness: fraction of cited values traceable to input data
  - Consistency:  Jaccard similarity across repeated runs (temperature=0)

Layer 3 metrics (end-to-end decision quality):
  - Ticket Precision:      flagged protocols that were truly stressed
  - Audit Completeness:    truly stressed protocols that were flagged
  - Target Stability:      Jaccard between consecutive weeks
  - Severity Correlation:  Spearman rho between recommended severity vs actual loss
  - Grounding Score:       reasons citing real numbers from data
  - Explanation Quality:   LLM-as-Judge score (1-5)

Prerequisites:
  1. SSH tunnel: ssh -f -N -L 8000:localhost:8000 gpu-server
  2. FM API running (verify: curl localhost:8000/health)
  3. ANTHROPIC_API_KEY set locally

Usage:
    python DeXposure_Agent/experiments/llm_eval_b5.py
    python DeXposure_Agent/experiments/llm_eval_b5.py --method m6_fm_llm --method m2_snapshot_llm
    python DeXposure_Agent/experiments/llm_eval_b5.py --resume   # skip completed weeks
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FM_API_URL = os.environ.get("FM_API_URL", "http://localhost:8000")
DECISION_MODEL = os.environ.get("LLM_EVAL_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "claude-haiku-4-5")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096
# Ground truth uses the same percentile-based definition as b5_decision_quality.py
# (top STRESS_PERCENTILE most-deteriorating active protocols per week). Keeping
# the two evaluators on identical ground truth is required for m5_fm_rules vs
# m6_fm_llm comparisons to be meaningful. The legacy absolute STRESS_THRESHOLD
# (0.50) is retained as a numeric fallback only.
STRESS_PERCENTILE = 0.05
STRESS_THRESHOLD = 0.50  # legacy; not used by detect_truly_stressed below.
STRESS_LOOKAHEAD = 4
CONSISTENCY_RUNS = 3
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# LLM API backend: OpenRouter (default) or Anthropic direct
LLM_API_URL = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_API_KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

# Cost per million tokens (OpenRouter listed prices, USD)
MODEL_COSTS = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
}
SEVERITY_ORDER = {"Monitor": 1, "Investigate": 2, "Recommend-Reduce": 3, "Contingency": 4}
GATED_LLM_METHODS = {"m7_fm_llm_gated"}


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
                logger.warning(f"GET {path} attempt {attempt+1} failed: {e}, retry in {wait}s")
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
                logger.warning(f"POST {path} attempt {attempt+1} failed: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise


def get_test_dates(test_split: str = "2025-01~2025-08") -> list[str]:
    data = _http_get("/dates")
    start, end = test_split.split("~")
    return [d for d in data["dates"] if d >= f"{start}-01" and d <= f"{end}-31"]


def get_snapshot(date: str) -> dict:
    return _http_get(f"/snapshot?date={date}")


def get_forecast(date: str, horizon: int) -> dict:
    return _http_post("/forecast", {"date": date, "horizon": horizon})


# ---------------------------------------------------------------------------
# Local metrics computation (no GPU deps)
# ---------------------------------------------------------------------------

def _gini(vals: list[float]) -> float:
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    s = sorted(vals)
    total = sum(s)
    cumsum = gs = 0.0
    for v in s:
        cumsum += v
        gs += cumsum
    return 1.0 - (2.0 * gs) / (n * total)


def compute_metrics(data: dict) -> dict:
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    n = len(nodes)
    if n == 0:
        return {}

    # Build forward adjacency (for out-weights) + reverse adjacency (for PageRank)
    out_weight: dict[str, float] = {nid: 0.0 for nid in nodes}
    adj_in: dict[str, list[tuple[str, float]]] = {nid: [] for nid in nodes}
    wdeg: dict[str, float] = {nid: 0.0 for nid in nodes}

    edge_agg: dict[tuple[str, str], float] = {}
    for e in edges:
        src, tgt, w = e["source"], e["target"], e["weight"]
        key = (src, tgt)
        edge_agg[key] = edge_agg.get(key, 0.0) + w
        if src in wdeg:
            wdeg[src] += w

    for (src, tgt), w in edge_agg.items():
        if src in out_weight:
            out_weight[src] += w
        if tgt in adj_in:
            adj_in[tgt].append((src, w))

    deg_vals = list(wdeg.values())
    total_wd = sum(deg_vals)

    # PageRank via reverse adjacency: O(V+E) per iteration instead of O(V^2)
    pr = {nid: 1.0 / n for nid in nodes}
    base = 0.15 / n
    for _ in range(20):
        new_pr = {}
        for node in nodes:
            rank = base
            for src, w in adj_in[node]:
                ow = out_weight[src]
                if ow > 0:
                    rank += 0.85 * pr[src] * w / ow
            new_pr[node] = rank
        pr = new_pr
    pr_vals = list(pr.values())

    return {
        "n_nodes": n,
        "n_edges": len(edges),
        "M1_max_pagerank": round(max(pr_vals), 6),
        "M3_hhi": round(sum((d / total_wd) ** 2 for d in deg_vals) if total_wd > 0 else 0, 6),
        "M4_density": round(len(edges) / (n * (n - 1)) if n > 1 else 0, 6),
        "M6_pagerank_gini": round(_gini(pr_vals), 6),
        "M7_degree_gini": round(_gini(deg_vals), 6),
    }


def run_scenarios(data: dict) -> list[dict]:
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
# Ground truth
# ---------------------------------------------------------------------------

def detect_truly_stressed(snap_current: dict, snap_future: dict,
                          pct: float = STRESS_PERCENTILE) -> set[str]:
    """Top-`pct` most-deteriorating active protocols (matches b5_decision)."""
    def _node_weights(data):
        nw: dict[str, float] = defaultdict(float)
        for e in data.get("edges", []):
            nw[e["source"]] += e["weight"]
            nw[e["target"]] += e["weight"]
        return nw

    wt = _node_weights(snap_current)
    wf = _node_weights(snap_future)
    drops: list[tuple[str, float]] = []
    for nid, w in wt.items():
        if w <= 0:
            continue
        drop = (w - wf.get(nid, 0.0)) / w
        if drop > 0.0:
            drops.append((nid, drop))
    if not drops:
        return set()
    drops.sort(key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(len(drops) * pct))
    return {nid for nid, _ in drops[:cutoff]}


def compute_actual_loss(snap_current: dict, snap_future: dict) -> dict[str, float]:
    """Per-protocol weight-drop fraction for severity correlation."""
    def _nw(data):
        nw: dict[str, float] = defaultdict(float)
        for e in data.get("edges", []):
            nw[e["source"]] += e["weight"]
            nw[e["target"]] += e["weight"]
        return nw

    wt = _nw(snap_current)
    wf = _nw(snap_future)
    losses = {}
    for nid, w in wt.items():
        if w > 0:
            losses[nid] = max(0.0, (w - wf.get(nid, 0.0)) / w)
    return losses


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
      "reason": "<specific explanation citing data values from the input>"
    }}
  ],
  "rationale": "<2-3 sentence overall assessment citing specific metrics and numbers>"
}}

Rules:
- Only flag protocols you genuinely believe are at elevated risk.
- Cite specific numbers from the provided data to justify each assessment.
- Use the exact protocol names as they appear in the data.
- Order target_protocols by risk_score descending.
- Recommend-Reduce requires elevated or critical overall risk and protocol risk_score >= 0.75.
- Contingency requires critical overall risk and protocol risk_score >= 0.90.
- If those action constraints are not met, use Investigate instead of an intervention."""

C0_LLM_USER_TEMPLATE = """Current DeFi network analysis ({date}), forecast horizon = {horizon} weeks.

== FM MODEL PREDICTIONS (predicted graph G_{{t+{horizon}}}) ==
Nodes: {n_nodes} protocols, Edges: {n_edges} weighted exposure links

Top-10 protocols by predicted exposure weight:
{top_protocols}

== PREDICTED NETWORK RISK METRICS ==
{metrics}

== STRESS SCENARIO ANALYSIS (applied to predicted graph) ==
{scenarios}

Based on these FM model predictions, metrics, and stress analysis, identify
which protocols are at elevated risk over the next {horizon} weeks."""

C3_USER_TEMPLATE = """Current DeFi network state ({date}), forecast horizon = {horizon} weeks.

NOTE: You do NOT have access to any predictive model. You must reason from the
current network snapshot only. There are no forward-looking predictions available.

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

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of DeFi risk analysis reports.
You will compare two risk assessments for the same week and rate the quality
of the SECOND one (Report B) on a scale of 1-5.

Respond with JSON only:
{{
  "quality_score": <int 1-5>,
  "reasoning": "<1-2 sentence justification>"
}}

Scoring guide:
  5: Excellent -- cites specific metrics, identifies correct risk factors, actionable
  4: Good -- mostly correct, cites some data, reasonable recommendations
  3: Adequate -- general assessment, few specific citations, partially correct
  2: Poor -- vague, misses key risks, few data references
  1: Very poor -- contradicts data, no grounding, hallucinated claims"""

JUDGE_USER_TEMPLATE = """Week: {date}, horizon: {horizon} weeks

== INPUT DATA SUMMARY ==
Network: {n_nodes} nodes, {n_edges} edges
Key metrics: {metrics_summary}

== GROUND TRUTH ==
{n_stressed} protocols are in the top-5% most-deteriorating set (largest weight drops over the horizon): {stressed_list}

== REPORT TO EVALUATE ==
Risk level: {risk_level}
Targets: {llm_targets}
Rationale: {rationale}

Rate the report's quality (1-5)."""


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
    system = SYSTEM_PROMPT.format(horizon=horizon)

    if method in ("m6_fm_llm", "m7_fm_llm_gated"):
        assert forecast is not None
        user = C0_LLM_USER_TEMPLATE.format(
            date=date, horizon=horizon,
            n_nodes=forecast.get("n_nodes", 0),
            n_edges=forecast.get("n_edges", 0),
            top_protocols=_format_top_protocols(forecast),
            metrics=_format_metrics(metrics),
            scenarios=_format_scenarios(scenarios or []),
        )
    elif method == "m2_snapshot_llm":
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
# LLM API caller (OpenRouter / OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------


def call_llm(system: str, user: str, model: str | None = None) -> dict:
    """Call LLM via OpenRouter (or any OpenAI-compatible endpoint).

    Returns parsed JSON + usage metadata.
    """
    import urllib.request, urllib.error

    model = model or DECISION_MODEL
    # OpenRouter needs provider prefix
    api_model = f"anthropic/{model}" if "/" not in model else model

    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    import http.client

    t0 = time.time()
    for attempt in range(5):
        try:
            req = urllib.request.Request(LLM_API_URL, method="POST")
            req.data = json.dumps(payload).encode()
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {LLM_API_KEY}")
            req.add_header("HTTP-Referer", "https://github.com/dexposure-agent")
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, http.client.RemoteDisconnected) as e:
            if attempt < 4:
                wait = 10 * (attempt + 1)
                logger.warning(f"LLM API attempt {attempt+1} failed: {type(e).__name__}: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"LLM API call failed after 5 attempts: {e}")
                return {"error": str(e), "risk_level": "unknown",
                        "target_protocols": [], "rationale": "", "raw_response": ""}
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:500]
            logger.error(f"LLM API HTTP {e.code}: {body}")
            if attempt < 4 and e.code in (429, 500, 502, 503):
                wait = 15 * (attempt + 1)
                logger.warning(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return {"error": f"HTTP {e.code}: {body}", "risk_level": "unknown",
                        "target_protocols": [], "rationale": "", "raw_response": ""}

    latency_ms = (time.time() - t0) * 1000

    raw = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # Parse JSON (handle markdown wrapper)
    text = raw
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed, raw={text[:200]}")
        parsed = {"risk_level": "unknown", "target_protocols": [],
                  "rationale": raw[:500]}

    usage = result.get("usage", {})
    parsed["raw_response"] = raw
    parsed["input_tokens"] = usage.get("prompt_tokens", 0)
    parsed["output_tokens"] = usage.get("completion_tokens", 0)
    parsed["latency_ms"] = round(latency_ms, 1)
    parsed["model"] = api_model
    return parsed


# ---------------------------------------------------------------------------
# Layer 2 metrics: LLM reasoning quality
# ---------------------------------------------------------------------------

def compute_grounding_score(llm_output: dict, input_data: str) -> float:
    """Fraction of target reasons that cite numeric data present in the input.

    A reason is "grounded" if it contains at least one number AND a metric
    keyword, and that number actually appears in the input prompt.
    """
    protos = llm_output.get("target_protocols", [])
    if not protos:
        return 0.0

    # Extract all numbers from input data for verification
    import re
    input_numbers = set()
    for match in re.findall(r'[\d]+\.[\d]+|[\d]+', input_data):
        input_numbers.add(match)

    metric_keywords = {"M1", "M3", "M4", "M6", "M7", "loss", "weight",
                       "pagerank", "gini", "hhi", "density", "%", "distressed",
                       "affected", "exposure"}

    grounded = 0
    for p in protos:
        reason = p.get("reason", "") + " " + llm_output.get("rationale", "")
        reason_numbers = set(re.findall(r'[\d]+\.[\d]+|[\d]+', reason))
        has_metric = any(kw.lower() in reason.lower() for kw in metric_keywords)
        has_traceable_number = bool(reason_numbers & input_numbers)
        if has_metric and has_traceable_number:
            grounded += 1
    return grounded / len(protos)


def compute_consistency(run_outputs: list[dict]) -> float:
    """Mean pairwise Jaccard of target sets across repeated runs."""
    if len(run_outputs) < 2:
        return 1.0
    tsets = [{p["protocol"] for p in o.get("target_protocols", [])} for o in run_outputs]
    jacs = [_jaccard(tsets[i], tsets[j])
            for i in range(len(tsets)) for j in range(i + 1, len(tsets))]
    return float(np.mean(jacs)) if jacs else 1.0


def apply_action_gate(llm_output: dict) -> dict:
    """Apply conservative intervention constraints to an LLM decision.

    This is deliberately a post-hoc safety gate, not a replacement for the
    deterministic rules engine. It lets us separate target-selection quality
    from the LLM's tendency to over-escalate intervention severity.
    """
    gated = json.loads(json.dumps(llm_output))
    risk_level = str(gated.get("risk_level", "")).lower()
    targets = gated.get("target_protocols", [])
    if not isinstance(targets, list):
        gated["target_protocols"] = []
        return gated

    for target in targets:
        if not isinstance(target, dict):
            continue
        action = target.get("action", "Monitor")
        try:
            risk_score = float(target.get("risk_score", 0.0))
        except (TypeError, ValueError):
            risk_score = 0.0

        if action == "Contingency":
            if risk_level != "critical" or risk_score < 0.90:
                target["action"] = "Investigate"
                target["gate_note"] = (
                    "Demoted from Contingency: Contingency requires critical "
                    "overall risk and protocol risk_score >= 0.90."
                )
        elif action == "Recommend-Reduce":
            if risk_level not in {"elevated", "critical"} or risk_score < 0.75:
                target["action"] = "Investigate"
                target["gate_note"] = (
                    "Demoted from Recommend-Reduce: intervention requires "
                    "elevated/critical overall risk and protocol risk_score >= 0.75."
                )

    return gated


# ---------------------------------------------------------------------------
# Layer 3 metrics: end-to-end decision quality
# ---------------------------------------------------------------------------

def normalize_protocol_name(name: str) -> str:
    """Normalize protocol names before matching with ground truth.

    LLM outputs may include a trailing category suffix such as
    "AAVE (Lending)" while ground truth uses "AAVE". Remove this suffix so
    formatting differences do not bias precision/recall.
    """
    if not isinstance(name, str):
        return ""
    normalized = name.strip()
    if not normalized:
        return ""
    normalized = re.sub(r"\s*\([^)]*\)\s*$", "", normalized)
    return normalized.strip()


def assess_week(llm_output: dict, truly_stressed: set[str],
                actual_losses: dict[str, float],
                user_prompt: str) -> dict:
    """Compute all Layer 2 + Layer 3 metrics for one week."""
    targets = {
        normalize_protocol_name(p.get("protocol", ""))
        for p in llm_output.get("target_protocols", [])
    }
    targets.discard("")
    truly_stressed_norm = {normalize_protocol_name(p) for p in truly_stressed}
    truly_stressed_norm.discard("")

    # Precision: fraction of flagged that are truly stressed
    precision = len(targets & truly_stressed_norm) / len(targets) if targets else 0.0
    # Recall / audit completeness
    completeness = (
        len(targets & truly_stressed_norm) / len(truly_stressed_norm)
        if truly_stressed_norm
        else 1.0
    )

    # Grounding score (Layer 2)
    grounding = compute_grounding_score(llm_output, user_prompt)

    # Severity correlation (Layer 3):
    # Does the LLM assign higher severity to protocols that actually lost more?
    severity_rho = float("nan")
    protos = llm_output.get("target_protocols", [])
    normalized_losses = {
        normalize_protocol_name(name): loss for name, loss in actual_losses.items()
    }
    normalized_losses.pop("", None)
    if len(protos) >= 3:
        sev_vals = []
        loss_vals = []
        for p in protos:
            name = normalize_protocol_name(p.get("protocol", ""))
            if not name:
                continue
            sev = SEVERITY_ORDER.get(p.get("action", "Monitor"), 1)
            loss = normalized_losses.get(name, 0.0)
            sev_vals.append(sev)
            loss_vals.append(loss)
        if len(set(sev_vals)) > 1:
            rho, _ = spearmanr(sev_vals, loss_vals)
            severity_rho = float(rho)

    # False intervention rate: flagging stable protocols with high-severity action
    false_interventions = 0
    intervention_targets = 0
    stable = set(normalized_losses.keys()) - truly_stressed_norm
    for p in protos:
        name = normalize_protocol_name(p.get("protocol", ""))
        if not name:
            continue
        if p.get("action") in ("Recommend-Reduce", "Contingency"):
            intervention_targets += 1
            if name in stable:
                false_interventions += 1
    fir = false_interventions / intervention_targets if intervention_targets > 0 else 0.0

    return {
        "targets": sorted(targets),
        "precision": precision,
        "completeness": completeness,
        "grounding_score": grounding,
        "severity_rho": severity_rho,
        "false_intervention_rate": fir,
    }


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


# ---------------------------------------------------------------------------
# LLM-as-Judge: explanation quality
# ---------------------------------------------------------------------------

def judge_explanation(date: str, horizon: int, metrics: dict,
                     truly_stressed: set[str], llm_output: dict) -> dict:
    """Rate the LLM's explanation quality using a cheaper judge model."""
    system = JUDGE_SYSTEM_PROMPT
    user = JUDGE_USER_TEMPLATE.format(
        date=date, horizon=horizon,
        n_nodes=metrics.get("n_nodes", 0),
        n_edges=metrics.get("n_edges", 0),
        metrics_summary=", ".join(f"{k}={v}" for k, v in metrics.items() if k.startswith("M")),
        n_stressed=len(truly_stressed),
        stressed_list=", ".join(sorted(truly_stressed)[:10]) or "none",
        risk_level=llm_output.get("risk_level", "?"),
        llm_targets=", ".join(p["protocol"] for p in llm_output.get("target_protocols", [])[:10]),
        rationale=llm_output.get("rationale", "N/A"),
    )

    result = call_llm(system, user, model=JUDGE_MODEL)
    score = result.get("quality_score", 0)
    if isinstance(score, (int, float)) and 1 <= score <= 5:
        return {"quality_score": int(score), "reasoning": result.get("reasoning", "")}
    return {"quality_score": 3, "reasoning": "parse_fallback"}


# ---------------------------------------------------------------------------
# Result dataclass
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
    severity_rho: float = float("nan")
    false_intervention_rate: float = 0.0
    explanation_quality: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Checkpoint / resume support
# ---------------------------------------------------------------------------

def _checkpoint_path(run_dir: Path, method: str) -> Path:
    return run_dir / f"checkpoint_{method}.json"


def _save_checkpoint(run_dir: Path, method: str, completed_dates: list[str],
                     week_results: list[WeekResult], raw_entries: list[dict]):
    cp = {
        "method": method,
        "completed_dates": completed_dates,
        "week_results": [asdict(wr) for wr in week_results],
        "raw_entries": raw_entries,
    }
    _checkpoint_path(run_dir, method).write_text(json.dumps(cp, indent=2, default=str))


def _load_checkpoint(run_dir: Path, method: str) -> tuple[set[str], list[WeekResult], list[dict]]:
    cp_path = _checkpoint_path(run_dir, method)
    if not cp_path.exists():
        return set(), [], []
    cp = json.loads(cp_path.read_text())
    wrs = [WeekResult(**wr) for wr in cp.get("week_results", [])]
    return set(cp.get("completed_dates", [])), wrs, cp.get("raw_entries", [])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    methods: list[str],
    test_split: str = "2025-01~2025-08",
    horizon: int = STRESS_LOOKAHEAD,
    consistency_runs: int = CONSISTENCY_RUNS,
    resume: bool = False,
    run_judge: bool = True,
) -> dict[str, list[WeekResult]]:
    """Run LLM evaluation pipeline locally.

    Connects to FM API via SSH tunnel, calls Claude API for decisions.
    """
    # Verify API key
    if not LLM_API_KEY:
        logger.error("No LLM API key set. Export OPENROUTER_API_KEY or ANTHROPIC_API_KEY.")
        sys.exit(1)

    # Verify FM API
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

    # Setup run directory (resume reuses the latest existing llm_eval_* dir)
    if resume:
        existing = sorted(RESULTS_DIR.glob("llm_eval_*/"), key=lambda p: p.name, reverse=True)
        if existing and any((existing[0] / f"checkpoint_{m}.json").exists() for m in methods):
            run_dir = existing[0]
            logger.info(f"Resuming from existing run: {run_dir.name}")
        else:
            run_dir = RESULTS_DIR / f"llm_eval_{time.strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = RESULTS_DIR / f"llm_eval_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "eval.log"
    log_id = logger.add(str(log_file), level="DEBUG",
                        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}")

    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Decision model: {DECISION_MODEL}")
    logger.info(f"Judge model: {JUDGE_MODEL}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Consistency runs: {consistency_runs}")

    # Get dates
    test_dates = get_test_dates(test_split)
    all_dates = _http_get("/dates")["dates"]
    logger.info(f"Test dates: {len(test_dates)} weeks ({test_dates[0]} to {test_dates[-1]})")

    # Results
    results: dict[str, list[WeekResult]] = {}
    raw_logs: dict[str, list[dict]] = {}
    total_cost = 0.0
    costs = MODEL_COSTS.get(DECISION_MODEL, {"input": 3.0, "output": 15.0})
    judge_costs = MODEL_COSTS.get(JUDGE_MODEL, {"input": 1.0, "output": 5.0})

    for method in methods:
        # Resume support
        done_dates, prev_results, prev_raw = set(), [], []
        if resume:
            done_dates, prev_results, prev_raw = _load_checkpoint(run_dir, method)
            if done_dates:
                logger.info(f"{method}: resuming, {len(done_dates)} weeks already done")
        results[method] = list(prev_results)
        raw_logs[method] = list(prev_raw)

        for wi, date_str in enumerate(test_dates):
            if date_str in done_dates:
                continue

            # Find future date for ground truth
            if date_str not in all_dates:
                continue
            t_idx = all_dates.index(date_str)
            future_idx = t_idx + horizon
            if future_idx >= len(all_dates):
                logger.info(f"[{wi+1}/{len(test_dates)}] {date_str}: no future snapshot, skip")
                continue
            future_date = all_dates[future_idx]

            # Fetch data from GPU server
            snap_current = get_snapshot(date_str)
            snap_future = get_snapshot(future_date)
            truly_stressed = detect_truly_stressed(snap_current, snap_future)
            actual_losses = compute_actual_loss(snap_current, snap_future)

            logger.info(f"[{wi+1}/{len(test_dates)}] {date_str} -> {future_date}: "
                        f"{len(truly_stressed)} truly stressed")

            # Collect data for prompt
            forecast = None
            scenarios = None
            if method in ("m6_fm_llm", "m7_fm_llm_gated"):
                forecast = get_forecast(date_str, horizon)
                metrics = compute_metrics(forecast)
                scenarios = run_scenarios(forecast)
            elif method == "m2_snapshot_llm":
                metrics = compute_metrics(snap_current)

            system, user = build_prompt(
                method, date_str, horizon, forecast, snap_current, metrics, scenarios)

            # Run LLM (multiple times for consistency)
            run_outputs = []
            for ri in range(consistency_runs):
                out = call_llm(system, user)
                if method in GATED_LLM_METHODS:
                    out = apply_action_gate(out)
                run_outputs.append(out)
                cost_usd = (out.get("input_tokens", 0) * costs["input"] +
                            out.get("output_tokens", 0) * costs["output"]) / 1_000_000
                total_cost += cost_usd

            primary = run_outputs[0]
            ev = assess_week(primary, truly_stressed, actual_losses, user)
            consistency = compute_consistency(run_outputs)

            # LLM-as-Judge
            explain_quality = 0
            if run_judge:
                judge_result = judge_explanation(
                    date_str, horizon, metrics, truly_stressed, primary)
                explain_quality = judge_result.get("quality_score", 0)
                j_cost = (judge_result.get("input_tokens", 0) * judge_costs["input"] +
                          judge_result.get("output_tokens", 0) * judge_costs["output"]) / 1_000_000
                total_cost += j_cost

            wr = WeekResult(
                date=date_str, method=method,
                risk_level=primary.get("risk_level", "unknown"),
                targets=ev["targets"],
                truly_stressed=sorted(truly_stressed),
                precision=ev["precision"],
                completeness=ev["completeness"],
                grounding_score=ev["grounding_score"],
                consistency=consistency,
                severity_rho=ev["severity_rho"],
                false_intervention_rate=ev["false_intervention_rate"],
                explanation_quality=explain_quality,
                input_tokens=primary.get("input_tokens", 0),
                output_tokens=primary.get("output_tokens", 0),
                latency_ms=primary.get("latency_ms", 0),
            )
            results[method].append(wr)

            raw_entry = {
                "date": date_str, "method": method,
                "system_prompt": system, "user_prompt": user,
                "llm_outputs": run_outputs,
                "action_gate_applied": method in GATED_LLM_METHODS,
                "truly_stressed": sorted(truly_stressed),
                "assessment": ev, "consistency": consistency,
                "explanation_quality": explain_quality,
            }
            raw_logs[method].append(raw_entry)

            logger.info(f"  {method}: prec={ev['precision']:.3f} "
                        f"compl={ev['completeness']:.3f} "
                        f"ground={ev['grounding_score']:.3f} "
                        f"consist={consistency:.3f} "
                        f"sev_rho={ev['severity_rho']:.3f} "
                        f"FIR={ev['false_intervention_rate']:.3f} "
                        f"explain={explain_quality}/5 "
                        f"targets={len(ev['targets'])}")

            # Checkpoint after each week
            done_dates.add(date_str)
            _save_checkpoint(run_dir, method,
                             sorted(done_dates), results[method], raw_logs[method])

    # ---------------------------------------------------------------------------
    # Aggregate summaries
    # ---------------------------------------------------------------------------

    summaries = {}
    for method, wrs in results.items():
        if not wrs:
            continue

        precs = [w.precision for w in wrs]
        compls = [w.completeness for w in wrs]
        grounds = [w.grounding_score for w in wrs]
        consists = [w.consistency for w in wrs]
        sev_rhos = [w.severity_rho for w in wrs if not np.isnan(w.severity_rho)]
        firs = [w.false_intervention_rate for w in wrs]
        explains = [w.explanation_quality for w in wrs if w.explanation_quality > 0]
        stabilities = [_jaccard(set(wrs[i - 1].targets), set(wrs[i].targets))
                       for i in range(1, len(wrs))]

        summary = {
            "method": method,
            "model": DECISION_MODEL,
            "n_weeks": len(wrs),
            # Layer 3
            "ticket_precision": round(float(np.mean(precs)), 4),
            "audit_completeness": round(float(np.mean(compls)), 4),
            "target_stability": round(float(np.mean(stabilities)), 4) if stabilities else 0.0,
            "severity_correlation": round(float(np.mean(sev_rhos)), 4) if sev_rhos else float("nan"),
            "false_intervention_rate": round(float(np.mean(firs)), 4),
            # Layer 2
            "grounding_score": round(float(np.mean(grounds)), 4),
            "consistency": round(float(np.mean(consists)), 4),
            "explanation_quality": round(float(np.mean(explains)), 2) if explains else 0.0,
            # Cost
            "total_input_tokens": sum(w.input_tokens for w in wrs),
            "total_output_tokens": sum(w.output_tokens for w in wrs),
        }
        summaries[method] = summary

        logger.info(f"\n{'='*60}\n{method} SUMMARY ({len(wrs)} weeks):")
        for k, v in summary.items():
            if k == "method":
                continue
            logger.info(f"  {k}: {v}")

    # Save results
    for method, summary in summaries.items():
        with open(run_dir / f"summary_{method}.json", "w") as f:
            json.dump(summary, f, indent=2)

    for method, raw in raw_logs.items():
        with open(run_dir / f"raw_{method}.json", "w") as f:
            json.dump(raw, f, indent=2, default=str)

    # Combined comparison table
    comparison = {"run_id": run_dir.name, "methods": summaries}
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison table
    _print_comparison_table(summaries)

    logger.info(f"\nTotal estimated cost: ${total_cost:.2f}")
    logger.info(f"Results saved to {run_dir}/")

    logger.remove(log_id)
    return results


def _print_comparison_table(summaries: dict[str, dict]):
    """Print a formatted comparison table to stdout and logger."""
    methods = list(summaries.keys())
    if not methods:
        return

    header = f"{'Method':<12} {'Prec':>6} {'Recall':>7} {'Stabil':>7} {'Ground':>7} {'Consist':>8} {'SevRho':>7} {'FIR':>6} {'Explain':>8}"
    sep = "-" * len(header)
    lines = ["\n" + sep, header, sep]

    for m in methods:
        s = summaries[m]
        sev = f"{s['severity_correlation']:.3f}" if not np.isnan(s.get("severity_correlation", float("nan"))) else "  N/A"
        lines.append(
            f"{m:<12} "
            f"{s['ticket_precision']:>6.3f} "
            f"{s['audit_completeness']:>7.3f} "
            f"{s['target_stability']:>7.3f} "
            f"{s['grounding_score']:>7.3f} "
            f"{s['consistency']:>8.3f} "
            f"{sev:>7} "
            f"{s['false_intervention_rate']:>6.3f} "
            f"{s.get('explanation_quality', 0):>7.1f}/5"
        )
    lines.append(sep)
    table_str = "\n".join(lines)
    print(table_str)
    logger.info(table_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="LLM Decision Evaluation Pipeline (runs locally)")
    parser.add_argument("--method", action="append", default=[],
                        help="m6_fm_llm, m7_fm_llm_gated, and/or m2_snapshot_llm (can repeat; default: all three)")
    parser.add_argument("--test-split", default="2025-01~2025-08")
    parser.add_argument("--horizon", type=int, default=STRESS_LOOKAHEAD)
    parser.add_argument("--consistency-runs", type=int, default=CONSISTENCY_RUNS)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-Judge evaluation")
    parser.add_argument("--model", default=None,
                        help=f"Decision model (default: {DECISION_MODEL})")
    parser.add_argument("--judge-model", default=None,
                        help=f"Judge model (default: {JUDGE_MODEL})")
    args = parser.parse_args()

    if not args.method:
        args.method = ["m6_fm_llm", "m7_fm_llm_gated", "m2_snapshot_llm"]
    if args.model:
        DECISION_MODEL = args.model
    if args.judge_model:
        JUDGE_MODEL = args.judge_model

    run_pipeline(
        methods=args.method,
        test_split=args.test_split,
        horizon=args.horizon,
        consistency_runs=args.consistency_runs,
        resume=args.resume,
        run_judge=not args.no_judge,
    )

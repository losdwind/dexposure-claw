#!/usr/bin/env python3
"""m2_snapshot_llm: Pure LLM-Agent competitor (no FM backbone).

Calls Claude with a text-only description of the current DeFi network.
No forward-looking FM predictions, no predicted graph, no scenario analysis
on predicted graph. The LLM must reason from the raw current snapshot alone.

Used by llm_eval_b5.py for the m2_snapshot_llm vs m6_fm_llm comparison.
Can also be called standalone for single-week analysis.

Usage:
    # Requires SSH tunnel: ssh -f -N -L 8000:localhost:8000 gpu-server
    # Requires ANTHROPIC_API_KEY
    python DeXposure_Agent/experiments/competitors/llm_agent.py --date 2025-03-03 --horizon 4
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


@dataclass
class LLMAgentConfig:
    model: str = "claude-sonnet-4-6"
    horizon: int = 4
    max_tokens: int = 4096
    temperature: float = 0.0
    api_key_env: str = "ANTHROPIC_API_KEY"


@dataclass
class LLMAgentPrediction:
    """Output format shared across all agent-level competitors."""
    method_id: str = "m2_snapshot_llm"
    horizon: int = 4
    risk_level: str = ""
    risk_scores: dict[str, float] = field(default_factory=dict)
    recommended_actions: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    raw_response: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


SYSTEM_PROMPT = """You are a DeFi systemic risk analyst for a regulatory supervisory body.
You will be given a summary of the CURRENT DeFi protocol network state as tabular metrics.
You do NOT have access to any predictive model. You must reason from current data only.
Your task is to identify protocols at elevated risk of distress over the specified horizon.

Respond with valid JSON only:
{{
  "risk_level": "low" | "moderate" | "elevated" | "critical",
  "target_protocols": [
    {{
      "protocol": "<name>",
      "risk_score": <float 0-1>,
      "action": "Monitor" | "Investigate" | "Recommend-Reduce" | "Contingency",
      "reason": "<explanation citing specific data>"
    }}
  ],
  "rationale": "<2-3 sentence overall assessment>"
}}"""


USER_TEMPLATE = """Current DeFi network state ({date}), forecast horizon = {horizon} weeks.

NOTE: No predictive model available. Reason from current snapshot only.

== CURRENT NETWORK ==
Protocols: {n_nodes} | Edges: {n_edges} | Total weight: {total_weight:.2f}

Top-10 protocols by exposure weight:
{top_protocols}

Category breakdown:
{category_summary}

== NETWORK METRICS ==
{metrics}

Identify protocols at elevated risk over the next {horizon} weeks."""


def _summarize_snapshot(snapshot: dict) -> dict[str, Any]:
    """Extract summary fields from a raw snapshot dict (from /snapshot API)."""
    nodes = snapshot.get("nodes", {})
    edges = snapshot.get("edges", [])
    nw: dict[str, float] = defaultdict(float)
    for e in edges:
        nw[e["source"]] += e["weight"]
        nw[e["target"]] += e["weight"]
    top = sorted(nw.items(), key=lambda x: x[1], reverse=True)[:10]

    top_lines = []
    for i, (nid, w) in enumerate(top, 1):
        cat = nodes.get(nid, {}).get("category", "?")
        top_lines.append(f"  {i}. {nid} ({cat}) -- weight: {w:.2f}")

    cats: dict[str, int] = defaultdict(int)
    for nid, nf in nodes.items():
        cats[nf.get("category", "Unknown")] += 1
    cat_lines = [f"  {c}: {n} protocols"
                 for c, n in sorted(cats.items(), key=lambda x: x[1], reverse=True)[:10]]

    # Simple metrics
    n = len(nodes)
    deg_vals = list(nw.values())
    total_wd = sum(deg_vals) if deg_vals else 1.0
    hhi = sum((d / total_wd) ** 2 for d in deg_vals) if total_wd > 0 else 0
    density = len(edges) / (n * (n - 1)) if n > 1 else 0

    return {
        "n_nodes": n,
        "n_edges": len(edges),
        "total_weight": sum(e["weight"] for e in edges),
        "top_protocols": "\n".join(top_lines),
        "category_summary": "\n".join(cat_lines),
        "metrics": f"  N2_hhi: {hhi:.6f}\n  N3_density: {density:.6f}\n  N5_degree_gini: {_gini(deg_vals):.6f}",
    }


def _gini(vals):
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


def run_llm_agent(
    snapshot: dict,
    config: LLMAgentConfig | None = None,
    date: str = "",
) -> LLMAgentPrediction:
    """Run the pure LLM agent (m2_snapshot_llm) on a raw snapshot dict.

    Args:
        snapshot: Raw snapshot dict from /snapshot API (nodes, edges).
        config: LLMAgentConfig (uses defaults if None).
        date: Snapshot date string.

    Returns:
        LLMAgentPrediction with risk scores and recommended actions.
    """
    if config is None:
        config = LLMAgentConfig()

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        logger.error(f"Environment variable {config.api_key_env!r} not set")
        return LLMAgentPrediction(rationale="API key not set")

    import anthropic
    client = anthropic.Anthropic()

    summary = _summarize_snapshot(snapshot)
    system = SYSTEM_PROMPT.format(horizon=config.horizon)
    user = USER_TEMPLATE.format(date=date, horizon=config.horizon, **summary)

    t0 = time.time()
    try:
        with client.messages.stream(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            response = stream.get_final_message()
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return LLMAgentPrediction(rationale=f"API error: {e}")

    latency_ms = (time.time() - t0) * 1000

    raw = ""
    for block in response.content:
        if block.type == "text":
            raw = block.text.strip()
            break

    # Parse JSON
    text = raw
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed: {text[:200]}")
        parsed = {"risk_level": "unknown", "target_protocols": [], "rationale": raw[:500]}

    risk_scores = {}
    actions = []
    for p in parsed.get("target_protocols", []):
        name = p.get("protocol", "")
        risk_scores[name] = p.get("risk_score", 0.5)
        actions.append({
            "protocol": name,
            "action": p.get("action", "Monitor"),
            "reason": p.get("reason", ""),
        })

    return LLMAgentPrediction(
        method_id="m2_snapshot_llm",
        horizon=config.horizon,
        risk_level=parsed.get("risk_level", "unknown"),
        risk_scores=risk_scores,
        recommended_actions=actions,
        rationale=parsed.get("rationale", ""),
        raw_response=raw,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=round(latency_ms, 1),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="m2_snapshot_llm: Pure LLM-Agent (no FM)")
    parser.add_argument("--date", required=True, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--fm-api", default="http://localhost:8000",
                        help="FM API URL for fetching snapshots")
    args = parser.parse_args()

    import urllib.request
    url = f"{args.fm_api}/snapshot?date={args.date}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        snapshot = json.loads(resp.read())

    cfg = LLMAgentConfig(horizon=args.horizon, model=args.model)
    pred = run_llm_agent(snapshot, cfg, date=args.date)
    print(json.dumps({
        "method": pred.method_id,
        "risk_level": pred.risk_level,
        "risk_scores": pred.risk_scores,
        "actions": pred.recommended_actions,
        "rationale": pred.rationale,
        "tokens": {"input": pred.input_tokens, "output": pred.output_tokens},
        "latency_ms": pred.latency_ms,
    }, indent=2))

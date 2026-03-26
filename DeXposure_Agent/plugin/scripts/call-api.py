#!/usr/bin/env python3
"""Unified API client for the DeXposure-FM server.

Wraps all FM API endpoints. Used by plugin skills, commands, and agents.

Usage:
    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py health
    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py dates
    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py predict --date 2025-01-06 --horizon 4
    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py predict --date 2025-01-06 --horizon 4 --compact
    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py metrics --date 2025-01-06 --horizon 4
    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py stress --date 2025-01-06 --horizon 4
    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py stress --date 2025-01-06 --horizon 4 --scenario S1
    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py batch --date 2025-01-06

Environment:
    FM_API_URL: Base URL for the FM server (default: http://localhost:8000)
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

try:
    import httpx
except ImportError:
    import urllib.request
    httpx = None

API_URL = os.environ.get("FM_API_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path):
    if httpx:
        with httpx.Client(base_url=API_URL, timeout=30) as c:
            r = c.get(path)
            r.raise_for_status()
            return r.json()
    else:
        req = urllib.request.Request(f"{API_URL}{path}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

def _post(path, data):
    if httpx:
        with httpx.Client(base_url=API_URL, timeout=180) as c:
            r = c.post(path, json=data)
            r.raise_for_status()
            return r.json()
    else:
        req = urllib.request.Request(f"{API_URL}{path}", method="POST")
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Metrics computation (self-contained, no external deps)
# ---------------------------------------------------------------------------

def _gini(values):
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    s = sorted(values)
    total = sum(s)
    cumsum = gini_sum = 0.0
    for v in s:
        cumsum += v
        gini_sum += cumsum
    return 1.0 - (2.0 * gini_sum) / (n * total)

def _compute_metrics(data):
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    n = len(nodes)
    if n == 0:
        return {"M1": 0, "M3": 0, "M4": 0, "M6": 0, "M7": 0}
    adj = {nid: {} for nid in nodes}
    wdeg = {nid: 0.0 for nid in nodes}
    for e in edges:
        src, tgt, w = e["source"], e["target"], e["weight"]
        if src in adj:
            adj[src][tgt] = adj[src].get(tgt, 0.0) + w
        if src in wdeg:
            wdeg[src] += w
    deg_vals = list(wdeg.values())
    total_wd = sum(deg_vals)
    # Simple PageRank
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
    return {
        "date": data.get("date", ""),
        "horizon": data.get("horizon", 0),
        "n_nodes": n,
        "n_edges": len(edges),
        "M1_max_pagerank": round(max(pr_vals), 8),
        "M3_hhi": round(sum((d/total_wd)**2 for d in deg_vals) if total_wd > 0 else 0, 8),
        "M4_density": round(len(edges) / (n*(n-1)) if n > 1 else 0, 8),
        "M6_pagerank_gini": round(_gini(pr_vals), 8),
        "M7_degree_gini": round(_gini(deg_vals), 8),
    }


# ---------------------------------------------------------------------------
# Stress scenario computation (self-contained)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "S1": {"name": "Top protocol failure", "type": "top_node", "shock_pct": 1.0, "count": 1},
    "S2": {"name": "Bridge cluster failure", "type": "category", "categories": ["Bridge", "Cross Chain"], "shock_pct": 1.0},
    "S3": {"name": "Stablecoin de-peg", "type": "category", "categories": ["Algo-Stables", "Decentralized Stablecoin", "CDP"], "shock_pct": 0.5},
    "S4": {"name": "Lending sector shock", "type": "category", "categories": ["Lending", "Uncollateralized Lending", "RWA Lending", "NFT Lending"], "shock_pct": 0.3},
    "S5": {"name": "Correlated stress (top-10)", "type": "top_nodes", "shock_pct": 0.2, "count": 10},
}

def _run_scenarios(data, scenario_filter=None):
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    nw = defaultdict(float)
    for e in edges:
        nw[e["source"]] += e["weight"]
        nw[e["target"]] += e["weight"]
    results = []
    for sid, spec in SCENARIOS.items():
        if scenario_filter and sid != scenario_filter:
            continue
        shock_type, shock_pct = spec["type"], spec["shock_pct"]
        if shock_type in ("top_node", "top_nodes"):
            cnt = spec.get("count", 1)
            shocked = {n for n, _ in sorted(nw.items(), key=lambda x: x[1], reverse=True)[:cnt]}
        elif shock_type == "category":
            cats = spec.get("categories", [spec["category"]] if "category" in spec else [])
            cats_lower = {c.lower() for c in cats}
            shocked = {nid for nid, nf in nodes.items() if nf.get("category", "").lower() in cats_lower}
        else:
            shocked = set()
        new_edges = []
        for e in edges:
            if e["source"] in shocked or e["target"] in shocked:
                nw2 = e["weight"] * (1 - shock_pct)
                if nw2 > 0:
                    new_edges.append({**e, "weight": nw2})
            else:
                new_edges.append(e)
        # Compute loss
        sum_orig = sum(nw.values())
        sw = defaultdict(float)
        for e in new_edges:
            sw[e["source"]] += e["weight"]
            sw[e["target"]] += e["weight"]
        sum_shock = sum(sw.values())
        loss = (sum_orig - sum_shock) / sum_orig if sum_orig > 0 else 0
        distressed = sum(1 for nid, ow in nw.items() if ow > 0 and (ow - sw.get(nid, 0)) / ow > 0.5)
        affected = sum(1 for nid in nw if nw[nid] - sw.get(nid, 0) > 0)
        depth = int(math.floor(math.log2(affected + 1))) + 1 if affected > 0 else 0
        top_aff = sorted(((nid, nw[nid] - sw.get(nid, 0)) for nid in nw if nw[nid] - sw.get(nid, 0) > 0),
                         key=lambda x: x[1], reverse=True)[:10]
        results.append({
            "scenario_id": sid, "scenario_name": spec["name"],
            "date": data.get("date", ""), "horizon": data.get("horizon", 0),
            "system_loss_pct": round(loss * 100, 4),
            "distressed_count": distressed, "propagation_depth": depth, "affected_nodes": affected,
            "top_affected": [{"protocol": nid, "weight_loss": round(d, 2),
                              "category": nodes.get(nid, {}).get("category", "unknown")} for nid, d in top_aff],
        })
    return results


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_health():
    print(json.dumps(_get("/health"), indent=2))

def cmd_dates():
    data = _get("/dates")
    test = [d for d in data["dates"] if d >= "2025-01-01"]
    print(json.dumps({"count": data["count"], "first": data["dates"][0], "last": data["dates"][-1],
                       "test_dates_count": len(test), "test_dates": test}, indent=2))

def cmd_predict(date, horizon, compact=False):
    data = _post("/forecast", {"date": date, "horizon": horizon})
    if compact:
        nw = defaultdict(float)
        cats = {}
        for nid, nf in data.get("nodes", {}).items():
            cats[nid] = nf.get("category", "unknown")
        for e in data.get("edges", []):
            nw[e["source"]] += e["weight"]
            nw[e["target"]] += e["weight"]
        top = sorted(nw.items(), key=lambda x: x[1], reverse=True)[:10]
        print(json.dumps({
            "date": data["date"], "horizon": data["horizon"],
            "n_nodes": data["n_nodes"], "n_edges": data["n_edges"], "elapsed_ms": data["elapsed_ms"],
            "top_10": [{"protocol": n, "weight": round(w, 2), "category": cats.get(n, "?")} for n, w in top],
        }, indent=2))
    else:
        print(json.dumps(data))

def cmd_metrics(date, horizon):
    data = _post("/forecast", {"date": date, "horizon": horizon})
    print(json.dumps(_compute_metrics(data), indent=2))

def cmd_stress(date, horizon, scenario=None):
    data = _post("/forecast", {"date": date, "horizon": horizon})
    print(json.dumps(_run_scenarios(data, scenario), indent=2))

def cmd_batch(date):
    data = _post("/batch-forecast", {"date": date, "horizons": [1, 4, 8, 12]})
    summary = [{"horizon": r["horizon"], "n_nodes": r["n_nodes"], "n_edges": r["n_edges"],
                "elapsed_ms": r["elapsed_ms"]} for r in data]
    print(json.dumps({"date": date, "predictions": summary}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DeXposure-FM API client")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("health")
    sub.add_parser("dates")

    pp = sub.add_parser("predict")
    pp.add_argument("--date", required=True)
    pp.add_argument("--horizon", type=int, default=4)
    pp.add_argument("--compact", action="store_true")

    pm = sub.add_parser("metrics")
    pm.add_argument("--date", required=True)
    pm.add_argument("--horizon", type=int, default=4)

    ps = sub.add_parser("stress")
    ps.add_argument("--date", required=True)
    ps.add_argument("--horizon", type=int, default=4)
    ps.add_argument("--scenario", choices=list(SCENARIOS.keys()))

    pb = sub.add_parser("batch")
    pb.add_argument("--date", required=True)

    args = p.parse_args()
    try:
        {"health": cmd_health, "dates": cmd_dates,
         "predict": lambda: cmd_predict(args.date, args.horizon, getattr(args, "compact", False)),
         "metrics": lambda: cmd_metrics(args.date, args.horizon),
         "stress": lambda: cmd_stress(args.date, args.horizon, getattr(args, "scenario", None)),
         "batch": lambda: cmd_batch(args.date),
        }[args.cmd]()
    except KeyError:
        p.print_help()
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

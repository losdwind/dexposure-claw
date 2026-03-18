#!/usr/bin/env python3
"""Thin httpx wrapper for calling the DeXposure-Agent API from plugin skills.

Usage:
    python call-api.py forecast --date 2025-01-01 --horizon 4
    python call-api.py health
    python call-api.py run-epoch --date 2025-01-01
"""
import argparse
import json
import sys

try:
    import httpx
except ImportError:
    import urllib.request
    # Fallback for environments without httpx
    httpx = None

DEFAULT_BASE = "http://gpu-server:8000"


def _request_fallback(method, url, data=None):
    """Fallback using urllib when httpx is not available."""
    req = urllib.request.Request(url, method=method)
    if data:
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="DeXposure-Agent API client")
    parser.add_argument("command", choices=["health", "forecast", "run-epoch", "models"])
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--date", help="Epoch date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--output", choices=["json", "summary"], default="summary")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    if httpx:
        client = httpx.Client(base_url=base, timeout=300)
        def get(path): return client.get(path).json()
        def post(path, data): return client.post(path, json=data).json()
    else:
        def get(path): return _request_fallback("GET", f"{base}{path}")
        def post(path, data): return _request_fallback("POST", f"{base}{path}", data)

    if args.command == "health":
        print(json.dumps(get("/health"), indent=2))
    elif args.command == "models":
        print(json.dumps(get("/models"), indent=2))
    elif args.command == "forecast":
        data = post("/forecast", {"date": args.date, "horizon": args.horizon})
        if args.output == "summary":
            n_edges = len(data.get("edge_probs", {}))
            print(f"Forecast for h={args.horizon}: {n_edges} edges predicted")
        else:
            print(json.dumps(data, indent=2))
    elif args.command == "run-epoch":
        data = post("/run-epoch", {"date": args.date})
        if args.output == "summary":
            n_alerts = len(data.get("alerts", []))
            n_tickets = len(data.get("tickets", []))
            print(f"Epoch {args.date}: {n_alerts} alerts, {n_tickets} tickets")
            if data.get("suppressed"):
                print("WARNING: Safe mode active — interventions suppressed")
        else:
            print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
